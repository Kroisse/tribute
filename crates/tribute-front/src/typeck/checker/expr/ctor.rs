//! Constructor applications, record literals, and their field shapes.

use rustc_hash::FxHashSet as HashSet;

use trunk_ir::Symbol;

use crate::ast::{Expr, FieldInit, NodeId, ResolvedRef, Type, TypeKind};
use crate::typeck::checker::TypeChecker;
use crate::typeck::func_context::{ConstructorShape, FunctionInferenceContext};
use crate::typeck::subst;

impl<'db> TypeChecker<'db> {
    /// Keep the full constructor instance, including field types and effect rows,
    /// stable across inference and conversion of this source occurrence.
    pub(super) fn instantiate_value_constructor_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node_id: NodeId,
        resolved: &ResolvedRef<'db>,
    ) -> Type<'db> {
        if let Some(ty) = ctx.get_constructor_reference_type(node_id) {
            return ty;
        }
        if let Some(deferred) = self.deferred_constructor(ctx, node_id) {
            return deferred.ctor_ty;
        }
        let resolved = &self.constructor_reference(ctx, node_id, resolved);
        let ty = match resolved {
            ResolvedRef::Module { path } => {
                self.report_module_reference(ctx, node_id, *path, "a constructor")
            }
            _ => self.infer_var_with_ctx(ctx, None, resolved),
        };
        ctx.record_constructor_reference_type(node_id, ty);
        ty
    }

    /// Infer a constructor application's type: `Some(x)`, `None`, or the
    /// implicit `.Some(x)`, whose constructor `expected` may give.
    pub(super) fn infer_constructor_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        ctor: &ResolvedRef<'db>,
        args: &[Expr<ResolvedRef<'db>>],
        expected: Option<Type<'db>>,
    ) -> Type<'db> {
        let shape = ConstructorShape::of_arity(args.len());
        let ctor = &self.decide_constructor(ctx, node, ctor, expected, shape, false);
        let ctor_ty = self.infer_var_with_ctx(ctx, Some(node), ctor);
        if args.is_empty() {
            // Unit constructor (e.g., None) - just return the constructor type
            ctor_ty
        } else {
            // Constructor with arguments (e.g., Some(x)) - treat as function call
            let arg_types = self.infer_call_args_with_ctx(ctx, ctor_ty, args);
            self.infer_call_with_ctx(ctx, ctor_ty, &arg_types, node)
        }
    }

    /// Infer a record literal's nominal type and constrain its fields and
    /// spread. An implicit constructor's type is `expected`, where known.
    pub(super) fn infer_record_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        record_id: NodeId,
        type_name: &ResolvedRef<'db>,
        fields: &[FieldInit<ResolvedRef<'db>>],
        spread: Option<&Expr<ResolvedRef<'db>>>,
        expected: Option<Type<'db>>,
    ) -> Type<'db> {
        let written: Vec<Symbol> = fields.iter().map(|f| f.name.clone()).collect();
        let type_name = &self.decide_constructor(
            ctx,
            record_id,
            type_name,
            expected,
            ConstructorShape::Fields {
                names: written,
                rest: spread.is_some(),
            },
            false,
        );
        if let Some(result) = self.infer_deferred_record_with_ctx(ctx, record_id, fields, spread) {
            return result;
        }
        let ctor_ty = self.instantiate_value_constructor_with_ctx(ctx, record_id, type_name);
        let struct_ty = if let TypeKind::Func { result, .. } = ctor_ty.kind(self.db()) {
            *result
        } else {
            ctor_ty
        };

        self.validate_record_shape_with_ctx(
            ctx,
            record_id,
            type_name,
            struct_ty,
            fields,
            spread.is_some(),
        );

        let written: Vec<Symbol> = fields.iter().map(|f| f.name.clone()).collect();
        let (_, _, variant_field_tys) =
            self.constructor_field_shape(ctx, record_id, type_name, &written);
        for (field, variant_field_ty) in fields.iter().zip(variant_field_tys) {
            let (field_name, field_expr) = (&field.name, &field.value);
            // Struct fields are read from the struct declaration; a named
            // variant's fields are its constructor instance's parameters.
            if let Some(expected_field_ty) = self
                .lookup_struct_field_type(struct_ty, field_name)
                .or(variant_field_ty)
            {
                let field_ty = self.infer_expr_with_expected(ctx, field_expr, expected_field_ty);
                ctx.constrain_coerce(field_ty, expected_field_ty, field_expr.id);
            } else {
                self.infer_expr_type_with_ctx(ctx, field_expr);
            }
        }
        if let Some(spread_expr) = spread {
            let spread_ty = self.infer_expr_type_with_ctx(ctx, spread_expr);
            ctx.constrain_coerce(spread_ty, struct_ty, spread_expr.id);
        }

        struct_ty
    }

    /// Validate declaration-owned field names once, independently of child typing.
    fn validate_record_shape_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        record_id: NodeId,
        type_name: &ResolvedRef<'db>,
        struct_ty: Type<'db>,
        fields: &[FieldInit<ResolvedRef<'db>>],
        has_spread: bool,
    ) {
        let ResolvedRef::Constructor { id, .. } = type_name else {
            return;
        };
        let (struct_id, _) = self.extract_struct_info(struct_ty);
        let Some(declared_fields) = struct_id.and_then(|id| self.env.lookup_struct_fields(id))
        else {
            self.validate_variant_record_shape_with_ctx(
                ctx, record_id, *id, struct_ty, fields, has_spread,
            );
            return;
        };
        if !ctx.mark_record_shape_checked(record_id) {
            return;
        }

        let declared: Vec<Symbol> = declared_fields
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        self.report_field_shape(
            record_id,
            format_args!("struct `{}`", self.names().path(id.path(self.db()))),
            &declared,
            fields.iter().map(|f| f.name.clone()),
            has_spread,
        );
    }

    /// Validate a record literal that constructs an enum variant: the variant
    /// must have named fields, every field must be written, and a spread is
    /// not allowed because an enum value's variant is not known statically.
    fn validate_variant_record_shape_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        record_id: NodeId,
        id: crate::ast::CtorId<'db>,
        result: Type<'db>,
        fields: &[FieldInit<ResolvedRef<'db>>],
        has_spread: bool,
    ) {
        let is_variant = matches!(
            result.kind(self.db()),
            TypeKind::Named { id, .. } if self.env.lookup_enum_variants(id.path(self.db())).is_some()
        );
        if !is_variant || !ctx.mark_record_shape_checked(record_id) {
            return;
        }
        let qualified = self.names().path(id.path(self.db()));
        let Some(declared) = self.env.lookup_constructor_field_names(id) else {
            self.report_type_error(
                record_id,
                format!(
                    "`{qualified}` has positional fields; construct it with `.{}(...)`",
                    id.name(self.db())
                ),
            );
            return;
        };
        if has_spread {
            self.report_type_error(
                record_id,
                format!(
                    "record spread is not allowed for variant `{qualified}`; write every field"
                ),
            );
        }
        self.report_field_shape(
            record_id,
            format_args!("variant `{qualified}`"),
            declared,
            fields.iter().map(|f| f.name.clone()),
            has_spread,
        );
    }

    /// Report written field names that do not fit the declared ones: unknown
    /// and duplicate fields in source order, then the first missing field in
    /// declaration order unless the rest are omitted explicitly.
    pub(in crate::typeck::checker) fn report_field_shape(
        &self,
        node: NodeId,
        owner: std::fmt::Arguments<'_>,
        declared: &[Symbol],
        written: impl IntoIterator<Item = Symbol>,
        omits_rest: bool,
    ) {
        let mut seen = HashSet::default();
        for name in written {
            if !declared.contains(&name) {
                self.report_type_error(node, format!("unknown field `{name}` for {owner}"));
            } else if !seen.insert(name.clone()) {
                self.report_type_error(node, format!("duplicate field `{name}`"));
            }
        }
        if !omits_rest && let Some(missing) = declared.iter().find(|name| !seen.contains(name)) {
            self.report_type_error(node, format!("missing field: {missing}"));
        }
    }

    /// Check that a positional constructor pattern names a constructor and
    /// has one sub-pattern per field. Reported once per pattern.
    pub(super) fn validate_variant_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        ctor: &ResolvedRef<'db>,
        ctor_ty: Type<'db>,
        written: usize,
    ) {
        match ctor {
            ResolvedRef::Constructor { .. }
                if matches!(ctor_ty.kind(self.db()), TypeKind::Error) => {}
            ResolvedRef::Constructor { id, .. } => {
                let declared = match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, .. } => params.len(),
                    _ => 0,
                };
                if written != declared && ctx.mark_record_shape_checked(pattern_id) {
                    let plural = if declared == 1 { "" } else { "s" };
                    self.report_type_error(
                        pattern_id,
                        format!(
                            "constructor `{}` expects {declared} field{plural}, but the pattern has {written}",
                            self.names().path(id.path(self.db()))
                        ),
                    );
                }
            }
            _ => self.report_non_constructor_pattern(ctx, pattern_id, ctor),
        }
    }

    /// Report a name that resolved to something other than a constructor in
    /// constructor position. Unresolved names and modules are reported where
    /// they are resolved.
    fn report_non_constructor_pattern(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        resolved: &ResolvedRef<'db>,
    ) {
        let name = match resolved {
            ResolvedRef::TypeDef { id } => self.names().path(id.path(self.db())).to_string(),
            ResolvedRef::Local { id, name } if !id.is_unresolved() => name.to_string(),
            _ => return,
        };
        if ctx.mark_record_shape_checked(pattern_id) {
            self.report_type_error(pattern_id, format!("`{name}` is not a constructor"));
        }
    }

    /// The constructor instance of a brace-form constructor pattern or record
    /// literal, the type it constructs, and the field type of each written
    /// field (`None` for a name the constructor does not declare).
    pub(super) fn constructor_field_shape(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        type_name: &ResolvedRef<'db>,
        written: &[Symbol],
    ) -> (Type<'db>, Type<'db>, Vec<Option<Type<'db>>>) {
        if let Some(deferred) = self.deferred_constructor(ctx, pattern_id) {
            let field_tys = deferred.field_types.iter().copied().map(Some).collect();
            return (deferred.ctor_ty, deferred.result_ty, field_tys);
        }
        let type_name = &self.constructor_reference(ctx, pattern_id, type_name);
        let ctor_ty = self.instantiate_value_constructor_with_ctx(ctx, pattern_id, type_name);
        let (params, result) = match ctor_ty.kind(self.db()) {
            TypeKind::Func { params, result, .. } => (params.as_slice(), *result),
            _ => (&[][..], ctor_ty),
        };
        let declared = match type_name {
            ResolvedRef::Constructor { id, .. } => self.env.lookup_constructor_field_names(*id),
            _ => None,
        };
        let field_tys = written
            .iter()
            .map(|name| {
                let index = declared?.iter().position(|declared| declared == name)?;
                params.get(index).copied()
            })
            .collect();
        (ctor_ty, result, field_tys)
    }

    /// Check a brace-form constructor pattern's field names against the
    /// constructor's declaration. Reported once per pattern.
    pub(super) fn validate_record_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        type_name: &ResolvedRef<'db>,
        result: Type<'db>,
        written: &[Symbol],
        omits_rest: bool,
    ) {
        let ResolvedRef::Constructor { id, .. } = type_name else {
            self.report_non_constructor_pattern(ctx, pattern_id, type_name);
            return;
        };
        if matches!(result.kind(self.db()), TypeKind::Error)
            || !ctx.mark_record_shape_checked(pattern_id)
        {
            return;
        }
        let qualified = self.names().path(id.path(self.db()));
        let Some(declared) = self.env.lookup_constructor_field_names(*id) else {
            self.report_type_error(
                pattern_id,
                format!(
                    "`{qualified}` has positional fields; match it with `.{}(...)`",
                    id.name(self.db())
                ),
            );
            return;
        };
        let is_variant = matches!(
            result.kind(self.db()),
            TypeKind::Named { id, .. } if self.env.lookup_enum_variants(id.path(self.db())).is_some()
        );
        let kind = if is_variant { "variant" } else { "struct" };
        self.report_field_shape(
            pattern_id,
            format_args!("{kind} `{qualified}`"),
            declared,
            written.iter().cloned(),
            omits_rest,
        );
    }

    /// Extract struct declaration identity and type arguments from a type.
    ///
    /// Returns (Some(struct_id), type_args) if the type is a Named type,
    /// otherwise (None, empty vec).
    fn extract_struct_info(
        &self,
        ty: Type<'db>,
    ) -> (Option<crate::ast::TypeDefId<'db>>, Vec<Type<'db>>) {
        match ty.kind(self.db()) {
            TypeKind::Named { id, args, .. } => (Some(*id), args.clone()),
            _ => (None, vec![]),
        }
    }

    /// Look up a struct field type from the receiver type.
    ///
    /// Given a receiver type like `Point` or `Point(Int)`, look up the field `x`
    /// and return its type with BoundVars substituted by the actual type arguments.
    pub(crate) fn lookup_struct_field_type(
        &self,
        receiver_ty: Type<'db>,
        field_name: &Symbol,
    ) -> Option<Type<'db>> {
        // Extract struct declaration identity from receiver type.
        let struct_id = match receiver_ty.kind(self.db()) {
            TypeKind::Named { id, .. } => *id,
            TypeKind::UniVar { .. } => {
                // Type not yet known - can't resolve field
                return None;
            }
            _ => return None,
        };

        // Look up field in ModuleTypeEnv
        let (type_params, field_ty) = self.env.lookup_struct_field(struct_id, field_name)?;

        let TypeKind::Named { row_args, args, .. } = receiver_ty.kind(self.db()) else {
            return None;
        };

        // Substitute BoundVars in field_ty with the type arguments
        let field_ty = if type_params.is_empty() {
            field_ty
        } else {
            self.substitute_bound_vars(field_ty, args)
        };
        // and the declaration's row binders with the row arguments.
        if row_args.is_empty() {
            return Some(field_ty);
        }
        let declaration = self.env.lookup_type_def(struct_id.path(self.db()))?;
        Some(subst::substitute_row_binders(
            self.db(),
            field_ty,
            declaration.effect_params(self.db()),
            row_args,
        ))
    }

    /// Substitute BoundVars in a type with actual types.
    ///
    /// Panics if a BoundVar index is out of bounds.
    pub(super) fn substitute_bound_vars(&self, ty: Type<'db>, args: &[Type<'db>]) -> Type<'db> {
        subst::substitute_bound_vars(self.db(), ty, args).unwrap_or_else(
            |subst::BoundVarOutOfBounds { index, max }| {
                panic!(
                    "BoundVar index out of range: index={}, subst.len()={}",
                    index, max
                )
            },
        )
    }
}
