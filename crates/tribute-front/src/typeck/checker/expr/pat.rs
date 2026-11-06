//! Patterns: their types, bindings, `let` generalization, and conversion.

use rustc_hash::FxHashMap as HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    Arm, Expr, ExprKind, FieldPattern, LiteralPattern, LocalId, NodeId, Pattern, PatternKind,
    ResolvedRef, Type, TypeKind, TypeScheme, TypedRef, collect_row_vars, reference_binding,
};
use crate::typeck::checker::{Mode, TypeChecker};
use crate::typeck::func_context::{ConstructorShape, FunctionInferenceContext, LetSchemeBinding};
use crate::typeck::solver::{RowSubst, TypeSolver, TypeSubst};

/// One local introduced by a pattern, together with its resolved binding type.
struct PatternBinding<'db> {
    name: Symbol,
    local_id: Option<LocalId>,
    scope: NodeId,
    ty: Type<'db>,
}

impl<'db> TypeChecker<'db> {
    pub(super) fn generalize_and_bind_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        value: &Expr<ResolvedRef<'db>>,
        value_ty: Type<'db>,
    ) {
        // A revisited `let` keeps the schemes of its first visit. Solving
        // again can fail once an unrelated error has been constrained, and a
        // monomorphic rebinding would then alias the quantifiers that
        // already-checked uses instantiated.
        if let Some(bindings) = ctx.let_schemes(pattern.id) {
            for LetSchemeBinding {
                name,
                local_id,
                scope,
                scheme,
            } in bindings
            {
                if let Some(local_id) = local_id {
                    ctx.bind_local_scheme(local_id, scheme);
                    ctx.record_local_binding_owner(local_id, scope);
                }
                ctx.bind_local_scheme_by_name(name, scheme);
            }
            return;
        }
        // A call whose receiver the statements so far have typed is resolved
        // here, so the binding generalizes over what the call performs.
        let mut solver = loop {
            let mut solver = TypeSolver::new(self.db());
            solver.reserve_row_vars(ctx.next_row_var());
            for method in ctx.deferred_methods() {
                solver.defer_producer(
                    method.node_id,
                    method.result_ty,
                    method.arg_types.clone(),
                    method.effect,
                );
            }
            for constructor in ctx.deferred_constructors() {
                constructor.defer_in(self.db(), &mut solver);
            }
            let solved = solver
                .solve(ctx.constraints_snapshot())
                .and_then(|()| solver.finalize_relations().map_err(|failure| failure.error));
            ctx.reserve_row_vars(solver.next_row_var());
            if solved.is_err() {
                self.bind_pattern_vars_with_ctx(ctx, pattern, value_ty);
                return;
            }
            if !self.resolve_typed_deferred_methods(ctx, &solver) {
                break solver;
            }
        };
        // The row of a call still to be resolved keeps its own name in the
        // scheme, so resolving the call reaches the scheme's uses.
        solver.make_row_representatives(
            ctx.deferred_methods()
                .iter()
                .filter_map(|method| method.effect.rest(self.db())),
        );

        let type_subst = solver.type_subst().clone();
        let row_subst = solver.row_subst().clone();
        let should_generalize = self.generalizes_rows(ctx, pattern, value);

        let (mut environment_type_vars, mut environment_row_vars) = solver.pending_variables();
        if should_generalize {
            for ty in ctx.annotation_type_parameters() {
                type_subst.collect_univars_from_type(
                    self.db(),
                    ty,
                    &row_subst,
                    &mut environment_type_vars,
                );
            }
            for scheme in ctx.visible_local_schemes() {
                let body =
                    type_subst.apply_with_rows(self.db(), *scheme.body(self.db()), &row_subst);
                type_subst.collect_univars_from_type(
                    self.db(),
                    body,
                    &row_subst,
                    &mut environment_type_vars,
                );
                for var in collect_row_vars(self.db(), body) {
                    if !scheme.effect_params(self.db()).contains(&var)
                        && !environment_row_vars.contains(&var)
                    {
                        environment_row_vars.push(var);
                    }
                }
            }
        }

        solver.expand_union_dependencies(&mut environment_type_vars, &mut environment_row_vars);
        let resolved_value = type_subst.apply_with_rows(self.db(), value_ty, &row_subst);
        let mut bindings = Vec::new();
        self.collect_pattern_bindings_with_ctx(
            ctx,
            pattern,
            resolved_value,
            &type_subst,
            &row_subst,
            &mut bindings,
        );

        // A `let` quantifies only ability-row variables. Its data-type
        // variables stay solver variables that every name and every use of
        // the binding shares (`let f as g = ...`, `State(s)`).
        let mut let_schemes = Vec::new();
        for PatternBinding {
            name,
            local_id,
            scope,
            ty,
        } in bindings
        {
            let scheme = if should_generalize {
                let retained = solver.row_unions_for_type(ty);
                let removals = solver.row_removals_for_type(ty);
                let (_, union_rows) = solver.row_union_variables(&retained);
                let (_, removed_rows) = solver.row_removal_variables(&removals);
                let no_types = HashMap::default();
                let mut effect_params = Vec::new();
                for var in collect_row_vars(self.db(), ty)
                    .into_iter()
                    .chain(union_rows)
                    .chain(removed_rows)
                {
                    if !environment_row_vars.contains(&var) && !effect_params.contains(&var) {
                        effect_params.push(var);
                    }
                }
                TypeScheme::builder(Vec::new(), effect_params, ty)
                    .row_unions(
                        retained
                            .iter()
                            .map(|union| solver.generalize_row_union(union, &no_types))
                            .collect(),
                    )
                    .row_removals(
                        removals
                            .iter()
                            .map(|r| solver.generalize_row_removal(r, &no_types))
                            .collect(),
                    )
                    .build(self.db())
            } else {
                TypeScheme::mono(self.db(), ty)
            };
            if let Some(local_id) = local_id {
                ctx.bind_local_scheme(local_id, scheme);
                ctx.record_local_binding_owner(local_id, scope);
            }
            ctx.bind_local_scheme_by_name(name.clone(), scheme);
            let_schemes.push(LetSchemeBinding {
                name,
                local_id,
                scope,
                scheme,
            });
        }
        ctx.record_let_schemes(pattern.id, let_schemes);
    }

    /// Whether a `let` of `value` quantifies rows: a use of a lambda or of a
    /// named-function alias chooses its own rows, and every other binding's
    /// uses share them (`new-plans/type-inference.md`, Generalization).
    fn generalizes_rows(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        value: &Expr<ResolvedRef<'db>>,
    ) -> bool {
        let Some((local, reference)) = reference_binding(pattern, value) else {
            return matches!(*value.kind, ExprKind::Lambda { .. });
        };
        let alias = match reference {
            ResolvedRef::Function { .. } => true,
            ResolvedRef::Local { id, .. } => ctx.is_function_alias(*id),
            _ => false,
        };
        if alias {
            ctx.record_function_alias(local);
        }
        alias
    }

    fn collect_pattern_bindings_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        ty: Type<'db>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        bindings: &mut Vec<PatternBinding<'db>>,
    ) {
        let resolve = |ty| type_subst.apply_with_rows(self.db(), ty, row_subst);
        match &*pattern.kind {
            PatternKind::Bind { name, local_id } => {
                bindings.push(PatternBinding {
                    name: name.clone(),
                    local_id: *local_id,
                    scope: pattern.id,
                    ty: resolve(ty),
                });
            }
            PatternKind::Tuple(patterns)
            | PatternKind::Variant {
                fields: patterns, ..
            }
            | PatternKind::List(patterns) => {
                for pattern in patterns {
                    let pattern_ty = ctx
                        .get_node_type(pattern.id)
                        .map(resolve)
                        .unwrap_or_else(|| resolve(ty));
                    self.collect_pattern_bindings_with_ctx(
                        ctx, pattern, pattern_ty, type_subst, row_subst, bindings,
                    );
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                let field_tys = self.record_pattern_field_types(ctx, pattern.id, type_name, fields);
                for (field, field_ty) in fields.iter().zip(field_tys) {
                    let field_ty = field_ty
                        .map(resolve)
                        .unwrap_or_else(|| ctx.fresh_type_var());
                    if let Some(pattern) = &field.pattern {
                        self.collect_pattern_bindings_with_ctx(
                            ctx, pattern, field_ty, type_subst, row_subst, bindings,
                        );
                    } else {
                        bindings.push(PatternBinding {
                            name: field.name.clone(),
                            local_id: None,
                            scope: field.id,
                            ty: field_ty,
                        });
                    }
                }
            }
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => {
                for pattern in head {
                    let pattern_ty = ctx
                        .get_node_type(pattern.id)
                        .map(resolve)
                        .unwrap_or_else(|| resolve(ty));
                    self.collect_pattern_bindings_with_ctx(
                        ctx, pattern, pattern_ty, type_subst, row_subst, bindings,
                    );
                }
                if let Some(name) = rest {
                    bindings.push(PatternBinding {
                        name: name.clone(),
                        local_id: *rest_local_id,
                        scope: pattern.id,
                        ty: resolve(ty),
                    });
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => {
                let resolved_ty = resolve(ty);
                bindings.push(PatternBinding {
                    name: name.clone(),
                    local_id: *local_id,
                    scope: pattern.id,
                    ty: resolved_ty,
                });
                self.collect_pattern_bindings_with_ctx(
                    ctx,
                    pattern,
                    resolved_ty,
                    type_subst,
                    row_subst,
                    bindings,
                );
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
        }
    }

    /// Infer the type that a pattern matches against. `expected` is the
    /// type of the value it matches, where known, which an implicit
    /// constructor takes its constructor from.
    pub(super) fn infer_pattern_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        expected: Option<Type<'db>>,
    ) -> Type<'db> {
        let ty = match &*pattern.kind {
            PatternKind::Wildcard | PatternKind::Bind { .. } => ctx.fresh_type_var(),
            PatternKind::Literal(lit) => match lit {
                LiteralPattern::Bool(_) => ctx.bool_type(),
                LiteralPattern::Nat(_) => ctx.nat_type(),
                LiteralPattern::Int(_) => ctx.int_type(),
                LiteralPattern::Float(_) => ctx.float_type(),
                LiteralPattern::String(_) => ctx.string_type(),
                LiteralPattern::Bytes(_) => ctx.bytes_type(),
                LiteralPattern::Rune(_) => ctx.rune_type(),
                LiteralPattern::Nil => ctx.nil_type(),
            },
            PatternKind::Variant { ctor, fields } => {
                let shape = ConstructorShape::of_arity(fields.len());
                let ctor = &self.decide_constructor(ctx, pattern.id, ctor, expected, shape, true);
                let ctor_ty = match ctor {
                    ResolvedRef::Module { path } => {
                        self.report_module_reference(ctx, pattern.id, *path, "a constructor")
                    }
                    ResolvedRef::ImplicitConstructor { .. } => {
                        self.infer_var_with_ctx(ctx, Some(pattern.id), ctor)
                    }
                    _ => ctx
                        .get_constructor_reference_type(pattern.id)
                        .unwrap_or_else(|| self.infer_var_with_ctx(ctx, None, ctor)),
                };
                ctx.record_node_type(pattern.id, ctor_ty);
                self.validate_variant_pattern_with_ctx(
                    ctx,
                    pattern.id,
                    ctor,
                    ctor_ty,
                    fields.len(),
                );

                match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, result, .. } => {
                        for (field_pat, param_ty) in fields.iter().zip(params.iter()) {
                            let field_ty =
                                self.infer_pattern_type_with_ctx(ctx, field_pat, Some(*param_ty));
                            ctx.constrain_eq(field_ty, *param_ty);
                        }
                        *result
                    }
                    _ => ctor_ty,
                }
            }
            PatternKind::Tuple(pats) => {
                let expected_elems = match expected.map(|ty| ty.kind(self.db())) {
                    Some(TypeKind::Tuple(elems)) if elems.len() == pats.len() => {
                        Some(elems.clone())
                    }
                    _ => None,
                };
                let elem_tys: Vec<_> = pats
                    .iter()
                    .enumerate()
                    .map(|(index, p)| {
                        let expected = expected_elems.as_ref().map(|elems| elems[index]);
                        self.infer_pattern_type_with_ctx(ctx, p, expected)
                    })
                    .collect();
                ctx.tuple_type(elem_tys)
            }
            PatternKind::List(pats) => {
                let elem_ty = ctx.fresh_type_var();
                for pat in pats {
                    let pat_ty = self.infer_pattern_type_with_ctx(ctx, pat, None);
                    ctx.constrain_eq(pat_ty, elem_ty);
                }
                ctx.canonical_list_type(elem_ty)
            }
            PatternKind::ListRest { head, .. } => {
                let elem_ty = ctx.fresh_type_var();
                for pat in head {
                    let pat_ty = self.infer_pattern_type_with_ctx(ctx, pat, None);
                    ctx.constrain_eq(pat_ty, elem_ty);
                }
                ctx.canonical_list_type(elem_ty)
            }
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let written: Vec<Symbol> = fields.iter().map(|field| field.name.clone()).collect();
                let type_name = &self.decide_constructor(
                    ctx,
                    pattern.id,
                    type_name,
                    expected,
                    ConstructorShape::Fields {
                        names: written.clone(),
                        rest: *rest,
                    },
                    true,
                );
                let (_, result, field_tys) =
                    self.constructor_field_shape(ctx, pattern.id, type_name, &written);
                self.validate_record_pattern_with_ctx(
                    ctx, pattern.id, type_name, result, &written, *rest,
                );
                for (field, field_ty) in fields.iter().zip(field_tys) {
                    if let Some(sub) = &field.pattern {
                        let sub_ty = self.infer_pattern_type_with_ctx(ctx, sub, field_ty);
                        if let Some(field_ty) = field_ty {
                            ctx.constrain_eq(sub_ty, field_ty);
                        }
                    }
                }
                result
            }
            PatternKind::As { pattern, .. } => {
                self.infer_pattern_type_with_ctx(ctx, pattern, expected)
            }
            PatternKind::Error => ctx.error_type(),
        };

        if !matches!(&*pattern.kind, PatternKind::Variant { .. }) {
            ctx.record_node_type(pattern.id, ty);
        }
        ty
    }

    /// Bind pattern variables to the given type.
    pub(super) fn bind_pattern_vars_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        ty: Type<'db>,
    ) {
        match &*pattern.kind {
            PatternKind::Bind { name, local_id } => {
                if let Some(id) = local_id {
                    ctx.bind_local(*id, ty);
                }
                ctx.bind_local_by_name(name.clone(), ty);
            }
            PatternKind::Tuple(pats) => {
                if let TypeKind::Tuple(elem_tys) = ty.kind(self.db()) {
                    for (pat, elem_ty) in pats.iter().zip(elem_tys.iter()) {
                        self.bind_pattern_vars_with_ctx(ctx, pat, *elem_ty);
                    }
                } else {
                    let fresh_vars: Vec<_> = pats.iter().map(|_| ctx.fresh_type_var()).collect();
                    for (pat, fresh_ty) in pats.iter().zip(fresh_vars) {
                        self.bind_pattern_vars_with_ctx(ctx, pat, fresh_ty);
                    }
                }
            }
            PatternKind::Variant { fields, .. } => {
                // Inference recorded the constructor's callable type on this
                // pattern and constrained each field to its parameter. A
                // field's own node type is not its value type when the field
                // is itself a constructor pattern.
                let params = match ctx.get_node_type(pattern.id).map(|ty| ty.kind(self.db())) {
                    Some(TypeKind::Func { params, .. }) if params.len() == fields.len() => {
                        Some(params.clone())
                    }
                    _ => None,
                };
                for (index, field) in fields.iter().enumerate() {
                    let field_ty = params
                        .as_ref()
                        .map(|params| params[index])
                        .or_else(|| ctx.get_node_type(field.id))
                        .unwrap_or_else(|| ctx.fresh_type_var());
                    self.bind_pattern_vars_with_ctx(ctx, field, field_ty);
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                let field_tys = self.record_pattern_field_types(ctx, pattern.id, type_name, fields);
                for (field, field_ty) in fields.iter().zip(field_tys) {
                    let field_ty = field_ty.unwrap_or_else(|| ctx.fresh_type_var());

                    if let Some(pat) = &field.pattern {
                        self.bind_pattern_vars_with_ctx(ctx, pat, field_ty);
                    } else {
                        // Shorthand { name } - bind the field name directly
                        ctx.bind_local_by_name(field.name.clone(), field_ty);
                    }
                }
            }
            PatternKind::List(pats) => {
                // Extract element type from the list type, or create a fresh var
                let elem_ty = self.extract_list_element_type(ty, ctx);
                for pat in pats {
                    self.bind_pattern_vars_with_ctx(ctx, pat, elem_ty);
                }
            }
            PatternKind::ListRest {
                head,
                rest_local_id,
                ..
            } => {
                // Extract element type from the list type, or create a fresh var
                let elem_ty = self.extract_list_element_type(ty, ctx);
                for pat in head {
                    self.bind_pattern_vars_with_ctx(ctx, pat, elem_ty);
                }
                if let Some(local_id) = rest_local_id {
                    // The rest is also a list of the same element type
                    let list_ty = ctx.canonical_list_type(elem_ty);
                    ctx.bind_local(*local_id, list_ty);
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => {
                ctx.bind_local_by_name(name.clone(), ty);
                if let Some(local_id) = local_id {
                    ctx.bind_local(*local_id, ty);
                }
                self.bind_pattern_vars_with_ctx(ctx, pattern, ty);
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
        }
    }

    /// Extract the element type from a List type.
    ///
    /// If the type is `List<T>`, returns `T`. Otherwise, creates a fresh type variable.
    pub(super) fn extract_list_element_type(
        &self,
        ty: Type<'db>,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
    ) -> Type<'db> {
        match ty.kind(self.db()) {
            TypeKind::Named { id, args, .. }
                if id.is_builtin_list(self.db()) && args.len() == 1 =>
            {
                args[0]
            }
            _ => ctx.fresh_type_var(),
        }
    }

    /// Convert a pattern.
    pub(super) fn convert_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match &*pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind {
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Literal(lit) => PatternKind::Literal(lit.clone()),
            PatternKind::Variant { ctor, fields } => PatternKind::Variant {
                ctor: TypedRef {
                    ty: ctx
                        .get_node_type(pattern.id)
                        .unwrap_or_else(|| self.infer_var_with_ctx(ctx, None, ctor)),
                    resolved: self.constructor_reference(ctx, pattern.id, ctor),
                },
                fields: fields
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            },
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let type_name = self.record_pattern_type_name(ctx, pattern.id, type_name);
                PatternKind::Record {
                    type_name,
                    fields: fields
                        .iter()
                        .map(|f| self.convert_field_pattern_with_ctx(ctx, f))
                        .collect(),
                    rest: *rest,
                }
            }
            PatternKind::Tuple(patterns) => PatternKind::Tuple(
                patterns
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            ),
            PatternKind::List(patterns) => PatternKind::List(
                patterns
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            ),
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => PatternKind::ListRest {
                head: head
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
                rest: rest.clone(),
                rest_local_id: *rest_local_id,
            },
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern_with_ctx(ctx, pattern),
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }

    /// The field type of each field of a record pattern, in source order,
    /// taken from its constructor instance.
    fn record_pattern_field_types<V>(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        type_name: &ResolvedRef<'db>,
        fields: &[FieldPattern<V>],
    ) -> Vec<Option<Type<'db>>> {
        let written: Vec<Symbol> = fields.iter().map(|field| field.name.clone()).collect();
        self.constructor_field_shape(ctx, pattern_id, type_name, &written)
            .2
    }

    /// The typed constructor reference of a brace-form constructor pattern,
    /// carrying the same constructor instance its fields were checked with.
    fn record_pattern_type_name(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        resolved: &ResolvedRef<'db>,
    ) -> TypedRef<'db> {
        let ty = self.instantiate_value_constructor_with_ctx(ctx, pattern_id, resolved);
        TypedRef {
            resolved: self.constructor_reference(ctx, pattern_id, resolved),
            ty,
        }
    }

    fn convert_field_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        fp: &FieldPattern<ResolvedRef<'db>>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name_id: fp.name_id,
            name: fp.name.clone(),
            pattern: fp
                .pattern
                .as_ref()
                .map(|p| self.convert_pattern_with_ctx(ctx, p)),
        }
    }

    /// Convert a field pattern with an expected type.
    fn convert_field_pattern_with_expected_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        fp: &FieldPattern<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name_id: fp.name_id,
            name: fp.name.clone(),
            pattern: fp
                .pattern
                .as_ref()
                .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, expected)),
        }
    }

    /// Convert a case arm with an expected scrutinee type.
    pub(super) fn convert_arm_with_scrutinee_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: &Arm<ResolvedRef<'db>>,
        scrutinee_ty: Type<'db>,
    ) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.convert_pattern_with_expected_ctx(ctx, &arm.pattern, scrutinee_ty),
            guard: arm.guard.as_ref().map(|g| {
                let bool_ty = ctx.bool_type();
                self.check_expr_with_ctx(ctx, g, Mode::Check(bool_ty))
            }),
            body: self.check_expr_with_ctx(ctx, &arm.body, Mode::Infer),
        }
    }

    /// Convert a pattern with an expected type.
    pub(super) fn convert_pattern_with_expected_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match &*pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind {
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Literal(lit) => PatternKind::Literal(lit.clone()),
            PatternKind::Variant { ctor, fields } => {
                let ctor_ty = ctx
                    .get_node_type(pattern.id)
                    .unwrap_or_else(|| self.infer_var_with_ctx(ctx, None, ctor));
                let ctor = &self.constructor_reference(ctx, pattern.id, ctor);

                match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, result, .. } => {
                        ctx.constrain_eq(*result, expected);
                        let fields = fields
                            .iter()
                            .zip(params.iter())
                            .map(|(p, param_ty)| {
                                self.convert_pattern_with_expected_ctx(ctx, p, *param_ty)
                            })
                            .collect();
                        PatternKind::Variant {
                            ctor: TypedRef {
                                resolved: ctor.clone(),
                                ty: ctor_ty,
                            },
                            fields,
                        }
                    }
                    _ => {
                        ctx.constrain_eq(ctor_ty, expected);
                        PatternKind::Variant {
                            ctor: TypedRef {
                                resolved: ctor.clone(),
                                ty: ctor_ty,
                            },
                            fields: vec![],
                        }
                    }
                }
            }
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let field_tys = self.record_pattern_field_types(ctx, pattern.id, type_name, fields);
                let type_name = self.record_pattern_type_name(ctx, pattern.id, type_name);
                let result = match type_name.ty.kind(self.db()) {
                    TypeKind::Func { result, .. } => *result,
                    _ => type_name.ty,
                };
                ctx.constrain_eq(result, expected);

                let converted_fields = fields
                    .iter()
                    .zip(field_tys)
                    .map(|(f, field_ty)| {
                        let field_expected = field_ty.unwrap_or_else(|| ctx.fresh_type_var());
                        self.convert_field_pattern_with_expected_ctx(ctx, f, field_expected)
                    })
                    .collect();

                PatternKind::Record {
                    type_name,
                    fields: converted_fields,
                    rest: *rest,
                }
            }
            PatternKind::Tuple(patterns) => {
                let elem_expectations: Vec<Type<'db>> =
                    if let TypeKind::Tuple(elems) = expected.kind(self.db()) {
                        elems.clone()
                    } else {
                        patterns.iter().map(|_| ctx.fresh_type_var()).collect()
                    };
                PatternKind::Tuple(
                    patterns
                        .iter()
                        .zip(elem_expectations)
                        .map(|(p, exp)| self.convert_pattern_with_expected_ctx(ctx, p, exp))
                        .collect(),
                )
            }
            PatternKind::List(patterns) => {
                let elem_ty = ctx.fresh_type_var();
                let list_ty = ctx.canonical_list_type(elem_ty);
                ctx.constrain_eq(expected, list_ty);
                PatternKind::List(
                    patterns
                        .iter()
                        .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, elem_ty))
                        .collect(),
                )
            }
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => {
                let elem_ty = ctx.fresh_type_var();
                let list_ty = ctx.canonical_list_type(elem_ty);
                ctx.constrain_eq(expected, list_ty);
                PatternKind::ListRest {
                    head: head
                        .iter()
                        .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, elem_ty))
                        .collect(),
                    rest: rest.clone(),
                    rest_local_id: *rest_local_id,
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern_with_expected_ctx(ctx, pattern, expected),
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }
}
