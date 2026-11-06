//! Type annotations written in a function body, converted with the body's
//! inference variables.

use trunk_ir::SymbolPath;

use crate::ast::{AbilityRow, Type, TypeKind};
use crate::typeck::checker::TypeChecker;
use crate::typeck::func_context::FunctionInferenceContext;

impl<'db> TypeChecker<'db> {
    /// Convert a type annotation to a Type within a function body.
    ///
    /// Uses FunctionInferenceContext for fresh type variables.
    pub(super) fn annotation_to_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        ann: &crate::ast::TypeAnnotation,
    ) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        if let Some(ty) = ctx.annotation_type(ann.id) {
            return ty;
        }
        let ty = match &ann.kind {
            TypeAnnotationKind::Named(name) if ctx.annotation_type_parameter(name).is_some() => ctx
                .annotation_type_parameter(name)
                .expect("known signature parameter"),
            TypeAnnotationKind::Named(name) => {
                if *name == "Int" {
                    ctx.int_type()
                } else if *name == "Nat" {
                    ctx.nat_type()
                } else if *name == "Float" {
                    ctx.float_type()
                } else if *name == "Bool" {
                    ctx.bool_type()
                } else if *name == "Bytes" {
                    ctx.bytes_type()
                } else if *name == "Rune" {
                    ctx.rune_type()
                } else if *name == "Nil" {
                    ctx.nil_type()
                } else if *name == "Never" {
                    ctx.never_type()
                } else {
                    ctx.named_type_in_scope(
                        SymbolPath::new([name.clone()]),
                        vec![],
                        self.current_prefix(),
                    )
                }
            }
            TypeAnnotationKind::Path(parts) => ctx.path_type(parts),
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_with_ctx(ctx, ctor);
                if let TypeKind::Named { id, name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_with_ctx(ctx, a))
                        .collect();
                    ctx.named_type_with_id(*id, name.clone(), arg_types)
                } else {
                    ctx.error_type()
                }
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| self.annotation_to_type_with_ctx(ctx, p))
                    .collect();
                let result_ty = self.annotation_to_type_with_ctx(ctx, result);
                let effect = self.ability_annotations_to_row_with_ctx(ctx, abilities);
                ctx.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.annotation_to_type_with_ctx(ctx, e))
                    .collect();
                ctx.tuple_type(elem_types)
            }
            TypeAnnotationKind::Infer => ctx.fresh_type_var(),
            TypeAnnotationKind::Error => ctx.error_type(),
        };
        ctx.record_annotation_type(ann.id, ty);
        ty
    }

    /// Convert the ability row written in a function type or a lambda's
    /// return annotation. Row-variable entries share one tail.
    pub(super) fn ability_annotations_to_row_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        abilities: &[crate::ast::TypeAnnotation],
    ) -> AbilityRow<'db> {
        use crate::ast::TypeAnnotationKind;

        let mut rows = Vec::new();
        for ability in abilities {
            let row = match &ability.kind {
                TypeAnnotationKind::Named(name) if crate::ast::is_type_variable(name) => {
                    ctx.annotation_row(name.clone())
                }
                TypeAnnotationKind::Infer => ctx.fresh_row_var(),
                _ => continue,
            };
            if !rows.contains(&row) {
                rows.push(row);
            }
        }
        let rest = match rows.as_slice() {
            [] => None,
            [row] => Some(*row),
            _ => {
                let result = ctx.fresh_row_var();
                ctx.constrain_row_union(crate::ast::RowUnion {
                    sources: rows
                        .into_iter()
                        .map(|row| AbilityRow::open(self.db(), row))
                        .collect(),
                    result: AbilityRow::open(self.db(), result),
                });
                Some(result)
            }
        };
        let converted = crate::ast::abilities_to_ability_row(
            self.db(),
            abilities,
            self.current_prefix(),
            &mut |a| self.annotation_to_type_with_ctx(ctx, a),
            || rest.expect("row annotations allocate a tail"),
        );
        AbilityRow::new(self.db(), converted.instances(self.db()), rest)
    }
}
