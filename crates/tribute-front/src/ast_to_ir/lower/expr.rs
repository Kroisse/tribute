//! Constructor coercion and source worker convention propagation.
//!
//! These helpers consume typechecked metadata; shared IR legalization owns CPS
//! construction and preserves the declared ability operation kind.

use super::IrBuilder;
use crate::ast::{CallingConvention, Expr, ExprKind, ResolvedRef, Stmt, TypeKind, TypedRef};
use trunk_ir::Symbol;
use trunk_ir::adt_layout::get_enum_variants;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

/// Coerce constructor arguments to the representation recorded in the enum
/// layout. Generic enum fields are erased to `anyref`, so primitive payloads
/// must cross an explicit conversion boundary before `adt.variant_new`.
pub(super) fn cast_variant_args<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    args: Vec<ValueRef>,
    enum_ty: TypeRef,
    variant: Symbol,
) -> Vec<ValueRef> {
    let field_types = get_enum_variants(builder.ir, enum_ty)
        .and_then(|variants| {
            variants
                .into_iter()
                .find_map(|(tag, fields)| (tag == variant).then_some(fields))
        })
        .expect("resolved constructor must exist in enum metadata");

    assert_eq!(
        args.len(),
        field_types.len(),
        "type checking must enforce constructor arity"
    );

    args.into_iter()
        .zip(field_types)
        .map(|(arg, field_ty)| builder.cast_if_needed(location, arg, field_ty))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EvaluationControlClass {
    Direct,
    Cps,
}

impl EvaluationControlClass {
    fn join(self, other: Self) -> Self {
        if self == Self::Cps || other == Self::Cps {
            Self::Cps
        } else {
            Self::Direct
        }
    }
}

pub(super) fn logical_evaluation_control_class<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> EvaluationControlClass {
    let children = |children: &[Expr<TypedRef<'db>>]| {
        children
            .iter()
            .fold(EvaluationControlClass::Direct, |class, child| {
                class.join(logical_evaluation_control_class(ctx, child))
            })
    };

    match &*expr.kind {
        ExprKind::Handle { .. } => EvaluationControlClass::Cps,
        ExprKind::Lambda { .. } => EvaluationControlClass::Direct,
        ExprKind::Resume { .. } => EvaluationControlClass::Cps,
        ExprKind::Call { callee, args } => {
            let call = if is_cps_call_expr(ctx, expr) {
                EvaluationControlClass::Cps
            } else {
                EvaluationControlClass::Direct
            };
            call.join(logical_evaluation_control_class(ctx, callee))
                .join(children(args))
        }
        ExprKind::Cons { args, .. } | ExprKind::Tuple(args) | ExprKind::List(args) => {
            children(args)
        }
        ExprKind::Record { fields, spread, .. } => {
            let spread = spread
                .as_ref()
                .map_or(EvaluationControlClass::Direct, |spread| {
                    logical_evaluation_control_class(ctx, spread)
                });
            fields.iter().fold(spread, |class, (_, field)| {
                class.join(logical_evaluation_control_class(ctx, field))
            })
        }
        ExprKind::Block { stmts, value } => {
            let statements = stmts
                .iter()
                .fold(EvaluationControlClass::Direct, |class, stmt| {
                    let expr = match stmt {
                        Stmt::Let { value, .. } => value,
                        Stmt::Expr { expr, .. } => expr,
                    };
                    class.join(logical_evaluation_control_class(ctx, expr))
                });
            statements.join(logical_evaluation_control_class(ctx, value))
        }
        ExprKind::BinOp { lhs, rhs, .. } => logical_evaluation_control_class(ctx, lhs)
            .join(logical_evaluation_control_class(ctx, rhs)),
        ExprKind::Case { scrutinee, arms } => arms.iter().fold(
            logical_evaluation_control_class(ctx, scrutinee),
            |class, arm| {
                let guard = arm
                    .guard
                    .as_ref()
                    .map_or(EvaluationControlClass::Direct, |guard| {
                        logical_evaluation_control_class(ctx, guard)
                    });
                class
                    .join(guard)
                    .join(logical_evaluation_control_class(ctx, &arm.body))
            },
        ),
        ExprKind::MethodCall { receiver, args, .. } => {
            logical_evaluation_control_class(ctx, receiver).join(children(args))
        }
        ExprKind::Var(_)
        | ExprKind::NatLit(_)
        | ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::Nil
        | ExprKind::RuneLit(_)
        | ExprKind::Error => EvaluationControlClass::Direct,
    }
}

fn calling_convention_for_type<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    ty: crate::ast::Type<'db>,
) -> CallingConvention {
    ctx.calling_convention_for_type(ty)
        .unwrap_or(CallingConvention::Cps)
}

fn is_cps_call_expr<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    expr: &Expr<TypedRef<'db>>,
) -> bool {
    let ExprKind::Call { callee, .. } = &*expr.kind else {
        return false;
    };
    let ExprKind::Var(tr) = &*callee.kind else {
        return ctx
            .get_node_type(callee.id)
            .copied()
            .is_none_or(|ty| calling_convention_for_type(ctx, ty) == CallingConvention::Cps);
    };
    match &tr.resolved {
        ResolvedRef::AbilityOp {
            kind: crate::ast::OpDeclKind::Op,
            ..
        } => true,
        ResolvedRef::AbilityOp { .. } => false,
        ResolvedRef::Local { .. } => {
            !matches!(tr.ty.kind(ctx.db), TypeKind::Continuation { .. })
                && calling_convention_for_type(ctx, tr.ty) == CallingConvention::Cps
        }
        _ => callee_requires_cps_by_definition(ctx, tr),
    }
}

/// Check whether a named callee's definition selects Cps.
fn callee_requires_cps_by_definition<'db>(
    ctx: &super::super::context::IrLoweringCtx<'db>,
    tr: &TypedRef<'db>,
) -> bool {
    let callee_name = match &tr.resolved {
        ResolvedRef::Function { id } => id.qualified(ctx.db),
        _ => {
            // For locals/closures, use the call-site function type.
            return ctx.calling_convention_for_type(tr.ty) == Some(CallingConvention::Cps);
        }
    };
    // Look up the function's TypeScheme (definition type)
    ctx.function_calling_convention(callee_name) == Some(CallingConvention::Cps)
}
