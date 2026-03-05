//! Case expression and pattern matching lowering.
//!
//! Lowers case expressions to a chain of `scf.if` operations, with
//! pattern checks generating boolean conditions and pattern bindings
//! extracted inside the matched region.

use trunk_ir::arena::context::{BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::{adt, arith, func, scf};
use trunk_ir::arena::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::arena::types::{Attribute, Location};

use crate::ast::{Arm, Expr, LiteralPattern, Pattern, PatternKind, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::{IrBuilder, get_or_create_tuple_type, is_irrefutable_pattern, resolve_enum_type_attr};

/// Lower a case expression as a chain of `scf.if` operations.
pub(super) fn lower_case_chain<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    result_ty: TypeRef,
    arms: &[Arm<TypedRef<'db>>],
    is_else_chain: bool,
) -> Option<ValueRef> {
    match arms {
        [] => {
            // No arms — exhaustiveness failure fallback, emit unreachable
            let unreachable_op = func::unreachable(builder.ir, location);
            builder.ir.push_op(builder.block, unreachable_op.op_ref());
            let nil_ty = builder.ctx.nil_type(builder.ir);
            let const_op = arith::r#const(builder.ir, location, nil_ty, Attribute::Unit);
            builder.ir.push_op(builder.block, const_op.op_ref());
            Some(const_op.result(builder.ir))
        }
        [last]
            if last.guard.is_none() && (is_else_chain || is_irrefutable_pattern(&last.pattern)) =>
        {
            // Unconditional arm: skip condition check
            builder.ctx.enter_scope();
            bind_pattern_fields(
                builder.ctx,
                builder.ir,
                builder.block,
                location,
                scrutinee,
                &last.pattern,
            );
            let result = super::expr::lower_expr(builder, last.body.clone());
            builder.ctx.exit_scope();
            result
        }
        [first, rest @ ..] => {
            // Multi-arm: pattern check → then/else regions → scf.if

            // 1. Emit pattern condition check
            let pattern_cond = emit_pattern_check(builder, location, scrutinee, &first.pattern)?;

            // 2. Build then region (handles guard internally if present)
            let then_region = if let Some(guard_expr) = &first.guard {
                build_guarded_arm_region(
                    builder.ctx,
                    builder.ir,
                    location,
                    scrutinee,
                    first,
                    guard_expr,
                    result_ty,
                    rest,
                )
            } else {
                build_arm_region(
                    builder.ctx,
                    builder.ir,
                    location,
                    scrutinee,
                    first,
                    result_ty,
                )
            };

            // 3. Build else region (recursive)
            let else_region = build_else_chain_region(
                builder.ctx,
                builder.ir,
                location,
                scrutinee,
                result_ty,
                rest,
            );

            // 4. Emit scf.if in current block
            let if_op = scf::r#if(
                builder.ir,
                location,
                pattern_cond,
                result_ty,
                then_region,
                else_region,
            );
            builder.ir.push_op(builder.block, if_op.op_ref());
            Some(if_op.result(builder.ir))
        }
    }
}

/// Emit a pattern check that produces a boolean condition value.
fn emit_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    pattern: &Pattern<TypedRef<'db>>,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);

    match &*pattern.kind {
        PatternKind::Wildcard | PatternKind::Bind { .. } | PatternKind::Error => {
            // Always matches
            let op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true));
            builder.ir.push_op(builder.block, op.op_ref());
            Some(op.result(builder.ir))
        }
        PatternKind::Variant { ctor, .. } => {
            // Test if scrutinee is of the specific variant
            let (variant_name, enum_ty) = match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => {
                    let ty = resolve_enum_type_attr(builder.ctx, builder.ir, ctor.ty);
                    (*variant, ty)
                }
                _ => {
                    unreachable!("non-constructor in variant pattern: {:?}", ctor.resolved)
                }
            };
            let op = adt::variant_is(
                builder.ir,
                location,
                scrutinee,
                bool_ty,
                enum_ty,
                variant_name,
            );
            builder.ir.push_op(builder.block, op.op_ref());
            Some(op.result(builder.ir))
        }
        PatternKind::Literal(lit) => emit_literal_check(builder, location, scrutinee, lit),
        PatternKind::Tuple(elements) => {
            // Tuple patterns: recursively check all element patterns
            let mut conditions = Vec::new();
            let any_ty = builder.ctx.any_type(builder.ir);
            let struct_ty = get_or_create_tuple_type(builder.ctx, builder.ir, pattern.id)
                .map(|(_, st)| st)
                .unwrap_or(any_ty);

            for (i, elem_pat) in elements.iter().enumerate() {
                let elem_op =
                    adt::struct_get(builder.ir, location, scrutinee, any_ty, struct_ty, i as u32);
                builder.ir.push_op(builder.block, elem_op.op_ref());
                let elem_val = elem_op.result(builder.ir);

                let cond = emit_pattern_check(builder, location, elem_val, elem_pat)?;
                conditions.push(cond);
            }

            // Combine all conditions with AND
            if conditions.is_empty() {
                let op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true));
                builder.ir.push_op(builder.block, op.op_ref());
                Some(op.result(builder.ir))
            } else {
                let mut result = conditions[0];
                for cond in conditions.into_iter().skip(1) {
                    let and_op = arith::and(builder.ir, location, result, cond, bool_ty);
                    builder.ir.push_op(builder.block, and_op.op_ref());
                    result = and_op.result(builder.ir);
                }
                Some(result)
            }
        }
        _ => {
            unreachable!("unsupported pattern in IR lowering: {:?}", pattern.kind)
        }
    }
}

/// Emit a literal equality check.
fn emit_literal_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    lit: &LiteralPattern,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let i32_ty = builder.ctx.i32_type(builder.ir);

    match lit {
        LiteralPattern::Nat(n) => {
            let value = super::validate_nat_i31(builder.db(), location, *n)?;
            let const_op = arith::r#const(
                builder.ir,
                location,
                i32_ty,
                Attribute::IntBits(value as u64),
            );
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmp_eq(builder.ir, location, scrutinee, const_val, bool_ty);
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::Int(n) => {
            let value = super::validate_int_i31(builder.db(), location, *n)?;
            let const_op = arith::r#const(
                builder.ir,
                location,
                i32_ty,
                Attribute::IntBits(value as u64),
            );
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmp_eq(builder.ir, location, scrutinee, const_val, bool_ty);
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::Bool(b) => {
            let const_op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(*b));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmp_eq(builder.ir, location, scrutinee, const_val, bool_ty);
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        _ => {
            unreachable!("unsupported literal pattern in IR lowering: {:?}", lit)
        }
    }
}

/// Build a then-region for a guarded arm (nested scf.if for guard).
#[allow(clippy::too_many_arguments)]
fn build_guarded_arm_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    scrutinee: ValueRef,
    arm: &Arm<TypedRef<'db>>,
    guard_expr: &Expr<TypedRef<'db>>,
    result_ty: TypeRef,
    rest: &[Arm<TypedRef<'db>>],
) -> trunk_ir::arena::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    // 1. Bind pattern fields (safe — we're inside the matched region)
    ctx.enter_scope();
    bind_pattern_fields(ctx, ir, block, location, scrutinee, &arm.pattern);

    // 2. Evaluate guard condition
    let guard_cond = {
        let mut builder = IrBuilder::new(ctx, ir, block);
        super::expr::lower_expr(&mut builder, guard_expr.clone())
    };
    let guard_cond = match guard_cond {
        Some(v) => v,
        None => {
            let bool_ty = ctx.bool_type(ir);
            let op = arith::r#const(ir, location, bool_ty, Attribute::Bool(false));
            ir.push_op(block, op.op_ref());
            op.result(ir)
        }
    };

    // 3. Build inner then region (arm body)
    let inner_then_region = {
        let inner_block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let result = {
            let mut builder = IrBuilder::new(ctx, ir, inner_block);
            let val = super::expr::lower_expr(&mut builder, arm.body.clone());
            val.map(|v| builder.cast_if_needed(location, v, result_ty))
        };
        let yield_val = match result {
            Some(v) => v,
            None => {
                let nil_ty = ctx.nil_type(ir);
                let op = arith::r#const(ir, location, nil_ty, Attribute::Unit);
                ir.push_op(inner_block, op.op_ref());
                op.result(ir)
            }
        };
        let yield_op = scf::r#yield(ir, location, [yield_val]);
        ir.push_op(inner_block, yield_op.op_ref());
        ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![inner_block],
            parent_op: None,
        })
    };

    ctx.exit_scope();

    // 4. Build inner else region (fall through to remaining arms)
    let inner_else_region = build_else_chain_region(ctx, ir, location, scrutinee, result_ty, rest);

    // 5. Emit inner scf.if for guard
    let inner_if_op = scf::r#if(
        ir,
        location,
        guard_cond,
        result_ty,
        inner_then_region,
        inner_else_region,
    );
    ir.push_op(block, inner_if_op.op_ref());
    let inner_result = inner_if_op.result(ir);

    // 6. Yield inner result
    let yield_op = scf::r#yield(ir, location, [inner_result]);
    ir.push_op(block, yield_op.op_ref());

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build a then-region for an unguarded arm.
fn build_arm_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    scrutinee: ValueRef,
    arm: &Arm<TypedRef<'db>>,
    result_ty: TypeRef,
) -> trunk_ir::arena::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    ctx.enter_scope();
    bind_pattern_fields(ctx, ir, block, location, scrutinee, &arm.pattern);

    let result = {
        let mut builder = IrBuilder::new(ctx, ir, block);
        let val = super::expr::lower_expr(&mut builder, arm.body.clone());
        val.map(|v| builder.cast_if_needed(location, v, result_ty))
    };

    ctx.exit_scope();

    let yield_val = match result {
        Some(v) => v,
        None => {
            let nil_ty = ctx.nil_type(ir);
            let op = arith::r#const(ir, location, nil_ty, Attribute::Unit);
            ir.push_op(block, op.op_ref());
            op.result(ir)
        }
    };
    let yield_op = scf::r#yield(ir, location, [yield_val]);
    ir.push_op(block, yield_op.op_ref());

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build an else-region that recursively chains remaining arms.
fn build_else_chain_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    scrutinee: ValueRef,
    result_ty: TypeRef,
    arms: &[Arm<TypedRef<'db>>],
) -> trunk_ir::arena::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    let result = {
        let mut builder = IrBuilder::new(ctx, ir, block);
        lower_case_chain(&mut builder, location, scrutinee, result_ty, arms, true)
    };

    let val = match result {
        Some(v) => v,
        None => {
            let nil_ty = ctx.nil_type(ir);
            let op = arith::r#const(ir, location, nil_ty, Attribute::Unit);
            ir.push_op(block, op.op_ref());
            op.result(ir)
        }
    };

    // Cast to result_ty if needed
    let yield_val = {
        let mut builder = IrBuilder::new(ctx, ir, block);
        builder.cast_if_needed(location, val, result_ty)
    };

    let yield_op = scf::r#yield(ir, location, [yield_val]);
    ir.push_op(block, yield_op.op_ref());

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Bind pattern variables to extracted values from the scrutinee.
///
/// Must only be called inside a region where the pattern has already matched.
pub(super) fn bind_pattern_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: BlockRef,
    location: Location,
    scrutinee: ValueRef,
    pattern: &Pattern<TypedRef<'db>>,
) {
    match &*pattern.kind {
        PatternKind::Bind {
            name,
            local_id: Some(id),
        } => {
            ctx.bind(*id, *name, scrutinee);
        }
        PatternKind::Variant { ctor, fields } => {
            let (variant_name, enum_ty) = match &ctor.resolved {
                ResolvedRef::Constructor { variant, .. } => {
                    let ty = resolve_enum_type_attr(ctx, ir, ctor.ty);
                    (*variant, ty)
                }
                _ => unreachable!("non-constructor in variant pattern: {:?}", ctor.resolved),
            };

            // Cast scrutinee to the specific variant type
            let cast_op =
                adt::variant_cast(ir, location, scrutinee, enum_ty, enum_ty, variant_name);
            ir.push_op(block, cast_op.op_ref());
            let cast_val = cast_op.result(ir);

            // Extract each field and recursively bind
            let any_ty = ctx.any_type(ir);
            for (i, field_pat) in fields.iter().enumerate() {
                let field_op = adt::variant_get(
                    ir,
                    location,
                    cast_val,
                    any_ty,
                    enum_ty,
                    variant_name,
                    i as u32,
                );
                ir.push_op(block, field_op.op_ref());
                let field_val = field_op.result(ir);
                bind_pattern_fields(ctx, ir, block, location, field_val, field_pat);
            }
        }
        PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {
            // No bindings needed
        }
        PatternKind::Bind { local_id: None, .. } => {
            // Bind pattern without local_id — no binding
        }
        PatternKind::Tuple(elements) => {
            let any_ty = ctx.any_type(ir);
            let struct_ty = get_or_create_tuple_type(ctx, ir, pattern.id)
                .map(|(_, st)| st)
                .unwrap_or(any_ty);
            for (i, elem_pat) in elements.iter().enumerate() {
                let elem_op = adt::struct_get(ir, location, scrutinee, any_ty, struct_ty, i as u32);
                ir.push_op(block, elem_op.op_ref());
                let elem_val = elem_op.result(ir);
                bind_pattern_fields(ctx, ir, block, location, elem_val, elem_pat);
            }
        }
        _ => {
            unreachable!(
                "unsupported pattern in bind_pattern_fields: {:?}",
                pattern.kind
            )
        }
    }
}
