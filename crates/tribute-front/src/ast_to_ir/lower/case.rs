//! Case expression and pattern matching lowering.
//!
//! Lowers case expressions to a chain of `scf.if` operations, with
//! pattern checks generating boolean conditions and pattern bindings
//! extracted inside the matched region.

use tribute_ir::dialect::list;
use trunk_ir::Symbol;
use trunk_ir::adt_layout::get_enum_variants;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, func, scf};
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use crate::ast::{Arm, Expr, LiteralPattern, Pattern, PatternKind, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::{IrBuilder, get_or_create_tuple_type, is_irrefutable_pattern, resolve_enum_type_attr};

fn lower_case_region_expr_with_local_done_k<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let outer_done_k = builder.ctx.done_k;
    if outer_done_k.is_some() && super::expr::contains_cps_call_in_evaluation(builder.ctx, &expr) {
        let identity_done_k = super::create_identity_done_k(builder, location);
        builder.ctx.done_k = Some(identity_done_k);
    }
    let result = super::expr::lower_block_cps_for_expr(builder, expr).map(|(value, _)| value);
    builder.ctx.done_k = outer_done_k;
    result
}

/// Lower an arm body to the value yielded by its `scf.if` region.
///
/// A CPS call inside the arm must first produce the arm's value, so isolate it
/// from the continuation surrounding the whole case expression.
fn lower_case_arm_body<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    lower_case_region_expr_with_local_done_k(builder, location, expr)
}

/// Lower a guard condition inside the already-matched arm region.
fn lower_case_guard_condition<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    expr: Expr<TypedRef<'db>>,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let value = lower_case_region_expr_with_local_done_k(builder, location, expr)?;
    Some(builder.cast_if_needed(location, value, bool_ty))
}

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
            let mut scope = builder.ctx.scope();
            bind_pattern_fields(
                &mut scope,
                builder.ir,
                builder.block,
                location,
                scrutinee,
                &last.pattern,
            );
            let mut builder = IrBuilder::new(&mut scope, builder.ir, builder.block);
            lower_case_arm_body(&mut builder, location, last.body.clone())
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
            let any_ty = builder.ctx.anyref_type(builder.ir);
            let struct_ty = get_or_create_tuple_type(builder.ctx, builder.ir, pattern.id)
                .map(|(_, st)| st)
                .unwrap_or(any_ty);

            for (i, elem_pat) in elements.iter().enumerate() {
                let elem_ty = builder
                    .ctx
                    .get_node_type(elem_pat.id)
                    .map(|ty| builder.ctx.convert_type(builder.ir, *ty))
                    .unwrap_or(any_ty);
                let elem_op = adt::struct_get(
                    builder.ir, location, scrutinee, elem_ty, struct_ty, i as u32,
                );
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
        PatternKind::List(elements) => {
            emit_list_pattern_check(builder, location, scrutinee, pattern, elements, true)
        }
        PatternKind::ListRest { head, .. } => {
            emit_list_pattern_check(builder, location, scrutinee, pattern, head, false)
        }
        PatternKind::As { pattern, .. } => {
            emit_pattern_check(builder, location, scrutinee, pattern)
        }
        _ => {
            unreachable!("unsupported pattern in IR lowering: {:?}", pattern.kind)
        }
    }
}

fn emit_list_pattern_check<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    scrutinee: ValueRef,
    whole_pattern: &Pattern<TypedRef<'db>>,
    elements: &[Pattern<TypedRef<'db>>],
    exact: bool,
) -> Option<ValueRef> {
    let list_ty = builder.ctx.anyref_type(builder.ir);
    let list_ast_ty = builder.ctx.get_node_type(whole_pattern.id).copied();
    let element_ast_ty = list_ast_ty.and_then(|ty| match ty.kind(builder.db()) {
        crate::ast::TypeKind::Named { id, args, .. }
            if id.is_builtin_list(builder.db()) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    });
    let element_ty = element_ast_ty
        .map(|ty| builder.ctx.convert_type(builder.ir, ty))
        .unwrap_or_else(|| builder.ctx.anyref_type(builder.ir));

    emit_list_pattern_suffix(
        builder, location, scrutinee, elements, exact, list_ty, element_ty,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_list_pattern_suffix<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    current: ValueRef,
    elements: &[Pattern<TypedRef<'db>>],
    exact: bool,
    list_ty: TypeRef,
    element_ty: TypeRef,
) -> Option<ValueRef> {
    let bool_ty = builder.ctx.bool_type(builder.ir);
    let Some((element, rest)) = elements.split_first() else {
        let terminal = if exact {
            list::is_empty(builder.ir, location, current, bool_ty, element_ty).op_ref()
        } else {
            arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true)).op_ref()
        };
        builder.ir.push_op(builder.block, terminal);
        return Some(builder.ir.op_result(terminal, 0));
    };

    let empty = list::is_empty(builder.ir, location, current, bool_ty, element_ty);
    builder.ir.push_op(builder.block, empty.op_ref());
    let true_value = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(true));
    builder.ir.push_op(builder.block, true_value.op_ref());
    let non_empty = arith::xor(
        builder.ir,
        location,
        empty.result(builder.ir),
        true_value.result(builder.ir),
        bool_ty,
    );
    builder.ir.push_op(builder.block, non_empty.op_ref());

    let then_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let then_value = {
        let mut nested = IrBuilder::new(builder.ctx, builder.ir, then_block);
        let result_ty = nested
            .ctx
            .get_node_type(element.id)
            .map(|ty| nested.ctx.convert_type(nested.ir, *ty))
            .unwrap_or(element_ty);
        let head = list::head(nested.ir, location, current, result_ty, element_ty);
        nested.ir.push_op(nested.block, head.op_ref());
        let head_value = head.result(nested.ir);
        let element_condition = emit_pattern_check(&mut nested, location, head_value, element)?;

        let match_block = nested.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let suffix_condition = {
            let mut matched = IrBuilder::new(nested.ctx, nested.ir, match_block);
            let tail = list::tail(matched.ir, location, current, list_ty, element_ty);
            matched.ir.push_op(matched.block, tail.op_ref());
            let tail_value = tail.result(matched.ir);
            emit_list_pattern_suffix(
                &mut matched,
                location,
                tail_value,
                rest,
                exact,
                list_ty,
                element_ty,
            )?
        };
        let match_yield = scf::r#yield(nested.ir, location, [suffix_condition]);
        nested.ir.push_op(match_block, match_yield.op_ref());
        let match_region = nested.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![match_block],
            parent_op: None,
        });

        let mismatch_block = nested.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let false_value = arith::r#const(nested.ir, location, bool_ty, Attribute::Bool(false));
        nested.ir.push_op(mismatch_block, false_value.op_ref());
        let mismatch_yield = scf::r#yield(nested.ir, location, [false_value.result(nested.ir)]);
        nested.ir.push_op(mismatch_block, mismatch_yield.op_ref());
        let mismatch_region = nested.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![mismatch_block],
            parent_op: None,
        });

        let guarded_suffix = scf::r#if(
            nested.ir,
            location,
            element_condition,
            bool_ty,
            match_region,
            mismatch_region,
        );
        nested.ir.push_op(nested.block, guarded_suffix.op_ref());
        guarded_suffix.result(nested.ir)
    };
    let then_yield = scf::r#yield(builder.ir, location, [then_value]);
    builder.ir.push_op(then_block, then_yield.op_ref());
    let then_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![then_block],
        parent_op: None,
    });

    let else_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let false_value = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(false));
    builder.ir.push_op(else_block, false_value.op_ref());
    let else_yield = scf::r#yield(builder.ir, location, [false_value.result(builder.ir)]);
    builder.ir.push_op(else_block, else_yield.op_ref());
    let else_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });

    let guarded = scf::r#if(
        builder.ir,
        location,
        non_empty.result(builder.ir),
        bool_ty,
        then_region,
        else_region,
    );
    builder.ir.push_op(builder.block, guarded.op_ref());
    Some(guarded.result(builder.ir))
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
            let const_op =
                arith::r#const(builder.ir, location, i32_ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmpi(
                builder.ir,
                location,
                scrutinee,
                const_val,
                bool_ty,
                Symbol::new("eq"),
            );
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::Int(n) => {
            let value = super::validate_int_i31(builder.db(), location, *n)?;
            let const_op =
                arith::r#const(builder.ir, location, i32_ty, Attribute::Int(value as i128));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmpi(
                builder.ir,
                location,
                scrutinee,
                const_val,
                bool_ty,
                Symbol::new("eq"),
            );
            builder.ir.push_op(builder.block, cmp_op.op_ref());
            Some(cmp_op.result(builder.ir))
        }
        LiteralPattern::Bool(b) => {
            let const_op = arith::r#const(builder.ir, location, bool_ty, Attribute::Bool(*b));
            builder.ir.push_op(builder.block, const_op.op_ref());
            let const_val = const_op.result(builder.ir);
            let cmp_op = arith::cmpi(
                builder.ir,
                location,
                scrutinee,
                const_val,
                bool_ty,
                Symbol::new("eq"),
            );
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
) -> trunk_ir::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    // 1. Bind pattern fields (safe — we're inside the matched region)
    let (guard_cond, inner_then_region) = {
        let mut scope = ctx.scope();
        bind_pattern_fields(&mut scope, ir, block, location, scrutinee, &arm.pattern);

        // 2. Evaluate guard condition
        let guard_cond = {
            let mut builder = IrBuilder::new(&mut scope, ir, block);
            lower_case_guard_condition(&mut builder, location, guard_expr.clone())
        };
        let guard_cond = match guard_cond {
            Some(v) => v,
            None => {
                let bool_ty = scope.bool_type(ir);
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
                let mut builder = IrBuilder::new(&mut scope, ir, inner_block);
                let val = lower_case_arm_body(&mut builder, location, arm.body.clone());
                val.map(|v| builder.cast_if_needed(location, v, result_ty))
            };
            let yield_val = match result {
                Some(v) => v,
                None => {
                    let nil_ty = scope.nil_type(ir);
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

        (guard_cond, inner_then_region)
    };

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
) -> trunk_ir::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    let result = {
        let mut scope = ctx.scope();
        bind_pattern_fields(&mut scope, ir, block, location, scrutinee, &arm.pattern);

        let mut builder = IrBuilder::new(&mut scope, ir, block);
        let val = lower_case_arm_body(&mut builder, location, arm.body.clone());
        val.map(|v| builder.cast_if_needed(location, v, result_ty))
    };

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
) -> trunk_ir::refs::RegionRef {
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
            let variant_fields = get_enum_variants(ir, enum_ty)
                .expect("variant pattern must lower from an ADT enum type")
                .into_iter()
                .find_map(|(tag, fields)| (tag == variant_name).then_some(fields))
                .expect("resolved constructor must exist in enum metadata");
            for (i, field_pat) in fields.iter().enumerate() {
                let field_ty = variant_fields
                    .get(i)
                    .copied()
                    .expect("type checking must reject out-of-range variant pattern fields");
                let field_op = adt::variant_get(
                    ir,
                    location,
                    cast_val,
                    field_ty,
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
            let any_ty = ctx.anyref_type(ir);
            let struct_ty = get_or_create_tuple_type(ctx, ir, pattern.id)
                .map(|(_, st)| st)
                .unwrap_or(any_ty);
            for (i, elem_pat) in elements.iter().enumerate() {
                let elem_ty = ctx
                    .get_node_type(elem_pat.id)
                    .map(|ty| ctx.convert_type(ir, *ty))
                    .unwrap_or(any_ty);
                let elem_op =
                    adt::struct_get(ir, location, scrutinee, elem_ty, struct_ty, i as u32);
                ir.push_op(block, elem_op.op_ref());
                let elem_val = elem_op.result(ir);
                bind_pattern_fields(ctx, ir, block, location, elem_val, elem_pat);
            }
        }
        PatternKind::List(elements) => {
            bind_list_pattern_fields(ctx, ir, block, location, scrutinee, pattern, elements, None);
        }
        PatternKind::ListRest {
            head,
            rest,
            rest_local_id,
        } => {
            bind_list_pattern_fields(
                ctx,
                ir,
                block,
                location,
                scrutinee,
                pattern,
                head,
                rest.zip(*rest_local_id),
            );
        }
        PatternKind::As {
            pattern,
            name,
            local_id,
        } => {
            if let Some(id) = local_id {
                ctx.bind(*id, *name, scrutinee);
            }
            bind_pattern_fields(ctx, ir, block, location, scrutinee, pattern);
        }
        _ => {
            unreachable!(
                "unsupported pattern in bind_pattern_fields: {:?}",
                pattern.kind
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_list_pattern_fields<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: BlockRef,
    location: Location,
    scrutinee: ValueRef,
    whole_pattern: &Pattern<TypedRef<'db>>,
    elements: &[Pattern<TypedRef<'db>>],
    rest: Option<(Symbol, crate::ast::LocalId)>,
) {
    let list_ty = ctx.anyref_type(ir);
    let element_ty = ctx
        .get_node_type(whole_pattern.id)
        .and_then(|ty| match ty.kind(ctx.db()) {
            crate::ast::TypeKind::Named { id, args, .. }
                if id.is_builtin_list(ctx.db()) && args.len() == 1 =>
            {
                Some(ctx.convert_type(ir, args[0]))
            }
            _ => None,
        })
        .unwrap_or_else(|| ctx.anyref_type(ir));
    let mut current = scrutinee;

    for element in elements {
        let result_ty = ctx
            .get_node_type(element.id)
            .map(|ty| ctx.convert_type(ir, *ty))
            .unwrap_or(element_ty);
        let head = list::head(ir, location, current, result_ty, element_ty);
        ir.push_op(block, head.op_ref());
        bind_pattern_fields(ctx, ir, block, location, head.result(ir), element);
        let tail = list::tail(ir, location, current, list_ty, element_ty);
        ir.push_op(block, tail.op_ref());
        current = tail.result(ir);
    }

    if let Some((name, id)) = rest {
        ctx.bind(id, name, current);
    }
}
