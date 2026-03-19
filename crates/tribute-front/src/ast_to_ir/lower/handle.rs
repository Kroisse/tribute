//! Handle expression and ability operation lowering.
//!
//! Lowers `handle` expressions using CPS-based effect handling:
//! - Body is wrapped in a `closure.lambda` that returns `YieldResult`
//! - Handler dispatch uses `ability.handle_dispatch`
//! - Continuation calls use `func.call_indirect` (not `cont.resume`)
//!
//! Also lowers ability operation calls to `cont.shift` (fallback for
//! non-CPS contexts).

use std::collections::HashSet;

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, cont, core, func, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use tribute_ir::dialect::{ability, closure};

use crate::ast::{Expr, HandlerArm, HandlerKind, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::IrBuilder;
use super::case::bind_pattern_fields;

/// Lower an ability operation call using CPS (`ability.perform`).
///
/// Creates an identity continuation that wraps the result in `YieldResult::Done`,
/// then emits `ability.perform` with this continuation. The returned value is
/// the `ability.perform` result (will become `YieldResult::Shift` after lowering).
///
/// Note: callers in expression position will have dead code after this op,
/// since `lower_ability_perform` adds `func.return` before the caller's
/// subsequent ops.
pub(super) fn lower_ability_op_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    ability: Symbol,
    op: Symbol,
    args: Vec<ValueRef>,
    _result_type: TypeRef,
) -> Option<ValueRef> {
    let anyref_ty = builder.ctx.anyref_type(builder.ir);
    let ability_ref = builder.ctx.ability_ref_type(builder.ir, ability, &[]);

    // Pack multiple arguments into a tuple if needed
    let packed_args = if args.len() > 1 {
        let tuple_op = adt::struct_new(builder.ir, location, args, anyref_ty, anyref_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        vec![tuple_op.result(builder.ir)]
    } else {
        args
    };

    // Build identity continuation: fn(result) { result }
    let entry_block = builder.ir.create_block(trunk_ir::context::BlockData {
        location,
        args: vec![trunk_ir::context::BlockArgData {
            ty: anyref_ty,
            attrs: Default::default(),
        }],
        ops: Default::default(),
        parent_region: None,
    });
    let param_val = builder.ir.block_arg(entry_block, 0);
    let ret = func::r#return(builder.ir, location, [param_val]);
    builder.ir.push_op(entry_block, ret.op_ref());

    let body_region = builder.ir.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &[anyref_ty], anyref_ty, None);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
    let lambda_op = closure::lambda(
        builder.ir,
        location,
        Vec::<ValueRef>::new(),
        closure_ty,
        body_region,
    );
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    let continuation = lambda_op.result(builder.ir);

    // Emit ability.perform
    let perform_op = ability::perform(
        builder.ir,
        location,
        continuation,
        packed_args,
        anyref_ty,
        ability_ref,
        op,
    );
    builder.ir.push_op(builder.block, perform_op.op_ref());
    Some(perform_op.result(builder.ir))
}

/// Lower a `handle` expression using CPS-based effect handling.
///
/// 1. Body is wrapped in a `closure.lambda` that returns `YieldResult`
/// 2. The body closure is called to produce a `YieldResult`
/// 3. `ability.handle_dispatch` dispatches on the `YieldResult`
/// 4. Handler arms use `func.call_indirect` for continuation calls
pub(super) fn lower_handle<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    body: &Expr<TypedRef<'db>>,
    handlers: &[HandlerArm<TypedRef<'db>>],
) -> Option<ValueRef> {
    // Generate a fresh prompt tag
    let tag = builder.ctx.push_prompt_tag();

    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Collect handled abilities for the body closure's effect annotation.
    // This ensures the body function is detected as effectful by evidence passes.
    let handled_ability_refs: Vec<_> = handlers
        .iter()
        .filter_map(|h| match &h.kind {
            HandlerKind::Fn { ability, .. } | HandlerKind::Op { ability, .. } => {
                let name = match &ability.resolved {
                    ResolvedRef::Ability { id } => {
                        Symbol::from_dynamic(&id.qualified_name(builder.db()).to_string())
                    }
                    ResolvedRef::TypeDef { id } => {
                        Symbol::from_dynamic(&id.qualified_name(builder.db()).to_string())
                    }
                    _ => return None,
                };
                Some(builder.ctx.ability_ref_type(builder.ir, name, &[]))
            }
            _ => None,
        })
        .collect();
    let effect_ty = if !handled_ability_refs.is_empty() {
        Some(
            builder
                .ctx
                .effect_row_type(builder.ir, &handled_ability_refs, 0),
        )
    } else {
        None
    };

    // 1. Build body as a CPS closure that returns anyref
    //    (tail calls handle effects; the final return goes to the handle frame)
    let body_yr = match build_cps_body(builder, location, body, anyref_ty, effect_ty) {
        Some(yr) => yr,
        None => {
            builder.ctx.pop_prompt_tag();
            return None;
        }
    };

    // Pop the prompt tag immediately after body.
    // Handlers are lowered with the outer prompt active.
    builder.ctx.pop_prompt_tag();

    // Compute the body's logical result type for the done handler cast.
    let logical_result_ty = builder
        .ctx
        .get_node_type(body.id)
        .map(|ty| builder.ctx.convert_type(builder.ir, *ty));

    // 2. Build handler dispatch body region (cont.done + cont.suspend ops)
    let handler_dispatch_body = build_cps_handler_dispatch_body(
        builder.ctx,
        builder.ir,
        location,
        handlers,
        anyref_ty,
        anyref_ty,
        logical_result_ty,
    );

    // 3. Build handler_dispatch closure: (k, op_idx, value) -> anyref
    let handler_fn_val = build_handler_dispatch_closure(
        builder, location, handlers, anyref_ty, anyref_ty, effect_ty,
    );

    // 4. Emit ability.handle_dispatch
    let dispatch_op = ability::handle_dispatch(
        builder.ir,
        location,
        body_yr,
        handler_fn_val,
        anyref_ty,
        tag,
        anyref_ty,
        handler_dispatch_body,
    );
    builder.ir.push_op(builder.block, dispatch_op.op_ref());
    Some(dispatch_op.result(builder.ir))
}

/// Build the handle body as a CPS closure and call it.
///
/// Returns the body result (anyref). In the tail-call CPS design,
/// the body function may tail-call handler_dispatch for effects;
/// the eventual return value arrives via the tail call chain.
fn build_cps_body<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    body: &Expr<TypedRef<'db>>,
    result_ty: TypeRef,
    effect: Option<TypeRef>,
) -> Option<ValueRef> {
    // Analyze captures for the body expression
    let mut free_vars = HashSet::new();
    super::lambda::collect_free_vars(body, &mut free_vars);
    let mut captures = Vec::new();
    for (local_id, name, value) in builder.ctx.all_bindings() {
        if free_vars.contains(&local_id) {
            captures.push(super::super::context::CaptureInfo {
                name,
                local_id,
                ty: builder.ir.value_ty(value),
                value,
            });
        }
    }

    // Build body closure: no params, returns anyref
    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    builder.ctx.enter_scope();
    let result = {
        let mut body_builder = IrBuilder::new(builder.ctx, builder.ir, entry_block);
        super::expr::lower_block_cps_for_expr(&mut body_builder, body.clone())
    };
    let Some((body_result, is_cps)) = result else {
        builder.ctx.exit_scope();
        return None;
    };

    // In the tail-call CPS design:
    // - CPS path (is_cps=true): ability.perform will be lowered to tail_call
    //   handler_dispatch. The body function never returns normally on this path.
    // - Pure path (is_cps=false): body returns the result directly.
    if !is_cps {
        let result = if builder.ir.value_ty(body_result) != result_ty {
            let cast =
                core::unrealized_conversion_cast(builder.ir, location, body_result, result_ty);
            builder.ir.push_op(entry_block, cast.op_ref());
            cast.result(builder.ir)
        } else {
            body_result
        };
        let ret = func::r#return(builder.ir, location, [result]);
        builder.ir.push_op(entry_block, ret.op_ref());
    }
    builder.ctx.exit_scope();

    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type: fn() ->{effect} anyref
    let closure_func_ty = builder
        .ctx
        .func_type_with_effect(builder.ir, &[], result_ty, effect);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    let capture_values: Vec<ValueRef> = captures.iter().map(|c| c.value).collect();
    let lambda_op = closure::lambda(
        builder.ir,
        location,
        capture_values,
        closure_ty,
        body_region,
    );
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    let body_closure = lambda_op.result(builder.ir);

    // Call the body closure → anyref result
    let call_op = func::call_indirect(builder.ir, location, body_closure, vec![], result_ty);
    builder.ir.push_op(builder.block, call_op.op_ref());

    Some(call_op.result(builder.ir))
}

/// Build the handler dispatch body region with done and suspend ops (CPS version).
fn build_cps_handler_dispatch_body<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handlers: &[HandlerArm<TypedRef<'db>>],
    result_ty: TypeRef,
    _yr_ty: TypeRef,
    logical_result_ty: Option<TypeRef>,
) -> trunk_ir::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    // Separate result handler from effect handlers
    let mut result_handler: Option<&HandlerArm<TypedRef<'db>>> = None;
    let mut effect_handlers: Vec<&HandlerArm<TypedRef<'db>>> = Vec::new();

    for handler in handlers {
        match &handler.kind {
            HandlerKind::Do { .. } => result_handler = Some(handler),
            HandlerKind::Fn { .. } | HandlerKind::Op { .. } => effect_handlers.push(handler),
        }
    }

    // cont.done child op
    let done_body = build_done_handler_region(
        ctx,
        ir,
        location,
        result_handler,
        result_ty,
        logical_result_ty,
    );
    let done_op = cont::done(ir, location, done_body);
    ir.push_op(block, done_op.op_ref());

    // cont.suspend child ops (handler arms use CPS mode for continuation calls)
    for effect_handler in &effect_handlers {
        let (ability_ref_ty, op_name) =
            extract_ability_ref_and_op_name(ctx, ir, location, effect_handler);
        let suspend_body =
            build_cps_suspend_handler_region(ctx, ir, location, effect_handler, _yr_ty);
        let suspend_op = cont::suspend(ir, location, ability_ref_ty, op_name, suspend_body);
        ir.push_op(block, suspend_op.op_ref());
    }

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build the done handler region (normal completion handler).
fn build_done_handler_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    result_handler: Option<&HandlerArm<TypedRef<'db>>>,
    result_ty: TypeRef,
    logical_result_ty: Option<TypeRef>,
) -> trunk_ir::refs::RegionRef {
    let block = ir.create_block(BlockData {
        location,
        args: vec![BlockArgData {
            ty: result_ty,
            attrs: Default::default(),
        }],
        ops: Default::default(),
        parent_region: None,
    });
    let done_value = ir.block_arg(block, 0);

    // Cast done_value from anyref to the body's logical result type so that
    // handler body code (e.g. `result + 1`) sees the correct operand type.
    let done_value = if let Some(logical_ty) = logical_result_ty {
        if logical_ty != result_ty {
            let cast = core::unrealized_conversion_cast(ir, location, done_value, logical_ty);
            ir.push_op(block, cast.op_ref());
            cast.result(ir)
        } else {
            done_value
        }
    } else {
        done_value
    };

    let result = if let Some(handler) = result_handler {
        ctx.enter_scope();

        if let HandlerKind::Do { binding } = &handler.kind {
            bind_pattern_fields(ctx, ir, block, location, done_value, binding);
        }

        let body_result = {
            let mut builder = IrBuilder::new(ctx, ir, block);
            super::expr::lower_expr(&mut builder, handler.body.clone())
        };

        ctx.exit_scope();
        body_result
    } else {
        Some(done_value)
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

/// Extract the ability ref type and op name from a handler arm.
fn extract_ability_ref_and_op_name<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handler: &HandlerArm<TypedRef<'db>>,
) -> (TypeRef, Symbol) {
    let db = ctx.db;
    let (ability, op) = match &handler.kind {
        HandlerKind::Fn { ability, op, .. } | HandlerKind::Op { ability, op, .. } => (ability, op),
        _ => unreachable!("extract_ability_ref_and_op_name called with non-effect handler"),
    };

    let ability_name = match &ability.resolved {
        ResolvedRef::Ability { id } => Symbol::from_dynamic(&id.qualified_name(db).to_string()),
        ResolvedRef::TypeDef { id } => Symbol::from_dynamic(&id.qualified_name(db).to_string()),
        other => {
            Diagnostic {
                message: format!(
                    "Expected ability type definition, got {:?}",
                    std::mem::discriminant(other)
                ),
                span: location.span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(db);
            Symbol::new("__unknown_ability__")
        }
    };

    let ability_ref_type = ctx.ability_ref_type(ir, ability_name, &[]);
    (ability_ref_type, *op)
}

/// Build a CPS suspend handler region (effect handler).
///
/// In CPS mode, continuation calls use `func.call_indirect` instead of
/// `cont.resume`, and handler arm results are wrapped in `YieldResult::Done`
/// if not already a `YieldResult`.
fn build_cps_suspend_handler_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handler: &HandlerArm<TypedRef<'db>>,
    _yr_ty: TypeRef,
) -> trunk_ir::refs::RegionRef {
    let (params, resume_local_id) = match &handler.kind {
        HandlerKind::Op {
            params,
            resume_local_id,
            ..
        } => (params, *resume_local_id),
        HandlerKind::Fn { params, .. } => (params, None),
        _ => unreachable!("build_cps_suspend_handler_region called with non-effect handler"),
    };

    let any_ty = ctx.anyref_type(ir);

    // Block args: [continuation (anyref), shift_value (anyref)]
    // In CPS, continuation is a closure (anyref), not a cont.continuation type.
    let block = ir.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: any_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: any_ty,
                attrs: Default::default(),
            },
        ],
        ops: Default::default(),
        parent_region: None,
    });
    let cont_value = ir.block_arg(block, 0);
    let shift_value = ir.block_arg(block, 1);

    ctx.enter_scope();

    // Enable CPS handler mode so that continuation calls use func.call_indirect
    let prev_cps_mode = ctx.cps_handler_mode;
    ctx.cps_handler_mode = true;

    // Bind resume continuation if this is an `op` arm with resume
    if let Some(k_local_id) = resume_local_id {
        ctx.bind(k_local_id, Symbol::new("resume"), cont_value);
    }

    // Bind params patterns
    if params.len() == 1 {
        bind_pattern_fields(ctx, ir, block, location, shift_value, &params[0]);
    } else if params.len() > 1 {
        // Multiple params - destructure as tuple
        for (i, param) in params.iter().enumerate() {
            let field_op = adt::struct_get(ir, location, shift_value, any_ty, any_ty, i as u32);
            ir.push_op(block, field_op.op_ref());
            let field_val = field_op.result(ir);
            bind_pattern_fields(ctx, ir, block, location, field_val, param);
        }
    }

    // Evaluate the handler body
    let body_result = {
        let mut builder = IrBuilder::new(ctx, ir, block);
        super::expr::lower_expr(&mut builder, handler.body.clone())
    };

    ctx.cps_handler_mode = prev_cps_mode;
    ctx.exit_scope();

    let body_val = match body_result {
        Some(v) => v,
        None => {
            let nil_ty = ctx.nil_type(ir);
            let op = arith::r#const(ir, location, nil_ty, Attribute::Unit);
            ir.push_op(block, op.op_ref());
            op.result(ir)
        }
    };

    // Cast to anyref if needed (no YieldResult wrapping — tail call CPS)
    let result_val = if ir.value_ty(body_val) != any_ty {
        let cast = core::unrealized_conversion_cast(ir, location, body_val, any_ty);
        ir.push_op(block, cast.op_ref());
        cast.result(ir)
    } else {
        body_val
    };

    let yield_op = scf::r#yield(ir, location, [result_val]);
    ir.push_op(block, yield_op.op_ref());

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build the handler_dispatch closure for tail-call-based CPS.
///
/// Creates a `closure.lambda` with signature `(k: anyref, op_idx: i32, value: anyref) -> YieldResult`
/// that dispatches to the appropriate handler arm based on op_idx.
///
/// The closure captures variables from the enclosing scope that the handler
/// arm bodies reference. `closure_lower` will later extract it to a top-level
/// function.
fn build_handler_dispatch_closure<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    handlers: &[HandlerArm<TypedRef<'db>>],
    anyref_ty: TypeRef,
    yr_ty: TypeRef,
    effect_ty: Option<TypeRef>,
) -> ValueRef {
    let i32_ty = builder.ctx.i32_type(builder.ir);

    // Collect effect handlers
    let effect_handlers: Vec<&HandlerArm<TypedRef<'db>>> = handlers
        .iter()
        .filter(|h| matches!(&h.kind, HandlerKind::Fn { .. } | HandlerKind::Op { .. }))
        .collect();

    // Analyze captures for all handler arm bodies
    let mut free_vars = HashSet::new();
    for handler in &effect_handlers {
        super::lambda::collect_free_vars(&handler.body, &mut free_vars);
    }
    let mut captures = Vec::new();
    for (local_id, name, value) in builder.ctx.all_bindings() {
        if free_vars.contains(&local_id) {
            captures.push(super::super::context::CaptureInfo {
                name,
                local_id,
                ty: builder.ir.value_ty(value),
                value,
            });
        }
    }

    // Build closure body: ^bb0(%k: anyref, %op_idx: i32, %value: anyref)
    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: anyref_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: anyref_ty,
                attrs: Default::default(),
            },
        ],
        ops: Default::default(),
        parent_region: None,
    });
    let k_val = builder.ir.block_arg(entry_block, 0);
    let op_idx_val = builder.ir.block_arg(entry_block, 1);
    let value_val = builder.ir.block_arg(entry_block, 2);

    // Build dispatch chain: if-else on op_idx for each effect handler arm
    if effect_handlers.is_empty() {
        // No effect handlers — unreachable
        let unreachable_op = func::unreachable(builder.ir, location);
        builder.ir.push_op(entry_block, unreachable_op.op_ref());
    } else {
        let dispatch_result = build_handler_dispatch_chain(
            builder.ctx,
            builder.ir,
            entry_block,
            location,
            &effect_handlers,
            k_val,
            op_idx_val,
            value_val,
            anyref_ty,
            yr_ty,
            i32_ty,
        );

        // Return the dispatch result
        let ret_op = func::r#return(builder.ir, location, [dispatch_result]);
        builder.ir.push_op(entry_block, ret_op.op_ref());
    }

    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type: fn(anyref, i32, anyref) ->{effect} YieldResult
    let closure_func_ty = builder.ctx.func_type_with_effect(
        builder.ir,
        &[anyref_ty, i32_ty, anyref_ty],
        yr_ty,
        effect_ty,
    );
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);

    let capture_values: Vec<ValueRef> = captures.iter().map(|c| c.value).collect();
    let lambda_op = closure::lambda(
        builder.ir,
        location,
        capture_values,
        closure_ty,
        body_region,
    );
    builder.ir.push_op(builder.block, lambda_op.op_ref());
    lambda_op.result(builder.ir)
}

/// Build the if-else dispatch chain for handler arms inside the handler_dispatch closure.
#[allow(clippy::too_many_arguments)]
fn build_handler_dispatch_chain<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: trunk_ir::refs::BlockRef,
    location: Location,
    effect_handlers: &[&HandlerArm<TypedRef<'db>>],
    k_val: ValueRef,
    op_idx_val: ValueRef,
    value_val: ValueRef,
    anyref_ty: TypeRef,
    yr_ty: TypeRef,
    i32_ty: TypeRef,
) -> ValueRef {
    use tribute_ir::dialect::ability::compute_op_idx;

    let i1_ty = ctx.bool_type(ir);
    let handler = effect_handlers[0];
    let is_last = effect_handlers.len() == 1;

    let (ability_ref_ty, op_name) = extract_ability_ref_and_op_name(ctx, ir, location, handler);

    // Compute expected op_idx
    let ability_data = ir.types.get(ability_ref_ty);
    let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(s)) => Some(*s),
        _ => None,
    };
    let expected_idx = compute_op_idx(ability_name, Some(op_name));

    // Build handler arm body region
    let arm_region = build_handler_arm_for_dispatch(
        ctx, ir, location, handler, k_val, value_val, anyref_ty, yr_ty,
    );

    if is_last {
        // Last arm: unconditional
        let true_const = arith::r#const(ir, location, i1_ty, Attribute::Int(1));
        ir.push_op(block, true_const.op_ref());

        let else_block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let unreachable = func::unreachable(ir, location);
        ir.push_op(else_block, unreachable.op_ref());
        let else_region = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![else_block],
            parent_op: None,
        });

        let if_op = scf::r#if(
            ir,
            location,
            true_const.result(ir),
            yr_ty,
            arm_region,
            else_region,
        );
        ir.push_op(block, if_op.op_ref());
        if_op.result(ir)
    } else {
        // Compare op_idx
        let expected_const =
            arith::r#const(ir, location, i32_ty, Attribute::Int(expected_idx as i128));
        ir.push_op(block, expected_const.op_ref());
        let cmp = arith::cmp_eq(ir, location, op_idx_val, expected_const.result(ir), i1_ty);
        ir.push_op(block, cmp.op_ref());

        // Build else region with remaining arms
        let else_block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let else_result = build_handler_dispatch_chain(
            ctx,
            ir,
            else_block,
            location,
            &effect_handlers[1..],
            k_val,
            op_idx_val,
            value_val,
            anyref_ty,
            yr_ty,
            i32_ty,
        );
        let else_yield = scf::r#yield(ir, location, [else_result]);
        ir.push_op(else_block, else_yield.op_ref());
        let else_region = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![else_block],
            parent_op: None,
        });

        let if_op = scf::r#if(ir, location, cmp.result(ir), yr_ty, arm_region, else_region);
        ir.push_op(block, if_op.op_ref());
        if_op.result(ir)
    }
}

/// Build a handler arm region for the handler_dispatch closure.
///
/// Similar to `build_cps_suspend_handler_region` but uses pre-existing
/// k and value values instead of block args, and wraps result in scf.yield.
#[allow(clippy::too_many_arguments)]
fn build_handler_arm_for_dispatch<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handler: &HandlerArm<TypedRef<'db>>,
    k_val: ValueRef,
    value_val: ValueRef,
    anyref_ty: TypeRef,
    _yr_ty: TypeRef,
) -> trunk_ir::refs::RegionRef {
    let (params, resume_local_id) = match &handler.kind {
        HandlerKind::Op {
            params,
            resume_local_id,
            ..
        } => (params, *resume_local_id),
        HandlerKind::Fn { params, .. } => (params, None),
        _ => unreachable!("build_handler_arm_for_dispatch called with non-effect handler"),
    };

    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    ctx.enter_scope();

    let prev_cps_mode = ctx.cps_handler_mode;
    ctx.cps_handler_mode = true;

    // Bind resume continuation if this is an `op` arm with resume
    if let Some(k_local_id) = resume_local_id {
        ctx.bind(k_local_id, Symbol::new("resume"), k_val);
    }

    // Bind params patterns
    if params.len() == 1 {
        bind_pattern_fields(ctx, ir, block, location, value_val, &params[0]);
    } else if params.len() > 1 {
        for (i, param) in params.iter().enumerate() {
            let field_op = adt::struct_get(ir, location, value_val, anyref_ty, anyref_ty, i as u32);
            ir.push_op(block, field_op.op_ref());
            let field_val = field_op.result(ir);
            bind_pattern_fields(ctx, ir, block, location, field_val, param);
        }
    }

    // Evaluate the handler body
    let body_result = {
        let mut inner_builder = IrBuilder::new(ctx, ir, block);
        super::expr::lower_expr(&mut inner_builder, handler.body.clone())
    };

    ctx.cps_handler_mode = prev_cps_mode;
    ctx.exit_scope();

    let body_val = match body_result {
        Some(v) => v,
        None => {
            let nil_ty = ctx.nil_type(ir);
            let op = arith::r#const(ir, location, nil_ty, Attribute::Unit);
            ir.push_op(block, op.op_ref());
            op.result(ir)
        }
    };

    // Cast to anyref if needed (no YieldResult wrapping — tail call CPS)
    let result_val = if ir.value_ty(body_val) != anyref_ty {
        let cast = core::unrealized_conversion_cast(ir, location, body_val, anyref_ty);
        ir.push_op(block, cast.op_ref());
        cast.result(ir)
    } else {
        body_val
    };

    let yield_op = scf::r#yield(ir, location, [result_val]);
    ir.push_op(block, yield_op.op_ref());

    ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}
