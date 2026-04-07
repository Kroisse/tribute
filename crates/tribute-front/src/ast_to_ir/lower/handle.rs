//! Handle expression and ability operation lowering.
//!
//! Lowers `handle` expressions using CPS-based effect handling:
//! - Body is wrapped in a `closure.lambda` that returns `YieldResult`
//! - Handler dispatch uses `ability.handle_dispatch`
//! - Continuation calls use `func.call_indirect` (not `ability.resume`)
//!
//! Ability operation calls are lowered to `ability.perform` with CPS
//! continuations.

use std::collections::HashSet;

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use tribute_ir::dialect::{ability, closure};

use crate::ast::{Expr, HandlerArm, HandlerKind, Pattern, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::IrBuilder;
use super::case::bind_pattern_fields;

/// Lower a `fn` (tail-resumptive) ability operation call using `ability.call`.
///
/// Unlike `lower_ability_op_call`, this does NOT create a continuation closure
/// or use CPS. The result flows inline — no `func.return` is inserted, and
/// subsequent code continues normally.
pub(super) fn lower_ability_fn_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    ability: Symbol,
    op: Symbol,
    args: Vec<ValueRef>,
    result_type: TypeRef,
) -> Option<ValueRef> {
    let anyref_ty = builder.ctx.anyref_type(builder.ir);
    let ability_ref = builder.ctx.ability_ref_type(builder.ir, ability, &[]);

    // Pack multiple arguments into a tuple if needed
    let packed_args = if args.len() > 1 {
        let tuple_ty = super::expr::ability_args_tuple_type(builder.ir, args.len());
        let tuple_op = adt::struct_new(builder.ir, location, args, anyref_ty, tuple_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        vec![tuple_op.result(builder.ir)]
    } else {
        args
    };

    // Emit ability.call (direct call, no continuation)
    let call_op = ability::call(
        builder.ir,
        location,
        packed_args,
        anyref_ty,
        ability_ref,
        op,
    );
    builder.ir.push_op(builder.block, call_op.op_ref());
    let result = call_op.result(builder.ir);

    // Cast result from anyref to the logical result type if needed
    let result = if result_type != anyref_ty {
        let cast = core::unrealized_conversion_cast(builder.ir, location, result, result_type);
        builder.ir.push_op(builder.block, cast.op_ref());
        cast.result(builder.ir)
    } else {
        result
    };

    Some(result)
}

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
        let tuple_ty = super::expr::ability_args_tuple_type(builder.ir, args.len());
        let tuple_op = adt::struct_new(builder.ir, location, args, anyref_ty, tuple_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        vec![tuple_op.result(builder.ir)]
    } else {
        args
    };

    // Build identity continuation: fn(result) { result }
    // Continuation closures are internal mechanism, not user lambdas.
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
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Generate a fresh prompt tag and build body inside the prompt scope.
    let (tag, body_yr, effect_ty) = {
        let mut prompt_scope = builder.ctx.prompt_tag_scope();
        let tag = prompt_scope.tag();

        // Collect handled abilities for the body closure's effect annotation.
        // This ensures the body function is detected as effectful by evidence passes.
        let handled_ability_refs: Vec<_> = handlers
            .iter()
            .filter_map(|h| match &h.kind {
                HandlerKind::Fn { ability, .. } | HandlerKind::Op { ability, .. } => {
                    let name = match &ability.resolved {
                        ResolvedRef::Ability { id } => id.qualified(prompt_scope.db),
                        ResolvedRef::TypeDef { id } => id.qualified(prompt_scope.db),
                        _ => return None,
                    };
                    Some(prompt_scope.ability_ref_type(builder.ir, name, &[]))
                }
                _ => None,
            })
            .collect();
        let effect_ty = if !handled_ability_refs.is_empty() {
            Some(prompt_scope.effect_row_type(builder.ir, &handled_ability_refs, 0))
        } else {
            None
        };

        // 1. Build body as a CPS closure that returns anyref
        //    (tail calls handle effects; the final return goes to the handle frame)
        let builder = &mut IrBuilder::new(&mut prompt_scope, builder.ir, builder.block);
        let body_yr = build_cps_body(builder, location, body, anyref_ty, effect_ty)?;

        (tag, body_yr, effect_ty)
    };
    // Prompt tag popped — handlers are lowered with the outer prompt active.

    // Compute the body's logical result type for the done handler cast.
    let logical_result_ty = builder
        .ctx
        .get_node_type(body.id)
        .map(|ty| builder.ctx.convert_type(builder.ir, *ty));

    // 2. Build handler dispatch body region (ability.done + ability.suspend ops)
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

    // 3b. Build tr_dispatch_fn closure: (op_idx, value) -> anyref
    //     Only includes fn handlers. If no fn handlers exist, use null.
    let tr_dispatch_fn_val =
        build_tr_dispatch_closure(builder, location, handlers, anyref_ty, anyref_ty, effect_ty);

    // 4. Emit ability.handle_dispatch
    let dispatch_op = ability::handle_dispatch(
        builder.ir,
        location,
        body_yr,
        handler_fn_val,
        tr_dispatch_fn_val,
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

    // Check if the body contains effectful function calls that need CPS.
    let body_needs_done_k = super::expr::body_contains_cps_call(builder.ctx, body);
    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Build body closure entry block.
    // If body needs done_k: fn(done_k: anyref) -> anyref
    // Otherwise: fn() -> anyref
    let entry_block_args = if body_needs_done_k {
        vec![BlockArgData {
            ty: anyref_ty,
            attrs: Default::default(),
        }]
    } else {
        vec![]
    };

    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: entry_block_args,
        ops: Default::default(),
        parent_region: None,
    });

    {
        let mut scope = builder.ctx.scope();

        let prev_done_k = scope.done_k;
        if body_needs_done_k {
            let done_k_val = builder.ir.block_arg(entry_block, 0);
            scope.done_k = Some(done_k_val);
        }

        let result = {
            let mut body_builder = IrBuilder::new(&mut scope, builder.ir, entry_block);
            super::expr::lower_block_cps_for_expr(&mut body_builder, body.clone())
        };
        let Some((body_result, _is_cps)) = result else {
            scope.done_k = prev_done_k;
            return None;
        };

        // Always add func.return to terminate the block.
        // For ability.perform: lower_ability_perform removes dead code after it.
        {
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

        scope.done_k = prev_done_k;
    }

    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type depends on whether body needs done_k
    let body_params: Vec<TypeRef> = if body_needs_done_k {
        vec![anyref_ty]
    } else {
        vec![]
    };
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &body_params, result_ty, effect);
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

    // Call the body closure
    if body_needs_done_k {
        // Build an identity done_k: fn(result) { return result }
        // Created at the handle expression level (sibling to body_closure,
        // not nested inside it), so lower_closure_lambda lifts both correctly.
        let done_k_val = super::create_identity_done_k(builder, location);

        let call_op = func::call_indirect(
            builder.ir,
            location,
            body_closure,
            vec![done_k_val],
            result_ty,
        );
        builder.ir.push_op(builder.block, call_op.op_ref());
        Some(call_op.result(builder.ir))
    } else {
        let call_op = func::call_indirect(builder.ir, location, body_closure, vec![], result_ty);
        builder.ir.push_op(builder.block, call_op.op_ref());
        Some(call_op.result(builder.ir))
    }
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

    // ability.done child op
    let done_body = build_done_handler_region(
        ctx,
        ir,
        location,
        result_handler,
        result_ty,
        logical_result_ty,
    );
    let done_op = ability::done(ir, location, done_body);
    ir.push_op(block, done_op.op_ref());

    // ability.suspend / ability.yield child ops (handler arms use CPS mode for continuation calls)
    for effect_handler in &effect_handlers {
        let (ability_ref_ty, op_name) =
            extract_ability_ref_and_op_name(ctx, ir, location, effect_handler);
        let handler_body =
            build_cps_suspend_handler_region(ctx, ir, location, effect_handler, _yr_ty);
        let handler_op_ref = match &effect_handler.kind {
            // fn handler: tail-resumptive guaranteed at compile time → ability.yield directly
            HandlerKind::Fn { .. } => {
                ability::r#yield(ir, location, ability_ref_ty, op_name, handler_body).op_ref()
            }
            // op handler: may capture continuation → ability.suspend
            HandlerKind::Op { .. } => {
                ability::suspend(ir, location, ability_ref_ty, op_name, handler_body).op_ref()
            }
            _ => unreachable!(),
        };
        ir.push_op(block, handler_op_ref);
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
        let mut scope = ctx.scope();

        if let HandlerKind::Do { binding } = &handler.kind {
            bind_pattern_fields(&mut scope, ir, block, location, done_value, binding);
        }

        let mut builder = IrBuilder::new(&mut scope, ir, block);
        super::expr::lower_expr(&mut builder, handler.body.clone())
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
        ResolvedRef::Ability { id } => id.qualified(db),
        ResolvedRef::TypeDef { id } => id.qualified(db),
        other => {
            Diagnostic::new(
                format!(
                    "Expected ability type definition, got {:?}",
                    std::mem::discriminant(other)
                ),
                location.span,
                DiagnosticSeverity::Error,
                CompilationPhase::Lowering,
            )
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
/// `ability.resume`, and handler arm results are wrapped in `YieldResult::Done`
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

    let body_result = {
        let mut scope = ctx.scope();

        // Enable CPS handler mode so that continuation calls use func.call_indirect
        let prev_cps_mode = scope.cps_handler_mode;
        scope.cps_handler_mode = true;

        // Bind resume continuation if this is an `op` arm with resume
        if let Some(k_local_id) = resume_local_id {
            scope.bind(k_local_id, Symbol::new("resume"), cont_value);
        }

        // Bind params patterns
        bind_handler_params(&mut scope, ir, block, location, shift_value, params, any_ty);

        // Create identity done_k for the handler arm body.
        // Handler arms may call effectful functions (e.g., run_state) that
        // expect done_k. Since the handler arm's result flows to scf.yield,
        // the identity done_k just returns the result through the call chain.
        let prev_done_k = scope.done_k;
        {
            let mut dk_builder = IrBuilder::new(&mut scope, ir, block);
            let identity_dk = super::create_identity_done_k(&mut dk_builder, location);
            scope.done_k = Some(identity_dk);
        }

        // Evaluate the handler body
        let body_result = {
            let mut builder = IrBuilder::new(&mut scope, ir, block);
            super::expr::lower_expr(&mut builder, handler.body.clone())
        };

        scope.done_k = prev_done_k;
        scope.cps_handler_mode = prev_cps_mode;
        body_result
    };

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

    // Build closure body: ^bb0(%cps_dk: anyref, %k: anyref, %op_idx: i32, %value: anyref)
    // CPS convention: done_k is first param (unused in dispatch closure)
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

/// Build the tr_dispatch_fn closure for `fn` (tail-resumptive) operations.
///
/// Creates a `closure.lambda` with signature `(op_idx: i32, value: anyref) -> anyref`
/// that dispatches to the appropriate `fn` handler arm based on op_idx.
/// If there are no `fn` handlers, emits a null anyref constant instead.
fn build_tr_dispatch_closure<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    handlers: &[HandlerArm<TypedRef<'db>>],
    anyref_ty: TypeRef,
    yr_ty: TypeRef,
    effect_ty: Option<TypeRef>,
) -> ValueRef {
    let i32_ty = builder.ctx.i32_type(builder.ir);

    // Collect fn handlers only
    let fn_handlers: Vec<&HandlerArm<TypedRef<'db>>> = handlers
        .iter()
        .filter(|h| matches!(&h.kind, HandlerKind::Fn { .. }))
        .collect();

    if fn_handlers.is_empty() {
        // No fn handlers — use null pointer
        let null_op = arith::r#const(builder.ir, location, anyref_ty, Attribute::Int(0));
        builder.ir.push_op(builder.block, null_op.op_ref());
        return null_op.result(builder.ir);
    }

    // Analyze captures for fn handler arm bodies
    let mut free_vars = HashSet::new();
    for handler in &fn_handlers {
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

    // Build closure body: ^bb0(%cps_dk: anyref, %op_idx: i32, %value: anyref)
    // CPS convention: done_k first (unused), then real params
    let entry_block = builder.ir.create_block(BlockData {
        location,
        args: vec![
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
    let op_idx_val = builder.ir.block_arg(entry_block, 0);
    let value_val = builder.ir.block_arg(entry_block, 1);

    // Build dispatch chain for fn handlers
    // Reuse the same dispatch chain builder, but without k_val
    let dispatch_result = build_tr_dispatch_chain(
        builder.ctx,
        builder.ir,
        entry_block,
        location,
        &fn_handlers,
        op_idx_val,
        value_val,
        anyref_ty,
        yr_ty,
        i32_ty,
    );

    // Return the dispatch result
    let ret_op = func::r#return(builder.ir, location, [dispatch_result]);
    builder.ir.push_op(entry_block, ret_op.op_ref());

    let body_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    // Closure type: fn(i32, anyref) ->{effect} anyref
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &[i32_ty, anyref_ty], yr_ty, effect_ty);
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

/// Build the if-else dispatch chain for `fn` handler arms (no continuation parameter).
#[allow(clippy::too_many_arguments)]
fn build_tr_dispatch_chain<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: trunk_ir::refs::BlockRef,
    location: Location,
    fn_handlers: &[&HandlerArm<TypedRef<'db>>],
    op_idx_val: ValueRef,
    value_val: ValueRef,
    anyref_ty: TypeRef,
    yr_ty: TypeRef,
    i32_ty: TypeRef,
) -> ValueRef {
    use tribute_ir::dialect::ability::compute_op_idx;

    let i1_ty = ctx.bool_type(ir);
    let handler = fn_handlers[0];
    let is_last = fn_handlers.len() == 1;

    let (ability_ref_ty, op_name) = extract_ability_ref_and_op_name(ctx, ir, location, handler);

    // Compute expected op_idx
    let ability_data = ir.types.get(ability_ref_ty);
    let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(s)) => Some(*s),
        _ => None,
    };
    let expected_idx = compute_op_idx(ability_name, Some(op_name));

    // Build handler arm body region (no k_val — fn handler doesn't use resume)
    let arm_region =
        build_fn_handler_arm_for_dispatch(ctx, ir, location, handler, value_val, anyref_ty, yr_ty);

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
        let cmp = arith::cmpi(
            ir,
            location,
            op_idx_val,
            expected_const.result(ir),
            i1_ty,
            Symbol::new("eq"),
        );
        ir.push_op(block, cmp.op_ref());

        // Build else region with remaining arms
        let else_block = ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let else_result = build_tr_dispatch_chain(
            ctx,
            ir,
            else_block,
            location,
            &fn_handlers[1..],
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

/// Build a `fn` handler arm region for the tr_dispatch closure.
///
/// Similar to `build_handler_arm_for_dispatch` but without k_val.
fn build_fn_handler_arm_for_dispatch<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handler: &HandlerArm<TypedRef<'db>>,
    value_val: ValueRef,
    anyref_ty: TypeRef,
    _yr_ty: TypeRef,
) -> trunk_ir::refs::RegionRef {
    let params = match &handler.kind {
        HandlerKind::Fn { params, .. } => params,
        _ => unreachable!("build_fn_handler_arm_for_dispatch called with non-fn handler"),
    };

    let block = ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });

    let body_result = {
        let mut scope = ctx.scope();

        // No resume binding for fn handlers

        // Bind params patterns
        bind_handler_params(
            &mut scope, ir, block, location, value_val, params, anyref_ty,
        );

        // Evaluate the handler body
        {
            let mut inner_builder = IrBuilder::new(&mut scope, ir, block);
            super::expr::lower_expr(&mut inner_builder, handler.body.clone())
        }
    };

    let body_val = match body_result {
        Some(v) => v,
        None => {
            let nil_ty = ctx.nil_type(ir);
            let op = arith::r#const(ir, location, nil_ty, Attribute::Unit);
            ir.push_op(block, op.op_ref());
            op.result(ir)
        }
    };

    // Cast to anyref if needed
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
        let cmp = arith::cmpi(
            ir,
            location,
            op_idx_val,
            expected_const.result(ir),
            i1_ty,
            Symbol::new("eq"),
        );
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

    let body_result = {
        let mut scope = ctx.scope();

        let prev_cps_mode = scope.cps_handler_mode;
        scope.cps_handler_mode = true;

        // Bind resume continuation if this is an `op` arm with resume
        if let Some(k_local_id) = resume_local_id {
            scope.bind(k_local_id, Symbol::new("resume"), k_val);
        }

        // Bind params patterns
        bind_handler_params(
            &mut scope, ir, block, location, value_val, params, anyref_ty,
        );

        // Create identity done_k for effectful calls in handler arm body.
        let prev_done_k = scope.done_k;
        {
            let mut dk_builder = super::IrBuilder::new(&mut scope, ir, block);
            let identity_dk = super::create_identity_done_k(&mut dk_builder, location);
            scope.done_k = Some(identity_dk);
        }

        // Evaluate the handler body
        let body_result = {
            let mut inner_builder = super::IrBuilder::new(&mut scope, ir, block);
            super::expr::lower_expr(&mut inner_builder, handler.body.clone())
        };

        scope.done_k = prev_done_k;
        scope.cps_handler_mode = prev_cps_mode;
        body_result
    };

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

/// Bind handler arm parameters from an ability operation's packed arguments.
///
/// Single-param ops pass the value directly; multi-param ops pack arguments
/// into an `anyref` tuple struct. This function destructures the tuple and
/// inserts `unrealized_conversion_cast` to convert each `anyref` field to
/// the pattern's actual IR type (e.g., `core.i32` for `Nat`).
fn bind_handler_params<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    block: trunk_ir::refs::BlockRef,
    location: Location,
    value: ValueRef,
    params: &[Pattern<TypedRef<'db>>],
    anyref_ty: TypeRef,
) {
    if params.len() == 1 {
        bind_pattern_fields(ctx, ir, block, location, value, &params[0]);
    } else if params.len() > 1 {
        let tuple_ty = super::expr::ability_args_tuple_type(ir, params.len());
        for (i, param) in params.iter().enumerate() {
            let field_op = adt::struct_get(ir, location, value, anyref_ty, tuple_ty, i as u32);
            ir.push_op(block, field_op.op_ref());
            let mut field_val = field_op.result(ir);

            // Cast anyref → actual param type (inserts unbox via unrealized_conversion_cast)
            let param_ty = ctx
                .get_node_type(param.id)
                .map(|ty| ctx.convert_type(ir, *ty))
                .unwrap_or(anyref_ty);
            if param_ty != anyref_ty {
                let cast_op = core::unrealized_conversion_cast(ir, location, field_val, param_ty);
                ir.push_op(block, cast_op.op_ref());
                field_val = cast_op.result(ir);
            }

            bind_pattern_fields(ctx, ir, block, location, field_val, param);
        }
    }
}
