//! Handle expression and ability operation lowering.
//!
//! Lowers `handle` expressions to `cont.push_prompt` + `cont.handler_dispatch`,
//! and ability operation calls to `cont.shift`.

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, cont, scf};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use crate::ast::{Expr, HandlerArm, HandlerKind, ResolvedRef, TypedRef};

use super::super::context::IrLoweringCtx;
use super::IrBuilder;
use super::case::bind_pattern_fields;

/// Lower an ability operation call to `cont.shift`.
pub(super) fn lower_ability_op_call<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    ability: Symbol,
    op: Symbol,
    args: Vec<ValueRef>,
    result_type: TypeRef,
) -> Option<ValueRef> {
    // Use the currently active prompt tag, or u32::MAX as sentinel
    let tag = builder.ctx.active_prompt_tag().unwrap_or(u32::MAX);

    // Create ability reference type for cont.shift
    let ability_ref = builder.ctx.ability_ref_type(builder.ir, ability, &[]);

    // Pack multiple arguments into a tuple if needed
    let shift_args = if args.len() > 1 {
        let any_ty = builder.ctx.any_type(builder.ir);
        let tuple_op = adt::struct_new(builder.ir, location, args, any_ty, any_ty);
        builder.ir.push_op(builder.block, tuple_op.op_ref());
        vec![tuple_op.result(builder.ir)]
    } else {
        args
    };

    // Create tag constant as operand for cont.shift
    let prompt_tag_ty = builder.ctx.prompt_tag_type(builder.ir);
    let tag_const = arith::r#const(
        builder.ir,
        location,
        prompt_tag_ty,
        Attribute::IntBits(tag as u64),
    );
    builder.ir.push_op(builder.block, tag_const.op_ref());
    let tag_value = tag_const.result(builder.ir);

    // Create empty handler region for shift
    let empty_block = builder.ir.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let handler_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![empty_block],
        parent_op: None,
    });

    // Generate cont.shift
    let shift_op = cont::shift(
        builder.ir,
        location,
        tag_value,
        shift_args,
        result_type,
        ability_ref,
        op,
        None, // op_table_index (set by resolve_evidence)
        None, // op_offset (set by resolve_evidence)
        handler_region,
    );
    builder.ir.push_op(builder.block, shift_op.op_ref());
    Some(shift_op.result(builder.ir))
}

/// Lower a `handle` expression.
pub(super) fn lower_handle<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    body: &Expr<TypedRef<'db>>,
    handlers: &[HandlerArm<TypedRef<'db>>],
) -> Option<ValueRef> {
    // Generate a fresh prompt tag
    let tag = builder.ctx.push_prompt_tag();

    let result_ty = builder.ctx.any_type(builder.ir);
    let step_ty = builder.ctx.any_type(builder.ir);

    // 1. Build the push_prompt body region
    let push_prompt_body = {
        let body_block = builder.ir.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        builder.ctx.enter_scope();
        let body_result = {
            let mut body_builder = IrBuilder::new(builder.ctx, builder.ir, body_block);
            super::expr::lower_expr(&mut body_builder, body.clone())
        };
        builder.ctx.exit_scope();

        let yield_val = match body_result {
            Some(v) => v,
            None => {
                let nil_ty = builder.ctx.nil_type(builder.ir);
                let op = arith::r#const(builder.ir, location, nil_ty, Attribute::Unit);
                builder.ir.push_op(body_block, op.op_ref());
                op.result(builder.ir)
            }
        };
        let yield_op = scf::r#yield(builder.ir, location, [yield_val]);
        builder.ir.push_op(body_block, yield_op.op_ref());

        builder.ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![body_block],
            parent_op: None,
        })
    };

    // Empty handlers region for push_prompt
    let empty_handlers = builder.ir.create_region(RegionData {
        location,
        blocks: Default::default(),
        parent_op: None,
    });

    // 2. Emit cont.push_prompt
    let push_prompt_op = cont::push_prompt(
        builder.ir,
        location,
        std::iter::empty(), // no args
        step_ty,
        Attribute::IntBits(tag as u64),
        push_prompt_body,
        empty_handlers,
    );
    builder.ir.push_op(builder.block, push_prompt_op.op_ref());
    let step_result = push_prompt_op.result(builder.ir);

    // Pop the prompt tag immediately after push_prompt.
    // Handlers are lowered with the outer prompt active.
    builder.ctx.pop_prompt_tag();

    // 3. Build handler_dispatch body region
    let handler_dispatch_body =
        build_handler_dispatch_body(builder.ctx, builder.ir, location, handlers, result_ty);

    // 4. Emit cont.handler_dispatch
    let handler_dispatch_op = cont::handler_dispatch(
        builder.ir,
        location,
        step_result,
        result_ty,
        tag,
        result_ty,
        handler_dispatch_body,
    );
    builder
        .ir
        .push_op(builder.block, handler_dispatch_op.op_ref());
    Some(handler_dispatch_op.result(builder.ir))
}

/// Build the handler dispatch body region with done and suspend ops.
fn build_handler_dispatch_body<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handlers: &[HandlerArm<TypedRef<'db>>],
    result_ty: TypeRef,
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
            HandlerKind::Result { .. } => result_handler = Some(handler),
            HandlerKind::Effect { .. } => effect_handlers.push(handler),
        }
    }

    // cont.done child op
    let done_body = build_done_handler_region(ctx, ir, location, result_handler, result_ty);
    let done_op = cont::done(ir, location, done_body);
    ir.push_op(block, done_op.op_ref());

    // cont.suspend child ops
    for effect_handler in &effect_handlers {
        let (ability_ref_ty, op_name) =
            extract_ability_ref_and_op_name(ctx, ir, location, effect_handler);
        let suspend_body = build_suspend_handler_region(ctx, ir, location, effect_handler);
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

    let result = if let Some(handler) = result_handler {
        ctx.enter_scope();

        if let HandlerKind::Result { binding } = &handler.kind {
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
    let HandlerKind::Effect { ability, op, .. } = &handler.kind else {
        unreachable!("extract_ability_ref_and_op_name called with non-effect handler");
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

/// Build a suspend handler region (effect handler).
fn build_suspend_handler_region<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    location: Location,
    handler: &HandlerArm<TypedRef<'db>>,
) -> trunk_ir::refs::RegionRef {
    let HandlerKind::Effect {
        params,
        continuation,
        continuation_local_id,
        ..
    } = &handler.kind
    else {
        unreachable!("build_suspend_handler_region called with non-effect handler");
    };

    let any_ty = ctx.any_type(ir);
    let cont_ty = ctx.continuation_type(ir, any_ty, any_ty, any_ty);

    let block = ir.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: cont_ty,
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

    // Bind continuation if named
    if let (Some(k_name), Some(k_local_id)) = (continuation, continuation_local_id) {
        ctx.bind(*k_local_id, *k_name, cont_value);
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

    ctx.exit_scope();

    let yield_val = match body_result {
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
