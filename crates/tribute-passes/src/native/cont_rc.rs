//! Continuation RC rewrite pass.
//!
//! Wraps raw resume pointers with RC root metadata by inserting
//! `clif.call @__tribute_cont_wrap_from_tls(k)` after each
//! `clif.call @__tribute_get_yield_continuation()`.
//!
//! The wrapped pointer (`TributeContinuation*`) is then used by the
//! existing `__tribute_resume` / `__tribute_resume_drop` calls, which
//! expect wrapped continuations.
//!
//! ## Pipeline Position
//!
//! Runs at Phase 2.85, after `insert_rc` (Phase 2.8) and before
//! `resolve_unrealized_casts` (Phase 3). At this point all calls are
//! `clif.call` operations.

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::arena::TypeDataBuilder;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::arena::rewrite::helpers::erase_op;
use trunk_ir::arena::{BlockRef, OpRef, RegionRef, ValueRef};

use tribute_ir::arena::dialect::tribute_rt as arena_tribute_rt;

/// Rewrite continuation operations to use RC-safe wrappers.
///
/// This pass ensures that captured continuation stacks properly manage
/// RC-protected heap objects by wrapping raw resume pointers in
/// `TributeContinuation` structs that carry RC root metadata.
pub fn rewrite_cont_rc(ctx: &mut IrContext, module: ArenaModule) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in module_ops {
        if let Ok(func_op) = arena_clif::Func::from_op(ctx, op) {
            let sym = func_op.sym_name(ctx);
            if sym.with_str(|s| s.starts_with(super::rtti::RELEASE_FN_PREFIX)) {
                continue;
            }
            let body = func_op.body(ctx);
            rewrite_region(ctx, body);
        }
    }
}

fn rewrite_region(ctx: &mut IrContext, region: RegionRef) {
    let mut cont_values: HashSet<ValueRef> = HashSet::new();

    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        rewrite_block(ctx, block, &mut cont_values);
    }
}

fn rewrite_block(ctx: &mut IrContext, block: BlockRef, cont_values: &mut HashSet<ValueRef>) {
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());

    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut ops_to_erase: Vec<OpRef> = Vec::new();

    for (idx, &op) in ops.iter().enumerate() {
        // Rewrite nested regions first
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            rewrite_region(ctx, region);
        }

        // Check for clif.call operations
        if let Ok(call_op) = arena_clif::Call::from_op(ctx, op) {
            let callee = call_op.callee(ctx);

            if callee == Symbol::new("__tribute_get_yield_continuation") {
                let raw_k = ctx.op_result(op, 0);

                // Insert wrap call after this op
                let wrap_call = arena_clif::call(
                    ctx,
                    ctx.op(op).location,
                    [raw_k],
                    ptr_ty,
                    Symbol::new("__tribute_cont_wrap_from_tls"),
                );
                let wrapped_k = wrap_call.result(ctx);
                let wrap_op = wrap_call.op_ref();

                // Insert wrap_call into block after the current op
                if idx + 1 < ops.len() {
                    ctx.insert_op_before(block, ops[idx + 1], wrap_op);
                } else {
                    ctx.push_op(block, wrap_op);
                }

                // Replace all uses of raw_k with wrapped_k, then fix wrap_call
                ctx.replace_all_uses(raw_k, wrapped_k);
                ctx.set_op_operand(wrap_op, 0, raw_k);

                cont_values.insert(wrapped_k);
                continue;
            }
        }

        // Remove retain on wrapped continuation pointers
        if let Ok(_retain_op) = arena_tribute_rt::Retain::from_op(ctx, op) {
            let operands = ctx.op_operands(op).to_vec();
            let ptr = operands[0];
            if cont_values.contains(&ptr) {
                let result = ctx.op_result(op, 0);
                ctx.replace_all_uses(result, ptr);
                ops_to_erase.push(op);
                continue;
            }
        }

        // Remove release on wrapped continuation pointers
        if let Ok(_release_op) = arena_tribute_rt::Release::from_op(ctx, op) {
            let operands = ctx.op_operands(op).to_vec();
            let ptr = operands[0];
            if cont_values.contains(&ptr) {
                ops_to_erase.push(op);
                continue;
            }
        }
    }

    for op in ops_to_erase {
        erase_op(ctx, op);
    }
}
