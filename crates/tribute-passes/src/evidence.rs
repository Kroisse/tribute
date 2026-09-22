//! Evidence utilities for the ability system.
//!
//! Helpers for working with evidence parameters on function types.

use tribute_ir::dialect::ability;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, ValueRef};

/// Find the evidence value from the enclosing `func.func`'s entry block.
///
/// Walks up the parent chain from the given op to find the containing
/// `func.func`, then returns its first block argument if it is an evidence type.
pub fn find_enclosing_evidence(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    let mut current = op;
    loop {
        let block = ctx.op(current).parent_block?;
        let region = ctx.block(block).parent_region?;
        let parent_op = ctx.region(region).parent_op?;
        if let Ok(func_op) = func::Func::from_op(ctx, parent_op) {
            let body = func_op.body(ctx);
            let entry = ctx.region(body).blocks[0];
            let args = ctx.block_args(entry);
            if !args.is_empty() && ability::is_evidence_type_ref(ctx, ctx.value_ty(args[0])) {
                return Some(args[0]);
            }
            return None;
        }
        current = parent_op;
    }
}
