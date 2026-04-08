//! Evidence utilities for the ability system.
//!
//! Helpers for working with evidence parameters on function types.

use tribute_ir::dialect::ability as arena_ability;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};

/// Check if a `core.func` type has evidence as its first parameter.
pub(crate) fn has_evidence_first_param(ctx: &IrContext, func_ty: TypeRef) -> bool {
    let data = ctx.types.get(func_ty);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return false;
    }
    // params[0] = return, params[1..] = param types
    if data.params.len() < 2 {
        return false;
    }
    arena_ability::is_evidence_type_ref(ctx, data.params[1])
}

/// Build a new `core.func` TypeRef with evidence prepended to params.
pub fn build_func_type_with_evidence(
    ctx: &mut IrContext,
    old_func_ty: TypeRef,
    ev_ty: TypeRef,
) -> TypeRef {
    let data = ctx.types.get(old_func_ty);
    debug_assert!(
        data.dialect == Symbol::new("core") && data.name == Symbol::new("func"),
        "build_func_type_with_evidence: expected core.func type, got {}.{} (ty={old_func_ty:?})",
        data.dialect,
        data.name,
    );
    // params[0] = return, params[1..] = param types
    let result_ty = data.params[0];
    let old_params = &data.params[1..];

    let mut new_params = Vec::with_capacity(old_params.len() + 1);
    new_params.push(ev_ty);
    new_params.extend_from_slice(old_params);

    trunk_ir::dialect::core::func(ctx, result_ty, new_params).as_type_ref()
}

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
        if let Ok(func_op) = arena_func::Func::from_op(ctx, parent_op) {
            let body = func_op.body(ctx);
            let entry = ctx.region(body).blocks[0];
            let args = ctx.block_args(entry);
            if !args.is_empty() && arena_ability::is_evidence_type_ref(ctx, ctx.value_ty(args[0])) {
                return Some(args[0]);
            }
            return None;
        }
        current = parent_op;
    }
}
