//! FFI function declarations for the libmprompt-based continuation lowering.
//!
//! These functions are provided at link time by `tribute-runtime`.
//! Each function is declared as an extern `func.func` with an unreachable body,
//! following the same pattern as `resolve_evidence::ensure_runtime_functions`.

use trunk_ir::Symbol;
use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::core as arena_core;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::rewrite::Module;
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::smallvec::smallvec;

/// Ensure all libmprompt FFI function declarations are present in the module.
///
/// Idempotent: skips declarations that already exist.
pub(super) fn ensure_libmprompt_ffi(ctx: &mut IrContext, module: Module) {
    let body = module.body(ctx).expect("module has body");
    let first_block = ctx.region(body).blocks[0];

    // Check which functions already exist
    let existing: std::collections::HashSet<Symbol> = ctx
        .block(first_block)
        .ops
        .iter()
        .filter_map(|&op| arena_func::Func::from_op(ctx, op).ok())
        .map(|f| f.sym_name(ctx))
        .collect();

    let loc = ctx.op(module.op()).location;

    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();
    let nil_ty = arena_core::nil(ctx).as_type_ref();

    let mut new_ops = Vec::new();

    let mut declare = |ctx: &mut IrContext,
                       name: &'static str,
                       params: &[trunk_ir::arena::refs::TypeRef],
                       result: trunk_ir::arena::refs::TypeRef| {
        if existing.contains(&Symbol::new(name)) {
            return;
        }
        let op = build_extern_func(ctx, loc, name, params, result);
        new_ops.push(op);
    };

    // __tribute_prompt(tag: i32, body_fn: ptr, env: ptr) -> ptr
    declare(ctx, "__tribute_prompt", &[i32_ty, ptr_ty, ptr_ty], ptr_ty);
    // __tribute_yield(tag: i32, op_idx: i32, shift_value: ptr) -> ptr
    declare(ctx, "__tribute_yield", &[i32_ty, i32_ty, ptr_ty], ptr_ty);
    // __tribute_resume(continuation: ptr, value: ptr) -> ptr
    declare(ctx, "__tribute_resume", &[ptr_ty, ptr_ty], ptr_ty);
    // __tribute_resume_drop(continuation: ptr) -> ()
    declare(ctx, "__tribute_resume_drop", &[ptr_ty], nil_ty);
    // __tribute_yield_active() -> i1
    declare(ctx, "__tribute_yield_active", &[], i1_ty);
    // __tribute_get_yield_op_idx() -> i32
    declare(ctx, "__tribute_get_yield_op_idx", &[], i32_ty);
    // __tribute_get_yield_continuation() -> ptr
    declare(ctx, "__tribute_get_yield_continuation", &[], ptr_ty);
    // __tribute_get_yield_shift_value() -> ptr
    declare(ctx, "__tribute_get_yield_shift_value", &[], ptr_ty);
    // __tribute_reset_yield_state() -> ()
    declare(ctx, "__tribute_reset_yield_state", &[], nil_ty);
    // __tribute_yield_set_rc_roots(roots: ptr, count: i32) -> ()
    declare(
        ctx,
        "__tribute_yield_set_rc_roots",
        &[ptr_ty, i32_ty],
        nil_ty,
    );
    // __tribute_cont_wrap_from_tls(resume: ptr) -> ptr
    declare(ctx, "__tribute_cont_wrap_from_tls", &[ptr_ty], ptr_ty);

    // Insert new declarations at the front of the module block
    for op in new_ops.into_iter().rev() {
        ctx.insert_op_before(first_block, ctx.block(first_block).ops[0], op);
    }
}

/// Build an extern `func.func` with an unreachable body.
fn build_extern_func(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    name: &str,
    params: &[trunk_ir::arena::refs::TypeRef],
    result: trunk_ir::arena::refs::TypeRef,
) -> trunk_ir::arena::refs::OpRef {
    // Build func type: core.func layout is params[0]=result, params[1..]=params
    let func_ty = arena_core::func(ctx, result, params.iter().copied(), None).as_type_ref();

    // Build entry block with params and unreachable
    let args: Vec<BlockArgData> = params
        .iter()
        .map(|&ty| BlockArgData {
            ty,
            attrs: std::collections::BTreeMap::new(),
        })
        .collect();

    let unreachable_op = arena_func::unreachable(ctx, loc);

    let entry_block = ctx.create_block(BlockData {
        location: loc,
        args,
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(entry_block, unreachable_op.op_ref());

    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![entry_block],
        parent_op: None,
    });

    let func_op = arena_func::func(ctx, loc, Symbol::from_dynamic(name), func_ty, body);

    // Add abi attribute
    let data = ctx.op_mut(func_op.op_ref());
    data.attributes
        .insert(Symbol::new("abi"), ArenaAttribute::String("C".to_string()));

    func_op.op_ref()
}
