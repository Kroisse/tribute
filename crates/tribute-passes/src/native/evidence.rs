//! Evidence runtime lowering for the native backend.
//!
//! This pass adapts evidence-related IR from `resolve_evidence` for native
//! (Cranelift) code generation.  The WASM backend (`evidence_to_wasm.rs`)
//! replaces the stubs with inline binary-search IR; the native backend instead
//! delegates to `extern "C"` functions in `tribute-runtime`.
//!
//! ## Transformations
//!
//! 1. **Stub replacement** — `func.func @__tribute_evidence_lookup` /
//!    `@__tribute_evidence_extend` stubs (with `unreachable` body) are replaced
//!    by extern declarations with native signatures.
//!
//! 2. **Empty evidence** — `adt.array_new(0, evidence_ty)` →
//!    `func.call @__tribute_evidence_empty()`.
//!
//! 3. **Extend call-site rewrite** — the 2-arg call
//!    `func.call @__tribute_evidence_extend(ev, marker)` where `marker` is
//!    produced by `adt.struct_new(ability_id, prompt_tag, op_table_index)` is
//!    rewritten to a 4-arg call passing the fields directly, and the now-dead
//!    `adt.struct_new` is removed.

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::arena::rewrite::helpers::erase_op;
use trunk_ir::arena::types::{
    Attribute as ArenaAttribute, Location as ArenaLocation, TypeDataBuilder,
};

/// Lower evidence operations for the native backend.
///
/// Must run AFTER `cont_to_libmprompt` and BEFORE DCE.
pub fn lower_evidence_to_native(ctx: &mut IrContext, module: ArenaModule) {
    replace_stubs_and_add_empty(ctx, module);
    rewrite_evidence_ops_in_module(ctx, module);
}

// =============================================================================
// Phase 1: Replace stubs + add __tribute_evidence_empty declaration
// =============================================================================

fn replace_stubs_and_add_empty(ctx: &mut IrContext, module: ArenaModule) {
    let first_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let loc = ctx.op(module.op()).location;
    let ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    let mut has_evidence_empty = false;
    let mut stubs_to_replace: Vec<(OpRef, &'static str)> = Vec::new();

    let lookup_sym = Symbol::new("__tribute_evidence_lookup");
    let extend_sym = Symbol::new("__tribute_evidence_extend");
    let empty_sym = Symbol::new("__tribute_evidence_empty");

    for &op in &ops {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if name == lookup_sym {
                stubs_to_replace.push((op, "__tribute_evidence_lookup"));
            } else if name == extend_sym {
                stubs_to_replace.push((op, "__tribute_evidence_extend"));
            } else if name == empty_sym {
                has_evidence_empty = true;
            }
        }
    }

    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    // Replace stubs with extern declarations
    for (old_op, name) in stubs_to_replace {
        let new_op = match name {
            "__tribute_evidence_lookup" => make_evidence_lookup_extern(ctx, loc, i64_ty, i32_ty),
            "__tribute_evidence_extend" => make_evidence_extend_extern(ctx, loc, i64_ty, i32_ty),
            _ => unreachable!(),
        };
        // Insert new before old, then remove old
        ctx.insert_op_before(first_block, old_op, new_op);
        ctx.remove_op_from_block(first_block, old_op);
        ctx.remove_op(old_op);
    }

    // Add __tribute_evidence_empty if missing
    if !has_evidence_empty {
        let empty_op = make_evidence_empty_extern(ctx, loc, i64_ty);
        // Insert at front of module block
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, empty_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, empty_op);
        }
    }
}

/// Build extern `fn __tribute_evidence_empty() -> i64`
fn make_evidence_empty_extern(ctx: &mut IrContext, loc: ArenaLocation, i64_ty: TypeRef) -> OpRef {
    super::build_extern_func(ctx, loc, "__tribute_evidence_empty", &[], i64_ty)
}

/// Build extern `fn __tribute_evidence_lookup(ev: i64, ability_id: i32) -> i32`
fn make_evidence_lookup_extern(
    ctx: &mut IrContext,
    loc: ArenaLocation,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        "__tribute_evidence_lookup",
        &[i64_ty, i32_ty],
        i32_ty,
    )
}

/// Build extern `fn __tribute_evidence_extend(ev: i64, ability_id: i32, prompt_tag: i32, op_table_index: i32) -> i64`
fn make_evidence_extend_extern(
    ctx: &mut IrContext,
    loc: ArenaLocation,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        "__tribute_evidence_extend",
        &[i64_ty, i32_ty, i32_ty, i32_ty],
        i64_ty,
    )
}

// =============================================================================
// Phase 2: Rewrite evidence ops inside function bodies
// =============================================================================

fn is_evidence_runtime_fn(name: Symbol) -> bool {
    name == Symbol::new("__tribute_evidence_lookup")
        || name == Symbol::new("__tribute_evidence_extend")
        || name == Symbol::new("__tribute_evidence_empty")
}

fn rewrite_evidence_ops_in_module(ctx: &mut IrContext, module: ArenaModule) {
    let first_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in ops {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if is_evidence_runtime_fn(name) {
                continue;
            }
            let body = func_op.body(ctx);
            rewrite_evidence_ops_in_region(ctx, body);
        }
    }
}

fn rewrite_evidence_ops_in_region(ctx: &mut IrContext, region: RegionRef) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        rewrite_evidence_ops_in_block(ctx, block);
    }
}

/// Check if a type is an evidence type in arena (adt.array<ability.evidence>).
fn is_evidence_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::arena::dialect::ability::is_evidence_type_ref(ctx, ty)
}

/// Check if a type is a Marker type in arena.
fn is_marker_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::arena::dialect::ability::is_marker_type_ref(ctx, ty)
}

fn rewrite_evidence_ops_in_block(ctx: &mut IrContext, block: BlockRef) {
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    // Track Marker struct_new results → their operands
    let mut marker_struct_operands: HashMap<ValueRef, Vec<ValueRef>> = HashMap::new();
    // Track evidence_lookup results for struct_get elimination
    let mut evidence_lookup_results: HashSet<ValueRef> = HashSet::new();
    // Ops to erase after processing
    let mut ops_to_erase: Vec<OpRef> = Vec::new();

    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        let op_data = ctx.op(op);
        let dialect = op_data.dialect;
        let name = op_data.name;
        let loc = op_data.location;

        // --- adt.array_new with evidence type → func.call @__tribute_evidence_empty ---
        if dialect == Symbol::new("adt") && name == Symbol::new("array_new") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                let old_result = ctx.op_result(op, 0);
                let call = arena_func::call(
                    ctx,
                    loc,
                    [],
                    i64_ty,
                    Symbol::new("__tribute_evidence_empty"),
                );
                let new_result = call.result(ctx);
                ctx.insert_op_before(block, op, call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- Track adt.struct_new that produces a Marker ---
        if dialect == Symbol::new("adt") && name == Symbol::new("struct_new") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_marker_type(ctx, result_types[0]) {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                let result_val = ctx.op_result(op, 0);
                marker_struct_operands.insert(result_val, operands);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- Rewrite func.call @__tribute_evidence_lookup → returns i32 ---
        if dialect == Symbol::new("func")
            && name == Symbol::new("call")
            && let Ok(call_op) = arena_func::Call::from_op(ctx, op)
        {
            let callee = call_op.callee(ctx);

            if callee == Symbol::new("__tribute_evidence_lookup") {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                let old_result = ctx.op_result(op, 0);
                let new_call = arena_func::call(
                    ctx,
                    loc,
                    operands,
                    i32_ty,
                    Symbol::new("__tribute_evidence_lookup"),
                );
                let new_result = new_call.result(ctx);
                ctx.insert_op_before(block, op, new_call.op_ref());
                evidence_lookup_results.insert(new_result);
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }

            // --- Rewrite func.call @__tribute_evidence_extend(ev, marker) → (ev, a, b, c) ---
            if callee == Symbol::new("__tribute_evidence_extend") {
                let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
                if operands.len() == 2 {
                    let ev_val = operands[0];
                    let marker_val = operands[1];

                    let fields = marker_struct_operands.get(&marker_val).unwrap_or_else(|| {
                        panic!(
                            "evidence_extend rewrite: missing marker decomposition for \
                             marker_val={marker_val:?} at op={op:?} (loc={loc:?}). \
                             The adt.struct_new that produced this marker was not recorded."
                        )
                    });
                    let mut args = vec![ev_val];
                    args.extend_from_slice(fields);
                    let old_result = ctx.op_result(op, 0);
                    let new_call = arena_func::call(
                        ctx,
                        loc,
                        args,
                        i64_ty,
                        Symbol::new("__tribute_evidence_extend"),
                    );
                    let new_result = new_call.result(ctx);
                    ctx.insert_op_before(block, op, new_call.op_ref());
                    ctx.replace_all_uses(old_result, new_result);
                    ops_to_erase.push(op);
                    continue;
                }
            }
        }

        // --- Eliminate adt.struct_get on evidence_lookup results ---
        if dialect == Symbol::new("adt") && name == Symbol::new("struct_get") {
            let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
            if !operands.is_empty() {
                let base_val = operands[0];
                if evidence_lookup_results.contains(&base_val) {
                    // Only eliminate struct_get for the prompt_tag field (index 1).
                    let field_attr = ctx.op(op).attributes.get(&Symbol::new("field"));
                    let field_idx = match field_attr {
                        Some(ArenaAttribute::IntBits(bits)) => *bits,
                        other => panic!(
                            "expected IntBits field attribute on adt.struct_get, got {other:?}"
                        ),
                    };
                    assert_eq!(
                        field_idx, 1,
                        "expected struct_get on evidence_lookup result to access prompt_tag (field 1), got field {field_idx}"
                    );
                    let old_result = ctx.op_result(op, 0);
                    ctx.replace_all_uses(old_result, base_val);
                    evidence_lookup_results.insert(old_result);
                    ops_to_erase.push(op);
                    continue;
                }
            }
        }

        // --- Recurse into nested regions ---
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            rewrite_evidence_ops_in_region(ctx, region);
        }
    }

    // Erase dead ops (in reverse to handle dependencies)
    for op in ops_to_erase.into_iter().rev() {
        erase_op(ctx, op);
    }
}
