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
//!    produced by `adt.struct_new(ability_id, prompt_tag, tr_dispatch_fn)` is
//!    rewritten to a 4-arg call passing the fields directly, and the now-dead
//!    `adt.struct_new` is removed.
//!
//! 4. **TR dispatch field** — `adt.struct_get(marker, 2)` on evidence_lookup
//!    results is rewritten to `func.call @__tribute_evidence_lookup_tr(ev, ability_id)`.

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::rewrite::helpers::erase_op;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

/// Lower evidence operations for the native backend.
///
/// Must run AFTER effect lowering passes and BEFORE DCE.
pub fn lower_evidence_to_native(ctx: &mut IrContext, module: Module) {
    replace_stubs_and_add_empty(ctx, module);
    rewrite_evidence_ops_in_module(ctx, module);
}

// =============================================================================
// Phase 1: Replace stubs + add __tribute_evidence_empty declaration
// =============================================================================

fn replace_stubs_and_add_empty(ctx: &mut IrContext, module: Module) {
    let first_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let loc = ctx.op(module.op()).location;
    let ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    let mut has_evidence_empty = false;
    let mut has_lookup_tr = false;
    let mut has_lookup_handler = false;
    let mut stubs_to_replace: Vec<(OpRef, &'static str)> = Vec::new();

    let lookup_sym = Symbol::new("__tribute_evidence_lookup");
    let extend_sym = Symbol::new("__tribute_evidence_extend");
    let empty_sym = Symbol::new("__tribute_evidence_empty");
    let lookup_tr_sym = Symbol::new("__tribute_evidence_lookup_tr");
    let lookup_handler_sym = Symbol::new("__tribute_evidence_lookup_handler");

    for &op in &ops {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx);
            if name == lookup_sym {
                stubs_to_replace.push((op, "__tribute_evidence_lookup"));
            } else if name == extend_sym {
                stubs_to_replace.push((op, "__tribute_evidence_extend"));
            } else if name == empty_sym {
                has_evidence_empty = true;
            } else if name == lookup_tr_sym {
                has_lookup_tr = true;
            } else if name == lookup_handler_sym {
                has_lookup_handler = true;
            }
        }
    }

    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    // Replace stubs with extern declarations
    for (old_op, name) in stubs_to_replace {
        let new_op = match name {
            "__tribute_evidence_lookup" => make_evidence_lookup_extern(ctx, loc, ptr_ty, i32_ty),
            "__tribute_evidence_extend" => make_evidence_extend_extern(ctx, loc, ptr_ty, i32_ty),
            _ => unreachable!(),
        };
        // Insert new before old, then remove old
        ctx.insert_op_before(first_block, old_op, new_op);
        ctx.remove_op_from_block(first_block, old_op);
        ctx.remove_op(old_op);
    }

    // Add __tribute_evidence_empty if missing
    if !has_evidence_empty {
        let empty_op = make_evidence_empty_extern(ctx, loc, ptr_ty);
        // Insert at front of module block
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, empty_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, empty_op);
        }
    }

    // Add __tribute_evidence_lookup_tr if missing
    if !has_lookup_tr {
        let lookup_tr_op = make_evidence_lookup_tr_extern(ctx, loc, ptr_ty, i32_ty);
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, lookup_tr_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, lookup_tr_op);
        }
    }

    // Add __tribute_evidence_lookup_handler if missing
    if !has_lookup_handler {
        let lookup_handler_op = make_evidence_lookup_handler_extern(ctx, loc, ptr_ty, i32_ty);
        let block_ops = &ctx.block(first_block).ops;
        if block_ops.is_empty() {
            ctx.push_op(first_block, lookup_handler_op);
        } else {
            let first_op = block_ops[0];
            ctx.insert_op_before(first_block, first_op, lookup_handler_op);
        }
    }
}

/// Build extern `fn __tribute_evidence_empty() -> ptr`
fn make_evidence_empty_extern(ctx: &mut IrContext, loc: Location, ptr_ty: TypeRef) -> OpRef {
    super::build_extern_func(ctx, loc, "__tribute_evidence_empty", &[], ptr_ty)
}

/// Build extern `fn __tribute_evidence_lookup(ev: ptr, ability_id: i32) -> i32`
fn make_evidence_lookup_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        "__tribute_evidence_lookup",
        &[ptr_ty, i32_ty],
        i32_ty,
    )
}

/// Build extern `fn __tribute_evidence_extend(ev: ptr, ability_id: i32, prompt_tag: i32, tr_dispatch_fn: ptr, handler_dispatch: ptr) -> ptr`
fn make_evidence_extend_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        "__tribute_evidence_extend",
        &[ptr_ty, i32_ty, i32_ty, ptr_ty, ptr_ty],
        ptr_ty,
    )
}

/// Build extern `fn __tribute_evidence_lookup_tr(ev: ptr, ability_id: i32) -> ptr`
fn make_evidence_lookup_tr_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        "__tribute_evidence_lookup_tr",
        &[ptr_ty, i32_ty],
        ptr_ty,
    )
}

/// Build extern `fn __tribute_evidence_lookup_handler(ev: ptr, ability_id: i32) -> ptr`
fn make_evidence_lookup_handler_extern(
    ctx: &mut IrContext,
    loc: Location,
    ptr_ty: TypeRef,
    i32_ty: TypeRef,
) -> OpRef {
    super::build_extern_func(
        ctx,
        loc,
        "__tribute_evidence_lookup_handler",
        &[ptr_ty, i32_ty],
        ptr_ty,
    )
}

// =============================================================================
// Phase 2: Rewrite evidence ops inside function bodies
// =============================================================================

fn is_evidence_runtime_fn(name: Symbol) -> bool {
    name == Symbol::new("__tribute_evidence_lookup")
        || name == Symbol::new("__tribute_evidence_extend")
        || name == Symbol::new("__tribute_evidence_empty")
        || name == Symbol::new("__tribute_evidence_lookup_tr")
        || name == Symbol::new("__tribute_evidence_lookup_handler")
}

fn rewrite_evidence_ops_in_module(ctx: &mut IrContext, module: Module) {
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
    tribute_ir::dialect::ability::is_evidence_type_ref(ctx, ty)
}

/// Check if a type is a Marker type in arena.
fn is_marker_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::ability::is_marker_type_ref(ctx, ty)
}

fn rewrite_evidence_ops_in_block(ctx: &mut IrContext, block: BlockRef) {
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    // Track Marker struct_new results → their operands
    let mut marker_struct_operands: HashMap<ValueRef, Vec<ValueRef>> = HashMap::new();
    // Track evidence_lookup results for struct_get elimination.
    // Maps result value → (ev, ability_id) operands for __tribute_evidence_lookup_tr calls.
    let mut evidence_lookup_results: HashMap<ValueRef, (ValueRef, ValueRef)> = HashMap::new();
    // Ops to erase after processing
    let mut ops_to_erase: Vec<OpRef> = Vec::new();

    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        let op_data = ctx.op(op);
        let dialect = op_data.dialect;
        let name = op_data.name;
        let loc = op_data.location;

        // --- adt.ref_null with evidence type → func.call @__tribute_evidence_empty ---
        // Closure lowering creates `adt.ref_null {type = evidence}` for null evidence.
        // Without this, the null ptr gets unboxed via `clif.load` which dereferences null.
        if dialect == Symbol::new("adt") && name == Symbol::new("ref_null") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                let old_result = ctx.op_result(op, 0);
                let call = arena_func::call(
                    ctx,
                    loc,
                    [],
                    ptr_ty,
                    Symbol::new("__tribute_evidence_empty"),
                );
                let new_result = call.result(ctx);
                ctx.insert_op_before(block, op, call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- adt.array_new with evidence type → func.call @__tribute_evidence_empty ---
        if dialect == Symbol::new("adt") && name == Symbol::new("array_new") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                // Evidence array_new must represent an empty evidence vector.
                // The only operand should be the size hint (arith.const 0).
                let operand_count = ctx.op_operands(op).len();
                assert!(
                    operand_count <= 1,
                    "evidence_to_native: adt.array_new with evidence type has {operand_count} \
                     operands; expected at most 1 (the size hint). Non-empty evidence arrays \
                     should not reach this pass."
                );
                let old_result = ctx.op_result(op, 0);
                let call = arena_func::call(
                    ctx,
                    loc,
                    [],
                    ptr_ty,
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
                let ev_val = operands[0];
                let ability_id_val = operands[1];
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
                evidence_lookup_results.insert(new_result, (ev_val, ability_id_val));
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
                        ptr_ty,
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
                if let Some(&(ev_val, ability_id_val)) = evidence_lookup_results.get(&base_val) {
                    let field_attr = ctx.op(op).attributes.get(&Symbol::new("field"));
                    let field_idx = match field_attr {
                        Some(Attribute::Int(bits)) => *bits,
                        other => {
                            panic!("expected Int field attribute on adt.struct_get, got {other:?}")
                        }
                    };
                    match field_idx {
                        1 => {
                            // Field 1: prompt_tag (i32) — __tribute_evidence_lookup already returns this
                            let old_result = ctx.op_result(op, 0);
                            ctx.replace_all_uses(old_result, base_val);
                            evidence_lookup_results.insert(old_result, (ev_val, ability_id_val));
                            ops_to_erase.push(op);
                        }
                        2 => {
                            // Field 2: tr_dispatch_fn (ptr) — call __tribute_evidence_lookup_tr
                            let old_result = ctx.op_result(op, 0);
                            let tr_call = arena_func::call(
                                ctx,
                                loc,
                                [ev_val, ability_id_val],
                                ptr_ty,
                                Symbol::new("__tribute_evidence_lookup_tr"),
                            );
                            let new_result = tr_call.result(ctx);
                            ctx.insert_op_before(block, op, tr_call.op_ref());
                            ctx.replace_all_uses(old_result, new_result);
                            ops_to_erase.push(op);
                        }
                        3 => {
                            // Field 3: handler_dispatch (ptr) — call __tribute_evidence_lookup_handler
                            let old_result = ctx.op_result(op, 0);
                            let handler_call = arena_func::call(
                                ctx,
                                loc,
                                [ev_val, ability_id_val],
                                ptr_ty,
                                Symbol::new("__tribute_evidence_lookup_handler"),
                            );
                            let new_result = handler_call.result(ctx);
                            ctx.insert_op_before(block, op, handler_call.op_ref());
                            ctx.replace_all_uses(old_result, new_result);
                            ops_to_erase.push(op);
                        }
                        _ => panic!(
                            "unexpected struct_get field {field_idx} on evidence_lookup result"
                        ),
                    }
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
