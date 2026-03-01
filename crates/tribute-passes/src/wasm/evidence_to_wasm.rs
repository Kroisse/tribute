//! Evidence runtime functions to WASM lowering (arena-based).
//!
//! This pass replaces evidence runtime function stubs with real implementations:
//!
//! - `__tribute_evidence_lookup(ev, ability_id)` -> binary search for marker
//! - `__tribute_evidence_extend(ev, marker)` -> sorted insertion with binary search
//!
//! ## Evidence Structure
//!
//! Evidence is represented as a WasmGC array of Marker structs, sorted by ability_id:
//!
//! ```text
//! Evidence = Array(Marker)
//! Marker = struct { ability_id: i32, prompt_tag: i32, op_table_index: i32 }
//! ```
//!
//! ## Implementation Strategy
//!
//! The pass replaces stub function declarations with real implementations that use
//! binary search (O(log n)) since the evidence array is maintained in sorted order
//! by ability_id.

use std::collections::BTreeMap;

use tribute_ir::arena::dialect::ability as arena_ability;
use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::ir::Symbol;
use trunk_ir::smallvec::smallvec;
use trunk_ir_wasm_backend::gc_types::{EVIDENCE_IDX, MARKER_IDX};

/// Lower evidence runtime functions to WASM implementations.
///
/// This pass:
/// 1. Replaces stub function declarations with real binary search implementations
/// 2. Lowers remaining `ability.evidence_lookup` and `ability.evidence_extend`
///    operations to calls to the generated functions.
pub fn lower_evidence_to_wasm(ctx: &mut IrContext, module: ArenaModule) {
    // Phase 1: Replace stubs with real implementations
    replace_evidence_function_stubs(ctx, module);

    // Phase 2: Pattern-based lowering for remaining ability ops
    let applicator = PatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(EvidenceLookupPattern)
        .add_pattern(EvidenceExtendPattern);
    applicator.apply_partial(ctx, module);
}

/// Replace evidence runtime function stubs with real implementations.
fn replace_evidence_function_stubs(ctx: &mut IrContext, module: ArenaModule) {
    let ops = module.ops(ctx);

    for op in ops {
        let data = ctx.op(op);
        let is_wasm_func = data.dialect == Symbol::new("wasm") && data.name == Symbol::new("func");
        let is_func_func = data.dialect == Symbol::new("func") && data.name == Symbol::new("func");

        if !is_wasm_func && !is_func_func {
            continue;
        }

        let sym_name = data.attributes.get(&Symbol::new("sym_name")).and_then(|a| {
            if let Attribute::Symbol(s) = a {
                Some(*s)
            } else {
                None
            }
        });

        let location = data.location;

        if sym_name == Some(Symbol::new("__tribute_evidence_lookup")) {
            let new_op = generate_evidence_lookup_function(ctx, location);
            replace_module_op(ctx, module, op, new_op);
        } else if sym_name == Some(Symbol::new("__tribute_evidence_extend")) {
            let new_op = generate_evidence_extend_function(ctx, location);
            replace_module_op(ctx, module, op, new_op);
        }
    }
}

/// Replace a top-level module operation with a new one.
fn replace_module_op(ctx: &mut IrContext, module: ArenaModule, old_op: OpRef, new_op: OpRef) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };

    // Insert new op before old, then remove old
    ctx.insert_op_before(first_block, old_op, new_op);
    ctx.remove_op_from_block(first_block, old_op);
    ctx.remove_op(old_op);
}

// =============================================================================
// Evidence Lookup Pattern
// =============================================================================

/// Pattern that matches `ability.evidence_lookup` and replaces it with
/// `wasm.call @__tribute_evidence_lookup`.
struct EvidenceLookupPattern;

impl ArenaRewritePattern for EvidenceLookupPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(lookup_op) = arena_ability::EvidenceLookup::from_op(ctx, op) else {
            return false;
        };

        let evidence_val = lookup_op.evidence(ctx);
        let result_ty = lookup_op.result_ty(ctx);
        let loc = ctx.op(op).location;

        // Extract ability_id from the ability_ref type attribute
        let ability_ref_ty = lookup_op.ability_ref(ctx);
        let ability_id = compute_ability_id(ctx, ability_ref_ty);

        let i32_ty = intern_i32(ctx);

        // Create: %id = wasm.i32_const(ability_id)
        let id_const = arena_wasm::i32_const(ctx, loc, i32_ty, ability_id);

        // Create: %result = wasm.call @__tribute_evidence_lookup(%ev, %id)
        let call_op = arena_wasm::call(
            ctx,
            loc,
            [evidence_val, id_const.result(ctx)],
            [result_ty],
            Symbol::new("__tribute_evidence_lookup"),
        );

        let call_result = call_op.results(ctx)[0];
        rewriter.insert_op(id_const.op_ref());
        rewriter.erase_op(vec![call_result]);
        true
    }

    fn name(&self) -> &'static str {
        "EvidenceLookupPattern"
    }
}

// =============================================================================
// Evidence Extend Pattern
// =============================================================================

/// Pattern that matches `ability.evidence_extend` and replaces it with
/// `wasm.call @__tribute_evidence_extend`.
struct EvidenceExtendPattern;

impl ArenaRewritePattern for EvidenceExtendPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(extend_op) = arena_ability::EvidenceExtend::from_op(ctx, op) else {
            return false;
        };

        let evidence_val = extend_op.evidence(ctx);
        let result_ty = extend_op.result_ty(ctx);
        let loc = ctx.op(op).location;

        // The marker value is already constructed by an earlier pass and passed
        // as the prompt_tag attribute. For evidence_extend, the actual marker
        // struct is expected to be provided as an operand or constructed before
        // this point. In the current pipeline, evidence_extend takes evidence
        // and produces new evidence. The marker is built from the prompt_tag
        // attribute and ability_ref.
        //
        // We need to construct the marker struct from ability_ref and prompt_tag,
        // then call __tribute_evidence_extend.

        let ability_ref_ty = extend_op.ability_ref(ctx);
        let ability_id = compute_ability_id(ctx, ability_ref_ty);
        let prompt_tag_attr = extend_op.prompt_tag(ctx);

        let i32_ty = intern_i32(ctx);
        let marker_ty = arena_ability::marker_adt_type_ref(ctx);

        // Create: %ability_id = wasm.i32_const(ability_id)
        let ability_id_const = arena_wasm::i32_const(ctx, loc, i32_ty, ability_id);

        // Create: %prompt_tag = wasm.i32_const(prompt_tag)
        let prompt_tag_val = match &prompt_tag_attr {
            Attribute::IntBits(v) => *v as i32,
            _ => 0,
        };
        let prompt_tag_const = arena_wasm::i32_const(ctx, loc, i32_ty, prompt_tag_val);

        // Create: %op_table_idx = wasm.i32_const(0)
        let op_table_const = arena_wasm::i32_const(ctx, loc, i32_ty, 0);

        // Create: %marker = wasm.struct_new(MARKER_IDX, %ability_id, %prompt_tag, %op_table_idx)
        let marker_op = arena_wasm::struct_new(
            ctx,
            loc,
            [
                ability_id_const.result(ctx),
                prompt_tag_const.result(ctx),
                op_table_const.result(ctx),
            ],
            marker_ty,
            MARKER_IDX,
        );

        // Create: %result = wasm.call @__tribute_evidence_extend(%ev, %marker)
        let call_op = arena_wasm::call(
            ctx,
            loc,
            [evidence_val, marker_op.result(ctx)],
            [result_ty],
            Symbol::new("__tribute_evidence_extend"),
        );

        let call_result = call_op.results(ctx)[0];
        rewriter.insert_op(ability_id_const.op_ref());
        rewriter.insert_op(prompt_tag_const.op_ref());
        rewriter.insert_op(op_table_const.op_ref());
        rewriter.insert_op(marker_op.op_ref());
        rewriter.erase_op(vec![call_result]);
        true
    }

    fn name(&self) -> &'static str {
        "EvidenceExtendPattern"
    }
}

// =============================================================================
// Function Generation
// =============================================================================

/// Local variable indices for binary search.
/// These start after function parameters (0, 1).
mod locals {
    pub const LOW: u32 = 2;
    pub const HIGH: u32 = 3;
}

/// Generate the `__tribute_evidence_lookup` function implementation.
///
/// Uses binary search to find a marker with the given ability_id.
/// Returns the marker, or traps with unreachable if not found (compiler bug).
fn generate_evidence_lookup_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let marker_sig_ty = arena_ability::marker_adt_type_ref(ctx);

    let func_ty = intern_func_type(ctx, &[evidence_ty, i32_ty], marker_sig_ty);

    // Create the function body block with arguments
    let body_block = ctx.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: evidence_ty,
                attrs: BTreeMap::new(),
            },
            BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            },
        ],
        ops: smallvec![],
        parent_region: None,
    });

    let ev_val = ctx.block_arg(body_block, 0);
    let target_id_val = ctx.block_arg(body_block, 1);

    // Initialize low = 0
    let zero = arena_wasm::i32_const(ctx, location, i32_ty, 0);
    ctx.push_op(body_block, zero.op_ref());
    let low_init = arena_wasm::local_set(ctx, location, zero.result(ctx), locals::LOW);
    ctx.push_op(body_block, low_init.op_ref());

    // Initialize high = array.len(ev)
    let len_op = arena_wasm::array_len(ctx, location, ev_val, i32_ty);
    ctx.push_op(body_block, len_op.op_ref());
    let high_init = arena_wasm::local_set(ctx, location, len_op.result(ctx), locals::HIGH);
    ctx.push_op(body_block, high_init.op_ref());

    // Build the search loop
    let nil_ty = intern_nil(ctx);
    let loop_region = build_lookup_loop_body(ctx, location, ev_val, target_id_val, i32_ty);
    let loop_op = arena_wasm::r#loop(ctx, location, [], nil_ty, loop_region);
    ctx.push_op(body_block, loop_op.op_ref());

    // unreachable after loop (should never reach here - loop always returns)
    let unreachable_op = arena_wasm::unreachable(ctx, location);
    ctx.push_op(body_block, unreachable_op.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![body_block],
        parent_op: None,
    });

    let func_op = arena_wasm::func(
        ctx,
        location,
        Symbol::new("__tribute_evidence_lookup"),
        func_ty,
        body,
    );
    func_op.op_ref()
}

/// Build the loop body for binary search lookup.
fn build_lookup_loop_body(
    ctx: &mut IrContext,
    location: Location,
    ev_val: ValueRef,
    target_id_val: ValueRef,
    i32_ty: TypeRef,
) -> RegionRef {
    let nil_ty = intern_nil(ctx);
    let marker_ty = arena_ability::marker_adt_type_ref(ctx);

    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // low = local.get LOW
    let get_low = arena_wasm::local_get(ctx, location, i32_ty, locals::LOW);
    let low = get_low.result(ctx);
    ctx.push_op(block, get_low.op_ref());

    // high = local.get HIGH
    let get_high = arena_wasm::local_get(ctx, location, i32_ty, locals::HIGH);
    let high = get_high.result(ctx);
    ctx.push_op(block, get_high.op_ref());

    // Check low >= high -> unreachable (ability not found = compiler bug)
    let ge_check = arena_wasm::i32_ge_s(ctx, location, low, high, i32_ty);
    ctx.push_op(block, ge_check.op_ref());

    let unreachable_then = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let unreachable_op = arena_wasm::unreachable(ctx, location);
        ctx.push_op(inner_block, unreachable_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let empty_else = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let bound_check_if = arena_wasm::r#if(
        ctx,
        location,
        ge_check.result(ctx),
        nil_ty,
        unreachable_then,
        empty_else,
    );
    ctx.push_op(block, bound_check_if.op_ref());

    // mid = (low + high) / 2
    let add_op = arena_wasm::i32_add(ctx, location, low, high, i32_ty);
    ctx.push_op(block, add_op.op_ref());
    let two = arena_wasm::i32_const(ctx, location, i32_ty, 2);
    ctx.push_op(block, two.op_ref());
    let mid_op = arena_wasm::i32_div_u(ctx, location, add_op.result(ctx), two.result(ctx), i32_ty);
    let mid = mid_op.result(ctx);
    ctx.push_op(block, mid_op.op_ref());

    // marker = array.get(ev, mid)
    let marker_op = arena_wasm::array_get(ctx, location, ev_val, mid, marker_ty, EVIDENCE_IDX);
    let marker = marker_op.result(ctx);
    ctx.push_op(block, marker_op.op_ref());

    // marker_ability_id = struct.get(marker, 0)
    let marker_id_op = arena_wasm::struct_get(ctx, location, marker, i32_ty, MARKER_IDX, 0);
    let marker_id = marker_id_op.result(ctx);
    ctx.push_op(block, marker_id_op.op_ref());

    // Check marker_ability_id == target -> return marker
    let eq_check = arena_wasm::i32_eq(ctx, location, marker_id, target_id_val, i32_ty);
    ctx.push_op(block, eq_check.op_ref());

    let found_then = {
        // Return marker
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let return_op = arena_wasm::r#return(ctx, location, [marker]);
        ctx.push_op(inner_block, return_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let continue_else = {
        // marker_ability_id < target ? low = mid + 1 : high = mid
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let lt_check = arena_wasm::i32_lt_s(ctx, location, marker_id, target_id_val, i32_ty);
        ctx.push_op(inner_block, lt_check.op_ref());

        let update_low = {
            // low = mid + 1
            let ub = ctx.create_block(BlockData {
                location,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let one = arena_wasm::i32_const(ctx, location, i32_ty, 1);
            ctx.push_op(ub, one.op_ref());
            let add_one = arena_wasm::i32_add(ctx, location, mid, one.result(ctx), i32_ty);
            ctx.push_op(ub, add_one.op_ref());
            let set_low = arena_wasm::local_set(ctx, location, add_one.result(ctx), locals::LOW);
            ctx.push_op(ub, set_low.op_ref());
            ctx.create_region(RegionData {
                location,
                blocks: smallvec![ub],
                parent_op: None,
            })
        };

        let update_high = {
            // high = mid
            let ub = ctx.create_block(BlockData {
                location,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let set_high = arena_wasm::local_set(ctx, location, mid, locals::HIGH);
            ctx.push_op(ub, set_high.op_ref());
            ctx.create_region(RegionData {
                location,
                blocks: smallvec![ub],
                parent_op: None,
            })
        };

        let update_if = arena_wasm::r#if(
            ctx,
            location,
            lt_check.result(ctx),
            nil_ty,
            update_low,
            update_high,
        );
        ctx.push_op(inner_block, update_if.op_ref());

        // br $loop (target = 0, since loop is innermost)
        let br_loop = arena_wasm::br(ctx, location, 0);
        ctx.push_op(inner_block, br_loop.op_ref());

        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let found_if = arena_wasm::r#if(
        ctx,
        location,
        eq_check.result(ctx),
        nil_ty,
        found_then,
        continue_else,
    );
    ctx.push_op(block, found_if.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

/// Generate the `__tribute_evidence_extend` function implementation.
///
/// Uses binary search to find the insertion point, then creates a new array
/// with the marker inserted at the correct position to maintain sorted order.
fn generate_evidence_extend_function(ctx: &mut IrContext, location: Location) -> OpRef {
    let evidence_ty = evidence_ref_type(ctx);
    let i32_ty = intern_i32(ctx);
    let marker_sig_ty = arena_ability::marker_adt_type_ref(ctx);

    let func_ty = intern_func_type(ctx, &[evidence_ty, marker_sig_ty], evidence_ty);

    // Create the function body block with arguments
    let body_block = ctx.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: evidence_ty,
                attrs: BTreeMap::new(),
            },
            BlockArgData {
                ty: marker_sig_ty,
                attrs: BTreeMap::new(),
            },
        ],
        ops: smallvec![],
        parent_region: None,
    });

    let ev_val = ctx.block_arg(body_block, 0);
    let marker_val = ctx.block_arg(body_block, 1);

    let nil_ty = intern_nil(ctx);

    // Get marker's ability_id for binary search
    let marker_id_op = arena_wasm::struct_get(ctx, location, marker_val, i32_ty, MARKER_IDX, 0);
    let marker_id = marker_id_op.result(ctx);
    ctx.push_op(body_block, marker_id_op.op_ref());

    // old_len = array.len(ev)
    let len_op = arena_wasm::array_len(ctx, location, ev_val, i32_ty);
    let old_len = len_op.result(ctx);
    ctx.push_op(body_block, len_op.op_ref());

    // Initialize low = 0
    let zero = arena_wasm::i32_const(ctx, location, i32_ty, 0);
    ctx.push_op(body_block, zero.op_ref());
    let low_init = arena_wasm::local_set(ctx, location, zero.result(ctx), locals::LOW);
    ctx.push_op(body_block, low_init.op_ref());

    // Initialize high = old_len
    let high_init = arena_wasm::local_set(ctx, location, old_len, locals::HIGH);
    ctx.push_op(body_block, high_init.op_ref());

    // Binary search loop to find insertion point
    // After loop, LOW contains the insertion index
    let search_loop = build_extend_search_loop(ctx, location, ev_val, marker_id, i32_ty);
    let loop_op = arena_wasm::r#loop(ctx, location, [], nil_ty, search_loop);

    // Wrap loop in block for br_if(..., 1) target
    let loop_wrapper_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    ctx.push_op(loop_wrapper_block, loop_op.op_ref());
    let loop_region = ctx.create_region(RegionData {
        location,
        blocks: smallvec![loop_wrapper_block],
        parent_op: None,
    });
    let block_op = arena_wasm::block(ctx, location, nil_ty, loop_region);
    ctx.push_op(body_block, block_op.op_ref());

    // insert_idx = local.get LOW
    let get_insert_idx = arena_wasm::local_get(ctx, location, i32_ty, locals::LOW);
    let insert_idx = get_insert_idx.result(ctx);
    ctx.push_op(body_block, get_insert_idx.op_ref());

    // new_len = old_len + 1
    let one = arena_wasm::i32_const(ctx, location, i32_ty, 1);
    ctx.push_op(body_block, one.op_ref());
    let add_len_op = arena_wasm::i32_add(ctx, location, old_len, one.result(ctx), i32_ty);
    let new_len = add_len_op.result(ctx);
    ctx.push_op(body_block, add_len_op.op_ref());

    // new_ev = array.new_default(new_len)
    let new_array_op =
        arena_wasm::array_new_default(ctx, location, new_len, evidence_ty, EVIDENCE_IDX);
    let new_ev = new_array_op.result(ctx);
    ctx.push_op(body_block, new_array_op.op_ref());

    // Copy elements before insertion point: array.copy(new_ev, 0, ev, 0, insert_idx)
    // Only if insert_idx > 0
    let zero2 = arena_wasm::i32_const(ctx, location, i32_ty, 0);
    ctx.push_op(body_block, zero2.op_ref());

    let gt_zero = arena_wasm::i32_gt_s(ctx, location, insert_idx, zero2.result(ctx), i32_ty);
    ctx.push_op(body_block, gt_zero.op_ref());

    let copy_prefix_then = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let zero3 = arena_wasm::i32_const(ctx, location, i32_ty, 0);
        ctx.push_op(inner_block, zero3.op_ref());
        let copy_op = arena_wasm::array_copy(
            ctx,
            location,
            new_ev,
            zero3.result(ctx),
            ev_val,
            zero3.result(ctx),
            insert_idx,
            EVIDENCE_IDX,
            EVIDENCE_IDX,
        );
        ctx.push_op(inner_block, copy_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let empty_else1 = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let copy_prefix_if = arena_wasm::r#if(
        ctx,
        location,
        gt_zero.result(ctx),
        nil_ty,
        copy_prefix_then,
        empty_else1,
    );
    ctx.push_op(body_block, copy_prefix_if.op_ref());

    // Set marker at insert_idx: array.set(new_ev, insert_idx, marker)
    let set_op = arena_wasm::array_set(ctx, location, new_ev, insert_idx, marker_val, EVIDENCE_IDX);
    ctx.push_op(body_block, set_op.op_ref());

    // Copy elements after insertion point: array.copy(new_ev, insert_idx+1, ev, insert_idx, old_len - insert_idx)
    // suffix_len = old_len - insert_idx
    let suffix_len_op = arena_wasm::i32_sub(ctx, location, old_len, insert_idx, i32_ty);
    let suffix_len = suffix_len_op.result(ctx);
    ctx.push_op(body_block, suffix_len_op.op_ref());

    // Only if suffix_len > 0
    let zero4 = arena_wasm::i32_const(ctx, location, i32_ty, 0);
    ctx.push_op(body_block, zero4.op_ref());
    let gt_zero2 = arena_wasm::i32_gt_s(ctx, location, suffix_len, zero4.result(ctx), i32_ty);
    ctx.push_op(body_block, gt_zero2.op_ref());

    let copy_suffix_then = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let one2 = arena_wasm::i32_const(ctx, location, i32_ty, 1);
        ctx.push_op(inner_block, one2.op_ref());
        let dst_offset_op =
            arena_wasm::i32_add(ctx, location, insert_idx, one2.result(ctx), i32_ty);
        ctx.push_op(inner_block, dst_offset_op.op_ref());
        let copy_op = arena_wasm::array_copy(
            ctx,
            location,
            new_ev,
            dst_offset_op.result(ctx),
            ev_val,
            insert_idx,
            suffix_len,
            EVIDENCE_IDX,
            EVIDENCE_IDX,
        );
        ctx.push_op(inner_block, copy_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let empty_else2 = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };
    let copy_suffix_if = arena_wasm::r#if(
        ctx,
        location,
        gt_zero2.result(ctx),
        nil_ty,
        copy_suffix_then,
        empty_else2,
    );
    ctx.push_op(body_block, copy_suffix_if.op_ref());

    // Return new_ev
    let return_op = arena_wasm::r#return(ctx, location, [new_ev]);
    ctx.push_op(body_block, return_op.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![body_block],
        parent_op: None,
    });

    let func_op = arena_wasm::func(
        ctx,
        location,
        Symbol::new("__tribute_evidence_extend"),
        func_ty,
        body,
    );
    func_op.op_ref()
}

/// Build the search loop for finding insertion position in evidence_extend.
///
/// This is a binary search that finds the first index where ev[i].ability_id >= marker_id.
/// After the loop, LOW contains the insertion index.
fn build_extend_search_loop(
    ctx: &mut IrContext,
    location: Location,
    ev_val: ValueRef,
    marker_id: ValueRef,
    i32_ty: TypeRef,
) -> RegionRef {
    let nil_ty = intern_nil(ctx);
    let marker_ty = arena_ability::marker_adt_type_ref(ctx);

    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // low = local.get LOW
    let get_low = arena_wasm::local_get(ctx, location, i32_ty, locals::LOW);
    let low = get_low.result(ctx);
    ctx.push_op(block, get_low.op_ref());

    // high = local.get HIGH
    let get_high = arena_wasm::local_get(ctx, location, i32_ty, locals::HIGH);
    let high = get_high.result(ctx);
    ctx.push_op(block, get_high.op_ref());

    // if low >= high: break (insertion point found at LOW)
    let ge_check = arena_wasm::i32_ge_s(ctx, location, low, high, i32_ty);
    ctx.push_op(block, ge_check.op_ref());

    // br_if to exit loop (target = 1 to break out of loop to outer block)
    let br_if_done = arena_wasm::br_if(ctx, location, ge_check.result(ctx), 1);
    ctx.push_op(block, br_if_done.op_ref());

    // mid = (low + high) / 2
    let add_op = arena_wasm::i32_add(ctx, location, low, high, i32_ty);
    ctx.push_op(block, add_op.op_ref());
    let two = arena_wasm::i32_const(ctx, location, i32_ty, 2);
    ctx.push_op(block, two.op_ref());
    let mid_op = arena_wasm::i32_div_u(ctx, location, add_op.result(ctx), two.result(ctx), i32_ty);
    let mid = mid_op.result(ctx);
    ctx.push_op(block, mid_op.op_ref());

    // mid_marker = array.get(ev, mid)
    let mid_marker_op = arena_wasm::array_get(ctx, location, ev_val, mid, marker_ty, EVIDENCE_IDX);
    let mid_marker = mid_marker_op.result(ctx);
    ctx.push_op(block, mid_marker_op.op_ref());

    // mid_ability_id = struct.get(mid_marker, 0)
    let mid_id_op = arena_wasm::struct_get(ctx, location, mid_marker, i32_ty, MARKER_IDX, 0);
    let mid_id = mid_id_op.result(ctx);
    ctx.push_op(block, mid_id_op.op_ref());

    // if mid_ability_id < marker_id: low = mid + 1
    // else: high = mid
    let lt_check = arena_wasm::i32_lt_s(ctx, location, mid_id, marker_id, i32_ty);
    ctx.push_op(block, lt_check.op_ref());

    let update_low = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let one = arena_wasm::i32_const(ctx, location, i32_ty, 1);
        ctx.push_op(inner_block, one.op_ref());
        let add_one = arena_wasm::i32_add(ctx, location, mid, one.result(ctx), i32_ty);
        ctx.push_op(inner_block, add_one.op_ref());
        let set_low = arena_wasm::local_set(ctx, location, add_one.result(ctx), locals::LOW);
        ctx.push_op(inner_block, set_low.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let update_high = {
        let inner_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let set_high = arena_wasm::local_set(ctx, location, mid, locals::HIGH);
        ctx.push_op(inner_block, set_high.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: smallvec![inner_block],
            parent_op: None,
        })
    };

    let update_if = arena_wasm::r#if(
        ctx,
        location,
        lt_check.result(ctx),
        nil_ty,
        update_low,
        update_high,
    );
    ctx.push_op(block, update_if.op_ref());

    // br $loop (continue searching)
    let br_loop = arena_wasm::br(ctx, location, 0);
    ctx.push_op(block, br_loop.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

// =============================================================================
// Helper functions
// =============================================================================

/// Get the WASM reference type for Evidence (wasm.arrayref).
fn evidence_ref_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("arrayref")).build())
}

/// Intern a `core.i32` type.
fn intern_i32(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Intern a `core.nil` type.
fn intern_nil(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
}

/// Intern a `core.func` type with the given parameter and return types.
fn intern_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
    let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).param(ret);
    for &p in params {
        builder = builder.param(p);
    }
    ctx.types.intern(builder.build())
}

/// Compute a stable ability ID hash from an ability type reference.
///
/// Uses the same hashing strategy as the Salsa-based pass: a deterministic
/// hash of the ability type data.
fn compute_ability_id(ctx: &IrContext, ability_ty: TypeRef) -> i32 {
    use std::hash::{Hash, Hasher};
    let data = ctx.types.get(ability_ty);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    data.dialect.hash(&mut hasher);
    data.name.hash(&mut hasher);
    // Use the first few params for more specific hashing
    for param in &data.params {
        param.hash(&mut hasher);
    }
    // Truncate to i32 and ensure positive for wasm i32 comparisons
    (hasher.finish() as i32).wrapping_abs()
}
