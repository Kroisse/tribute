//! Evidence runtime functions to WASM lowering.
//!
//! This pass replaces evidence runtime function stubs with real implementations:
//!
//! - `__tribute_evidence_lookup(ev, ability_id)` → binary search for marker
//! - `__tribute_evidence_extend(ev, marker)` → sorted insertion with binary search
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

use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{
    Block, BlockArg, BlockId, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol,
    Type, Value, ValueDef,
};

use super::type_converter::wasm_type_converter;
use trunk_ir_wasm_backend::gc_types::{EVIDENCE_IDX, MARKER_IDX};

/// Lower evidence runtime functions to WASM implementations.
///
/// This pass:
/// 1. Replaces stub function declarations with real binary search implementations
/// 2. Keeps function calls unchanged (they now call the real implementations)
#[salsa::tracked]
pub fn lower_evidence_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Phase 1: Replace stubs with real implementations
    let module = replace_evidence_function_stubs(db, module);

    // Phase 2: Pattern applicator (currently no-op, but kept for future extensions)
    let mut applicator = PatternApplicator::new(wasm_type_converter());
    applicator = applicator
        .add_pattern(EvidenceLookupPattern)
        .add_pattern(EvidenceExtendPattern);

    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
}

/// Replace evidence runtime function stubs with real implementations.
fn replace_evidence_function_stubs<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    let mut changed = false;
    let mut new_ops: Vec<Operation<'db>> = Vec::new();

    for op in entry_block.operations(db).iter() {
        // Replace func.func stubs
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.sym_name(db);
            let location = op.location(db);

            if name == Symbol::new("__tribute_evidence_lookup") {
                new_ops.push(generate_evidence_lookup_function(db, location));
                changed = true;
                continue;
            }
            if name == Symbol::new("__tribute_evidence_extend") {
                new_ops.push(generate_evidence_extend_function(db, location));
                changed = true;
                continue;
            }
        }

        // Replace wasm.func stubs
        if let Ok(func_op) = wasm::Func::from_operation(db, *op) {
            let name = func_op.sym_name(db);
            let location = op.location(db);

            if name == Symbol::new("__tribute_evidence_lookup") {
                new_ops.push(generate_evidence_lookup_function(db, location));
                changed = true;
                continue;
            }
            if name == Symbol::new("__tribute_evidence_extend") {
                new_ops.push(generate_evidence_extend_function(db, location));
                changed = true;
                continue;
            }
        }

        new_ops.push(*op);
    }

    if !changed {
        return module;
    }

    let new_entry_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        IdVec::from(new_ops),
    );

    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_entry_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

// =============================================================================
// Evidence Lookup Pattern
// =============================================================================

/// Pattern for `wasm.call @__tribute_evidence_lookup(ev, ability_id)`.
///
/// This pattern keeps the call unchanged because the stub function is replaced
/// with a real implementation by `replace_evidence_function_stubs`.
///
/// Algorithm (binary search in the generated function):
/// ```text
/// fn evidence_lookup(ev: Evidence, ability_id: i32) -> Marker {
///     let low = 0
///     let high = array.len(ev)
///     loop $search {
///         if low >= high { unreachable }  // ability not found = compiler bug
///         let mid = (low + high) / 2
///         let marker = array.get(ev, mid)
///         let marker_ability_id = struct.get(marker, 0)
///         if marker_ability_id == ability_id { return marker }
///         if marker_ability_id < ability_id {
///             low = mid + 1
///         } else {
///             high = mid
///         }
///         br $search
///     }
/// }
/// ```
struct EvidenceLookupPattern;

impl<'db> RewritePattern<'db> for EvidenceLookupPattern {
    fn match_and_rewrite<'a>(
        &self,
        _db: &'a dyn salsa::Database,
        _op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        // Keep the call unchanged - the stub is replaced with a real implementation
        // by replace_evidence_function_stubs
        RewriteResult::Unchanged
    }
}

// =============================================================================
// Evidence Extend Pattern
// =============================================================================

/// Pattern for `wasm.call @__tribute_evidence_extend(ev, marker)`.
///
/// This pattern keeps the call unchanged because the stub function is replaced
/// with a real implementation by `replace_evidence_function_stubs`.
///
/// Algorithm (binary search for sorted insertion in the generated function):
/// ```text
/// fn evidence_extend(ev: Evidence, marker: Marker) -> Evidence {
///     let old_len = array.len(ev)
///     let new_len = old_len + 1
///     let ability_id = struct.get(marker, 0)
///
///     // Binary search for insertion point to maintain sorted order
///     let insert_idx = binary_search_insert_position(ev, ability_id)
///
///     // Create new array
///     let new_ev = array.new_default(new_len)
///
///     // Copy elements before insertion point
///     if insert_idx > 0 {
///         array.copy(new_ev, 0, ev, 0, insert_idx)
///     }
///
///     // Insert new marker
///     array.set(new_ev, insert_idx, marker)
///
///     // Copy elements after insertion point
///     if insert_idx < old_len {
///         array.copy(new_ev, insert_idx + 1, ev, insert_idx, old_len - insert_idx)
///     }
///
///     return new_ev
/// }
/// ```
struct EvidenceExtendPattern;

impl<'db> RewritePattern<'db> for EvidenceExtendPattern {
    fn match_and_rewrite<'a>(
        &self,
        _db: &'a dyn salsa::Database,
        _op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        // Keep the call unchanged - the stub is replaced with a real implementation
        // by replace_evidence_function_stubs
        RewriteResult::Unchanged
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
fn generate_evidence_lookup_function<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
) -> Operation<'db> {
    let evidence_ty = evidence_ref_type(db);
    let i32_ty = core::I32::new(db).as_type();
    let marker_ty = wasm::Structref::new(db).as_type();

    let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, i32_ty]), marker_ty);

    let body_block_id = BlockId::fresh();

    // Block arguments = function parameters
    let ev_arg = BlockArg::of_type(db, evidence_ty);
    let target_id_arg = BlockArg::of_type(db, i32_ty);

    // References to block arguments
    let ev_val = Value::new(db, ValueDef::BlockArg(body_block_id), 0);
    let target_id_val = Value::new(db, ValueDef::BlockArg(body_block_id), 1);

    let mut ops = Vec::new();

    // Initialize low = 0
    let zero = wasm::i32_const(db, location, i32_ty, 0);
    ops.push(zero.as_operation());
    let low_init = wasm::local_set(db, location, zero.result(db), locals::LOW);
    ops.push(low_init.as_operation());

    // Initialize high = array.len(ev)
    let len_op = wasm::array_len(db, location, ev_val, i32_ty);
    ops.push(len_op.as_operation());
    let high_init = wasm::local_set(db, location, len_op.result(db), locals::HIGH);
    ops.push(high_init.as_operation());

    // Build the search loop
    let nil_ty = core::Nil::new(db).as_type();
    let loop_region = build_lookup_loop_body(db, location, ev_val, target_id_val, i32_ty);
    let loop_op = wasm::r#loop(db, location, nil_ty, loop_region);
    ops.push(loop_op.as_operation());

    // unreachable after loop (should never reach here - loop always returns)
    let unreachable_op = wasm::unreachable(db, location);
    ops.push(unreachable_op.as_operation());

    let body_block = Block::new(
        db,
        body_block_id,
        location,
        IdVec::from(vec![ev_arg, target_id_arg]),
        IdVec::from(ops),
    );
    let body = Region::new(db, location, IdVec::from(vec![body_block]));

    wasm::func(
        db,
        location,
        Symbol::new("__tribute_evidence_lookup"),
        *func_ty,
        body,
    )
    .as_operation()
}

/// Build the loop body for binary search lookup.
fn build_lookup_loop_body<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    ev_val: Value<'db>,
    target_id_val: Value<'db>,
    i32_ty: Type<'db>,
) -> Region<'db> {
    let nil_ty = core::Nil::new(db).as_type();
    let marker_ty = wasm::Structref::new(db).as_type();
    let block_id = BlockId::fresh();
    let mut ops = Vec::new();

    // low = local.get LOW
    let get_low = wasm::local_get(db, location, i32_ty, locals::LOW);
    let low = get_low.result(db);
    ops.push(get_low.as_operation());

    // high = local.get HIGH
    let get_high = wasm::local_get(db, location, i32_ty, locals::HIGH);
    let high = get_high.result(db);
    ops.push(get_high.as_operation());

    // Check low >= high -> unreachable (ability not found = compiler bug)
    let ge_check = wasm::i32_ge_s(db, location, low, high, i32_ty);
    ops.push(ge_check.as_operation());

    let unreachable_then = {
        let unreachable_op = wasm::unreachable(db, location);
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![unreachable_op.as_operation()]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };
    let empty_else = {
        let block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        Region::new(db, location, IdVec::from(vec![block]))
    };
    let bound_check_if = wasm::r#if(
        db,
        location,
        ge_check.result(db),
        nil_ty,
        unreachable_then,
        empty_else,
    );
    ops.push(bound_check_if.as_operation());

    // mid = (low + high) / 2
    let add_op = wasm::i32_add(db, location, low, high, i32_ty);
    ops.push(add_op.as_operation());
    let two = wasm::i32_const(db, location, i32_ty, 2);
    ops.push(two.as_operation());
    let mid_op = wasm::i32_div_u(db, location, add_op.result(db), two.result(db), i32_ty);
    let mid = mid_op.result(db);
    ops.push(mid_op.as_operation());

    // marker = array.get(ev, mid)
    let marker_op = wasm::array_get(db, location, ev_val, mid, marker_ty, EVIDENCE_IDX);
    let marker = marker_op.result(db);
    ops.push(marker_op.as_operation());

    // marker_ability_id = struct.get(marker, 0)
    let marker_id_op = wasm::struct_get(db, location, marker, i32_ty, MARKER_IDX, 0);
    let marker_id = marker_id_op.result(db);
    ops.push(marker_id_op.as_operation());

    // Check marker_ability_id == target -> return marker
    let eq_check = wasm::i32_eq(db, location, marker_id, target_id_val, i32_ty);
    ops.push(eq_check.as_operation());

    let found_then = {
        // Cast marker from structref to the expected type and return
        let return_op = wasm::r#return(db, location, vec![marker]);
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![return_op.as_operation()]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };

    let continue_else = {
        // marker_ability_id < target ? low = mid + 1 : high = mid
        let lt_check = wasm::i32_lt_s(db, location, marker_id, target_id_val, i32_ty);

        let update_low = {
            // low = mid + 1
            let one = wasm::i32_const(db, location, i32_ty, 1);
            let add_one = wasm::i32_add(db, location, mid, one.result(db), i32_ty);
            let set_low = wasm::local_set(db, location, add_one.result(db), locals::LOW);
            let block = Block::new(
                db,
                BlockId::fresh(),
                location,
                IdVec::new(),
                IdVec::from(vec![
                    one.as_operation(),
                    add_one.as_operation(),
                    set_low.as_operation(),
                ]),
            );
            Region::new(db, location, IdVec::from(vec![block]))
        };

        let update_high = {
            // high = mid
            let set_high = wasm::local_set(db, location, mid, locals::HIGH);
            let block = Block::new(
                db,
                BlockId::fresh(),
                location,
                IdVec::new(),
                IdVec::from(vec![set_high.as_operation()]),
            );
            Region::new(db, location, IdVec::from(vec![block]))
        };

        let update_if = wasm::r#if(
            db,
            location,
            lt_check.result(db),
            nil_ty,
            update_low,
            update_high,
        );

        // br $loop (target = 0, since loop is innermost)
        let br_loop = wasm::br(db, location, 0);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                lt_check.as_operation(),
                update_if.as_operation(),
                br_loop.as_operation(),
            ]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };

    let found_if = wasm::r#if(
        db,
        location,
        eq_check.result(db),
        nil_ty,
        found_then,
        continue_else,
    );
    ops.push(found_if.as_operation());

    let block = Block::new(db, block_id, location, IdVec::new(), IdVec::from(ops));
    Region::new(db, location, IdVec::from(vec![block]))
}

/// Generate the `__tribute_evidence_extend` function implementation.
///
/// Uses binary search to find the insertion point, then creates a new array
/// with the marker inserted at the correct position to maintain sorted order.
fn generate_evidence_extend_function<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
) -> Operation<'db> {
    let evidence_ty = evidence_ref_type(db);
    let i32_ty = core::I32::new(db).as_type();
    let marker_ty = wasm::Structref::new(db).as_type();

    let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, marker_ty]), evidence_ty);

    let body_block_id = BlockId::fresh();

    // Block arguments = function parameters
    let ev_arg = BlockArg::of_type(db, evidence_ty);
    let marker_arg = BlockArg::of_type(db, marker_ty);

    // References to block arguments
    let ev_val = Value::new(db, ValueDef::BlockArg(body_block_id), 0);
    let marker_val = Value::new(db, ValueDef::BlockArg(body_block_id), 1);

    let mut ops = Vec::new();

    // Get marker's ability_id for binary search
    let marker_id_op = wasm::struct_get(db, location, marker_val, i32_ty, MARKER_IDX, 0);
    let marker_id = marker_id_op.result(db);
    ops.push(marker_id_op.as_operation());

    // old_len = array.len(ev)
    let len_op = wasm::array_len(db, location, ev_val, i32_ty);
    let old_len = len_op.result(db);
    ops.push(len_op.as_operation());

    // Initialize low = 0
    let zero = wasm::i32_const(db, location, i32_ty, 0);
    ops.push(zero.as_operation());
    let low_init = wasm::local_set(db, location, zero.result(db), locals::LOW);
    ops.push(low_init.as_operation());

    // Initialize high = old_len
    let high_init = wasm::local_set(db, location, old_len, locals::HIGH);
    ops.push(high_init.as_operation());

    // Binary search loop to find insertion point
    // After loop, LOW contains the insertion index
    let nil_ty = core::Nil::new(db).as_type();
    let search_loop = build_extend_search_loop(db, location, ev_val, marker_id, i32_ty);
    let loop_op = wasm::r#loop(db, location, nil_ty, search_loop);

    // Wrap loop in block for br_if(..., 1) target
    let loop_block = Block::new(
        db,
        BlockId::fresh(),
        location,
        IdVec::new(),
        IdVec::from(vec![loop_op.as_operation()]),
    );
    let loop_region = Region::new(db, location, IdVec::from(vec![loop_block]));
    let block_op = wasm::block(db, location, nil_ty, loop_region);
    ops.push(block_op.as_operation());

    // insert_idx = local.get LOW
    let get_insert_idx = wasm::local_get(db, location, i32_ty, locals::LOW);
    let insert_idx = get_insert_idx.result(db);
    ops.push(get_insert_idx.as_operation());

    // new_len = old_len + 1
    let one = wasm::i32_const(db, location, i32_ty, 1);
    ops.push(one.as_operation());
    let add_len_op = wasm::i32_add(db, location, old_len, one.result(db), i32_ty);
    let new_len = add_len_op.result(db);
    ops.push(add_len_op.as_operation());

    // new_ev = array.new_default(new_len)
    let new_array_op = wasm::array_new_default(db, location, new_len, evidence_ty, EVIDENCE_IDX);
    let new_ev = new_array_op.result(db);
    ops.push(new_array_op.as_operation());

    // Copy elements before insertion point: array.copy(new_ev, 0, ev, 0, insert_idx)
    // Only if insert_idx > 0
    let zero2 = wasm::i32_const(db, location, i32_ty, 0);
    ops.push(zero2.as_operation());

    let gt_zero = wasm::i32_gt_s(db, location, insert_idx, zero2.result(db), i32_ty);
    ops.push(gt_zero.as_operation());

    let copy_prefix_then = {
        let zero3 = wasm::i32_const(db, location, i32_ty, 0);
        let copy_op = wasm::array_copy(
            db,
            location,
            new_ev,
            zero3.result(db),
            ev_val,
            zero3.result(db),
            insert_idx,
            EVIDENCE_IDX,
            EVIDENCE_IDX,
        );
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![zero3.as_operation(), copy_op.as_operation()]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };
    let empty_else1 = {
        let block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        Region::new(db, location, IdVec::from(vec![block]))
    };
    let copy_prefix_if = wasm::r#if(
        db,
        location,
        gt_zero.result(db),
        nil_ty,
        copy_prefix_then,
        empty_else1,
    );
    ops.push(copy_prefix_if.as_operation());

    // Set marker at insert_idx: array.set(new_ev, insert_idx, marker)
    let set_op = wasm::array_set(db, location, new_ev, insert_idx, marker_val, EVIDENCE_IDX);
    ops.push(set_op.as_operation());

    // Copy elements after insertion point: array.copy(new_ev, insert_idx+1, ev, insert_idx, old_len - insert_idx)
    // suffix_len = old_len - insert_idx
    let suffix_len_op = wasm::i32_sub(db, location, old_len, insert_idx, i32_ty);
    let suffix_len = suffix_len_op.result(db);
    ops.push(suffix_len_op.as_operation());

    // Only if suffix_len > 0
    let zero4 = wasm::i32_const(db, location, i32_ty, 0);
    ops.push(zero4.as_operation());
    let gt_zero2 = wasm::i32_gt_s(db, location, suffix_len, zero4.result(db), i32_ty);
    ops.push(gt_zero2.as_operation());

    let copy_suffix_then = {
        let one2 = wasm::i32_const(db, location, i32_ty, 1);
        let dst_offset_op = wasm::i32_add(db, location, insert_idx, one2.result(db), i32_ty);
        let copy_op = wasm::array_copy(
            db,
            location,
            new_ev,
            dst_offset_op.result(db),
            ev_val,
            insert_idx,
            suffix_len,
            EVIDENCE_IDX,
            EVIDENCE_IDX,
        );
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                one2.as_operation(),
                dst_offset_op.as_operation(),
                copy_op.as_operation(),
            ]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };
    let empty_else2 = {
        let block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        Region::new(db, location, IdVec::from(vec![block]))
    };
    let copy_suffix_if = wasm::r#if(
        db,
        location,
        gt_zero2.result(db),
        nil_ty,
        copy_suffix_then,
        empty_else2,
    );
    ops.push(copy_suffix_if.as_operation());

    // Return new_ev
    let return_op = wasm::r#return(db, location, vec![new_ev]);
    ops.push(return_op.as_operation());

    let body_block = Block::new(
        db,
        body_block_id,
        location,
        IdVec::from(vec![ev_arg, marker_arg]),
        IdVec::from(ops),
    );
    let body = Region::new(db, location, IdVec::from(vec![body_block]));

    wasm::func(
        db,
        location,
        Symbol::new("__tribute_evidence_extend"),
        *func_ty,
        body,
    )
    .as_operation()
}

/// Build the search loop for finding insertion position in evidence_extend.
///
/// This is a binary search that finds the first index where ev[i].ability_id >= marker_id.
/// After the loop, LOW contains the insertion index.
fn build_extend_search_loop<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    ev_val: Value<'db>,
    marker_id: Value<'db>,
    i32_ty: Type<'db>,
) -> Region<'db> {
    let nil_ty = core::Nil::new(db).as_type();
    let marker_ty = wasm::Structref::new(db).as_type();
    let block_id = BlockId::fresh();
    let mut ops = Vec::new();

    // low = local.get LOW
    let get_low = wasm::local_get(db, location, i32_ty, locals::LOW);
    let low = get_low.result(db);
    ops.push(get_low.as_operation());

    // high = local.get HIGH
    let get_high = wasm::local_get(db, location, i32_ty, locals::HIGH);
    let high = get_high.result(db);
    ops.push(get_high.as_operation());

    // if low >= high: break (insertion point found at LOW)
    let ge_check = wasm::i32_ge_s(db, location, low, high, i32_ty);
    ops.push(ge_check.as_operation());

    // br_if to exit loop (target = 1 to break out of loop to outer block)
    let br_if_done = wasm::br_if(db, location, ge_check.result(db), 1);
    ops.push(br_if_done.as_operation());

    // mid = (low + high) / 2
    let add_op = wasm::i32_add(db, location, low, high, i32_ty);
    ops.push(add_op.as_operation());
    let two = wasm::i32_const(db, location, i32_ty, 2);
    ops.push(two.as_operation());
    let mid_op = wasm::i32_div_u(db, location, add_op.result(db), two.result(db), i32_ty);
    let mid = mid_op.result(db);
    ops.push(mid_op.as_operation());

    // mid_marker = array.get(ev, mid)
    let mid_marker_op = wasm::array_get(db, location, ev_val, mid, marker_ty, EVIDENCE_IDX);
    let mid_marker = mid_marker_op.result(db);
    ops.push(mid_marker_op.as_operation());

    // mid_ability_id = struct.get(mid_marker, 0)
    let mid_id_op = wasm::struct_get(db, location, mid_marker, i32_ty, MARKER_IDX, 0);
    let mid_id = mid_id_op.result(db);
    ops.push(mid_id_op.as_operation());

    // if mid_ability_id < marker_id: low = mid + 1
    // else: high = mid
    let lt_check = wasm::i32_lt_s(db, location, mid_id, marker_id, i32_ty);
    ops.push(lt_check.as_operation());

    let update_low = {
        let one = wasm::i32_const(db, location, i32_ty, 1);
        let add_one = wasm::i32_add(db, location, mid, one.result(db), i32_ty);
        let set_low = wasm::local_set(db, location, add_one.result(db), locals::LOW);
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![
                one.as_operation(),
                add_one.as_operation(),
                set_low.as_operation(),
            ]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };

    let update_high = {
        let set_high = wasm::local_set(db, location, mid, locals::HIGH);
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![set_high.as_operation()]),
        );
        Region::new(db, location, IdVec::from(vec![block]))
    };

    let update_if = wasm::r#if(
        db,
        location,
        lt_check.result(db),
        nil_ty,
        update_low,
        update_high,
    );
    ops.push(update_if.as_operation());

    // br $loop (continue searching)
    let br_loop = wasm::br(db, location, 0);
    ops.push(br_loop.as_operation());

    let block = Block::new(db, block_id, location, IdVec::new(), IdVec::from(ops));
    Region::new(db, location, IdVec::from(vec![block]))
}

// =============================================================================
// Helper functions
// =============================================================================

/// Get the WASM reference type for Evidence (arrayref).
fn evidence_ref_type(db: &dyn salsa::Database) -> Type<'_> {
    wasm::Arrayref::new(db).as_type()
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tribute_ir::ModulePathExt;
    use trunk_ir::dialect::core;
    use trunk_ir::{BlockArg, BlockId, DialectType, Location, PathId, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_evidence_lookup_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);

        // Create a module with __tribute_evidence_lookup declaration
        let evidence_ty = wasm::Anyref::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let marker_ty = wasm::Structref::new(db).as_type();

        let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, i32_ty]), marker_ty);
        let unreachable_op = func::unreachable(db, location);
        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::from(vec![
                BlockArg::of_type(db, evidence_ty),
                BlockArg::of_type(db, i32_ty),
            ]),
            IdVec::from(vec![unreachable_op.as_operation()]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));
        let lookup_func = func::func(
            db,
            location,
            Symbol::new("__tribute_evidence_lookup"),
            *func_ty,
            body,
        );

        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![lookup_func.as_operation()]),
        );
        let module_body = Region::new(db, location, IdVec::from(vec![entry_block]));
        core::Module::create(db, location, "test".into(), module_body)
    }

    #[salsa_test]
    fn test_evidence_runtime_function_replacement(db: &salsa::DatabaseImpl) {
        // Test that evidence runtime function stubs are replaced with real implementations
        let module = make_evidence_lookup_module(db);

        // Lower
        let lowered = lower_evidence_to_wasm(db, module);

        // Check that the function is still present (replaced, not removed)
        let new_body = lowered.body(db);
        let new_entry = new_body.blocks(db).first().unwrap();
        assert_eq!(
            new_entry.operations(db).len(),
            1,
            "Evidence runtime function should be replaced with real implementation"
        );

        // Verify it's a wasm.func (not func.func stub)
        let func_op = new_entry.operations(db).first().unwrap();
        let wasm_func = wasm::Func::from_operation(db, *func_op)
            .expect("expected wasm.func for evidence lookup implementation");
        assert_eq!(
            wasm_func.sym_name(db),
            Symbol::new("__tribute_evidence_lookup"),
            "function name should be preserved"
        );
    }

    #[salsa::tracked]
    fn make_other_function_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);

        // Create a module with a regular function
        let i32_ty = core::I32::new(db).as_type();
        let func_ty = core::Func::new(db, IdVec::new(), i32_ty);
        let unreachable_op = func::unreachable(db, location);
        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![unreachable_op.as_operation()]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));
        let other_func = func::func(db, location, Symbol::new("other_function"), *func_ty, body);

        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![other_func.as_operation()]),
        );
        let module_body = Region::new(db, location, IdVec::from(vec![entry_block]));
        core::Module::create(db, location, "test".into(), module_body)
    }

    #[salsa_test]
    fn test_other_functions_preserved(db: &salsa::DatabaseImpl) {
        // Test that non-evidence functions are preserved
        let module = make_other_function_module(db);

        // Lower
        let lowered = lower_evidence_to_wasm(db, module);

        // Check that the function was preserved
        let new_body = lowered.body(db);
        let new_entry = new_body.blocks(db).first().unwrap();
        assert_eq!(
            new_entry.operations(db).len(),
            1,
            "Other functions should be preserved"
        );
    }

    /// Create a module with a function that calls __tribute_evidence_lookup.
    #[salsa::tracked]
    fn make_evidence_lookup_call_module(db: &dyn salsa::Database) -> core::Module<'_> {
        use trunk_ir::{Value, ValueDef};

        let location = test_location(db);
        let evidence_ty = wasm::Arrayref::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let marker_ty = wasm::Structref::new(db).as_type();

        // Create a function that calls __tribute_evidence_lookup
        let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, i32_ty]), marker_ty);

        // Build caller function body
        let body_block_id = BlockId::fresh();
        let ev_arg = BlockArg::of_type(db, evidence_ty);
        let ability_id_arg = BlockArg::of_type(db, i32_ty);

        // wasm.call @__tribute_evidence_lookup(ev, ability_id)
        let ev_val = Value::new(db, ValueDef::BlockArg(body_block_id), 0);
        let ability_id_val = Value::new(db, ValueDef::BlockArg(body_block_id), 1);
        let call_op = wasm::call(
            db,
            location,
            vec![ev_val, ability_id_val],
            vec![marker_ty],
            Symbol::new("__tribute_evidence_lookup"),
        );

        // Return the result
        let return_op = func::r#return(db, location, vec![call_op.result(db, 0)]);

        let body_block = Block::new(
            db,
            body_block_id,
            location,
            IdVec::from(vec![ev_arg, ability_id_arg]),
            IdVec::from(vec![call_op.as_operation(), return_op.as_operation()]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));
        let caller_func = func::func(db, location, Symbol::new("test_caller"), *func_ty, body);

        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![caller_func.as_operation()]),
        );
        let module_body = Region::new(db, location, IdVec::from(vec![entry_block]));
        core::Module::create(db, location, "test".into(), module_body)
    }

    #[salsa_test]
    fn test_evidence_lookup_call_preserved(db: &salsa::DatabaseImpl) {
        // Test that __tribute_evidence_lookup calls are preserved (not inlined)
        // because ability_id is a hash value, not an array index
        let module = make_evidence_lookup_call_module(db);

        // Lower
        let lowered = lower_evidence_to_wasm(db, module);

        // Find the caller function and check that the call is still there
        let body = lowered.body(db);
        let entry = body.blocks(db).first().unwrap();

        // Should have the caller function
        assert_eq!(entry.operations(db).len(), 1);

        let func_op = entry.operations(db).first().unwrap();
        let func = func::Func::from_operation(db, *func_op).expect("expected func.func");

        // Check the function body still has the wasm.call
        let func_body = func.body(db);
        let func_entry = func_body.blocks(db).first().unwrap();

        // Should have: wasm.call + func.return
        assert_eq!(func_entry.operations(db).len(), 2);

        // First op should still be wasm.call
        let first_op = func_entry.operations(db).first().unwrap();
        let call = wasm::Call::from_operation(db, *first_op).expect("expected wasm.call");
        assert_eq!(
            call.callee(db).last_segment(),
            Symbol::new("__tribute_evidence_lookup"),
            "evidence_lookup call should be preserved"
        );
    }
}
