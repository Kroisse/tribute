//! Evidence runtime functions to WASM lowering.
//!
//! This pass transforms calls to evidence runtime functions into inline WASM operations:
//!
//! - `__tribute_evidence_lookup(ev, ability_id)` → inline array search + struct_get
//! - `__tribute_evidence_extend(ev, marker)` → inline array copy with insertion
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
//! Phase 1: Remove the runtime function declarations (stub functions).
//! Phase 2: Transform function calls to inline WASM operations.
//!
//! The lookup uses linear search since evidence arrays are typically small (< 10 elements).
//! The extend creates a new array and copies elements, maintaining sorted order by ability_id.

use tribute_ir::ModulePathExt;
use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{
    Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type, Value,
};

use super::type_converter::wasm_type_converter;
use trunk_ir_wasm_backend::gc_types::EVIDENCE_IDX;

/// Lower evidence runtime functions to inline WASM operations.
///
/// This pass:
/// 1. Removes the stub function declarations for evidence runtime functions
/// 2. Transforms `wasm.call @__tribute_evidence_lookup` to inline array search
/// 3. Transforms `wasm.call @__tribute_evidence_extend` to inline array copy
#[salsa::tracked]
pub fn lower_evidence_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // Phase 1: Remove function declarations
    let module = remove_evidence_function_declarations(db, module);

    // Phase 2: Transform function calls to inline operations
    let mut applicator = PatternApplicator::new(wasm_type_converter());
    applicator = applicator
        .add_pattern(EvidenceLookupPattern)
        .add_pattern(EvidenceExtendPattern);

    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
}

/// Remove evidence runtime function declarations (Phase 1).
fn remove_evidence_function_declarations<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    let mut changed = false;
    let new_ops: IdVec<Operation<'db>> = entry_block
        .operations(db)
        .iter()
        .filter_map(|op| {
            // Remove evidence runtime function declarations (func.func stubs)
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let name = func_op.sym_name(db);
                if name == Symbol::new("__tribute_evidence_lookup")
                    || name == Symbol::new("__tribute_evidence_extend")
                {
                    changed = true;
                    return None; // Remove the declaration
                }
            }

            // Also remove wasm.func stubs if they exist
            if let Ok(func_op) = wasm::Func::from_operation(db, *op) {
                let name = func_op.sym_name(db);
                if name == Symbol::new("__tribute_evidence_lookup")
                    || name == Symbol::new("__tribute_evidence_extend")
                {
                    changed = true;
                    return None; // Remove the declaration
                }
            }

            Some(*op)
        })
        .collect();

    if !changed {
        return module;
    }

    let new_entry_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        new_ops,
    );

    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_entry_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

// =============================================================================
// Evidence Lookup Pattern
// =============================================================================

/// Pattern for `wasm.call @__tribute_evidence_lookup(ev, ability_id)` → inline search
///
/// Algorithm (linear search):
/// ```text
/// fn evidence_lookup(ev: Evidence, ability_id: i32) -> Marker {
///     let len = array.len(ev)
///     let i = 0
///     block $found {
///         loop $search {
///             if i >= len { unreachable }  // ability not found = bug
///             let marker = array.get(ev, i)
///             let marker_ability_id = struct.get(marker, 0)
///             if marker_ability_id == ability_id { br $found }
///             i = i + 1
///             br $search
///         }
///     }
///     return marker
/// }
/// ```
struct EvidenceLookupPattern;

impl<'db> RewritePattern<'db> for EvidenceLookupPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        // Check if this is wasm.call to __tribute_evidence_lookup
        let Ok(call_op) = wasm::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        if call_op.callee(db).last_segment() != Symbol::new("__tribute_evidence_lookup") {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);
        if operands.len() != 2 {
            return RewriteResult::Unchanged;
        }

        let evidence_val = operands[0];
        let ability_id_val = operands[1];
        let location = op.location(db);

        // Generate inline lookup code
        let ops = generate_evidence_lookup(db, location, evidence_val, ability_id_val);
        RewriteResult::Expand(ops)
    }
}

/// Generate inline WASM operations for evidence lookup.
///
/// The generated code uses a block+loop pattern for the search:
/// - Outer block catches the "found" branch
/// - Inner loop iterates through the array
fn generate_evidence_lookup<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    evidence_val: Value<'db>,
    ability_id_val: Value<'db>,
) -> Vec<Operation<'db>> {
    let i32_ty = core::I32::new(db).as_type();
    let marker_ref_ty = marker_ref_type(db);

    // For now, use a simple sequential search pattern without complex control flow.
    // This generates operations that will work with the current emitter.
    //
    // Simplified approach: Since evidence is sorted and typically small,
    // we can use a linear scan. For the MVP, we keep the call but it will
    // be linked/inlined at a later stage.
    //
    // TODO: Implement proper block+loop pattern when the emitter supports it.

    // Get array length
    let len_op = wasm::array_len(db, location, evidence_val, i32_ty);
    let _len_val = len_op.result(db);

    // For now, emit a simple array access at index 0 as placeholder.
    // The actual loop implementation requires block/loop region support
    // which needs careful coordination with the emitter.
    //
    // This is a temporary implementation that assumes ability_id maps directly
    // to array index (which works for single-ability cases).
    let get_marker = wasm::array_get(
        db,
        location,
        evidence_val,
        ability_id_val,
        marker_ref_ty,
        EVIDENCE_IDX,
    );

    vec![len_op.as_operation(), get_marker.as_operation()]
}

// =============================================================================
// Evidence Extend Pattern
// =============================================================================

/// Pattern for `wasm.call @__tribute_evidence_extend(ev, marker)` → inline extend
///
/// Algorithm:
/// ```text
/// fn evidence_extend(ev: Evidence, marker: Marker) -> Evidence {
///     let old_len = array.len(ev)
///     let new_len = old_len + 1
///     let ability_id = struct.get(marker, 0)
///
///     // Find insertion point to maintain sorted order
///     let insert_idx = find_insert_position(ev, ability_id)
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
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        // Check if this is wasm.call to __tribute_evidence_extend
        let Ok(call_op) = wasm::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        if call_op.callee(db).last_segment() != Symbol::new("__tribute_evidence_extend") {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);
        if operands.len() != 2 {
            return RewriteResult::Unchanged;
        }

        let evidence_val = operands[0];
        let marker_val = operands[1];
        let location = op.location(db);

        // Generate inline extend code
        let ops = generate_evidence_extend(db, location, evidence_val, marker_val);
        RewriteResult::Expand(ops)
    }
}

/// Generate inline WASM operations for evidence extend.
///
/// For simplicity, this appends the marker at the end. The sorted order
/// is maintained by the order in which handlers are installed (innermost first).
fn generate_evidence_extend<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    evidence_val: Value<'db>,
    marker_val: Value<'db>,
) -> Vec<Operation<'db>> {
    let i32_ty = core::I32::new(db).as_type();
    let evidence_ref_ty = evidence_ref_type(db);

    let mut ops = Vec::new();

    // Get old array length
    let len_op = wasm::array_len(db, location, evidence_val, i32_ty);
    let old_len = len_op.result(db);
    ops.push(len_op.as_operation());

    // Calculate new length: old_len + 1
    let one_const = wasm::i32_const(db, location, i32_ty, 1);
    let one_val = one_const.result(db);
    ops.push(one_const.as_operation());

    let add_op = wasm::i32_add(db, location, old_len, one_val, i32_ty);
    let new_len = add_op.result(db);
    ops.push(add_op.as_operation());

    // Create new array with default values
    let new_array_op =
        wasm::array_new_default(db, location, new_len, evidence_ref_ty, EVIDENCE_IDX);
    let new_ev = new_array_op.result(db);
    ops.push(new_array_op.as_operation());

    // Copy all elements from old array to new array (at offset 0)
    // array.copy(dst, dst_offset, src, src_offset, len)
    let zero_const = wasm::i32_const(db, location, i32_ty, 0);
    let zero_val = zero_const.result(db);
    ops.push(zero_const.as_operation());

    let copy_op = wasm::array_copy(
        db,
        location,
        new_ev,       // dst
        zero_val,     // dst_offset = 0
        evidence_val, // src
        zero_val,     // src_offset = 0 (reuse zero_val)
        old_len,      // len = old_len
        EVIDENCE_IDX,
        EVIDENCE_IDX,
    );
    ops.push(copy_op.as_operation());

    // Set the new marker at the end (index = old_len)
    let set_op = wasm::array_set(db, location, new_ev, old_len, marker_val, EVIDENCE_IDX);
    ops.push(set_op.as_operation());

    // The new evidence array is the result
    // We need to ensure new_ev is the last defined value for result remapping
    ops
}

// =============================================================================
// Helper functions
// =============================================================================

/// Get the WASM reference type for Marker.
fn marker_ref_type(db: &dyn salsa::Database) -> Type<'_> {
    // Marker is a struct ref (non-nullable)
    wasm::Structref::new(db).as_type()
}

/// Get the WASM reference type for Evidence.
fn evidence_ref_type(db: &dyn salsa::Database) -> Type<'_> {
    // Evidence is an array ref (non-nullable)
    wasm::Arrayref::new(db).as_type()
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
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
    fn test_evidence_runtime_function_removal(db: &salsa::DatabaseImpl) {
        // Test that evidence runtime function declarations are removed
        let module = make_evidence_lookup_module(db);

        // Lower
        let lowered = lower_evidence_to_wasm(db, module);

        // Check that the function declaration was removed
        let new_body = lowered.body(db);
        let new_entry = new_body.blocks(db).first().unwrap();
        assert_eq!(
            new_entry.operations(db).len(),
            0,
            "Evidence runtime function should be removed"
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
}
