//! Lower ability.handler_table to wasm.table + wasm.elem.
//!
//! This pass transforms the `ability.handler_table` operation (which captures
//! handler dispatch table structure) into WebAssembly table and element segment
//! definitions.
//!
//! ## Transformation
//!
//! ```text
//! // Before (after resolve_evidence pass)
//! ability.handler_table { max_ops_per_handler = 8 }
//!   entries {
//!     ability.handler_entry { tag = 0, op_count = 2 }
//!       funcs { func.constant @__handler_0_op_0, func.constant @__handler_0_op_1 }
//!     ability.handler_entry { tag = 1, op_count = 1 }
//!       funcs { func.constant @__handler_1_op_0 }
//!   }
//!
//! // After (WASM dialect)
//! wasm.table { reftype = "funcref", min = 16, max = 16 }  // 2 handlers * 8 max ops
//! wasm.elem { table = 0, offset = 0 }
//!   funcs {
//!     wasm.ref_func @__handler_0_op_0
//!     wasm.ref_func @__handler_0_op_1
//!     wasm.ref_null  // padding
//!     ...
//!     wasm.ref_func @__handler_1_op_0
//!     ...
//!   }
//! ```
//!
//! ## Table Layout
//!
//! The table uses a flat layout where each handler occupies `max_ops_per_handler`
//! slots. This enables O(1) dispatch:
//!
//! ```text
//! table_index = op_table_index * max_ops_per_handler + op_offset
//! ```

use tribute_ir::dialect::ability;
use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::{Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol};

/// Lower `ability.handler_table` to `wasm.table` + `wasm.elem`.
///
/// This pass:
/// 1. Finds `ability.handler_table` in module body
/// 2. Extracts handler entries and function references
/// 3. Creates `wasm.table` with appropriate size
/// 4. Creates `wasm.elem` with flattened function references (with padding)
/// 5. Removes the original `ability.handler_table` operation
///
/// If no `ability.handler_table` is found, the module is returned unchanged.
#[salsa::tracked]
pub fn lower_handler_table<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    // Find ability.handler_table operation
    let handler_table_idx = entry_block
        .operations(db)
        .iter()
        .position(|op| ability::HandlerTable::from_operation(db, *op).is_ok());

    let Some(table_idx) = handler_table_idx else {
        return module;
    };

    let handler_table_op = entry_block.operations(db)[table_idx];
    let handler_table = ability::HandlerTable::from_operation(db, handler_table_op)
        .expect("Already verified this is a handler_table");

    let location = handler_table_op.location(db);
    let max_ops = handler_table.max_ops_per_handler(db);

    // Extract handler entries
    let entries = extract_handler_entries(db, &handler_table);

    if entries.is_empty() {
        // No entries, just remove the handler_table operation
        return remove_handler_table(db, module, table_idx);
    }

    // Calculate table size: number of handlers * max_ops_per_handler
    let table_size = entries.len() as u32 * max_ops;

    // Generate wasm.table operation
    let table_op = wasm::table(
        db,
        location,
        Symbol::new("funcref"),
        table_size,
        Some(table_size),
    );

    // Generate wasm.elem with flattened function references
    let elem_op = build_elem_segment(db, location, &entries, max_ops);

    // Replace ability.handler_table with wasm.table + wasm.elem
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    for (i, op) in entry_block.operations(db).iter().enumerate() {
        if i == table_idx {
            // Replace handler_table with table + elem
            new_ops.push(table_op.as_operation());
            new_ops.push(elem_op);
        } else {
            new_ops.push(*op);
        }
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

/// Information about a handler entry extracted from ability.handler_entry.
struct HandlerEntryInfo<'db> {
    /// Handler tag (op_table_index)
    tag: u32,
    /// Number of operations in this handler
    _op_count: u32,
    /// Function reference symbols for each operation
    func_refs: Vec<Symbol>,
    /// Location for error reporting
    _location: Location<'db>,
}

/// Extract handler entry information from ability.handler_table.
fn extract_handler_entries<'db>(
    db: &'db dyn salsa::Database,
    handler_table: &ability::HandlerTable<'db>,
) -> Vec<HandlerEntryInfo<'db>> {
    let mut entries = Vec::new();
    let entries_region = handler_table.entries(db);

    for block in entries_region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(entry) = ability::HandlerEntry::from_operation(db, *op) {
                let tag = entry.tag(db);
                let op_count = entry.op_count(db);
                let location = op.location(db);

                // Extract func_refs from funcs region
                let func_refs = extract_func_refs(db, &entry);

                entries.push(HandlerEntryInfo {
                    tag,
                    _op_count: op_count,
                    func_refs,
                    _location: location,
                });
            }
        }
    }

    // Sort by tag to ensure consistent ordering
    entries.sort_by_key(|e| e.tag);
    entries
}

/// Extract function reference symbols from handler_entry's funcs region.
fn extract_func_refs<'db>(
    db: &'db dyn salsa::Database,
    entry: &ability::HandlerEntry<'db>,
) -> Vec<Symbol> {
    let mut func_refs = Vec::new();
    let funcs_region = entry.funcs(db);

    for block in funcs_region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(constant) = func::Constant::from_operation(db, *op) {
                func_refs.push(constant.func_ref(db));
            }
        }
    }

    func_refs
}

/// Build wasm.elem segment with flattened function references and padding.
fn build_elem_segment<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    entries: &[HandlerEntryInfo<'db>],
    max_ops_per_handler: u32,
) -> Operation<'db> {
    let funcref_ty = wasm::Funcref::new(db).as_type();

    // Build flattened function references with padding
    let mut func_ops: Vec<Operation<'db>> = Vec::new();

    for entry in entries {
        // Add function references for this handler
        for func_ref in &entry.func_refs {
            let ref_func_op = wasm::ref_func(db, location, funcref_ty, *func_ref);
            func_ops.push(ref_func_op.as_operation());
        }

        // Add padding (null references) to fill up to max_ops_per_handler
        let func_count = entry.func_refs.len();
        if func_count > max_ops_per_handler as usize {
            panic!(
                "Handler has {} operations, exceeds max_ops_per_handler ({}). Handler tag: {}",
                func_count, max_ops_per_handler, entry.tag
            );
        }
        let padding_count = max_ops_per_handler as usize - func_count;
        for _ in 0..padding_count {
            let null_op = wasm::ref_null(db, location, funcref_ty, funcref_ty, None);
            func_ops.push(null_op.as_operation());
        }
    }

    // Create funcs region
    let funcs_block = Block::new(
        db,
        trunk_ir::BlockId::fresh(),
        location,
        IdVec::new(),
        IdVec::from(func_ops),
    );
    let funcs_region = Region::new(db, location, IdVec::from(vec![funcs_block]));

    // Create wasm.elem operation
    // table = 0 (first table), offset = 0 (start of table)
    wasm::elem(db, location, Some(0), Some(0), funcs_region).as_operation()
}

/// Remove ability.handler_table from module (used when entries is empty).
fn remove_handler_table<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    table_idx: usize,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);
    let entry_block = blocks.first().expect("Module should have entry block");

    let new_ops: IdVec<Operation<'db>> = entry_block
        .operations(db)
        .iter()
        .enumerate()
        .filter_map(|(i, op)| if i == table_idx { None } else { Some(*op) })
        .collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve_evidence::{MAX_OPS_PER_HANDLER, OpTableRegistry, emit_handler_table};
    use salsa_test_macros::salsa_test;
    use trunk_ir::{BlockId, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn run_lower_handler_table_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);

        // Create a module with handler_table (via emit_handler_table)
        let entry_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        let body = Region::new(db, location, IdVec::from(vec![entry_block]));
        let module = core::Module::create(db, location, Symbol::new("test"), body);

        // Create registry with handlers
        let mut registry = OpTableRegistry::new();
        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        registry.register(
            vec![state_ref.as_type()],
            vec![
                (state_ref.as_type(), Symbol::new("get")),
                (state_ref.as_type(), Symbol::new("set")),
            ],
            location,
        );

        // Emit handler_table
        let module_with_table = emit_handler_table(db, module, &registry);

        // Lower to WASM
        let lowered = lower_handler_table(db, module_with_table);

        // Verify wasm.table was created
        let body = lowered.body(db);
        let entry = body.blocks(db).first().unwrap();
        let ops = entry.operations(db);

        // Find wasm.table
        let table_op = ops
            .iter()
            .find(|op| wasm::Table::from_operation(db, **op).is_ok());
        let Some(table_op) = table_op else {
            return Err("Should have wasm.table operation".to_string());
        };

        let table = wasm::Table::from_operation(db, *table_op).unwrap();
        let expected_size = MAX_OPS_PER_HANDLER; // 1 handler * max_ops
        if table.min(db) != expected_size {
            return Err(format!(
                "Table min should be {}, got {}",
                expected_size,
                table.min(db)
            ));
        }

        // Find wasm.elem
        let elem_op = ops
            .iter()
            .find(|op| wasm::Elem::from_operation(db, **op).is_ok());
        let Some(elem_op) = elem_op else {
            return Err("Should have wasm.elem operation".to_string());
        };

        let elem = wasm::Elem::from_operation(db, *elem_op).unwrap();
        if elem.table(db) != Some(0) {
            return Err("Elem should reference table 0".to_string());
        }

        // Verify funcs region has correct number of operations (2 + padding)
        let funcs_region = elem.funcs(db);
        let funcs_block = funcs_region.blocks(db).first().unwrap();
        let func_ops = funcs_block.operations(db);

        let expected_ops = MAX_OPS_PER_HANDLER as usize; // 2 refs + padding
        if func_ops.len() != expected_ops {
            return Err(format!(
                "Should have {} func ops (with padding), got {}",
                expected_ops,
                func_ops.len()
            ));
        }

        // First two should be ref_func
        for i in 0..2 {
            if wasm::RefFunc::from_operation(db, func_ops[i]).is_err() {
                return Err(format!("Op {} should be wasm.ref_func", i));
            }
        }

        // Rest should be ref_null (padding)
        for i in 2..expected_ops {
            if wasm::RefNull::from_operation(db, func_ops[i]).is_err() {
                return Err(format!("Op {} should be wasm.ref_null (padding)", i));
            }
        }

        // Verify handler_table was removed
        let handler_table_exists = ops
            .iter()
            .any(|op| ability::HandlerTable::from_operation(db, *op).is_ok());
        if handler_table_exists {
            return Err("ability.handler_table should be removed after lowering".to_string());
        }

        Ok(())
    }

    #[salsa_test]
    fn test_lower_handler_table(db: &salsa::DatabaseImpl) {
        let result = run_lower_handler_table_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }

    #[salsa::tracked]
    fn run_lower_handler_table_no_table_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);

        // Create a module without handler_table
        let ret_op = func::r#return(db, location, None).as_operation();
        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(vec![ret_op]),
        );
        let body = Region::new(db, location, IdVec::from(vec![entry_block]));
        let module = core::Module::create(db, location, Symbol::new("test"), body);

        // Lower (should be no-op)
        let lowered = lower_handler_table(db, module);

        // Verify module unchanged
        let body = lowered.body(db);
        let entry = body.blocks(db).first().unwrap();
        let ops = entry.operations(db);

        if ops.len() != 1 {
            return Err(format!("Should have 1 op, got {}", ops.len()));
        }

        if func::Return::from_operation(db, ops[0]).is_err() {
            return Err("Should still have return op".to_string());
        }

        Ok(())
    }

    #[salsa_test]
    fn test_lower_handler_table_no_table(db: &salsa::DatabaseImpl) {
        let result = run_lower_handler_table_no_table_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }
}
