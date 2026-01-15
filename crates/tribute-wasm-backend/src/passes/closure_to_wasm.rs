//! Lower closure dialect operations to wasm dialect.
//!
//! This pass converts closure operations to WasmGC struct operations:
//!
//! ## Closure Representation
//!
//! Closures are represented as WasmGC structs with two fields:
//! - Field 0: function table index (i32) - index into function table
//! - Field 1: environment (anyref) - captured variables as struct
//!
//! ## Transformations
//!
//! - `closure.new(env) @func_ref` -> `wasm.struct_new` with func table index and env
//! - `closure.func(closure)` -> `wasm.struct_get` field 0
//! - `closure.env(closure)` -> `wasm.struct_get` field 1
//!
//! ## Function Table
//!
//! Functions referenced by closures are collected and placed in a function table.
//! This pass generates `wasm.table` and `wasm.elem` operations for the table.

use std::collections::HashMap;

use tribute_ir::dialect::closure;
use trunk_ir::dialect::func;
use trunk_ir::dialect::{arith, core::Module, wasm};
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Symbol,
};

use crate::gc_types::CLOSURE_STRUCT_IDX;
use crate::type_converter::wasm_type_converter;

/// Lower closure dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // 1. Collect functions referenced by closure.new operations
    let closure_funcs = collect_closure_func_refs(db, &module);

    if closure_funcs.is_empty() {
        // No closures - just apply patterns without table generation
        return PatternApplicator::new(wasm_type_converter())
            .add_pattern(ClosureNewPattern {
                table_indices: HashMap::new(),
            })
            .add_pattern(ClosureFuncPattern)
            .add_pattern(ClosureEnvPattern)
            .apply(db, module)
            .module;
    }

    // 2. Assign table indices (sorted for deterministic ordering)
    let mut sorted_funcs: Vec<_> = closure_funcs.into_iter().collect();
    sorted_funcs.sort_by(|a, b| a.with_str(|a_str| b.with_str(|b_str| a_str.cmp(b_str))));

    let table_indices: HashMap<Symbol, u32> = sorted_funcs
        .iter()
        .enumerate()
        .map(|(idx, sym)| (*sym, idx as u32))
        .collect();

    let table_size = sorted_funcs.len() as u32;

    // 3. Apply patterns to transform closure operations
    let result = PatternApplicator::new(wasm_type_converter())
        .add_pattern(ClosureNewPattern {
            table_indices: table_indices.clone(),
        })
        .add_pattern(ClosureFuncPattern)
        .add_pattern(ClosureEnvPattern)
        .apply(db, module);

    // 4. Add wasm.table and wasm.elem to the module
    let module = result.module;
    add_closure_table(db, module, &sorted_funcs, table_size)
}

/// Collect all function symbols referenced by closure.new operations.
fn collect_closure_func_refs<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> Vec<Symbol> {
    let mut funcs = Vec::new();
    collect_from_region(db, &module.body(db), &mut funcs);

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    funcs.retain(|sym| seen.insert(*sym));

    funcs
}

fn collect_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    funcs: &mut Vec<Symbol>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check for closure.new
            if let Ok(closure_new) = closure::New::from_operation(db, *op) {
                funcs.push(closure_new.func_ref(db));
            }

            // Recurse into nested regions
            for nested in op.regions(db).iter() {
                collect_from_region(db, nested, funcs);
            }
        }
    }
}

/// Add wasm.table and wasm.elem operations to the module.
fn add_closure_table<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    funcs: &[Symbol],
    table_size: u32,
) -> Module<'db> {
    let location = module.location(db);

    // Create wasm.table for closure functions
    let table_op = wasm::table(
        db,
        location,
        Symbol::new("funcref"),
        table_size,
        Some(table_size),
    );

    // Create func.constant operations for each function in the element segment
    let func_constants: IdVec<Operation<'db>> = funcs
        .iter()
        .map(|func_sym| {
            let funcref_ty = wasm::Funcref::new(db).as_type();
            func::constant(db, location, funcref_ty, *func_sym).as_operation()
        })
        .collect();

    // Create the funcs region for wasm.elem
    let funcs_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), func_constants);
    let funcs_region = Region::new(db, location, IdVec::from(vec![funcs_block]));

    // Create wasm.elem with table 0 and offset 0
    let elem_op = wasm::elem(db, location, Some(0), Some(0), funcs_region);

    // Prepend table and elem operations to the module body
    let body = module.body(db);
    let first_block = &body.blocks(db)[0];
    let mut new_ops: IdVec<Operation<'db>> = IdVec::new();
    new_ops.push(table_op.as_operation());
    new_ops.push(elem_op.as_operation());
    new_ops.extend(first_block.operations(db).iter().copied());

    let new_block = Block::new(
        db,
        first_block.id(db),
        first_block.location(db),
        first_block.args(db).clone(),
        new_ops,
    );

    // Rebuild region with modified first block
    let mut new_blocks: IdVec<Block<'db>> = IdVec::new();
    new_blocks.push(new_block);
    new_blocks.extend(body.blocks(db).iter().skip(1).copied());

    let new_body = Region::new(db, body.location(db), new_blocks);
    Module::create(db, location, module.sym_name(db), new_body)
}

/// Pattern for `closure.new(env) @func_ref` -> `wasm.struct_new`
///
/// Creates a closure struct with:
/// - Field 0: function table index (i32)
/// - Field 1: environment struct
struct ClosureNewPattern {
    table_indices: HashMap<Symbol, u32>,
}

impl<'db> RewritePattern<'db> for ClosureNewPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(closure_new) = closure::New::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let func_ref = closure_new.func_ref(db);
        let env = closure_new.env(db);

        // Get the closure type from the result for the result type
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("closure.new must have exactly one result type");

        // Look up the table index for this function
        let table_idx = self
            .table_indices
            .get(&func_ref)
            .copied()
            .unwrap_or_else(|| panic!("Function {:?} not found in table_indices", func_ref));

        // Create arith.const with the table index
        let i32_ty = trunk_ir::dialect::core::I32::new(db).as_type();
        let idx_const = arith::r#const(db, location, i32_ty, Attribute::IntBits(table_idx as u64));
        let idx_val = idx_const.result(db);

        // Create wasm.struct_new with function table index and environment.
        //
        // Layout: (func_idx: i32, env: anyref)
        //
        // Use CLOSURE_STRUCT_IDX directly since closure structs always have 2 fields.
        let struct_new = wasm::struct_new(
            db,
            location,
            IdVec::from(vec![idx_val, env]),
            result_ty,
            CLOSURE_STRUCT_IDX,
        )
        .as_operation();

        RewriteResult::Expand(vec![idx_const.as_operation(), struct_new])
    }
}

/// Pattern for `closure.func(closure)` -> `wasm.struct_get` field 0
///
/// Extracts the function table index from a closure struct.
struct ClosureFuncPattern;

impl<'db> RewritePattern<'db> for ClosureFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_closure_func) = closure::Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Create wasm.struct_get for field 0 (function table index)
        // Use CLOSURE_STRUCT_IDX directly instead of a placeholder to ensure
        // the struct_get uses the builtin closure struct type (index 4).
        let ref_operand = op.operands(db)[0];
        let result_type = op.results(db)[0];
        let struct_get = wasm::struct_get(
            db,
            location,
            ref_operand,
            result_type,
            CLOSURE_STRUCT_IDX,
            0,
        )
        .as_operation();

        RewriteResult::Replace(struct_get)
    }
}

/// Pattern for `closure.env(closure)` -> `wasm.struct_get` field 1
///
/// Extracts the environment struct from a closure.
struct ClosureEnvPattern;

impl<'db> RewritePattern<'db> for ClosureEnvPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_closure_env) = closure::Env::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Create wasm.struct_get for field 1 (environment)
        // Use CLOSURE_STRUCT_IDX directly instead of a placeholder to ensure
        // the struct_get uses the builtin closure struct type (index 4).
        let ref_operand = op.operands(db)[0];
        let result_type = op.results(db)[0];
        let struct_get = wasm::struct_get(
            db,
            location,
            ref_operand,
            result_type,
            CLOSURE_STRUCT_IDX,
            1,
        )
        .as_operation();

        RewriteResult::Replace(struct_get)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;
    use trunk_ir::{DialectType, Location, PathId, Span, Value, ValueDef, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_closure_new_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, i32_ty).as_type();

        // Create a dummy env value using arith.const
        let env_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let env_val = env_op.result(db);

        // Create closure.new
        let closure_new = closure::new(db, location, env_val, closure_ty, Symbol::new("test_func"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![env_op.as_operation(), closure_new.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_and_check_names(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_closure_new_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_closure_new_module(db);
        let op_names = lower_and_check_names(db, module);

        // Should have wasm.table, wasm.elem, arith.const (for table idx), wasm.struct_new
        assert!(op_names.iter().any(|n| n == "wasm.table"));
        assert!(op_names.iter().any(|n| n == "wasm.elem"));
        assert!(op_names.iter().any(|n| n == "arith.const"));
        assert!(op_names.iter().any(|n| n == "wasm.struct_new"));
        assert!(!op_names.iter().any(|n| n == "closure.new"));
    }

    #[salsa::tracked]
    fn make_closure_func_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, i32_ty).as_type();

        // Create a dummy env value using arith.const
        let env_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let env_val = env_op.result(db);

        // Create closure.new
        let closure_new = closure::new(db, location, env_val, closure_ty, Symbol::new("test_func"));
        let closure_val = Value::new(db, ValueDef::OpResult(closure_new.as_operation()), 0);

        // Create closure.func
        let closure_func = closure::func(db, location, closure_val, i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                env_op.as_operation(),
                closure_new.as_operation(),
                closure_func.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_closure_env_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, i32_ty).as_type();

        // Create a dummy env value using arith.const
        let env_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let env_val = env_op.result(db);

        // Create closure.new
        let closure_new = closure::new(db, location, env_val, closure_ty, Symbol::new("test_func"));
        let closure_val = Value::new(db, ValueDef::OpResult(closure_new.as_operation()), 0);

        // Create closure.env
        let closure_env = closure::env(db, location, closure_val, i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                env_op.as_operation(),
                closure_new.as_operation(),
                closure_env.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    /// Helper to check operation attributes
    #[salsa::tracked]
    fn lower_and_get_field_idx(db: &dyn salsa::Database, module: Module<'_>) -> Option<i64> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        // Find the wasm.struct_get operation
        for op in ops.iter() {
            if op.full_name(db) == "wasm.struct_get"
                && let Some(Attribute::IntBits(idx)) =
                    op.attributes(db).get(&Symbol::new("field_idx"))
            {
                return Some(*idx as i64);
            }
        }
        None
    }

    #[salsa_test]
    fn test_closure_func_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_closure_func_module(db);
        let op_names = lower_and_check_names(db, module);

        // closure.func should become wasm.struct_get
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(!op_names.iter().any(|n| n == "closure.func"));

        // Verify field_idx is 0 (function table index)
        let field_idx = lower_and_get_field_idx(db, module);
        assert_eq!(field_idx, Some(0));
    }

    #[salsa_test]
    fn test_closure_env_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_closure_env_module(db);
        let op_names = lower_and_check_names(db, module);

        // closure.env should become wasm.struct_get
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(!op_names.iter().any(|n| n == "closure.env"));

        // Verify field_idx is 1 (environment)
        let field_idx = lower_and_get_field_idx(db, module);
        assert_eq!(field_idx, Some(1));
    }
}
