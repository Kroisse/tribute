//! Lower func dialect operations to wasm dialect.
//!
//! This pass converts function-level operations to wasm operations:
//! - `func.func` -> `wasm.func`
//! - `func.call` -> `wasm.call`
//! - `func.call_indirect` -> `wasm.call_indirect`
//! - `func.return` -> `wasm.return`
//! - `func.tail_call` -> `wasm.return_call`
//! - `func.unreachable` -> `wasm.unreachable`
//! - `func.constant` -> `wasm.i32_const` (function table index)
//!
//! For closures, this pass also:
//! - Collects all functions referenced by `func.constant`
//! - Creates a function table with those functions
//! - Generates `wasm.table` and `wasm.elem` operations

use std::collections::HashMap;

use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::{func, wasm};
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Symbol,
};

use crate::type_converter::wasm_type_converter;

/// Lower func dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // 1. Collect all functions referenced by func.constant operations
    let func_refs = collect_func_constant_refs(db, &module);

    if func_refs.is_empty() {
        // No func.constant operations - just apply patterns without table generation
        return PatternApplicator::new(wasm_type_converter())
            .add_pattern(FuncFuncPattern)
            .add_pattern(FuncCallPattern)
            .add_pattern(FuncCallIndirectPattern)
            .add_pattern(FuncReturnPattern)
            .add_pattern(FuncTailCallPattern)
            .add_pattern(FuncUnreachablePattern)
            .add_pattern(FuncConstantPattern {
                table_indices: HashMap::new(),
            })
            .apply(db, module)
            .module;
    }

    // 2. Assign table indices (sorted for deterministic ordering)
    let mut sorted_funcs: Vec<_> = func_refs.into_iter().collect();
    sorted_funcs.sort_by(|a, b| a.with_str(|a_str| b.with_str(|b_str| a_str.cmp(b_str))));

    let table_indices: HashMap<Symbol, u32> = sorted_funcs
        .iter()
        .enumerate()
        .map(|(idx, sym)| (*sym, idx as u32))
        .collect();

    let table_size = sorted_funcs.len() as u32;

    // 3. Apply patterns to transform operations
    let result = PatternApplicator::new(wasm_type_converter())
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern {
            table_indices: table_indices.clone(),
        })
        .apply(db, module);

    // 4. Add wasm.table and wasm.elem to the module
    add_function_table(db, result.module, &sorted_funcs, table_size)
}

/// Collect all function symbols referenced by func.constant operations.
fn collect_func_constant_refs<'db>(
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
            // Check for func.constant
            if let Ok(const_op) = func::Constant::from_operation(db, *op) {
                funcs.push(const_op.func_ref(db));
            }

            // Recurse into nested regions
            for nested in op.regions(db).iter() {
                collect_from_region(db, nested, funcs);
            }
        }
    }
}

/// Add wasm.table and wasm.elem operations to the module for the function table.
fn add_function_table<'db>(
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

    // Create wasm.ref_func operations for each function in the element segment
    let func_refs: IdVec<Operation<'db>> = funcs
        .iter()
        .map(|func_sym| {
            let funcref_ty = wasm::Funcref::new(db).as_type();
            wasm::ref_func(db, location, funcref_ty, *func_sym).as_operation()
        })
        .collect();

    // Create the funcs region for wasm.elem
    let funcs_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), func_refs);
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

    // Reconstruct module with other blocks unchanged
    let mut new_blocks: IdVec<Block<'db>> = IdVec::new();
    new_blocks.push(new_block);
    for block in body.blocks(db).iter().skip(1) {
        new_blocks.push(*block);
    }

    let new_body = Region::new(db, body.location(db), new_blocks);
    Module::create(db, location, module.sym_name(db), new_body)
}

/// Pattern for `func.func` -> `wasm.func`
struct FuncFuncPattern;

impl<'db> RewritePattern<'db> for FuncFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_func_op) = func::Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // wasm.func has the same structure: sym_name, type attributes, body region
        // PatternApplicator will recursively process the body region
        let new_op = op.modify(db).dialect_str("wasm").name_str("func").build();

        // Debug: verify type attribute is preserved
        if let Some(Attribute::Type(ty)) = new_op.attributes(db).get(&trunk_ir::Symbol::new("type"))
            && let Some(fn_ty) = trunk_ir::dialect::core::Func::from_type(db, *ty)
        {
            tracing::debug!(
                "FuncFuncPattern: {} -> wasm.func with params={:?}, result={}.{}",
                _func_op.sym_name(db),
                fn_ty
                    .params(db)
                    .iter()
                    .map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
                    .collect::<Vec<_>>(),
                fn_ty.result(db).dialect(db),
                fn_ty.result(db).name(db)
            );
        }

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.call` -> `wasm.call`
struct FuncCallPattern;

impl<'db> RewritePattern<'db> for FuncCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(call_op) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Build wasm.call with same callee and operands
        // Note: we use modify() but need to update the callee attribute format
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("call")
            .attr("callee", Attribute::Symbol(call_op.callee(db)))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.call_indirect` -> `wasm.call_indirect`
///
/// Transforms indirect function calls for closures.
/// The callee (funcref) is the first operand, followed by arguments.
struct FuncCallIndirectPattern;

impl<'db> RewritePattern<'db> for FuncCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_call_indirect) = func::CallIndirect::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Build wasm.call_indirect with same operands
        // The emit phase will resolve the type_idx and table attributes
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("call_indirect")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.return` -> `wasm.return`
struct FuncReturnPattern;

impl<'db> RewritePattern<'db> for FuncReturnPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_return_op) = func::Return::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op.modify(db).dialect_str("wasm").name_str("return").build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.tail_call` -> `wasm.return_call`
struct FuncTailCallPattern;

impl<'db> RewritePattern<'db> for FuncTailCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(tail_call_op) = func::TailCall::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Build wasm.return_call with same callee and operands
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("return_call")
            .attr("callee", Attribute::Symbol(tail_call_op.callee(db)))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.unreachable` -> `wasm.unreachable`
struct FuncUnreachablePattern;

impl<'db> RewritePattern<'db> for FuncUnreachablePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_unreachable_op) = func::Unreachable::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("unreachable")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.constant` -> `arith.const` (i32 table index)
///
/// Transforms function constant references to i32 table indices.
/// Used for closures where lifted functions are stored via function table.
struct FuncConstantPattern {
    table_indices: HashMap<Symbol, u32>,
}

impl<'db> RewritePattern<'db> for FuncConstantPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(const_op) = func::Constant::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let func_ref = const_op.func_ref(db);

        // Look up the table index for this function
        let Some(&table_idx) = self.table_indices.get(&func_ref) else {
            // If not in table_indices, this is a func.constant not used in closures
            // (e.g., in element segment generation). Keep as-is or convert to ref_func.
            // For element segments we still need ref_func.
            let funcref_ty = wasm::Funcref::new(db).as_type();
            let new_op = op
                .modify(db)
                .dialect_str("wasm")
                .name_str("ref_func")
                .attr("func_name", Attribute::Symbol(func_ref))
                .results(IdVec::from(vec![funcref_ty]))
                .build();
            return RewriteResult::Replace(new_op);
        };

        // Transform to wasm.i32_const with table index
        let i32_ty = core::I32::new(db).as_type();
        let new_op = wasm::i32_const(db, op.location(db), i32_ty, table_idx as i32);

        RewriteResult::Replace(new_op.as_operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core};
    use trunk_ir::{
        Attribute, Block, BlockId, DialectType, Location, PathId, Region, Span, Symbol, idvec,
    };

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_func_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a simple func.call
        let func_call = func::call(db, location, vec![], i32_ty, Symbol::new("foo"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_call.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_func_func_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create func.return inside func.func body
        let func_return = func::r#return(db, location, vec![]);

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_return.as_operation()],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Create func.func using typed helper
        let func_func =
            func::func(db, location, Symbol::new("test_fn"), func_ty, body_region).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_func]);
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

    #[salsa::tracked]
    fn lower_and_check_nested(db: &dyn salsa::Database, module: Module<'_>) -> (String, String) {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let func_op = &body.blocks(db)[0].operations(db)[0];
        let func_name = func_op.full_name(db);

        // Get the operation inside the function body
        let func_body = func_op.regions(db)[0].blocks(db)[0].operations(db);
        let inner_name = func_body[0].full_name(db);

        (func_name, inner_name)
    }

    #[salsa_test]
    fn test_func_call_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_func_call_module(db);
        let op_names = lower_and_check_names(db, module);

        assert!(op_names.iter().any(|n| n == "wasm.call"));
        assert!(!op_names.iter().any(|n| n == "func.call"));
    }

    #[salsa_test]
    fn test_func_func_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_func_func_module(db);
        let (func_name, inner_name) = lower_and_check_nested(db, module);

        // func.func should become wasm.func
        assert_eq!(func_name, "wasm.func");
        // func.return inside should become wasm.return
        assert_eq!(inner_name, "wasm.return");
    }

    #[salsa::tracked]
    fn make_call_indirect_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create dummy values using arith.const
        let callee_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let callee_val = callee_op.result(db);

        let arg_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let arg_val = arg_op.result(db);

        // Create func.call_indirect
        let call_indirect = func::call_indirect(db, location, callee_val, vec![arg_val], i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                callee_op.as_operation(),
                arg_op.as_operation(),
                call_indirect.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_call_indirect_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_call_indirect_module(db);
        let op_names = lower_and_check_names(db, module);

        // func.call_indirect should become wasm.call_indirect
        assert!(op_names.iter().any(|n| n == "wasm.call_indirect"));
        assert!(!op_names.iter().any(|n| n == "func.call_indirect"));
    }

    #[salsa::tracked]
    fn make_func_constant_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let func_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();

        // Create func.constant
        let func_constant = func::constant(db, location, func_ty, Symbol::new("test_func"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_constant.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_constant_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_func_constant_module(db);
        let op_names = lower_and_check_names(db, module);

        // func.constant should become wasm.i32_const (table index)
        // wasm.table and wasm.elem should be added for function table
        assert!(
            op_names.iter().any(|n| n == "wasm.table"),
            "expected wasm.table"
        );
        assert!(
            op_names.iter().any(|n| n == "wasm.elem"),
            "expected wasm.elem"
        );
        assert!(
            op_names.iter().any(|n| n == "wasm.i32_const"),
            "expected wasm.i32_const"
        );
        assert!(
            !op_names.iter().any(|n| n == "func.constant"),
            "func.constant should be lowered"
        );
    }
}
