//! Concretize types in wasm dialect operations.
//!
//! This pass resolves placeholder types (`tribute.type_var`) to concrete types
//! before the emit phase. This allows emit to be a simple 1:1 translation
//! without runtime type inference.
//!
//! ## What this pass does
//!
//! 1. **Replace `type_var` in operation results** with concrete types:
//!    - `wasm.call`: Use callee's return type from function signature
//!    - `wasm.call_indirect`: Use enclosing function's return type as hint
//!    - `wasm.if`/`wasm.block`/`wasm.loop`: Infer from branch result types
//!
//! 2. **Insert explicit boxing operations** at polymorphic call sites:
//!    - When a concrete type (Int, Float) is passed to a generic parameter
//!    - Replace with: `tribute_rt.box_int(%value)` before the call
//!
//! 3. **Insert explicit unboxing operations** for generic return values:
//!    - When a generic function returns `anyref` but concrete type is expected
//!    - Replace with: `tribute_rt.unbox_int(%result)` after the call

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::dialect::tribute;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value, ValueDef};

use crate::type_converter::wasm_type_converter;

/// Concretize types in wasm operations.
///
/// This pass runs after `tribute_rt_to_wasm` and before emit.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: core::Module<'db>) -> core::Module<'db> {
    // First pass: collect function signatures for type lookup
    let func_return_types = collect_func_return_types(db, module);

    let applicator = PatternApplicator::new(wasm_type_converter())
        .add_pattern(CallResultTypePattern {
            func_return_types: func_return_types.clone(),
        })
        .add_pattern(CallIndirectResultTypePattern {
            func_return_types: func_return_types.clone(),
        })
        .add_pattern(IfResultTypePattern)
        .add_pattern(BlockResultTypePattern)
        .add_pattern(LoopResultTypePattern);
    applicator.apply(db, module).module
}

/// Collect function return types from the module.
fn collect_func_return_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> HashMap<Symbol, Type<'db>> {
    let mut func_return_types = HashMap::new();

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check for wasm.func operations
            if let Ok(func_op) = wasm::Func::from_operation(db, *op) {
                let sym_name = func_op.sym_name(db);
                let func_ty = func_op.r#type(db);

                // Extract return type from function type
                if let Some(func) = core::Func::from_type(db, func_ty) {
                    let return_ty = func.result(db);
                    func_return_types.insert(sym_name, return_ty);
                }
            }
        }
    }

    func_return_types
}

/// Pattern to concretize result types of wasm.call operations.
///
/// If a call's result type is `tribute.type_var`, replace it with
/// the callee's declared return type.
struct CallResultTypePattern<'db> {
    /// Map of function name -> return type.
    func_return_types: HashMap<Symbol, Type<'db>>,
}

impl<'db> RewritePattern<'db> for CallResultTypePattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.call operations
        let Ok(call_op) = wasm::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get the result type
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        // Only process if result type is a type variable
        if !tribute::is_type_var(db, result_ty) {
            return RewriteResult::Unchanged;
        }

        // Look up the callee's return type
        let callee = call_op.callee(db);
        let Some(&return_ty) = self.func_return_types.get(&callee) else {
            debug!(
                "wasm_type_concrete: callee {} not found in func_return_types",
                callee
            );
            return RewriteResult::Unchanged;
        };

        // If callee also returns type_var, we can't concretize further
        if tribute::is_type_var(db, return_ty) {
            debug!(
                "wasm_type_concrete: callee {} returns type_var, cannot concretize",
                callee
            );
            return RewriteResult::Unchanged;
        }

        debug!(
            "wasm_type_concrete: concretizing wasm.call {} result from type_var to {}.{}",
            callee,
            return_ty.dialect(db),
            return_ty.name(db)
        );

        // Create a new call operation with the concrete result type
        let new_op = op.modify(db).results(IdVec::from(vec![return_ty])).build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.call_indirect operations.
///
/// If a call_indirect's result type is `tribute.type_var`, try to infer it from:
/// 1. The callee's function type (if it's a known funcref)
/// 2. For `wasm.ref_func` callees, look up the referenced function's return type
struct CallIndirectResultTypePattern<'db> {
    /// Map of function name -> return type.
    func_return_types: HashMap<Symbol, Type<'db>>,
}

impl<'db> RewritePattern<'db> for CallIndirectResultTypePattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.call_indirect operations
        if !wasm::CallIndirect::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Get the result type
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        // Only process if result type is a type variable
        if !tribute::is_type_var(db, result_ty) {
            return RewriteResult::Unchanged;
        }

        // The callee is the last operand (funcref)
        let operands = op.operands(db);
        let Some(&callee_val) = operands.last() else {
            return RewriteResult::Unchanged;
        };

        // Try to infer type from callee
        if let Some(concrete_ty) = infer_type_from_callee(db, callee_val, &self.func_return_types)
            && !tribute::is_type_var(db, concrete_ty)
        {
            debug!(
                "wasm_type_concrete: concretizing wasm.call_indirect result from type_var to {}.{}",
                concrete_ty.dialect(db),
                concrete_ty.name(db)
            );

            let new_op = op
                .modify(db)
                .results(IdVec::from(vec![concrete_ty]))
                .build();

            return RewriteResult::Replace(new_op);
        }

        RewriteResult::Unchanged
    }
}

/// Try to infer the return type from a callee value.
///
/// Handles cases like:
/// - wasm.ref_func: look up the referenced function's return type
/// - Values with core.func type: extract return type from the type
fn infer_type_from_callee<'db>(
    db: &'db dyn salsa::Database,
    callee: Value<'db>,
    func_return_types: &HashMap<Symbol, Type<'db>>,
) -> Option<Type<'db>> {
    match callee.def(db) {
        ValueDef::OpResult(def_op) => {
            // Check if it's a wasm.ref_func operation
            if let Ok(ref_func) = wasm::RefFunc::from_operation(db, def_op) {
                let func_name = ref_func.func_name(db);
                if let Some(&return_ty) = func_return_types.get(&func_name) {
                    return Some(return_ty);
                }
            }

            // Try to get type from the operation's result
            let index = callee.index(db);
            if let Some(callee_ty) = def_op.results(db).get(index).copied() {
                // If it's a function type, extract return type
                if let Some(func_ty) = core::Func::from_type(db, callee_ty) {
                    return Some(func_ty.result(db));
                }
            }

            None
        }
        ValueDef::BlockArg(_) => None,
    }
}

/// Pattern to concretize result types of wasm.if operations.
///
/// If an if's result type is `tribute.type_var`, try to infer it from
/// the yield operations in its then/else branches.
struct IfResultTypePattern;

impl<'db> RewritePattern<'db> for IfResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.if operations
        if !wasm::If::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Get the result type
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        // Only process if result type is a type variable
        if !tribute::is_type_var(db, result_ty) {
            return RewriteResult::Unchanged;
        }

        // Try to infer concrete type from regions
        let regions = op.regions(db);
        let inferred = infer_type_from_regions(db, regions);

        let Some(concrete_ty) = inferred else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.if result from type_var to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![concrete_ty]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.block operations.
struct BlockResultTypePattern;

impl<'db> RewritePattern<'db> for BlockResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.block operations
        if !wasm::Block::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Get the result type
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        // Only process if result type is a type variable
        if !tribute::is_type_var(db, result_ty) {
            return RewriteResult::Unchanged;
        }

        // Try to infer concrete type from the body region
        let regions = op.regions(db);
        let inferred = infer_type_from_regions(db, regions);

        let Some(concrete_ty) = inferred else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.block result from type_var to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![concrete_ty]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern to concretize result types of wasm.loop operations.
struct LoopResultTypePattern;

impl<'db> RewritePattern<'db> for LoopResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle wasm.loop operations
        if !wasm::Loop::matches(db, *op) {
            return RewriteResult::Unchanged;
        }

        // Get the result type
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };

        // Only process if result type is a type variable
        if !tribute::is_type_var(db, result_ty) {
            return RewriteResult::Unchanged;
        }

        // Try to infer concrete type from the body region
        let regions = op.regions(db);
        let inferred = infer_type_from_regions(db, regions);

        let Some(concrete_ty) = inferred else {
            return RewriteResult::Unchanged;
        };

        debug!(
            "wasm_type_concrete: concretizing wasm.loop result from type_var to {}.{}",
            concrete_ty.dialect(db),
            concrete_ty.name(db)
        );

        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![concrete_ty]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Try to infer a concrete type from regions by looking at yield operations.
fn infer_type_from_regions<'db>(
    db: &'db dyn salsa::Database,
    regions: &IdVec<Region<'db>>,
) -> Option<Type<'db>> {
    for region in regions.iter() {
        if let Some(ty) = infer_type_from_region(db, region) {
            return Some(ty);
        }
    }
    None
}

/// Try to infer a concrete type from a region's yield operations.
fn infer_type_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Type<'db>> {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Look for wasm.yield operations
            if let Ok(yield_op) = wasm::Yield::from_operation(db, *op) {
                let yielded_value = yield_op.value(db);
                if let Some(ty) = get_value_type(db, yielded_value)
                    && !tribute::is_type_var(db, ty)
                {
                    return Some(ty);
                }
            }
        }
    }
    None
}

/// Get the type of a value from its definition.
fn get_value_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(def_op) => {
            let index = value.index(db);
            def_op.results(db).get(index).copied()
        }
        ValueDef::BlockArg(_block_id) => {
            // For block arguments, we'd need the block's argument types
            // which we don't easily have access to here.
            // Return None for now - these cases are less common.
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_and_lower_module_with_type_var_call(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let type_var = tribute::type_var_with_id(db, 0);

        // Create a function that returns i32
        let func_ty = core::Func::new(db, idvec![], i32_ty).as_type();
        let func_body_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
        let func_body = Region::new(db, location, idvec![func_body_block]);
        let func_op = wasm::func(db, location, Symbol::new("callee"), func_ty, func_body);

        // Create a call to that function with type_var result
        let call_op = wasm::call(db, location, None, Some(type_var), Symbol::new("callee"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_op.as_operation(), call_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = core::Module::create(db, location, "test".into(), region);

        // Lower the module within the tracked function
        lower(db, module)
    }

    #[salsa_test]
    fn test_call_result_type_concretization(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_module_with_type_var_call(db);

        // Find the call operation
        let block = lowered.body(db).blocks(db).first().unwrap();
        let call_op = block
            .operations(db)
            .iter()
            .find(|op| op.dialect(db) == wasm::DIALECT_NAME() && op.name(db) == wasm::CALL())
            .expect("call operation not found");

        // Check that result type is now i32, not type_var
        let result_ty = call_op.results(db).first().copied().unwrap();
        assert!(
            !tribute::is_type_var(db, result_ty),
            "result type should be concrete, not type_var"
        );
        assert!(
            core::I32::from_type(db, result_ty).is_some(),
            "result type should be i32"
        );
    }
}
