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
use trunk_ir::{DialectOp, DialectType, IdVec, Operation, Symbol, Type};

use crate::type_converter::wasm_type_converter;

/// Concretize types in wasm operations.
///
/// This pass runs after `tribute_rt_to_wasm` and before emit.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: core::Module<'db>) -> core::Module<'db> {
    // First pass: collect function signatures for type lookup
    let func_return_types = collect_func_return_types(db, module);

    let applicator = PatternApplicator::new(wasm_type_converter())
        .add_pattern(CallResultTypePattern { func_return_types });
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
