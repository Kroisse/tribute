//! Normalize tribute_rt primitive types to core types.
//!
//! This pass converts high-level `tribute_rt` primitive types to their
//! corresponding `core` types early in the WASM pipeline, ensuring that
//! downstream passes and the emit phase don't need to handle tribute_rt types.
//!
//! ## Type Conversions
//!
//! | Source Type       | Target Type   |
//! |-------------------|---------------|
//! | `tribute_rt.int`  | `core.i32`    |
//! | `tribute_rt.nat`  | `core.i32`    |
//! | `tribute_rt.bool` | `core.i32`    |
//! | `tribute_rt.float`| `core.f64`    |
//!
//! ## What this pass normalizes
//!
//! - Function signatures (parameter and return types)
//! - Operation result types
//! - Block argument types
//!
//! The pass uses the existing `wasm_type_converter()` which already defines
//! these conversions and their materializations (boxing/unboxing operations).

use tracing::debug;
use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::{func, wasm};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region, Type};

use crate::type_converter::wasm_type_converter;

/// Normalize tribute_rt primitive types to core types.
///
/// This pass should run early in the WASM pipeline, before `trampoline_to_wasm`.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let type_converter = wasm_type_converter();

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(NormalizeFuncFuncPattern)
        .add_pattern(NormalizeWasmFuncPattern)
        .add_pattern(NormalizeCallPattern)
        .add_pattern(NormalizeCallIndirectPattern)
        .add_pattern(NormalizeOpResultPattern);

    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
}

/// Convert a primitive type to its core equivalent.
/// Returns None if the type doesn't need conversion.
fn convert_primitive_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Type<'db>> {
    // tribute_rt.int, tribute_rt.nat, tribute_rt.bool -> core.i32
    if tribute_rt::Int::from_type(db, ty).is_some()
        || tribute_rt::Nat::from_type(db, ty).is_some()
        || tribute_rt::Bool::from_type(db, ty).is_some()
    {
        return Some(core::I32::new(db).as_type());
    }

    // tribute_rt.float -> core.f64
    if tribute_rt::Float::from_type(db, ty).is_some() {
        return Some(core::F64::new(db).as_type());
    }

    None
}

/// Normalize a function type by converting all primitive types in its signature.
fn normalize_func_type<'db>(db: &'db dyn salsa::Database, func_ty: Type<'db>) -> Option<Type<'db>> {
    let core_func = core::Func::from_type(db, func_ty)?;

    let params = core_func.params(db);
    let result = core_func.result(db);

    let mut changed = false;

    // Normalize parameter types
    let new_params: IdVec<Type<'db>> = params
        .iter()
        .map(|&param_ty| {
            if let Some(converted) = convert_primitive_type(db, param_ty) {
                changed = true;
                converted
            } else {
                param_ty
            }
        })
        .collect();

    // Normalize result type
    let new_result = if let Some(converted) = convert_primitive_type(db, result) {
        changed = true;
        converted
    } else {
        result
    };

    if changed {
        Some(core::Func::new(db, new_params, new_result).as_type())
    } else {
        None
    }
}

// ============================================================================
// Patterns
// ============================================================================

/// Normalize func.func function signatures.
struct NormalizeFuncFuncPattern;

impl<'db> RewritePattern<'db> for NormalizeFuncFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let func_op = func::Func::from_operation(db, *op).ok();
        let func_op = match func_op {
            Some(f) => f,
            None => return RewriteResult::Unchanged,
        };

        let func_ty = func_op.r#type(db);
        let new_func_ty = match normalize_func_type(db, func_ty) {
            Some(ty) => ty,
            None => return RewriteResult::Unchanged,
        };

        debug!(
            "normalize_primitive_types: func.func {} signature normalized",
            func_op.sym_name(db)
        );

        // Also normalize block arguments in the body region
        let body = func_op.body(db);
        let new_body = normalize_region_block_args(db, body);

        let new_op = func::func(
            db,
            op.location(db),
            func_op.sym_name(db),
            new_func_ty,
            new_body,
        );

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Normalize wasm.func function signatures.
struct NormalizeWasmFuncPattern;

impl<'db> RewritePattern<'db> for NormalizeWasmFuncPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let func_op = wasm::Func::from_operation(db, *op).ok();
        let func_op = match func_op {
            Some(f) => f,
            None => return RewriteResult::Unchanged,
        };

        let func_ty = func_op.r#type(db);
        let new_func_ty = match normalize_func_type(db, func_ty) {
            Some(ty) => ty,
            None => return RewriteResult::Unchanged,
        };

        debug!(
            "normalize_primitive_types: wasm.func {} signature normalized",
            func_op.sym_name(db)
        );

        // Also normalize block arguments in the body region
        let body = func_op.body(db);
        let new_body = normalize_region_block_args(db, body);

        let new_op = wasm::func(
            db,
            op.location(db),
            func_op.sym_name(db),
            new_func_ty,
            new_body,
        );

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Normalize func.call operation result types.
struct NormalizeCallPattern;

impl<'db> RewritePattern<'db> for NormalizeCallPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let call_op = func::Call::from_operation(db, *op).ok();
        let call_op = match call_op {
            Some(c) => c,
            None => return RewriteResult::Unchanged,
        };

        let results = op.results(db);
        if results.is_empty() {
            return RewriteResult::Unchanged;
        }

        let result_ty = results[0];
        let new_result_ty = match convert_primitive_type(db, result_ty) {
            Some(ty) => ty,
            None => return RewriteResult::Unchanged,
        };

        debug!(
            "normalize_primitive_types: func.call {} result type normalized from {}.{} to {}.{}",
            call_op.callee(db),
            result_ty.dialect(db),
            result_ty.name(db),
            new_result_ty.dialect(db),
            new_result_ty.name(db)
        );

        let new_op = func::call(
            db,
            op.location(db),
            adaptor.operands().clone(),
            new_result_ty,
            call_op.callee(db),
        );

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Normalize func.call_indirect operation result types.
struct NormalizeCallIndirectPattern;

impl<'db> RewritePattern<'db> for NormalizeCallIndirectPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let _call_op = func::CallIndirect::from_operation(db, *op).ok();
        let _call_op = match _call_op {
            Some(c) => c,
            None => return RewriteResult::Unchanged,
        };

        let results = op.results(db);
        if results.is_empty() {
            return RewriteResult::Unchanged;
        }

        let result_ty = results[0];
        let new_result_ty = match convert_primitive_type(db, result_ty) {
            Some(ty) => ty,
            None => return RewriteResult::Unchanged,
        };

        debug!(
            "normalize_primitive_types: func.call_indirect result type normalized from {}.{} to {}.{}",
            result_ty.dialect(db),
            result_ty.name(db),
            new_result_ty.dialect(db),
            new_result_ty.name(db)
        );

        let operands = adaptor.operands();
        let callee = operands[0];
        let args: IdVec<_> = operands.iter().skip(1).copied().collect();

        let new_op = func::call_indirect(db, op.location(db), callee, args, new_result_ty);

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Normalize general operation result types.
///
/// This is a catch-all pattern for operations with primitive result types
/// that weren't handled by more specific patterns.
struct NormalizeOpResultPattern;

impl<'db> RewritePattern<'db> for NormalizeOpResultPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let results = op.results(db);
        if results.is_empty() {
            return RewriteResult::Unchanged;
        }

        // Check if any result types need normalization
        let mut any_changed = false;
        let new_results: IdVec<Type<'db>> = results
            .iter()
            .map(|&result_ty| {
                if let Some(converted) = convert_primitive_type(db, result_ty) {
                    any_changed = true;
                    converted
                } else {
                    result_ty
                }
            })
            .collect();

        if !any_changed {
            return RewriteResult::Unchanged;
        }

        // Skip operations already handled by other patterns
        let dialect = op.dialect(db);
        let name = op.name(db);
        if (dialect == func::DIALECT_NAME() && name == func::FUNC())
            || (dialect == wasm::DIALECT_NAME() && name == wasm::FUNC())
            || (dialect == func::DIALECT_NAME() && name == func::CALL())
            || (dialect == func::DIALECT_NAME() && name == func::CALL_INDIRECT())
        {
            return RewriteResult::Unchanged;
        }

        debug!(
            "normalize_primitive_types: {}.{} result type normalized",
            dialect, name
        );

        let new_op = op.modify(db).results(new_results).build();

        RewriteResult::Replace(new_op)
    }
}

/// Normalize block argument types in a region.
fn normalize_region_block_args<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
) -> Region<'db> {
    let blocks = region.blocks(db);
    let mut any_changed = false;

    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| {
            let args = block.args(db);
            let mut args_changed = false;

            let new_args: IdVec<BlockArg<'db>> = args
                .iter()
                .map(|arg| {
                    let arg_ty = arg.ty(db);
                    if let Some(converted) = convert_primitive_type(db, arg_ty) {
                        args_changed = true;
                        BlockArg::new(db, converted, arg.attrs(db).clone())
                    } else {
                        *arg
                    }
                })
                .collect();

            if args_changed {
                any_changed = true;
                // Recursively normalize nested regions in operations
                let new_ops: IdVec<Operation<'db>> = block
                    .operations(db)
                    .iter()
                    .map(|op| {
                        if op.regions(db).is_empty() {
                            *op
                        } else {
                            let new_regions: IdVec<Region<'db>> = op
                                .regions(db)
                                .iter()
                                .map(|r| normalize_region_block_args(db, *r))
                                .collect();
                            op.modify(db).regions(new_regions).build()
                        }
                    })
                    .collect();

                Block::new(db, block.id(db), block.location(db), new_args, new_ops)
            } else {
                *block
            }
        })
        .collect();

    if any_changed {
        Region::new(db, region.location(db), new_blocks)
    } else {
        region
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{BlockId, Location, PathId, Span, Symbol, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_and_lower_module_with_int_func(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        // Create a function: fn test() -> tribute_rt.int
        let int_ty = tribute_rt::Int::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], int_ty).as_type();

        let body_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
        let body = Region::new(db, location, idvec![body_block]);
        let func_op = func::func(db, location, Symbol::new("test"), func_ty, body);

        let module_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_op.as_operation()],
        );
        let module_body = Region::new(db, location, idvec![module_block]);

        let module = Module::create(db, location, Symbol::new("test_module"), module_body);
        lower(db, module)
    }

    #[salsa_test]
    fn test_normalize_func_return_type(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_module_with_int_func(db);

        // Find the function and check its return type
        let body = lowered.body(db);
        let block = &body.blocks(db)[0];
        let op = &block.operations(db)[0];

        let func_op = func::Func::from_operation(db, *op).expect("should be func.func");
        let func_ty = func_op.r#type(db);
        let core_func = core::Func::from_type(db, func_ty).expect("should be core.func type");

        // Result type should now be core.i32
        let result_ty = core_func.result(db);
        assert!(
            core::I32::from_type(db, result_ty).is_some(),
            "Expected core.i32, got {}.{}",
            result_ty.dialect(db),
            result_ty.name(db)
        );
    }

    #[salsa::tracked]
    fn make_and_lower_module_with_int_param(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);

        // Create a function: fn test(x: tribute_rt.int) -> core.nil
        let int_ty = tribute_rt::Int::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![int_ty], nil_ty).as_type();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![BlockArg::of_type(db, int_ty)],
            idvec![],
        );
        let body = Region::new(db, location, idvec![body_block]);
        let func_op = func::func(db, location, Symbol::new("test"), func_ty, body);

        let module_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_op.as_operation()],
        );
        let module_body = Region::new(db, location, idvec![module_block]);

        let module = Module::create(db, location, Symbol::new("test_module"), module_body);
        lower(db, module)
    }

    #[salsa_test]
    fn test_normalize_func_param_type(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_module_with_int_param(db);

        // Find the function and check its param type
        let body = lowered.body(db);
        let block = &body.blocks(db)[0];
        let op = &block.operations(db)[0];

        let func_op = func::Func::from_operation(db, *op).expect("should be func.func");
        let func_ty = func_op.r#type(db);
        let core_func = core::Func::from_type(db, func_ty).expect("should be core.func type");

        // Param type should now be core.i32
        let param_ty = core_func.params(db)[0];
        assert!(
            core::I32::from_type(db, param_ty).is_some(),
            "Expected core.i32, got {}.{}",
            param_ty.dialect(db),
            param_ty.name(db)
        );

        // Block arg should also be normalized
        let func_body = func_op.body(db);
        let first_block = &func_body.blocks(db)[0];
        let block_arg_ty = first_block.args(db)[0].ty(db);
        assert!(
            core::I32::from_type(db, block_arg_ty).is_some(),
            "Block arg should be core.i32, got {}.{}",
            block_arg_ty.dialect(db),
            block_arg_ty.name(db)
        );
    }
}
