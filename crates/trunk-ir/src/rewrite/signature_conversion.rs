//! Function signature conversion patterns for dialect lowering.
//!
//! This module provides MLIR-style signature conversion patterns that
//! automatically convert function signature types using a TypeConverter.
//!
//! # Overview
//!
//! During dialect lowering, function signatures often need type conversion:
//! - Parameter types need to be converted to target dialect types
//! - Return types need to be converted
//! - Entry block arguments must match the converted parameter types
//!
//! The patterns in this module handle all of this automatically:
//!
//! - [`FuncSignatureConversionPattern`]: Converts `func.func` signatures
//!
//! # Usage
//!
//! ```no_run
//! use trunk_ir::rewrite::{
//!     FuncSignatureConversionPattern, PatternApplicator, TypeConverter, ConversionTarget
//! };
//! use trunk_ir::dialect::core;
//!
//! let converter = TypeConverter::new()
//!     .add_conversion(|db, ty| {
//!         // Convert some_type to target_type
//!         None
//!     });
//!
//! let applicator = PatternApplicator::new(converter)
//!     .add_pattern(FuncSignatureConversionPattern);
//! ```

use crate::dialect::{core, func, wasm};
use crate::{Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region};

use super::rewriter::PatternRewriter;
use super::{RewritePattern, TypeConverter};

/// Helper to convert function signature types.
///
/// Returns (new_params, new_result, params_changed, result_changed)
fn convert_func_signature<'db>(
    db: &'db dyn salsa::Database,
    func_ty: &core::Func<'db>,
    converter: &TypeConverter,
) -> (IdVec<crate::Type<'db>>, crate::Type<'db>, bool, bool) {
    let old_params = func_ty.params(db);
    let old_result = func_ty.result(db);

    let new_params: IdVec<_> = old_params
        .iter()
        .map(|&ty| converter.convert_type(db, ty).unwrap_or(ty))
        .collect();

    let new_result = converter.convert_type(db, old_result).unwrap_or(old_result);

    let params_changed = new_params
        .iter()
        .zip(old_params.iter())
        .any(|(new, old)| new != old);
    let result_changed = new_result != old_result;

    (new_params, new_result, params_changed, result_changed)
}

/// Helper to rebuild entry block with converted argument types.
fn rebuild_entry_block<'db>(
    db: &'db dyn salsa::Database,
    body: Region<'db>,
    new_params: &IdVec<crate::Type<'db>>,
) -> Region<'db> {
    let blocks = body.blocks(db);

    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .enumerate()
        .map(|(block_idx, block)| {
            if block_idx == 0 {
                // Entry block: update argument types
                let args = block.args(db);
                let new_args: IdVec<BlockArg<'db>> = args
                    .iter()
                    .enumerate()
                    .map(|(i, arg)| {
                        if i < new_params.len() && new_params[i] != arg.ty(db) {
                            BlockArg::new(db, new_params[i], arg.attrs(db).clone())
                        } else {
                            *arg
                        }
                    })
                    .collect();

                Block::new(
                    db,
                    block.id(db),
                    block.location(db),
                    new_args,
                    block.operations(db).clone(),
                )
            } else {
                *block
            }
        })
        .collect();

    Region::new(db, body.location(db), new_blocks)
}

/// Pattern that converts `func.func` operation signatures using a TypeConverter.
///
/// This pattern:
/// 1. Matches `func.func` operations
/// 2. Converts parameter and result types using the `PatternApplicator`'s TypeConverter
/// 3. Updates the entry block argument types to match
/// 4. Rebuilds the function with the converted signature
///
/// This is similar to MLIR's `populateFunctionOpInterfaceTypeConversionPattern`.
pub struct FuncSignatureConversionPattern;

impl<'db> RewritePattern<'db> for FuncSignatureConversionPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_operation(db, *op) else {
            return false;
        };

        let func_type = func_op.r#type(db);
        let Some(func_ty) = core::Func::from_type(db, func_type) else {
            return false;
        };

        // Convert parameter and result types using the applicator's TypeConverter
        let converter = rewriter.type_converter();
        let (new_params, new_result, params_changed, result_changed) =
            convert_func_signature(db, &func_ty, converter);

        if !params_changed && !result_changed {
            return false;
        }

        // Rebuild function type
        let effect = func_ty.effect(db);
        let new_func_type = core::Func::with_effect(db, new_params.clone(), new_result, effect);

        // Rebuild entry block with converted argument types
        let body = func_op.body(db);
        let new_body = rebuild_entry_block(db, body, &new_params);

        // Rebuild operation with new type and body
        let new_op = op
            .modify(db)
            .attr("type", Attribute::Type(new_func_type.as_type()))
            .regions(vec![new_body].into_iter().collect())
            .build();

        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "FuncSignatureConversionPattern"
    }
}

/// Pattern that converts `wasm.func` operation signatures using a TypeConverter.
///
/// This pattern is similar to [`FuncSignatureConversionPattern`] but targets
/// `wasm.func` operations which are the lowered form of `func.func` in the
/// WASM backend.
pub struct WasmFuncSignatureConversionPattern;

impl<'db> RewritePattern<'db> for WasmFuncSignatureConversionPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(wasm_func_op) = wasm::Func::from_operation(db, *op) else {
            return false;
        };

        let func_type = wasm_func_op.r#type(db);
        let Some(func_ty) = core::Func::from_type(db, func_type) else {
            return false;
        };

        // Convert parameter and result types using the applicator's TypeConverter
        let converter = rewriter.type_converter();
        let (new_params, new_result, params_changed, result_changed) =
            convert_func_signature(db, &func_ty, converter);

        if !params_changed && !result_changed {
            return false;
        }

        // Rebuild function type
        let effect = func_ty.effect(db);
        let new_func_type = core::Func::with_effect(db, new_params.clone(), new_result, effect);

        // Rebuild entry block with converted argument types
        let body = wasm_func_op.body(db);
        let new_body = rebuild_entry_block(db, body, &new_params);

        // Rebuild operation with new type and body
        let new_op = op
            .modify(db)
            .attr("type", Attribute::Type(new_func_type.as_type()))
            .regions(vec![new_body].into_iter().collect())
            .build();

        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "WasmFuncSignatureConversionPattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::core;
    use crate::parser::parse_test_module;
    use crate::rewrite::{ConversionTarget, PatternApplicator};
    use salsa_test_macros::salsa_test;

    /// Create a test module with a function that has i32 params and result.
    #[salsa::tracked]
    fn make_i32_func_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test {
  func.func @test_fn(%arg0: core.i32) -> core.i32 {
    func.return %arg0
  }
}"#,
        )
    }

    /// Apply i32 -> i64 signature conversion pattern.
    #[salsa::tracked]
    fn apply_i32_to_i64_signature_conversion<'db>(
        db: &'db dyn salsa::Database,
        module: core::Module<'db>,
    ) -> (bool, core::Module<'db>) {
        // The pattern uses the same converter from PatternApplicator
        let converter = TypeConverter::new().add_conversion(|db, ty| {
            core::I32::from_type(db, ty).map(|_| core::I64::new(db).as_type())
        });

        let applicator =
            PatternApplicator::new(converter).add_pattern(FuncSignatureConversionPattern);
        let target = ConversionTarget::new();
        let result = applicator.apply_partial(db, module, target);
        (result.reached_fixpoint, result.module)
    }

    #[salsa_test]
    fn test_func_signature_conversion(db: &salsa::DatabaseImpl) {
        let module = make_i32_func_module(db);
        let (reached_fixpoint, result_module) = apply_i32_to_i64_signature_conversion(db, module);

        assert!(reached_fixpoint);

        // Check that the function signature was converted
        let body = result_module.body(db);
        let func_op = body.blocks(db)[0].operations(db)[0];
        let converted_func = func::Func::from_operation(db, func_op).unwrap();
        let converted_type = core::Func::from_type(db, converted_func.r#type(db)).unwrap();

        let i64_ty = core::I64::new(db).as_type();
        assert_eq!(converted_type.result(db), i64_ty);
        assert_eq!(converted_type.params(db)[0], i64_ty);

        // Check entry block arguments were also converted
        let converted_body = converted_func.body(db);
        let entry_block = &converted_body.blocks(db)[0];
        assert_eq!(entry_block.args(db)[0].ty(db), i64_ty);
    }

    /// Apply f32 -> f64 signature conversion pattern (won't match i32).
    #[salsa::tracked]
    fn apply_f32_to_f64_signature_conversion<'db>(
        db: &'db dyn salsa::Database,
        module: core::Module<'db>,
    ) -> (bool, core::Module<'db>) {
        // The pattern uses the same converter from PatternApplicator
        let converter = TypeConverter::new().add_conversion(|db, ty| {
            core::F32::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        });

        let applicator =
            PatternApplicator::new(converter).add_pattern(FuncSignatureConversionPattern);
        let target = ConversionTarget::new();
        let result = applicator.apply_partial(db, module, target);
        (result.reached_fixpoint, result.module)
    }

    #[salsa_test]
    fn test_no_conversion_when_types_unchanged(db: &salsa::DatabaseImpl) {
        // Create a function with i32 params (won't be converted by f32 -> f64 converter)
        let module = make_i32_func_module(db);
        let (reached_fixpoint, _result_module) = apply_f32_to_f64_signature_conversion(db, module);

        // Should reach fixpoint immediately since no conversions apply
        assert!(reached_fixpoint);
    }
}
