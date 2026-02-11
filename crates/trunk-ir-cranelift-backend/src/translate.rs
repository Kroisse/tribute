//! Cranelift native object file translation.
//!
//! This module provides functions for validating and emitting native object files
//! from TrunkIR modules that have already been lowered to the clif dialect.

use trunk_ir::dialect::core::Module;

use crate::{CompilationResult, validate_clif_ir};

/// Emit a native object file from a lowered TrunkIR module.
///
/// This function assumes the module has already been lowered to clif dialect
/// and all type conversions have been resolved. It:
/// 1. Validates the IR (checks for non-clif ops)
/// 2. Emits native code via Cranelift (stub: returns empty bytes)
///
/// For Tribute-specific compilation (including lowering from high-level IR),
/// use the orchestration in the main crate's pipeline.
#[salsa::tracked]
pub fn emit_module_to_native<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> CompilationResult<Vec<u8>> {
    // Validate IR (check for non-clif ops)
    validate_clif_ir(db, module)?;

    // TODO(#344): Emit native code via Cranelift
    // For now, return empty bytes as a stub
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{clif, core};
    use trunk_ir::{
        Block, BlockId, DialectOp, DialectType, Location, PathId, Region, Span, Symbol, idvec,
    };

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_clif_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        let iconst_op = clif::iconst(db, location, i64_ty, 42).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![iconst_op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_emit_module_stub(db: &salsa::DatabaseImpl) {
        let module = make_clif_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_ok());
        // Stub returns empty bytes
        assert!(result.unwrap().is_empty());
    }

    #[salsa::tracked]
    fn make_invalid_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        let const_op =
            trunk_ir::dialect::func::constant(db, location, i64_ty, Symbol::new("some_func"))
                .as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![const_op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_emit_rejects_non_clif_module(db: &salsa::DatabaseImpl) {
        let module = make_invalid_module(db);
        let result = emit_module_to_native(db, module);
        assert!(result.is_err());
    }
}
