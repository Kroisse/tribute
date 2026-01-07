//! Lower tribute_rt dialect operations to wasm dialect.
//!
//! This pass converts boxing/unboxing operations to their wasm equivalents:
//! - `tribute_rt.box_int` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_int` -> `wasm.i31_get_s` (i31ref -> i32)
//!
//! ## Type Mappings
//!
//! - `tribute_rt.int` -> `core.i32`
//! - `tribute_rt.intref` -> `wasm.i31ref`
//! - `tribute_rt.any` -> `wasm.anyref`

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{DialectOp, DialectType, Operation};

use crate::type_converter::wasm_type_converter;

/// Lower tribute_rt dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new(wasm_type_converter())
        .add_pattern(BoxIntPattern)
        .add_pattern(UnboxIntPattern)
        .apply(db, module)
        .module
}

/// Pattern for `tribute_rt.box_int` -> `wasm.ref_i31`
///
/// Boxing converts an unboxed i32 to an i31ref.
/// Since tribute_rt.int is 31-bit, no truncation is needed.
struct BoxIntPattern;

impl RewritePattern for BoxIntPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(box_op) = tribute_rt::BoxInt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let value = box_op.value(db);

        // Result type is i31ref
        let i31ref_ty = wasm::I31ref::new(db).as_type();

        // wasm.ref_i31: i32 -> i31ref
        let new_op = wasm::ref_i31(db, location, value, i31ref_ty);

        RewriteResult::Replace(new_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_int` -> `wasm.i31_get_s`
///
/// Unboxing extracts the signed i32 from an i31ref.
struct UnboxIntPattern;

impl RewritePattern for UnboxIntPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxInt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let value = unbox_op.value(db);

        // Result type is i32
        let i32_ty = trunk_ir::dialect::core::I32::new(db).as_type();

        // wasm.i31_get_s: i31ref -> i32 (signed)
        let new_op = wasm::i31_get_s(db, location, value, i32_ty);

        RewriteResult::Replace(new_op.as_operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_and_lower_box_int_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let intref_ty = tribute_rt::intref_type(db);

        // Create: %0 = wasm.i32_const 42
        //         %1 = tribute_rt.box_int %0
        let const_op = wasm::i32_const(db, location, i32_ty, 42);
        let box_op = tribute_rt::box_int(
            db,
            location,
            const_op.as_operation().result(db, 0),
            intref_ty,
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op.as_operation(), box_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = core::Module::create(db, location, "test".into(), region);

        // Lower within the tracked function
        lower(db, module)
    }

    #[salsa_test]
    fn test_box_int_lowering(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_box_int_module(db);

        // Check result: should have wasm.ref_i31 instead of tribute_rt.box_int
        let ops: Vec<_> = lowered
            .body(db)
            .blocks(db)
            .first()
            .unwrap()
            .operations(db)
            .iter()
            .copied()
            .collect();

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[1].dialect(db), wasm::DIALECT_NAME());
        assert_eq!(ops[1].name(db), wasm::REF_I31());
    }

    #[salsa::tracked]
    fn make_and_lower_unbox_int_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i31ref_ty = wasm::I31ref::new(db).as_type();

        // Create a dummy i31ref value (use i32_const as placeholder for test)
        // In real usage, this would come from box_int or another source
        let placeholder = wasm::i32_const(db, location, i31ref_ty, 42);
        let unbox_op = tribute_rt::unbox_int(
            db,
            location,
            placeholder.as_operation().result(db, 0),
            i32_ty,
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![placeholder.as_operation(), unbox_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = core::Module::create(db, location, "test".into(), region);

        // Lower within the tracked function
        lower(db, module)
    }

    #[salsa_test]
    fn test_unbox_int_lowering(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_unbox_int_module(db);

        // Check result: should have wasm.i31_get_s instead of tribute_rt.unbox_int
        let ops: Vec<_> = lowered
            .body(db)
            .blocks(db)
            .first()
            .unwrap()
            .operations(db)
            .iter()
            .copied()
            .collect();

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[1].dialect(db), wasm::DIALECT_NAME());
        assert_eq!(ops[1].name(db), wasm::I31_GET_S());
    }
}
