//! Lower tribute_rt dialect operations to clif dialect for native backend.
//!
//! This pass converts boxing/unboxing operations to their native equivalents:
//! - `tribute_rt.box_int` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_int` → `clif.load`
//! - `tribute_rt.box_nat` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_nat` → `clif.load`
//! - `tribute_rt.box_float` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_float` → `clif.load`
//! - `tribute_rt.box_bool` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_bool` → `clif.load`
//!
//! ## Allocation Strategy
//!
//! Each boxed value is heap-allocated via `__tribute_alloc(size)` and the
//! primitive value is stored at offset 0. Phase 3 (RC) will prepend an 8-byte
//! header (refcount + rtti_idx) before the payload.

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{clif, core};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{DialectOp, DialectType, Operation, Symbol, Value};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Generate boxing operations: allocate + store value.
///
/// Returns a sequence of clif ops where the last op produces the pointer result.
///
/// ```text
/// %size = clif.iconst(<payload_size>)
/// %ptr  = clif.call @__tribute_alloc(%size)
/// clif.store(%value, %ptr, offset=0)
/// %zero = clif.iconst(0)
/// %result = clif.iadd(%ptr, %zero)   // identity for result mapping
/// ```
fn box_value<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    value: Value<'db>,
    payload_size: i64,
) -> Vec<Operation<'db>> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();

    let mut ops = Vec::new();

    // 1. Allocation size
    let size_op = clif::iconst(db, location, i64_ty, payload_size);
    let size_val = size_op.result(db);
    ops.push(size_op.as_operation());

    // 2. Allocate heap memory
    let call_op = clif::call(db, location, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
    let ptr_val = call_op.result(db);
    ops.push(call_op.as_operation());

    // 3. Store value at offset 0
    let store_op = clif::store(db, location, value, ptr_val, 0);
    ops.push(store_op.as_operation());

    // 4. Identity pass-through so the last op produces the ptr result.
    //    Cranelift will optimize away iadd(ptr, 0).
    let zero_op = clif::iconst(db, location, i64_ty, 0);
    let zero_val = zero_op.result(db);
    ops.push(zero_op.as_operation());

    let identity_op = clif::iadd(db, location, ptr_val, zero_val, ptr_ty);
    ops.push(identity_op.as_operation());

    ops
}

/// Lower tribute_rt boxing/unboxing operations to clif dialect.
///
/// This is a partial lowering: only box/unbox operations are converted.
/// `retain`/`release` ops pass through (handled by a future RC lowering pass).
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Module<'db> {
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("tribute_rt");

    PatternApplicator::new(type_converter)
        .add_pattern(BoxIntPattern)
        .add_pattern(UnboxIntPattern)
        .add_pattern(BoxNatPattern)
        .add_pattern(UnboxNatPattern)
        .add_pattern(BoxFloatPattern)
        .add_pattern(UnboxFloatPattern)
        .add_pattern(BoxBoolPattern)
        .add_pattern(UnboxBoolPattern)
        .apply_partial(db, module, target)
        .module
}

// =============================================================================
// Boxing Patterns (primitive → heap pointer)
// =============================================================================

/// Pattern for `tribute_rt.box_int` → alloc + store (4 bytes for i32).
struct BoxIntPattern;

impl<'db> RewritePattern<'db> for BoxIntPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(box_op) = tribute_rt::BoxInt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let ops = box_value(db, op.location(db), box_op.value(db), 4);
        RewriteResult::Expand(ops)
    }
}

/// Pattern for `tribute_rt.box_nat` → alloc + store (4 bytes for i32).
struct BoxNatPattern;

impl<'db> RewritePattern<'db> for BoxNatPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(box_op) = tribute_rt::BoxNat::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let ops = box_value(db, op.location(db), box_op.value(db), 4);
        RewriteResult::Expand(ops)
    }
}

/// Pattern for `tribute_rt.box_bool` → alloc + store (4 bytes for i32).
struct BoxBoolPattern;

impl<'db> RewritePattern<'db> for BoxBoolPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(box_op) = tribute_rt::BoxBool::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let ops = box_value(db, op.location(db), box_op.value(db), 4);
        RewriteResult::Expand(ops)
    }
}

/// Pattern for `tribute_rt.box_float` → alloc + store (8 bytes for f64).
struct BoxFloatPattern;

impl<'db> RewritePattern<'db> for BoxFloatPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(box_op) = tribute_rt::BoxFloat::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let ops = box_value(db, op.location(db), box_op.value(db), 8);
        RewriteResult::Expand(ops)
    }
}

// =============================================================================
// Unboxing Patterns (heap pointer → primitive)
// =============================================================================

/// Pattern for `tribute_rt.unbox_int` → `clif.load` (i32 from offset 0).
struct UnboxIntPattern;

impl<'db> RewritePattern<'db> for UnboxIntPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxInt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let i32_ty = core::I32::new(db).as_type();
        let load_op = clif::load(db, op.location(db), unbox_op.value(db), i32_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_nat` → `clif.load` (i32 from offset 0).
struct UnboxNatPattern;

impl<'db> RewritePattern<'db> for UnboxNatPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxNat::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let i32_ty = core::I32::new(db).as_type();
        let load_op = clif::load(db, op.location(db), unbox_op.value(db), i32_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_bool` → `clif.load` (i32 from offset 0).
struct UnboxBoolPattern;

impl<'db> RewritePattern<'db> for UnboxBoolPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxBool::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let i32_ty = core::I32::new(db).as_type();
        let load_op = clif::load(db, op.location(db), unbox_op.value(db), i32_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_float` → `clif.load` (f64 from offset 0).
struct UnboxFloatPattern;

impl<'db> RewritePattern<'db> for UnboxFloatPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxFloat::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let f64_ty = core::F64::new(db).as_type();
        let load_op = clif::load(db, op.location(db), unbox_op.value(db), f64_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::rewrite::TypeConverter;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn get_ops<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) -> Vec<Operation<'db>> {
        module.body(db).blocks(db)[0]
            .operations(db)
            .iter()
            .copied()
            .collect()
    }

    // === box_int ===

    #[salsa::tracked]
    fn make_and_lower_box_int(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let const_op = clif::iconst(db, location, i32_ty, 42);
        let box_op = tribute_rt::box_int(db, location, const_op.result(db), ptr_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op.as_operation(), box_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = Module::create(db, location, "test".into(), region);
        lower(db, module, test_converter())
    }

    #[salsa_test]
    fn test_box_int_to_clif(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_box_int(db);
        let ops = get_ops(db, &lowered);

        // iconst(42) + iconst(4) + call alloc + store + iconst(0) + iadd = 6 ops
        assert_eq!(ops.len(), 6);
        assert_eq!(ops[0].full_name(db), "clif.iconst");
        assert_eq!(ops[1].full_name(db), "clif.iconst"); // size=4
        assert_eq!(ops[2].full_name(db), "clif.call"); // alloc
        assert_eq!(ops[3].full_name(db), "clif.store"); // store value
        assert_eq!(ops[4].full_name(db), "clif.iconst"); // zero
        assert_eq!(ops[5].full_name(db), "clif.iadd"); // identity

        // Last op should produce ptr type
        let result_ty = ops[5].results(db)[0];
        assert_eq!(result_ty, core::Ptr::new(db).as_type());
    }

    // === unbox_int ===

    #[salsa::tracked]
    fn make_and_lower_unbox_int(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let placeholder = clif::iconst(db, location, ptr_ty, 0);
        let unbox_op = tribute_rt::unbox_int(db, location, placeholder.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![placeholder.as_operation(), unbox_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = Module::create(db, location, "test".into(), region);
        lower(db, module, test_converter())
    }

    #[salsa_test]
    fn test_unbox_int_to_clif(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_unbox_int(db);
        let ops = get_ops(db, &lowered);

        // placeholder + load = 2 ops
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].full_name(db), "clif.iconst");
        assert_eq!(ops[1].full_name(db), "clif.load");

        // Load should produce i32
        let result_ty = ops[1].results(db)[0];
        assert_eq!(result_ty, core::I32::new(db).as_type());
    }

    // === box_float ===

    #[salsa::tracked]
    fn make_and_lower_box_float(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let f64_ty = core::F64::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let const_op = clif::f64const(db, location, f64_ty, 2.5);
        let box_op = tribute_rt::box_float(db, location, const_op.result(db), ptr_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op.as_operation(), box_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = Module::create(db, location, "test".into(), region);
        lower(db, module, test_converter())
    }

    #[salsa_test]
    fn test_box_float_to_clif(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_box_float(db);
        let ops = get_ops(db, &lowered);

        // f64const + iconst(8) + call alloc + store + iconst(0) + iadd = 6 ops
        assert_eq!(ops.len(), 6);
        assert_eq!(ops[0].full_name(db), "clif.f64const");
        assert_eq!(ops[1].full_name(db), "clif.iconst"); // size=8
        assert_eq!(ops[2].full_name(db), "clif.call"); // alloc
        assert_eq!(ops[3].full_name(db), "clif.store");
        assert_eq!(ops[5].full_name(db), "clif.iadd");

        let result_ty = ops[5].results(db)[0];
        assert_eq!(result_ty, core::Ptr::new(db).as_type());
    }

    // === unbox_float ===

    #[salsa::tracked]
    fn make_and_lower_unbox_float(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let f64_ty = core::F64::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let placeholder = clif::iconst(db, location, ptr_ty, 0);
        let unbox_op = tribute_rt::unbox_float(db, location, placeholder.result(db), f64_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![placeholder.as_operation(), unbox_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        let module = Module::create(db, location, "test".into(), region);
        lower(db, module, test_converter())
    }

    #[salsa_test]
    fn test_unbox_float_to_clif(db: &salsa::DatabaseImpl) {
        let lowered = make_and_lower_unbox_float(db);
        let ops = get_ops(db, &lowered);

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].full_name(db), "clif.iconst");
        assert_eq!(ops[1].full_name(db), "clif.load");

        let result_ty = ops[1].results(db)[0];
        assert_eq!(result_ty, core::F64::new(db).as_type());
    }
}
