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

use tribute_ir::dialect::tribute_rt::{self, RC_HEADER_SIZE};
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{clif, core};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{DialectOp, DialectType, Operation, Symbol, Value};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Generate boxing operations: allocate + store RC header + store value.
///
/// Returns a sequence of clif ops where the last op produces the payload pointer.
///
/// ```text
/// %alloc_size = clif.iconst(<payload_size + 8>)
/// %raw_ptr    = clif.call @__tribute_alloc(%alloc_size)
/// %rc_one     = clif.iconst(1)           // initial refcount
/// clif.store(%rc_one, %raw_ptr, offset=0)
/// %rtti_zero  = clif.iconst(0)           // rtti_idx placeholder
/// clif.store(%rtti_zero, %raw_ptr, offset=4)
/// %hdr_size   = clif.iconst(8)
/// %payload    = clif.iadd(%raw_ptr, %hdr_size)  // payload_ptr = raw + 8
/// clif.store(%value, %payload, offset=0)
/// %zero       = clif.iconst(0)
/// %result     = clif.iadd(%payload, %zero)      // identity for result mapping
/// ```
fn box_value<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    value: Value<'db>,
    payload_size: u64,
    rtti_idx: u32,
) -> Vec<Operation<'db>> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();

    let mut ops = Vec::new();

    // 1. Allocation size (payload + RC header)
    let alloc_size = payload_size
        .checked_add(RC_HEADER_SIZE)
        .expect("allocation size overflow: payload_size + RC_HEADER_SIZE exceeds u64::MAX");
    let alloc_size_i64 = i64::try_from(alloc_size).expect("allocation size does not fit in i64");
    let size_op = clif::iconst(db, location, i64_ty, alloc_size_i64);
    let size_val = size_op.result(db);
    ops.push(size_op.as_operation());

    // 2. Allocate heap memory
    let call_op = clif::call(db, location, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
    let raw_ptr = call_op.result(db);
    ops.push(call_op.as_operation());

    // 3. Store refcount = 1 at raw_ptr + 0
    let rc_one = clif::iconst(db, location, i32_ty, 1);
    ops.push(rc_one.as_operation());
    let store_rc = clif::store(db, location, rc_one.result(db), raw_ptr, 0);
    ops.push(store_rc.as_operation());

    // 4. Store rtti_idx at raw_ptr + 4
    let rtti_val = clif::iconst(db, location, i32_ty, rtti_idx as i64);
    ops.push(rtti_val.as_operation());
    let store_rtti = clif::store(db, location, rtti_val.result(db), raw_ptr, 4);
    ops.push(store_rtti.as_operation());

    // 5. Compute payload pointer = raw_ptr + 8
    let hdr_size = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_size.as_operation());
    let payload_ptr = clif::iadd(db, location, raw_ptr, hdr_size.result(db), ptr_ty);
    ops.push(payload_ptr.as_operation());

    // 6. Store value at payload offset 0
    let store_val = clif::store(db, location, value, payload_ptr.result(db), 0);
    ops.push(store_val.as_operation());

    // 7. Identity pass-through so the last op produces the payload ptr result.
    //    Cranelift will optimize away iadd(ptr, 0).
    let zero_op = clif::iconst(db, location, ptr_ty, 0);
    let zero_val = zero_op.result(db);
    ops.push(zero_op.as_operation());

    let identity_op = clif::iadd(db, location, payload_ptr.result(db), zero_val, ptr_ty);
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
        .illegal_dialect("tribute_rt")
        .legal_op("tribute_rt", "retain")
        .legal_op("tribute_rt", "release");

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
        let ops = box_value(
            db,
            op.location(db),
            box_op.value(db),
            4,
            super::rtti::RTTI_INT,
        );
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
        let ops = box_value(
            db,
            op.location(db),
            box_op.value(db),
            4,
            super::rtti::RTTI_NAT,
        );
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
        let ops = box_value(
            db,
            op.location(db),
            box_op.value(db),
            4,
            super::rtti::RTTI_BOOL,
        );
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
        let ops = box_value(
            db,
            op.location(db),
            box_op.value(db),
            8,
            super::rtti::RTTI_FLOAT,
        );
        RewriteResult::Expand(ops)
    }
}

// =============================================================================
// Unboxing Patterns (heap pointer → primitive)
// =============================================================================

/// Pattern for `tribute_rt.unbox_int` → `clif.load` (i32 from offset 0).
///
/// If the input value is already `core.i32` (e.g., from a variant field that
/// was stored as a raw integer), the unbox is a no-op.
struct UnboxIntPattern;

impl<'db> RewritePattern<'db> for UnboxIntPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxInt::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let value = unbox_op.value(db);
        let i32_ty = core::I32::new(db).as_type();

        // If the input is already i32, the unbox is a no-op (value was stored raw).
        if let Some(val_ty) = adaptor.get_value_type(db, value) {
            if val_ty == i32_ty {
                return RewriteResult::Erase {
                    replacement_values: vec![value],
                };
            }
        }

        let load_op = clif::load(db, op.location(db), value, i32_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_nat` → `clif.load` (i32 from offset 0).
///
/// If the input value is already `core.i32`, the unbox is a no-op.
struct UnboxNatPattern;

impl<'db> RewritePattern<'db> for UnboxNatPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxNat::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let value = unbox_op.value(db);
        let i32_ty = core::I32::new(db).as_type();

        // If the input is already i32, the unbox is a no-op.
        if let Some(val_ty) = adaptor.get_value_type(db, value) {
            if val_ty == i32_ty {
                return RewriteResult::Erase {
                    replacement_values: vec![value],
                };
            }
        }

        let load_op = clif::load(db, op.location(db), value, i32_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_bool` → `clif.load` (i32 from offset 0).
///
/// If the input value is already `core.i32` or `core.i1`, the unbox is a no-op.
struct UnboxBoolPattern;

impl<'db> RewritePattern<'db> for UnboxBoolPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxBool::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let value = unbox_op.value(db);
        let i32_ty = core::I32::new(db).as_type();

        // If the input is already a small integer type, the unbox is a no-op.
        if let Some(val_ty) = adaptor.get_value_type(db, value) {
            if val_ty == i32_ty || core::I1::from_type(db, val_ty).is_some() {
                return RewriteResult::Erase {
                    replacement_values: vec![value],
                };
            }
        }

        let load_op = clif::load(db, op.location(db), value, i32_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `tribute_rt.unbox_float` → `clif.load` (f64 from offset 0).
///
/// If the input value is already `core.f64`, the unbox is a no-op.
struct UnboxFloatPattern;

impl<'db> RewritePattern<'db> for UnboxFloatPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(unbox_op) = tribute_rt::UnboxFloat::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };
        let value = unbox_op.value(db);
        let f64_ty = core::F64::new(db).as_type();

        // If the input is already f64, the unbox is a no-op.
        if let Some(val_ty) = adaptor.get_value_type(db, value) {
            if val_ty == f64_ty {
                return RewriteResult::Erase {
                    replacement_values: vec![value],
                };
            }
        }

        let load_op = clif::load(db, op.location(db), value, f64_ty, 0);
        RewriteResult::Replace(load_op.as_operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_op;
    use trunk_ir::rewrite::TypeConverter;

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    #[salsa::input]
    struct TextInput {
        #[returns(ref)]
        text: String,
    }

    #[salsa::tracked]
    fn do_lower(db: &dyn salsa::Database, input: TextInput) -> Module<'_> {
        let module = parse_test_module(db, input.text(db));
        lower(db, module, test_converter())
    }

    fn run_lower(db: &salsa::DatabaseImpl, ir: &str) -> String {
        let input = TextInput::new(db, ir.to_string());
        let result = do_lower(db, input);
        print_op(db, result.as_operation())
    }

    // === box_int ===

    #[salsa_test]
    fn test_box_int_to_clif(db: &salsa::DatabaseImpl) {
        let ir = run_lower(
            db,
            r#"
            core.module @test {
                %val = clif.iconst {value = 42} : core.i32
                %ptr = tribute_rt.box_int %val : core.ptr
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }

    // === unbox_int ===

    #[salsa_test]
    fn test_unbox_int_to_clif(db: &salsa::DatabaseImpl) {
        let ir = run_lower(
            db,
            r#"
            core.module @test {
                %ptr = clif.iconst {value = 0} : core.ptr
                %val = tribute_rt.unbox_int %ptr : core.i32
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }

    // === box_float ===

    #[salsa_test]
    fn test_box_float_to_clif(db: &salsa::DatabaseImpl) {
        let ir = run_lower(
            db,
            r#"
            core.module @test {
                %val = clif.f64const {value = 2.5} : core.f64
                %ptr = tribute_rt.box_float %val : core.ptr
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }

    // === unbox_float ===

    #[salsa_test]
    fn test_unbox_float_to_clif(db: &salsa::DatabaseImpl) {
        let ir = run_lower(
            db,
            r#"
            core.module @test {
                %ptr = clif.iconst {value = 0} : core.ptr
                %val = tribute_rt.unbox_float %ptr : core.f64
            }
            "#,
        );
        insta::assert_snapshot!(ir);
    }

    // === retain/release passthrough ===

    #[salsa_test]
    fn test_retain_release_pass_through(db: &salsa::DatabaseImpl) {
        let ir = run_lower(
            db,
            r#"
            core.module @test {
                %0 = clif.iconst {value = 0} : core.ptr
                %1 = tribute_rt.retain %0 : core.ptr
                tribute_rt.release %1 {alloc_size = 0}
            }
            "#,
        );

        // retain and release should survive the boxing lowering pass
        assert!(
            ir.contains("tribute_rt.retain"),
            "retain should pass through"
        );
        assert!(
            ir.contains("tribute_rt.release"),
            "release should pass through"
        );
    }
}
