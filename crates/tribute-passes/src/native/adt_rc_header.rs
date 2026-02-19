//! Lower `adt.struct_new` and `adt.variant_new` operations with RC header initialization.
//!
//! This pass converts ADT construction ops to `clif.*` operations that:
//! 1. Allocate memory via `__tribute_alloc(payload_size + RC_HEADER_SIZE)`
//! 2. Store refcount = 1 and rtti_idx in the RC header
//! 3. Store field values in the payload area
//!
//! This is a Tribute-specific pass that must run before the language-agnostic
//! `adt_to_clif` pass (which handles field access operations).
//!
//! ## Pipeline Position
//!
//! Runs at Phase 1.95, after RTTI assignment (Phase 1.9) and before
//! `adt_to_clif` (Phase 2).

use std::collections::HashMap;

use tracing::warn;

use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use trunk_ir::adt_layout::{compute_enum_layout, compute_struct_layout, find_variant_layout};
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, clif, core};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{DialectOp, DialectType, Operation, Symbol, Type};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Lower `adt.struct_new` operations to clif dialect with RC headers.
///
/// The `rtti_map` maps struct types to their RTTI indices for storing
/// in RC headers. Types not in the map get rtti_idx = 0.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
    rtti_map: &HashMap<Type<'db>, u32>,
) -> Module<'db> {
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("adt");

    PatternApplicator::new(type_converter)
        .add_pattern(StructNewPattern {
            rtti_map: rtti_map.clone(),
        })
        .add_pattern(VariantNewPattern {
            rtti_map: rtti_map.clone(),
        })
        .apply_partial(db, module, target)
        .module
}

/// Pattern for `adt.struct_new(fields...)` -> heap allocation + RC header + stores.
///
/// Generates:
/// ```text
/// %size      = clif.iconst(layout.total_size + 8)
/// %raw_ptr   = clif.call @__tribute_alloc(%size)
/// %rc_one    = clif.iconst(1)
/// clif.store(%rc_one, %raw_ptr, offset=0)        // refcount
/// %rtti_idx  = clif.iconst(<rtti_idx>)
/// clif.store(%rtti_idx, %raw_ptr, offset=4)      // rtti_idx
/// %hdr_size  = clif.iconst(8)
/// %payload   = clif.iadd(%raw_ptr, %hdr_size)    // payload_ptr
/// clif.store(%field0, %payload, offset=0)
/// clif.store(%field1, %payload, offset=4)
/// ...
/// %result    = clif.iadd(%payload, %zero)         // identity
/// ```
struct StructNewPattern<'db> {
    rtti_map: HashMap<Type<'db>, u32>,
}

impl<'db> RewritePattern<'db> for StructNewPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(struct_new) = adt::StructNew::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let struct_ty = struct_new.r#type(db);
        let type_converter = adaptor.type_converter();

        let Some(layout) = compute_struct_layout(db, struct_ty, type_converter) else {
            warn!(
                "adt_rc_header: cannot compute layout for struct_new type at {:?}",
                op.location(db)
            );
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let fields = adaptor.operands();

        let mut ops: Vec<Operation<'db>> = Vec::new();

        // 1. Compute allocation size (payload + RC header)
        let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = clif::iconst(db, location, i64_ty, alloc_size as i64);
        let size_val = size_op.result(db);
        ops.push(size_op.as_operation());

        // 2. Call __tribute_alloc
        let call_op = clif::call(db, location, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
        let raw_ptr = call_op.result(db);
        ops.push(call_op.as_operation());

        // 3. Store refcount = 1 at raw_ptr + 0
        let rc_one = clif::iconst(db, location, i32_ty, 1);
        ops.push(rc_one.as_operation());
        let store_rc = clif::store(db, location, rc_one.result(db), raw_ptr, 0);
        ops.push(store_rc.as_operation());

        // 4. Store rtti_idx at raw_ptr + 4
        let rtti_idx = self.rtti_map.get(&struct_ty).copied().unwrap_or(0) as i64;
        let rtti_val = clif::iconst(db, location, i32_ty, rtti_idx);
        ops.push(rtti_val.as_operation());
        let store_rtti = clif::store(db, location, rtti_val.result(db), raw_ptr, 4);
        ops.push(store_rtti.as_operation());

        // 5. Compute payload pointer = raw_ptr + 8
        let hdr_size = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
        ops.push(hdr_size.as_operation());
        let payload_ptr = clif::iadd(db, location, raw_ptr, hdr_size.result(db), ptr_ty);
        let payload_val = payload_ptr.result(db);
        ops.push(payload_ptr.as_operation());

        // 6. Store each field at its computed offset (relative to payload)
        for (i, &field_val) in fields.iter().enumerate() {
            if i < layout.field_offsets.len() {
                let offset = layout.field_offsets[i] as i32;
                let store_op = clif::store(db, location, field_val, payload_val, offset);
                ops.push(store_op.as_operation());
            }
        }

        // 7. Identity pass-through so the last op produces the payload ptr result.
        let zero_op = clif::iconst(db, location, i64_ty, 0);
        let zero_val = zero_op.result(db);
        ops.push(zero_op.as_operation());

        let identity_op = clif::iadd(db, location, payload_val, zero_val, ptr_ty);
        ops.push(identity_op.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Pattern for `adt.variant_new(fields...)` -> heap allocation + RC header + tag + stores.
///
/// Generates:
/// ```text
/// %size      = clif.iconst(enum_layout.total_size + 8)
/// %raw_ptr   = clif.call @__tribute_alloc(%size)
/// %rc_one    = clif.iconst(1)
/// clif.store(%rc_one, %raw_ptr, offset=0)        // refcount
/// %rtti_idx  = clif.iconst(<rtti_idx>)
/// clif.store(%rtti_idx, %raw_ptr, offset=4)      // rtti_idx
/// %hdr_size  = clif.iconst(8)
/// %payload   = clif.iadd(%raw_ptr, %hdr_size)    // payload_ptr
/// %tag       = clif.iconst(<tag_value>)
/// clif.store(%tag, %payload, offset=0)            // discriminant
/// clif.store(%field0, %payload, offset=8)         // first field
/// clif.store(%field1, %payload, offset=16)        // second field
/// ...
/// %result    = clif.iadd(%payload, %zero)         // identity
/// ```
struct VariantNewPattern<'db> {
    rtti_map: HashMap<Type<'db>, u32>,
}

impl<'db> RewritePattern<'db> for VariantNewPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(variant_new) = adt::VariantNew::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let enum_ty = variant_new.r#type(db);
        let tag = variant_new.tag(db);
        let type_converter = adaptor.type_converter();

        let Some(enum_layout) = compute_enum_layout(db, enum_ty, type_converter) else {
            warn!(
                "adt_rc_header: cannot compute enum layout for variant_new at {:?}",
                op.location(db)
            );
            return RewriteResult::Unchanged;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!(
                "adt_rc_header: unknown variant tag {:?} at {:?}",
                tag,
                op.location(db)
            );
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let fields = adaptor.operands();

        let mut ops: Vec<Operation<'db>> = Vec::new();

        // 1. Compute allocation size (payload + RC header)
        let alloc_size = enum_layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = clif::iconst(db, location, i64_ty, alloc_size as i64);
        let size_val = size_op.result(db);
        ops.push(size_op.as_operation());

        // 2. Call __tribute_alloc
        let call_op = clif::call(db, location, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
        let raw_ptr = call_op.result(db);
        ops.push(call_op.as_operation());

        // 3. Store refcount = 1 at raw_ptr + 0
        let rc_one = clif::iconst(db, location, i32_ty, 1);
        ops.push(rc_one.as_operation());
        let store_rc = clif::store(db, location, rc_one.result(db), raw_ptr, 0);
        ops.push(store_rc.as_operation());

        // 4. Store rtti_idx at raw_ptr + 4
        let rtti_idx = self.rtti_map.get(&enum_ty).copied().unwrap_or(0) as i64;
        let rtti_val = clif::iconst(db, location, i32_ty, rtti_idx);
        ops.push(rtti_val.as_operation());
        let store_rtti = clif::store(db, location, rtti_val.result(db), raw_ptr, 4);
        ops.push(store_rtti.as_operation());

        // 5. Compute payload pointer = raw_ptr + 8
        let hdr_size = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
        ops.push(hdr_size.as_operation());
        let payload_ptr = clif::iadd(db, location, raw_ptr, hdr_size.result(db), ptr_ty);
        let payload_val = payload_ptr.result(db);
        ops.push(payload_ptr.as_operation());

        // 6. Store tag at payload + 0
        let tag_val = clif::iconst(db, location, i32_ty, variant_layout.tag_value as i64);
        ops.push(tag_val.as_operation());
        let store_tag = clif::store(db, location, tag_val.result(db), payload_val, 0);
        ops.push(store_tag.as_operation());

        // 7. Store each field at its computed offset (relative to payload + fields_offset)
        for (i, &field_val) in fields.iter().enumerate() {
            if i < variant_layout.field_offsets.len() {
                let offset = (enum_layout.fields_offset + variant_layout.field_offsets[i]) as i32;
                let store_op = clif::store(db, location, field_val, payload_val, offset);
                ops.push(store_op.as_operation());
            }
        }

        // 8. Identity pass-through so the last op produces the payload ptr result.
        let zero_op = clif::iconst(db, location, i64_ty, 0);
        let zero_val = zero_op.result(db);
        ops.push(zero_op.as_operation());

        let identity_op = clif::iadd(db, location, payload_val, zero_val, ptr_ty);
        ops.push(identity_op.as_operation());

        RewriteResult::expand(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::printer::print_op;
    use trunk_ir::{Block, BlockId, DialectOp, DialectType, Location, PathId, Region, Span, idvec};

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn lower_and_print(db: &dyn salsa::Database, module: Module<'_>) -> String {
        let lowered = lower(db, module, test_converter(), &HashMap::new());
        print_op(db, lowered.as_operation())
    }

    #[salsa::tracked]
    fn make_struct_new_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );

        let const1 = clif::iconst(db, location, i32_ty, 10);
        let const2 = clif::iconst(db, location, i32_ty, 20);
        let struct_new_op = adt::struct_new(
            db,
            location,
            vec![const1.result(db), const2.result(db)],
            struct_ty,
            struct_ty,
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                const1.as_operation(),
                const2.as_operation(),
                struct_new_op.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn do_lower_struct_new(db: &dyn salsa::Database) -> String {
        lower_and_print(db, make_struct_new_module(db))
    }

    #[salsa_test]
    fn test_struct_new_to_clif(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_struct_new(db));
    }

    #[salsa::tracked]
    fn make_struct_new_empty_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let struct_ty = adt::struct_type(db, Symbol::new("Unit"), vec![]);
        let struct_new_op = adt::struct_new(db, location, vec![], struct_ty, struct_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![struct_new_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn do_lower_struct_new_empty(db: &dyn salsa::Database) -> String {
        lower_and_print(db, make_struct_new_empty_module(db))
    }

    #[salsa_test]
    fn test_struct_new_empty(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_struct_new_empty(db));
    }
}
