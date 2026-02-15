//! Lower ADT dialect struct operations to clif dialect.
//!
//! This pass converts struct-related ADT operations to their Cranelift equivalents:
//! - `adt.struct_new(fields...)` -> `clif.call @__tribute_alloc` + `clif.store` per field
//! - `adt.struct_get(ref, field)` -> `clif.load(ref + offset)`
//! - `adt.struct_set(ref, value, field)` -> `clif.store(value, ref + offset)`
//!
//! ## Allocation strategy
//!
//! Struct allocation uses `__tribute_alloc(size: i64) -> ptr`, an imported runtime
//! function. Phase 1 is a simple malloc wrapper; Phase 3 will add RC headers.
//!
//! ## Limitations
//!
//! - Only struct operations are lowered (variant/array/ref ops are left unchanged)

use std::collections::HashMap;

use tracing::warn;

use crate::adt_layout::compute_struct_layout;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, clif, core};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{DialectOp, DialectType, Operation, Symbol, Type};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// RC header size: 4 bytes refcount + 4 bytes rtti_idx = 8 bytes.
/// Canonical definition: `tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE`.
const RC_HEADER_SIZE: i64 = 8;

/// Lower ADT struct operations to clif dialect.
///
/// This is a partial lowering: only struct_new, struct_get, and struct_set are converted.
/// Other ADT operations (variant, array, ref) pass through unchanged.
///
/// The `type_converter` parameter is used to determine field sizes for
/// layout computation. The optional `rtti_map` maps struct types to their
/// RTTI indices for storing in RC headers.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Result<Module<'db>, ConversionError> {
    lower_with_rtti(db, module, type_converter, &HashMap::new())
}

/// Lower ADT struct operations with RTTI index mapping.
pub fn lower_with_rtti<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
    rtti_map: &HashMap<Type<'db>, u32>,
) -> Result<Module<'db>, ConversionError> {
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("adt");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(StructNewPattern {
            rtti_map: rtti_map.clone(),
        })
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .apply_partial(db, module, target)
        .module)
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
                "adt_to_clif: cannot compute layout for struct_new type at {:?}",
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
        let alloc_size = layout.total_size as i64 + RC_HEADER_SIZE;
        let size_op = clif::iconst(db, location, i64_ty, alloc_size);
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
        let hdr_size = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE);
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

/// Pattern for `adt.struct_get(ref, field_idx)` -> `clif.load(ref, offset)`.
struct StructGetPattern;

impl<'db> RewritePattern<'db> for StructGetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(struct_get) = adt::StructGet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let struct_ty = struct_get.r#type(db);
        let field_idx = struct_get.field(db) as usize;
        let type_converter = adaptor.type_converter();

        let Some(layout) = compute_struct_layout(db, struct_ty, type_converter) else {
            warn!(
                "adt_to_clif: cannot compute layout for struct_get type at {:?}",
                op.location(db)
            );
            return RewriteResult::Unchanged;
        };

        if field_idx >= layout.field_offsets.len() {
            warn!(
                "adt_to_clif: field index {} out of bounds (struct has {} fields)",
                field_idx,
                layout.field_offsets.len()
            );
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let offset = layout.field_offsets[field_idx] as i32;

        // Get the remapped ref operand
        let ref_val = adaptor.operand(0).unwrap_or_else(|| struct_get.r#ref(db));

        // Result type: use the converted result type from the adaptor
        let result_ty = adaptor
            .result_type(db, 0)
            .unwrap_or_else(|| op.results(db)[0]);

        let load_op = clif::load(db, location, ref_val, result_ty, offset);
        RewriteResult::Replace(load_op.as_operation())
    }
}

/// Pattern for `adt.struct_set(ref, value, field_idx)` -> `clif.store(value, ref, offset)`.
struct StructSetPattern;

impl<'db> RewritePattern<'db> for StructSetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(struct_set) = adt::StructSet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let struct_ty = struct_set.r#type(db);
        let field_idx = struct_set.field(db) as usize;
        let type_converter = adaptor.type_converter();

        let Some(layout) = compute_struct_layout(db, struct_ty, type_converter) else {
            warn!(
                "adt_to_clif: cannot compute layout for struct_set type at {:?}",
                op.location(db)
            );
            return RewriteResult::Unchanged;
        };

        if field_idx >= layout.field_offsets.len() {
            warn!(
                "adt_to_clif: field index {} out of bounds (struct has {} fields)",
                field_idx,
                layout.field_offsets.len()
            );
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let offset = layout.field_offsets[field_idx] as i32;

        // Get remapped operands: ref (0) and value (1)
        let ref_val = adaptor.operand(0).unwrap_or_else(|| struct_set.r#ref(db));
        let value_val = adaptor.operand(1).unwrap_or_else(|| struct_set.value(db));

        let store_op = clif::store(db, location, value_val, ref_val, offset);
        RewriteResult::Replace(store_op.as_operation())
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
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
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
    fn make_struct_get_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );

        let ref_op = clif::iconst(db, location, ptr_ty, 0);
        let struct_get_op = adt::struct_get(db, location, ref_op.result(db), i32_ty, struct_ty, 1);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![ref_op.as_operation(), struct_get_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn do_lower_struct_get(db: &dyn salsa::Database) -> String {
        lower_and_print(db, make_struct_get_module(db))
    }

    #[salsa_test]
    fn test_struct_get_to_clif(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_struct_get(db));
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

    #[salsa::tracked]
    fn make_struct_set_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );

        let ref_op = clif::iconst(db, location, ptr_ty, 0);
        let val_op = clif::iconst(db, location, i32_ty, 42);
        let struct_set_op = adt::struct_set(
            db,
            location,
            ref_op.result(db),
            val_op.result(db),
            struct_ty,
            0,
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                ref_op.as_operation(),
                val_op.as_operation(),
                struct_set_op.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn do_lower_struct_set(db: &dyn salsa::Database) -> String {
        lower_and_print(db, make_struct_set_module(db))
    }

    #[salsa_test]
    fn test_struct_set_to_clif(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_struct_set(db));
    }
}
