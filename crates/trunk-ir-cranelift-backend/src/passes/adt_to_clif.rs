//! Lower ADT dialect struct access operations to clif dialect.
//!
//! This pass converts struct access operations to their Cranelift equivalents:
//! - `adt.struct_get(ref, field)` -> `clif.load(ref + offset)`
//! - `adt.struct_set(ref, value, field)` -> `clif.store(value, ref + offset)`
//!
//! ## Note
//!
//! `adt.struct_new` is handled by a separate Tribute-specific pass
//! (`tribute_passes::native::adt_rc_header`) that initializes RC headers.
//! This pass only handles field access operations.
//!
//! ## Limitations
//!
//! - Only struct access operations are lowered (variant/array/ref ops are left unchanged)

use tracing::warn;

use crate::adt_layout::compute_struct_layout;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, clif};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{DialectOp, Operation};

/// Lower ADT struct access operations to clif dialect.
///
/// This is a partial lowering: only struct_get and struct_set are converted.
/// Other ADT operations (struct_new, variant, array, ref) pass through unchanged.
///
/// The `type_converter` parameter is used to determine field sizes for
/// layout computation.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Result<Module<'db>, ConversionError> {
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("adt");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .apply_partial(db, module, target)
        .module)
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
    use trunk_ir::dialect::core;
    use trunk_ir::printer::print_op;
    use trunk_ir::{
        Block, BlockId, DialectOp, DialectType, Location, PathId, Region, Span, Symbol, idvec,
    };

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
