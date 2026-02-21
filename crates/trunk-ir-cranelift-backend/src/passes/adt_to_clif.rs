//! Lower ADT dialect operations to clif dialect.
//!
//! This pass converts ADT operations to their Cranelift equivalents:
//! - `adt.struct_get(ref, field)` -> `clif.load(ref + offset)`
//! - `adt.struct_set(ref, value, field)` -> `clif.store(value, ref + offset)`
//! - `adt.variant_is(ref, type, tag)` -> `clif.load(ref, 0)` + `clif.icmp(eq, tag_val, expected)`
//! - `adt.variant_cast(ref, type, tag)` -> identity (native pointers are untyped)
//! - `adt.variant_get(ref, type, tag, field)` -> `clif.load(ref, fields_offset + field_offset)`
//! - `adt.ref_null(type)` -> `clif.iconst(0)` (null pointer)
//! - `adt.ref_cast(ref, type)` -> identity (native pointers are untyped)
//! - `adt.ref_is_null(ref)` -> `clif.icmp(eq, ref, 0)`
//!
//! ## Note
//!
//! `adt.struct_new` and `adt.variant_new` are handled by a separate
//! Tribute-specific pass (`tribute_passes::native::adt_rc_header`) that
//! initializes RC headers. This pass only handles field access and
//! reference operations.
//!
//! ## Limitations
//!
//! - Array ops are left unchanged

use tracing::warn;

use crate::adt_layout::{compute_enum_layout, compute_struct_layout, find_variant_layout};
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, clif, core};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::{DialectOp, DialectType, Operation, Symbol};

/// Lower ADT operations to clif dialect.
///
/// This is a partial lowering: struct access and reference operations are converted.
/// Other ADT operations (struct_new, variant, array) pass through unchanged.
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
        .add_pattern(VariantIsPattern)
        .add_pattern(VariantCastPattern)
        .add_pattern(VariantGetPattern)
        .add_pattern(RefNullPattern)
        .add_pattern(RefCastPattern)
        .add_pattern(RefIsNullPattern)
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(struct_get) = adt::StructGet::from_operation(db, *op) else {
            return false;
        };

        let struct_ty = struct_get.r#type(db);
        let field_idx = struct_get.field(db) as usize;
        let type_converter = rewriter.type_converter();

        let Some(layout) = compute_struct_layout(db, struct_ty, type_converter) else {
            warn!(
                "adt_to_clif: cannot compute layout for struct_get type at {:?}",
                op.location(db)
            );
            return false;
        };

        if field_idx >= layout.field_offsets.len() {
            warn!(
                "adt_to_clif: field index {} out of bounds (struct has {} fields)",
                field_idx,
                layout.field_offsets.len()
            );
            return false;
        }

        let location = op.location(db);
        let offset = layout.field_offsets[field_idx] as i32;

        // Get the remapped ref operand
        let ref_val = rewriter.operand(0).unwrap_or_else(|| struct_get.r#ref(db));

        // Result type: use the converted result type from the rewriter
        let result_ty = rewriter
            .result_type(db, op, 0)
            .unwrap_or_else(|| op.results(db)[0]);

        let load_op = clif::load(db, location, ref_val, result_ty, offset);
        rewriter.replace_op(load_op.as_operation());
        true
    }
}

/// Pattern for `adt.struct_set(ref, value, field_idx)` -> `clif.store(value, ref, offset)`.
struct StructSetPattern;

impl<'db> RewritePattern<'db> for StructSetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(struct_set) = adt::StructSet::from_operation(db, *op) else {
            return false;
        };

        let struct_ty = struct_set.r#type(db);
        let field_idx = struct_set.field(db) as usize;
        let type_converter = rewriter.type_converter();

        let Some(layout) = compute_struct_layout(db, struct_ty, type_converter) else {
            warn!(
                "adt_to_clif: cannot compute layout for struct_set type at {:?}",
                op.location(db)
            );
            return false;
        };

        if field_idx >= layout.field_offsets.len() {
            warn!(
                "adt_to_clif: field index {} out of bounds (struct has {} fields)",
                field_idx,
                layout.field_offsets.len()
            );
            return false;
        }

        let location = op.location(db);
        let offset = layout.field_offsets[field_idx] as i32;

        // Get remapped operands: ref (0) and value (1)
        let ref_val = rewriter.operand(0).unwrap_or_else(|| struct_set.r#ref(db));
        let value_val = rewriter.operand(1).unwrap_or_else(|| struct_set.value(db));

        let store_op = clif::store(db, location, value_val, ref_val, offset);
        rewriter.replace_op(store_op.as_operation());
        true
    }
}

// =============================================================================
// Variant Patterns
// =============================================================================

/// Pattern for `adt.variant_is(ref, type, tag)` -> load tag + compare.
///
/// Loads the tag (i32) at offset 0 from the payload pointer and compares
/// with the expected discriminant value.
struct VariantIsPattern;

impl<'db> RewritePattern<'db> for VariantIsPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(variant_is) = adt::VariantIs::from_operation(db, *op) else {
            return false;
        };

        let enum_ty = variant_is.r#type(db);
        let tag = variant_is.tag(db);
        let type_converter = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout(db, enum_ty, type_converter) else {
            warn!(
                "adt_to_clif: cannot compute enum layout for variant_is at {:?}",
                op.location(db)
            );
            return false;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!(
                "adt_to_clif: unknown variant tag {:?} at {:?}",
                tag,
                op.location(db)
            );
            return false;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        let ref_val = rewriter.operand(0).unwrap_or_else(|| variant_is.r#ref(db));

        // Load tag from payload_ptr + 0
        let tag_load = clif::load(db, location, ref_val, i32_ty, 0);
        let tag_val = tag_load.result(db);

        // Compare with expected discriminant
        let expected = clif::iconst(db, location, i32_ty, variant_layout.tag_value as i64);
        let cmp_op = clif::icmp(
            db,
            location,
            tag_val,
            expected.result(db),
            i1_ty,
            Symbol::new("eq"),
        );

        rewriter.insert_op(tag_load.as_operation());
        rewriter.insert_op(expected.as_operation());
        rewriter.replace_op(cmp_op.as_operation());
        true
    }
}

/// Pattern for `adt.variant_cast(ref, type, tag)` -> identity (no-op).
///
/// In native code, all references are opaque pointers, so variant casts
/// are purely a type-system concept and require no runtime operation.
struct VariantCastPattern;

impl<'db> RewritePattern<'db> for VariantCastPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(variant_cast) = adt::VariantCast::from_operation(db, *op) else {
            return false;
        };

        let ref_val = rewriter
            .operand(0)
            .unwrap_or_else(|| variant_cast.r#ref(db));
        rewriter.erase_op(vec![ref_val]);
        true
    }
}

/// Pattern for `adt.variant_get(ref, type, tag, field)` -> `clif.load(ref, offset)`.
///
/// Uses the enum layout and variant tag to compute the field offset:
/// `fields_offset + variant_field_offsets[field_idx]`
struct VariantGetPattern;

impl<'db> RewritePattern<'db> for VariantGetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(variant_get) = adt::VariantGet::from_operation(db, *op) else {
            return false;
        };

        let enum_ty = variant_get.r#type(db);
        let tag = variant_get.tag(db);
        let field_idx = variant_get.field(db) as usize;
        let type_converter = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout(db, enum_ty, type_converter) else {
            warn!(
                "adt_to_clif: cannot compute enum layout for variant_get at {:?}",
                op.location(db)
            );
            return false;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!(
                "adt_to_clif: unknown variant tag {:?} at {:?}",
                tag,
                op.location(db)
            );
            return false;
        };

        if field_idx >= variant_layout.field_offsets.len() {
            warn!(
                "adt_to_clif: field index {} out of bounds for variant {:?} ({} fields)",
                field_idx,
                tag,
                variant_layout.field_offsets.len()
            );
            return false;
        }

        let location = op.location(db);
        let offset = (enum_layout.fields_offset + variant_layout.field_offsets[field_idx]) as i32;

        let ref_val = rewriter.operand(0).unwrap_or_else(|| variant_get.r#ref(db));

        // Determine the load type from the enum type definition.
        // The field was stored with its native type, so we must load with the
        // same type rather than the type-erased result type (which may be
        // core.ptr when the IR uses tribute_rt.any).
        let load_ty = adt::get_enum_variants(db, enum_ty)
            .and_then(|variants| {
                variants
                    .iter()
                    .find(|(name, _)| *name == tag)
                    .and_then(|(_, fields)| fields.get(field_idx).copied())
            })
            .map(|field_ty| {
                // Convert the field type to native (e.g., tribute_rt.any → core.ptr).
                // If already native (e.g., core.i32), keep as-is.
                type_converter
                    .convert_type(db, field_ty)
                    .unwrap_or(field_ty)
            })
            .unwrap_or_else(|| {
                rewriter
                    .result_type(db, op, 0)
                    .unwrap_or_else(|| op.results(db)[0])
            });

        let load_op = clif::load(db, location, ref_val, load_ty, offset);
        rewriter.replace_op(load_op.as_operation());
        true
    }
}

/// Pattern for `adt.ref_null(type)` -> `clif.iconst(0)` (null pointer).
///
/// In native code, a null reference is simply a zero-valued pointer.
struct RefNullPattern;

impl<'db> RewritePattern<'db> for RefNullPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        if adt::RefNull::from_operation(db, *op).is_err() {
            return false;
        }

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();

        let iconst_op = clif::iconst(db, location, ptr_ty, 0);
        rewriter.replace_op(iconst_op.as_operation());
        true
    }
}

/// Pattern for `adt.ref_cast(ref, type)` -> identity (no-op on native).
///
/// In native code, all references are opaque pointers, so type casts
/// are purely a type-system concept and require no runtime operation.
struct RefCastPattern;

impl<'db> RewritePattern<'db> for RefCastPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(ref_cast) = adt::RefCast::from_operation(db, *op) else {
            return false;
        };

        // Pass the input operand directly as the result — no runtime work needed.
        let ref_val = rewriter.operand(0).unwrap_or_else(|| ref_cast.r#ref(db));
        rewriter.erase_op(vec![ref_val]);
        true
    }
}

/// Pattern for `adt.ref_is_null(ref)` -> `clif.icmp(eq, ref, 0)`.
///
/// Compares the reference pointer against zero (null).
struct RefIsNullPattern;

impl<'db> RewritePattern<'db> for RefIsNullPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(ref_is_null) = adt::RefIsNull::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        let ref_val = rewriter.operand(0).unwrap_or_else(|| ref_is_null.r#ref(db));

        // Create a null constant to compare against
        let null_op = clif::iconst(db, location, ptr_ty, 0);
        let icmp_op = clif::icmp(
            db,
            location,
            ref_val,
            null_op.result(db),
            i1_ty,
            Symbol::new("eq"),
        );
        rewriter.insert_op(null_op.as_operation());
        rewriter.replace_op(icmp_op.as_operation());
        true
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

    #[salsa::tracked]
    fn do_lower_ref_null(db: &dyn salsa::Database) -> String {
        let module = trunk_ir::parser::parse_test_module(
            db,
            r#"core.module @test {
  %0 = adt.ref_null {type = adt.struct() {name = @Env, fields = [@x]}} : core.ptr
}"#,
        );
        lower_and_print(db, module)
    }

    #[salsa_test]
    fn test_ref_null_to_clif(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_ref_null(db));
    }

    #[salsa::tracked]
    fn do_lower_ref_cast(db: &dyn salsa::Database) -> String {
        let module = trunk_ir::parser::parse_test_module(
            db,
            r#"core.module @test {
  %0 = clif.iconst {value = 100} : core.ptr
  %1 = adt.ref_cast %0 {type = adt.struct() {name = @Env, fields = [@x]}} : core.ptr
}"#,
        );
        lower_and_print(db, module)
    }

    #[salsa_test]
    fn test_ref_cast_to_clif(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_ref_cast(db));
    }

    #[salsa::tracked]
    fn do_lower_ref_is_null(db: &dyn salsa::Database) -> String {
        let module = trunk_ir::parser::parse_test_module(
            db,
            r#"core.module @test {
  %0 = clif.iconst {value = 42} : core.ptr
  %1 = adt.ref_is_null %0 : core.i1
}"#,
        );
        lower_and_print(db, module)
    }

    #[salsa_test]
    fn test_ref_is_null_to_clif(db: &salsa::DatabaseImpl) {
        insta::assert_snapshot!(do_lower_ref_is_null(db));
    }
}
