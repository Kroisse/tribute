//! Lower adt dialect operations to wasm dialect.
//!
//! This pass converts ADT (Algebraic Data Type) operations to wasm operations.
//!
//! ## Struct Operations
//! - `adt.struct_new` -> `wasm.struct_new`
//! - `adt.struct_get` -> `wasm.struct_get`
//! - `adt.struct_set` -> `wasm.struct_set`
//!
//! ## Variant Operations (WasmGC Subtyping Approach)
//!
//! Variants use WasmGC's nominal subtyping for discrimination instead of
//! explicit tag fields. Each variant gets its own struct type (e.g., `Expr$Add`,
//! `Expr$Num`) that is a subtype of the base enum type. The type itself serves
//! as the discriminant via `ref.test` and `ref.cast` instructions.
//!
//! - `adt.variant_new` -> `wasm.struct_new` with variant-specific type
//!   - Creates a struct with only the variant's fields (no tag field)
//!   - Result type is marked with `is_variant=true` and `variant_tag` attributes
//! - `adt.variant_is` -> `wasm.ref_test` (tests if ref is of specific variant type)
//! - `adt.variant_cast` -> `wasm.ref_cast` (casts to specific variant type)
//! - `adt.variant_get` -> `wasm.struct_get` (direct field access, no offset)
//! - `adt.variant_tag` -> DEPRECATED (issues warning, kept for compatibility)
//!
//! ## Array Operations
//! - `adt.array_new` -> `wasm.array_new` or `wasm.array_new_default`
//! - `adt.array_get` -> `wasm.array_get`
//! - `adt.array_set` -> `wasm.array_set`
//! - `adt.array_len` -> `wasm.array_len`
//!
//! ## Reference Operations
//! - `adt.ref_null` -> `wasm.ref_null`
//! - `adt.ref_is_null` -> `wasm.ref_is_null`
//! - `adt.ref_cast` -> `wasm.ref_cast`
//!
//! Note: `adt.string_const` and `adt.bytes_const` are handled by WasmLowerer
//! because they require data segment allocation.
//!
//! Type information is preserved as `type` attribute when available from the source operation.
//! For operations where the result type is set explicitly (e.g., variant_new, variant_cast),
//! emit can infer type_idx from result/operand types without the attribute.

use tracing::warn;

use tribute_ir::{ModulePathExt as _, dialect::adt};
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, IdVec, Operation, Symbol, Type};

use crate::type_converter::wasm_type_converter;

/// Lower adt dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::with_type_converter(wasm_type_converter())
        .add_pattern(StructNewPattern)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .add_pattern(VariantNewPattern)
        .add_pattern(VariantTagPattern) // deprecated, kept for compatibility
        .add_pattern(VariantIsPattern)
        .add_pattern(VariantCastPattern)
        .add_pattern(VariantGetPattern)
        .add_pattern(ArrayNewPattern)
        .add_pattern(ArrayGetPattern)
        .add_pattern(ArraySetPattern)
        .add_pattern(ArrayLenPattern)
        .add_pattern(RefNullPattern)
        .add_pattern(RefIsNullPattern)
        .add_pattern(RefCastPattern)
        .apply(db, module)
        .module
}

/// Pattern for `adt.struct_new` -> `wasm.struct_new`
struct StructNewPattern;

impl RewritePattern for StructNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_struct_new) = adt::StructNew::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Keep type attribute, emit will convert to type_idx
        // Note: Result type is preserved as-is; emit phase uses type_to_field_type
        // for wasm type conversion. The wasm_type_converter is available for
        // IR-level conversions where type information needs to be transformed.
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("struct_new")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.struct_get` -> `wasm.struct_get`
struct StructGetPattern;

impl RewritePattern for StructGetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(struct_get) = adt::StructGet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get field index (can be IntBits or String name)
        let Attribute::IntBits(field_idx) = struct_get.field(db) else {
            #[cfg(debug_assertions)]
            warn!(
                "StructGetPattern expects IntBits field index, got {:?}",
                struct_get.field(db)
            );
            return RewriteResult::Unchanged;
        };

        // Build wasm.struct_get: change dialect/name and add field_idx attr
        // Preserve type attribute (if present) for cases where emit can't infer from operand types
        // Note: original 'field' attr remains but emit only looks for 'field_idx'
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("struct_get")
            .attr("field_idx", Attribute::IntBits(field_idx))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.struct_set` -> `wasm.struct_set`
struct StructSetPattern;

impl RewritePattern for StructSetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(struct_set) = adt::StructSet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get field index
        let Attribute::IntBits(field_idx) = struct_set.field(db) else {
            #[cfg(debug_assertions)]
            warn!(
                "StructSetPattern expects IntBits field index, got {:?}",
                struct_set.field(db)
            );
            return RewriteResult::Unchanged;
        };

        // Build wasm.struct_set: change dialect/name and add field_idx attr
        // Preserve type attribute (if present) for cases where emit can't infer from operand types
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("struct_set")
            .attr("field_idx", Attribute::IntBits(field_idx))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.variant_new` -> `wasm.struct_new`
///
/// With WasmGC subtyping, variants are represented as separate struct types
/// without an explicit tag field. The type itself serves as the discriminant.
struct VariantNewPattern;

impl RewritePattern for VariantNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(variant_new) = adt::VariantNew::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let tag_sym = variant_new.tag(db);
        let base_type = variant_new.r#type(db);

        // Create variant-specific type: Expr + Add -> Expr$Add
        let variant_type = make_variant_type(db, base_type, tag_sym);

        // Create wasm.struct_new with variant-specific type (no tag field)
        // Result type is the variant-specific type - emit infers type_idx from it
        // Clear original attrs (type, tag) and set only what wasm.struct_new needs
        let struct_new = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("struct_new")
            .attrs(trunk_ir::Attrs::new()) // Clear original type/tag attrs
            .attr("type", Attribute::Type(variant_type))
            .results(IdVec::from(vec![variant_type]))
            .build();

        RewriteResult::Replace(struct_new)
    }
}

/// Create a variant-specific type by combining base type name with variant tag.
/// e.g., base type `adt.Expr` + tag `Add` -> `adt.Expr$Add`
///
/// The resulting type carries attributes for variant detection:
/// - `is_variant = true` - marks this as a variant instance type
/// - `variant_tag = Symbol` - the variant tag (e.g., `Add`, `Num`)
fn make_variant_type<'db>(
    db: &'db dyn salsa::Database,
    base_type: Type<'db>,
    tag: Symbol,
) -> Type<'db> {
    let dialect = base_type.dialect(db);

    // For adt.typeref types, extract the actual type name from the name attribute
    let base_name = if adt::is_typeref(db, base_type) {
        adt::get_type_name(db, base_type)
            .map(|name| name.last_segment())
            .unwrap_or_else(|| base_type.name(db))
    } else {
        base_type.name(db)
    };

    let variant_name = Symbol::from_dynamic(&format!("{}${}", base_name, tag));

    // Convert &[Type] to IdVec<Type>
    let params: IdVec<Type<'db>> = base_type.params(db).iter().copied().collect();

    // Add variant type attributes for proper detection (instead of name-based heuristics)
    let mut attrs = trunk_ir::Attrs::new();
    attrs.insert(adt::ATTR_IS_VARIANT(), Attribute::Bool(true));
    attrs.insert(adt::ATTR_BASE_ENUM(), Attribute::Type(base_type));
    attrs.insert(adt::ATTR_VARIANT_TAG(), Attribute::Symbol(tag));

    Type::new(db, dialect, variant_name, params, attrs)
}

/// Pattern for `adt.variant_tag` -> `wasm.struct_get` (field 0)
/// DEPRECATED: This pattern is kept for compatibility but should not be used
/// with the new WasmGC subtyping approach. Use `adt.variant_is` instead.
struct VariantTagPattern;

impl RewritePattern for VariantTagPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_variant_tag) = adt::VariantTag::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        warn!("adt.variant_tag is deprecated, use adt.variant_is instead");

        // Tag is always field 0
        // Preserve type attribute (if present) for cases where emit can't infer from operand types
        let struct_get = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("struct_get")
            .attr("field_idx", Attribute::IntBits(0))
            .build();

        RewriteResult::Replace(struct_get)
    }
}

/// Pattern for `adt.variant_is` -> `wasm.ref_test`
///
/// Tests if a variant reference is of a specific variant type.
struct VariantIsPattern;

impl RewritePattern for VariantIsPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(variant_is) = adt::VariantIs::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let tag = variant_is.tag(db);

        // Get the enum type - prefer operand type over attribute type
        // (attribute may have unsubstituted type.var, operand has concrete type)
        // Using OpAdaptor.operand_type() handles both OpResult and BlockArg cases
        let enum_type = adaptor
            .operand_type(0)
            .unwrap_or_else(|| variant_is.r#type(db));

        // Create variant-specific type for the ref.test
        let variant_type = make_variant_type(db, enum_type, tag);

        // Create wasm.ref_test with variant-specific type
        // Override 'type' attribute with variant type (original has base enum type)
        let ref_test = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("ref_test")
            .attr("type", Attribute::Type(variant_type))
            .build();

        RewriteResult::Replace(ref_test)
    }
}

/// Pattern for `adt.variant_cast` -> `wasm.ref_cast`
///
/// Casts a variant reference to a specific variant type after pattern matching.
struct VariantCastPattern;

impl RewritePattern for VariantCastPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(variant_cast) = adt::VariantCast::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let tag = variant_cast.tag(db);

        // Get the enum type - prefer operand type over attribute type
        // (attribute may have unsubstituted type.var, operand has concrete type)
        // Using OpAdaptor.operand_type() handles both OpResult and BlockArg cases
        let enum_type = adaptor
            .operand_type(0)
            .unwrap_or_else(|| variant_cast.r#type(db));

        // Create variant-specific type for the ref.cast
        let variant_type = make_variant_type(db, enum_type, tag);

        // Create wasm.ref_cast with variant-specific type
        // Result type is the variant-specific type - emit infers type_idx from it
        // Override 'type' attribute with variant type (original has base enum type)
        let ref_cast = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("ref_cast")
            .attr("type", Attribute::Type(variant_type))
            .results(IdVec::from(vec![variant_type]))
            .build();

        RewriteResult::Replace(ref_cast)
    }
}

/// Pattern for `adt.variant_get` -> `wasm.struct_get`
///
/// With WasmGC subtyping, variant structs no longer have a tag field,
/// so field indices are used directly without offset.
/// The type for struct.get comes from the operand (the variant_cast result).
struct VariantGetPattern;

impl RewritePattern for VariantGetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(variant_get) = adt::VariantGet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get field index directly (no offset - tag field removed in WasmGC subtyping)
        let Attribute::IntBits(field_idx) = variant_get.field(db) else {
            #[cfg(debug_assertions)]
            warn!(
                "VariantGetPattern expects IntBits field index, got {:?}",
                variant_get.field(db)
            );
            return RewriteResult::Unchanged;
        };

        // Infer type from the operand (the cast result has the variant-specific type)
        // Set as type attribute for emit to use; clear any original attrs via fresh builder
        // Using OpAdaptor.operand_type() handles both OpResult and BlockArg cases
        let mut builder = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("struct_get")
            .attr("field_idx", Attribute::IntBits(field_idx));

        // Override type attribute with operand type (original might have base enum type)
        if let Some(ref_type) = adaptor.operand_type(0) {
            builder = builder.attr("type", Attribute::Type(ref_type));
        }

        RewriteResult::Replace(builder.build())
    }
}

/// Pattern for `adt.array_new` -> `wasm.array_new` or `wasm.array_new_default`
struct ArrayNewPattern;

impl RewritePattern for ArrayNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::ARRAY_NEW() {
            return RewriteResult::Unchanged;
        }

        let operands = op.operands(db);

        // If only size operand, use array_new_default; otherwise array_new
        let wasm_name = if operands.len() <= 1 {
            "array_new_default"
        } else {
            "array_new"
        };

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str(wasm_name)
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.array_get` -> `wasm.array_get`
struct ArrayGetPattern;

impl RewritePattern for ArrayGetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::ARRAY_GET() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("array_get")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.array_set` -> `wasm.array_set`
struct ArraySetPattern;

impl RewritePattern for ArraySetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::ARRAY_SET() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("array_set")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.array_len` -> `wasm.array_len`
struct ArrayLenPattern;

impl RewritePattern for ArrayLenPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::ARRAY_LEN() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("array_len")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.ref_null` -> `wasm.ref_null`
struct RefNullPattern;

impl RewritePattern for RefNullPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::REF_NULL() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("ref_null")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.ref_is_null` -> `wasm.ref_is_null`
struct RefIsNullPattern;

impl RewritePattern for RefIsNullPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::REF_IS_NULL() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("ref_is_null")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.ref_cast` -> `wasm.ref_cast`
struct RefCastPattern;

impl RewritePattern for RefCastPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != adt::DIALECT_NAME() || op.name(db) != adt::REF_CAST() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("ref_cast")
            .build();

        RewriteResult::Replace(new_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;
    use trunk_ir::{Block, BlockId, DialectType, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_struct_new_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create adt.struct_new with type attribute
        let struct_new = adt::struct_new(db, location, vec![], i32_ty, i32_ty).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![struct_new]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_and_check_names(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_struct_new_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_struct_new_module(db);
        let op_names = lower_and_check_names(db, module);

        assert!(op_names.iter().any(|n| n == "wasm.struct_new"));
        assert!(!op_names.iter().any(|n| n == "adt.struct_new"));
    }
}
