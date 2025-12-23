//! Lower adt dialect operations to wasm dialect.
//!
//! This pass converts ADT (Algebraic Data Type) operations to wasm operations:
//! - `adt.struct_new` -> `wasm.struct_new`
//! - `adt.struct_get` -> `wasm.struct_get`
//! - `adt.struct_set` -> `wasm.struct_set`
//! - `adt.variant_new` -> `wasm.i32_const` + `wasm.struct_new` (tag + fields)
//! - `adt.variant_tag` -> `wasm.struct_get` (field 0)
//! - `adt.variant_get` -> `wasm.struct_get` (field + 1)
//! - `adt.array_new` -> `wasm.array_new` or `wasm.array_new_default`
//! - `adt.array_get` -> `wasm.array_get`
//! - `adt.array_set` -> `wasm.array_set`
//! - `adt.array_len` -> `wasm.array_len`
//! - `adt.ref_null` -> `wasm.ref_null`
//! - `adt.ref_is_null` -> `wasm.ref_is_null`
//! - `adt.ref_cast` -> `wasm.ref_cast`
//!
//! Note: `adt.string_const` and `adt.bytes_const` are handled by WasmLowerer
//! because they require data segment allocation.
//!
//! Type attributes are preserved and converted to type_idx at emit time.

use trunk_ir::dialect::adt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, idvec};

/// Lower adt dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(StructNewPattern)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .add_pattern(VariantNewPattern)
        .add_pattern(VariantTagPattern)
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
    ) -> RewriteResult<'db> {
        let Ok(_struct_new) = adt::StructNew::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Keep type attribute, emit will convert to type_idx
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
    ) -> RewriteResult<'db> {
        let Ok(struct_get) = adt::StructGet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get field index (can be IntBits or String name)
        let Attribute::IntBits(field_idx) = struct_get.field(db) else {
            #[cfg(debug_assertions)]
            eprintln!("WARNING: StructGetPattern expects IntBits field index, got {:?}", struct_get.field(db));
            return RewriteResult::Unchanged;
        };

        // Build wasm.struct_get with renamed attributes
        let mut builder = Operation::of_name(db, op.location(db), "wasm.struct_get")
            .operands(op.operands(db).clone())
            .results(op.results(db).clone())
            .attr("field_idx", Attribute::IntBits(field_idx));

        // Preserve type attribute (will be converted to type_idx at emit time)
        if let Some(ty_attr) = op.attributes(db).get(&Symbol::new("type")) {
            builder = builder.attr("type", ty_attr.clone());
        }

        RewriteResult::Replace(builder.build())
    }
}

/// Pattern for `adt.struct_set` -> `wasm.struct_set`
struct StructSetPattern;

impl RewritePattern for StructSetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(struct_set) = adt::StructSet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Get field index
        let Attribute::IntBits(field_idx) = struct_set.field(db) else {
            #[cfg(debug_assertions)]
            eprintln!("WARNING: StructSetPattern expects IntBits field index, got {:?}", struct_set.field(db));
            return RewriteResult::Unchanged;
        };

        // Build wasm.struct_set with renamed attributes
        let mut builder = Operation::of_name(db, op.location(db), "wasm.struct_set")
            .operands(op.operands(db).clone())
            .attr("field_idx", Attribute::IntBits(field_idx));

        // Preserve type attribute
        if let Some(ty_attr) = op.attributes(db).get(&Symbol::new("type")) {
            builder = builder.attr("type", ty_attr.clone());
        }

        RewriteResult::Replace(builder.build())
    }
}

/// Pattern for `adt.variant_new` -> `wasm.i32_const` + `wasm.struct_new`
///
/// Variants are represented as structs with tag (field 0) + payload fields.
struct VariantNewPattern;

impl RewritePattern for VariantNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(variant_new) = adt::VariantNew::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let tag_sym = variant_new.tag(db);
        let tag = name_hash_u32(&tag_sym.to_string());

        let i32_ty = core::I32::new(db).as_type();

        // Create tag constant
        let tag_const = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(u64::from(tag)));

        // Build operands: tag + original fields
        let mut variant_fields = idvec![tag_const.result(db)];
        for &operand in variant_new.fields(db).iter() {
            variant_fields.push(operand);
        }

        // Create wasm.struct_new with type attribute preserved
        let struct_new = Operation::of_name(db, location, "wasm.struct_new")
            .operands(variant_fields)
            .results(op.results(db).clone())
            .attr("type", Attribute::Type(variant_new.r#type(db)))
            .build();

        RewriteResult::Expand(vec![tag_const.operation(), struct_new])
    }
}

/// Pattern for `adt.variant_tag` -> `wasm.struct_get` (field 0)
struct VariantTagPattern;

impl RewritePattern for VariantTagPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_variant_tag) = adt::VariantTag::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Tag is always field 0
        let mut struct_get = Operation::of_name(db, location, "wasm.struct_get")
            .operands(op.operands(db).clone())
            .attr("field_idx", Attribute::IntBits(0))
            .results(op.results(db).clone());

        // Preserve type attribute
        if let Some(ty_attr) = op.attributes(db).get(&Symbol::new("type")) {
            struct_get = struct_get.attr("type", ty_attr.clone());
        }

        RewriteResult::Replace(struct_get.build())
    }
}

/// Pattern for `adt.variant_get` -> `wasm.struct_get` (field + 1)
struct VariantGetPattern;

impl RewritePattern for VariantGetPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(variant_get) = adt::VariantGet::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Get field index and add 1 (to skip tag field)
        let Attribute::IntBits(idx) = variant_get.field(db) else {
            #[cfg(debug_assertions)]
            eprintln!("WARNING: VariantGetPattern expects IntBits field index, got {:?}", variant_get.field(db));
            return RewriteResult::Unchanged;
        };
        let field_idx = idx + 1;

        let mut struct_get = Operation::of_name(db, location, "wasm.struct_get")
            .operands(op.operands(db).clone())
            .attr("field_idx", Attribute::IntBits(field_idx))
            .results(op.results(db).clone());

        // Preserve type attribute
        if let Some(ty_attr) = op.attributes(db).get(&Symbol::new("type")) {
            struct_get = struct_get.attr("type", ty_attr.clone());
        }

        RewriteResult::Replace(struct_get.build())
    }
}

/// Pattern for `adt.array_new` -> `wasm.array_new` or `wasm.array_new_default`
struct ArrayNewPattern;

impl RewritePattern for ArrayNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
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

/// Hash a name string to a u32 for variant tags.
fn name_hash_u32(name: &str) -> u32 {
    name.as_bytes()
        .iter()
        .fold(0u32, |h, &b| h.wrapping_mul(31).wrapping_add(u32::from(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_struct_new_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create adt.struct_new with type attribute
        let struct_new = Operation::of_name(db, location, "adt.struct_new")
            .attr("type", Attribute::Type(i32_ty))
            .results(idvec![i32_ty])
            .build();

        let block = Block::new(db, location, idvec![], idvec![struct_new]);
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
