//! Lower ADT dialect struct operations to clif dialect.
//!
//! This pass converts struct-related ADT operations to their Cranelift equivalents:
//! - `adt.struct_new(fields...)` -> `clif.call @tribute_rt_alloc` + `clif.store` per field
//! - `adt.struct_get(ref, field)` -> `clif.load(ref + offset)`
//! - `adt.struct_set(ref, value, field)` -> `clif.store(value, ref + offset)`
//!
//! ## Allocation strategy
//!
//! Struct allocation uses `tribute_rt_alloc(size: i64) -> ptr`, an imported runtime
//! function. Phase 1 is a simple malloc wrapper; Phase 3 will add RC headers.
//!
//! ## Limitations
//!
//! - Only struct operations are lowered (variant/array/ref ops are left unchanged)

use tracing::warn;

use crate::adt_layout::compute_struct_layout;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, clif, core};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{DialectOp, DialectType, Operation, Symbol};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "tribute_rt_alloc";

/// Lower ADT struct operations to clif dialect.
///
/// This is a partial lowering: only struct_new, struct_get, and struct_set are converted.
/// Other ADT operations (variant, array, ref) pass through unchanged.
///
/// The `type_converter` parameter is used to determine field sizes for
/// layout computation.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Result<Module<'db>, ConversionError> {
    // Partial lowering: mark adt as illegal so patterns are applied, but use
    // apply_partial (no verification) since only struct ops are handled here.
    // Remaining adt ops (variant, array, ref) will be handled by future passes.
    let target = ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("adt");

    Ok(PatternApplicator::new(type_converter)
        .add_pattern(StructNewPattern)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .apply_partial(db, module, target)
        .module)
}

/// Pattern for `adt.struct_new(fields...)` -> heap allocation + stores.
///
/// Generates:
/// ```text
/// %size   = clif.iconst(layout.total_size)
/// %ptr    = clif.call @tribute_rt_alloc(%size)
/// clif.store(%field0, %ptr, offset=0)
/// clif.store(%field1, %ptr, offset=4)
/// ...
/// %result = clif.iadd(%ptr, %zero)   // identity (last op for result mapping)
/// ```
struct StructNewPattern;

impl<'db> RewritePattern<'db> for StructNewPattern {
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
        let fields = adaptor.operands();

        let mut ops: Vec<Operation<'db>> = Vec::new();

        // 1. Compute allocation size
        let size_op = clif::iconst(db, location, i64_ty, layout.total_size as i64);
        let size_val = size_op.result(db);
        ops.push(size_op.as_operation());

        // 2. Call tribute_rt_alloc
        let call_op = clif::call(db, location, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
        let ptr_val = call_op.result(db);
        ops.push(call_op.as_operation());

        // 3. Store each field at its computed offset
        for (i, &field_val) in fields.iter().enumerate() {
            if i < layout.field_offsets.len() {
                let offset = layout.field_offsets[i] as i32;
                let store_op = clif::store(db, location, field_val, ptr_val, offset);
                ops.push(store_op.as_operation());
            }
        }

        // 4. Identity pass-through so the last op produces the result.
        //    Cranelift will optimize away iadd(ptr, 0).
        let zero_op = clif::iconst(db, location, i64_ty, 0);
        let zero_val = zero_op.result(db);
        ops.push(zero_op.as_operation());

        let identity_op = clif::iadd(db, location, ptr_val, zero_val, ptr_ty);
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
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{
        Attribute, Block, BlockId, DialectOp, DialectType, Location, PathId, Region, Span, idvec,
    };

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn format_module_ops(db: &dyn salsa::Database, module: &Module<'_>) -> String {
        let body = module.body(db);
        let ops = &body.blocks(db)[0].operations(db);
        ops.iter()
            .map(|op| format_op(db, op))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_op(db: &dyn salsa::Database, op: &Operation<'_>) -> String {
        let name = op.full_name(db);
        let operands = op.operands(db);
        let results = op.results(db);
        let attrs = op.attributes(db);

        let mut parts = vec![name];

        for (key, attr) in attrs.iter() {
            match attr {
                Attribute::IntBits(v) if *key == "value" || *key == "offset" => {
                    parts.push(format!("{}={}", key, *v as i64));
                }
                Attribute::Symbol(s) if *key == "callee" => {
                    parts.push(format!("callee={}", s));
                }
                _ => {}
            }
        }

        if !operands.is_empty() {
            parts.push(format!("operands={}", operands.len()));
        }

        if !results.is_empty() {
            let result_types: Vec<_> = results.iter().map(|t| t.name(db).to_string()).collect();
            parts.push(format!("-> {}", result_types.join(", ")));
        }

        parts.join(" ")
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
    fn format_lowered_module<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> String {
        let lowered = lower(db, module, test_converter()).expect("conversion should succeed");
        format_module_ops(db, &lowered)
    }

    #[salsa_test]
    fn test_struct_new_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_struct_new_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=10 -> i32
        clif.iconst value=20 -> i32
        clif.iconst value=8 -> i64
        clif.call callee=tribute_rt_alloc operands=1 -> ptr
        clif.store offset=0 operands=2
        clif.store offset=4 operands=2
        clif.iconst value=0 -> i64
        clif.iadd operands=2 -> ptr
        ");
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

        // Simulate a struct reference (use iconst as placeholder)
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

    #[salsa_test]
    fn test_struct_get_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_struct_get_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=0 -> ptr
        clif.load offset=4 operands=1 -> i32
        ");
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

    #[salsa_test]
    fn test_struct_new_empty(db: &salsa::DatabaseImpl) {
        let module = make_struct_new_empty_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=0 -> i64
        clif.call callee=tribute_rt_alloc operands=1 -> ptr
        clif.iconst value=0 -> i64
        clif.iadd operands=2 -> ptr
        ");
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

        // Simulate: ref = some pointer, value = 42, set field 0
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

    #[salsa_test]
    fn test_struct_set_to_clif(db: &salsa::DatabaseImpl) {
        let module = make_struct_set_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r"
        clif.iconst value=0 -> ptr
        clif.iconst value=42 -> i32
        clif.store offset=0 operands=2
        ");
    }
}
