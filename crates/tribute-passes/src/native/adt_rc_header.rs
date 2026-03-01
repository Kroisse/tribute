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
    ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(struct_new) = adt::StructNew::from_operation(db, *op) else {
            return false;
        };

        let struct_ty = struct_new.r#type(db);
        let type_converter = rewriter.type_converter();

        let Some(layout) = compute_struct_layout(db, struct_ty, type_converter) else {
            warn!(
                "adt_rc_header: cannot compute layout for struct_new type at {:?}",
                op.location(db)
            );
            return false;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let fields = rewriter.operands();

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

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(variant_new) = adt::VariantNew::from_operation(db, *op) else {
            return false;
        };

        let enum_ty = variant_new.r#type(db);
        let tag = variant_new.tag(db);
        let type_converter = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout(db, enum_ty, type_converter) else {
            warn!(
                "adt_rc_header: cannot compute enum layout for variant_new at {:?}",
                op.location(db)
            );
            return false;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!(
                "adt_rc_header: unknown variant tag {:?} at {:?}",
                tag,
                op.location(db)
            );
            return false;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let fields = rewriter.operands();

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

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Arena-based implementation
// ============================================================================

use trunk_ir::adt_layout::{compute_enum_layout_arena, compute_struct_layout_arena};
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::rewrite::rewriter::PatternRewriter as ArenaPatternRewriter;
use trunk_ir::arena::rewrite::type_converter::ArenaTypeConverter;
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, PatternApplicator as ArenaPatternApplicator,
};
use trunk_ir::arena::types::TypeDataBuilder;

/// Lower `adt.struct_new` and `adt.variant_new` operations with RC headers (arena version).
///
/// The `rtti_map` maps arena type refs to their RTTI indices.
pub fn lower_arena(
    ctx: &mut IrContext,
    module: ArenaModule,
    type_converter: ArenaTypeConverter,
    rtti_map: &HashMap<TypeRef, u32>,
) {
    // Pre-intern types
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

    let applicator = ArenaPatternApplicator::new(type_converter)
        .add_pattern(ArenaStructNewPattern {
            rtti_map: rtti_map.clone(),
            ptr_ty,
            i64_ty,
            i32_ty,
        })
        .add_pattern(ArenaVariantNewPattern {
            rtti_map: rtti_map.clone(),
            ptr_ty,
            i64_ty,
            i32_ty,
        });

    applicator.apply_partial(ctx, module);
}

/// Arena pattern for `adt.struct_new(fields...)` -> heap allocation + RC header + stores.
struct ArenaStructNewPattern {
    rtti_map: HashMap<TypeRef, u32>,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for ArenaStructNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_new) = arena_adt::StructNew::from_op(ctx, op) else {
            return false;
        };

        let struct_ty = struct_new.r#type(ctx);
        let tc = rewriter.type_converter();

        let Some(layout) = compute_struct_layout_arena(ctx, struct_ty, tc) else {
            warn!("adt_rc_header arena: cannot compute layout for struct_new type");
            return false;
        };

        let loc = ctx.op(op).location;
        let fields: Vec<_> = ctx.op_operands(op).to_vec();

        let mut ops: Vec<OpRef> = Vec::new();

        // 1. Compute allocation size (payload + RC header)
        let alloc_size = layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = arena_clif::iconst(ctx, loc, self.i64_ty, alloc_size as i64);
        let size_val = size_op.result(ctx);
        ops.push(size_op.op_ref());

        // 2. Call __tribute_alloc
        let call_op = arena_clif::call(ctx, loc, [size_val], self.ptr_ty, Symbol::new(ALLOC_FN));
        let raw_ptr = call_op.result(ctx);
        ops.push(call_op.op_ref());

        // 3. Store refcount = 1 at raw_ptr + 0
        let rc_one = arena_clif::iconst(ctx, loc, self.i32_ty, 1);
        let rc_one_val = rc_one.result(ctx);
        ops.push(rc_one.op_ref());
        let store_rc = arena_clif::store(ctx, loc, rc_one_val, raw_ptr, 0);
        ops.push(store_rc.op_ref());

        // 4. Store rtti_idx at raw_ptr + 4
        let rtti_idx = self.rtti_map.get(&struct_ty).copied().unwrap_or(0) as i64;
        let rtti_val = arena_clif::iconst(ctx, loc, self.i32_ty, rtti_idx);
        let rtti_val_v = rtti_val.result(ctx);
        ops.push(rtti_val.op_ref());
        let store_rtti = arena_clif::store(ctx, loc, rtti_val_v, raw_ptr, 4);
        ops.push(store_rtti.op_ref());

        // 5. Compute payload pointer = raw_ptr + 8
        let hdr_size = arena_clif::iconst(ctx, loc, self.i64_ty, RC_HEADER_SIZE as i64);
        let hdr_size_val = hdr_size.result(ctx);
        ops.push(hdr_size.op_ref());
        let payload_ptr = arena_clif::iadd(ctx, loc, raw_ptr, hdr_size_val, self.ptr_ty);
        let payload_val = payload_ptr.result(ctx);
        ops.push(payload_ptr.op_ref());

        // 6. Store each field at its computed offset (relative to payload)
        for (i, &field_val) in fields.iter().enumerate() {
            if i < layout.field_offsets.len() {
                let offset = layout.field_offsets[i] as i32;
                let store_op = arena_clif::store(ctx, loc, field_val, payload_val, offset);
                ops.push(store_op.op_ref());
            }
        }

        // 7. Identity pass-through
        let zero_op = arena_clif::iconst(ctx, loc, self.i64_ty, 0);
        let zero_val = zero_op.result(ctx);
        ops.push(zero_op.op_ref());

        let identity_op = arena_clif::iadd(ctx, loc, payload_val, zero_val, self.ptr_ty);
        ops.push(identity_op.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

/// Arena pattern for `adt.variant_new(fields...)` -> heap allocation + RC header + tag + stores.
struct ArenaVariantNewPattern {
    rtti_map: HashMap<TypeRef, u32>,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for ArenaVariantNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_new) = arena_adt::VariantNew::from_op(ctx, op) else {
            return false;
        };

        let enum_ty = variant_new.r#type(ctx);
        let tag = variant_new.tag(ctx);
        let tc = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout_arena(ctx, enum_ty, tc) else {
            warn!("adt_rc_header arena: cannot compute enum layout for variant_new");
            return false;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!("adt_rc_header arena: unknown variant tag {:?}", tag);
            return false;
        };

        let loc = ctx.op(op).location;
        let fields: Vec<_> = ctx.op_operands(op).to_vec();

        let mut ops: Vec<OpRef> = Vec::new();

        // 1. Compute allocation size (payload + RC header)
        let alloc_size = enum_layout.total_size as u64 + RC_HEADER_SIZE;
        let size_op = arena_clif::iconst(ctx, loc, self.i64_ty, alloc_size as i64);
        let size_val = size_op.result(ctx);
        ops.push(size_op.op_ref());

        // 2. Call __tribute_alloc
        let call_op = arena_clif::call(ctx, loc, [size_val], self.ptr_ty, Symbol::new(ALLOC_FN));
        let raw_ptr = call_op.result(ctx);
        ops.push(call_op.op_ref());

        // 3. Store refcount = 1 at raw_ptr + 0
        let rc_one = arena_clif::iconst(ctx, loc, self.i32_ty, 1);
        let rc_one_val = rc_one.result(ctx);
        ops.push(rc_one.op_ref());
        let store_rc = arena_clif::store(ctx, loc, rc_one_val, raw_ptr, 0);
        ops.push(store_rc.op_ref());

        // 4. Store rtti_idx at raw_ptr + 4
        let rtti_idx = self.rtti_map.get(&enum_ty).copied().unwrap_or(0) as i64;
        let rtti_val = arena_clif::iconst(ctx, loc, self.i32_ty, rtti_idx);
        let rtti_val_v = rtti_val.result(ctx);
        ops.push(rtti_val.op_ref());
        let store_rtti = arena_clif::store(ctx, loc, rtti_val_v, raw_ptr, 4);
        ops.push(store_rtti.op_ref());

        // 5. Compute payload pointer = raw_ptr + 8
        let hdr_size = arena_clif::iconst(ctx, loc, self.i64_ty, RC_HEADER_SIZE as i64);
        let hdr_size_val = hdr_size.result(ctx);
        ops.push(hdr_size.op_ref());
        let payload_ptr = arena_clif::iadd(ctx, loc, raw_ptr, hdr_size_val, self.ptr_ty);
        let payload_val = payload_ptr.result(ctx);
        ops.push(payload_ptr.op_ref());

        // 6. Store tag at payload + 0
        let tag_const = arena_clif::iconst(ctx, loc, self.i32_ty, variant_layout.tag_value as i64);
        let tag_val = tag_const.result(ctx);
        ops.push(tag_const.op_ref());
        let store_tag = arena_clif::store(ctx, loc, tag_val, payload_val, 0);
        ops.push(store_tag.op_ref());

        // 7. Store each field at its computed offset (relative to payload + fields_offset)
        for (i, &field_val) in fields.iter().enumerate() {
            if i < variant_layout.field_offsets.len() {
                let offset = (enum_layout.fields_offset + variant_layout.field_offsets[i]) as i32;
                let store_op = arena_clif::store(ctx, loc, field_val, payload_val, offset);
                ops.push(store_op.op_ref());
            }
        }

        // 8. Identity pass-through
        let zero_op = arena_clif::iconst(ctx, loc, self.i64_ty, 0);
        let zero_val = zero_op.result(ctx);
        ops.push(zero_op.op_ref());

        let identity_op = arena_clif::iadd(ctx, loc, payload_val, zero_val, self.ptr_ty);
        ops.push(identity_op.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
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
