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

use crate::adt_layout::{
    compute_enum_layout_arena, compute_struct_layout_arena, find_variant_layout,
    get_enum_variants_arena,
};
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter,
    PatternApplicator as ArenaPatternApplicator, PatternRewriter as ArenaPatternRewriter,
};
use trunk_ir::arena::types::TypeDataBuilder;
use trunk_ir::ir::Symbol;

/// Lower ADT operations to clif dialect.
///
/// This is a partial lowering: struct access and reference operations are converted.
/// Other ADT operations (struct_new, variant, array) pass through unchanged.
///
/// The `type_converter` parameter is used to determine field sizes for
/// layout computation.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    use trunk_ir::arena::rewrite::ArenaConversionTarget;

    let mut target = ArenaConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_illegal_dialect("adt");

    let applicator = ArenaPatternApplicator::new(type_converter)
        .with_target(target)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .add_pattern(VariantIsPattern)
        .add_pattern(VariantCastPattern)
        .add_pattern(VariantGetPattern)
        .add_pattern(RefNullPattern)
        .add_pattern(RefCastPattern)
        .add_pattern(RefIsNullPattern);
    applicator.apply_partial(ctx, module);
}

fn intern_i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

fn intern_i1_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build())
}

fn intern_ptr_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
}

struct StructGetPattern;

impl ArenaRewritePattern for StructGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_get) = arena_adt::StructGet::from_op(ctx, op) else {
            return false;
        };

        let struct_ty = struct_get.r#type(ctx);
        let field_idx = struct_get.field(ctx) as usize;
        let tc = rewriter.type_converter();

        let Some(layout) = compute_struct_layout_arena(ctx, struct_ty, tc) else {
            warn!("adt_to_clif arena: cannot compute layout for struct_get");
            return false;
        };

        if field_idx >= layout.field_offsets.len() {
            warn!("adt_to_clif arena: field index {} out of bounds", field_idx);
            return false;
        }

        let loc = ctx.op(op).location;
        let Ok(offset) = i32::try_from(layout.field_offsets[field_idx]) else {
            warn!("adt_to_clif arena: field offset exceeds i32");
            return false;
        };
        let ref_val = struct_get.r#ref(ctx);

        // Convert result type through type converter (e.g. tribute_rt.any -> core.ptr)
        let result_types = ctx.op_result_types(op);
        let Some(result_ty) = result_types.first().copied() else {
            warn!("adt_to_clif arena: struct_get has no result type");
            return false;
        };
        let result_ty = tc.convert_type_or_identity(ctx, result_ty);

        let load_op = arena_clif::load(ctx, loc, ref_val, result_ty, offset);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct StructSetPattern;

impl ArenaRewritePattern for StructSetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_set) = arena_adt::StructSet::from_op(ctx, op) else {
            return false;
        };

        let struct_ty = struct_set.r#type(ctx);
        let field_idx = struct_set.field(ctx) as usize;
        let tc = rewriter.type_converter();

        let Some(layout) = compute_struct_layout_arena(ctx, struct_ty, tc) else {
            warn!("adt_to_clif arena: cannot compute layout for struct_set");
            return false;
        };

        if field_idx >= layout.field_offsets.len() {
            return false;
        }

        let loc = ctx.op(op).location;
        let Ok(offset) = i32::try_from(layout.field_offsets[field_idx]) else {
            warn!("adt_to_clif arena: field offset exceeds i32");
            return false;
        };
        let ref_val = struct_set.r#ref(ctx);
        let value_val = struct_set.value(ctx);

        let store_op = arena_clif::store(ctx, loc, value_val, ref_val, offset);
        rewriter.replace_op(store_op.op_ref());
        true
    }
}

struct VariantIsPattern;

impl ArenaRewritePattern for VariantIsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_is) = arena_adt::VariantIs::from_op(ctx, op) else {
            return false;
        };

        let enum_ty = variant_is.r#type(ctx);
        let tag = variant_is.tag(ctx);
        let tc = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout_arena(ctx, enum_ty, tc) else {
            warn!("adt_to_clif arena: cannot compute enum layout for variant_is");
            return false;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!("adt_to_clif arena: unknown variant tag {:?}", tag);
            return false;
        };

        let loc = ctx.op(op).location;
        let i32_ty = intern_i32_type(ctx);
        let i1_ty = intern_i1_type(ctx);
        let ref_val = variant_is.r#ref(ctx);

        // Load tag from payload_ptr + 0
        let tag_load = arena_clif::load(ctx, loc, ref_val, i32_ty, 0);
        let tag_val = tag_load.result(ctx);

        // Compare with expected discriminant
        let expected = arena_clif::iconst(ctx, loc, i32_ty, variant_layout.tag_value as i64);
        let cmp_op = arena_clif::icmp(
            ctx,
            loc,
            tag_val,
            expected.result(ctx),
            i1_ty,
            Symbol::new("eq"),
        );

        rewriter.insert_op(tag_load.op_ref());
        rewriter.insert_op(expected.op_ref());
        rewriter.replace_op(cmp_op.op_ref());
        true
    }
}

struct VariantCastPattern;

impl ArenaRewritePattern for VariantCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_cast) = arena_adt::VariantCast::from_op(ctx, op) else {
            return false;
        };
        let ref_val = variant_cast.r#ref(ctx);
        rewriter.erase_op(vec![ref_val]);
        true
    }
}

struct VariantGetPattern;

impl ArenaRewritePattern for VariantGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_get) = arena_adt::VariantGet::from_op(ctx, op) else {
            return false;
        };

        let enum_ty = variant_get.r#type(ctx);
        let tag = variant_get.tag(ctx);
        let field_idx = variant_get.field(ctx) as usize;
        let tc = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout_arena(ctx, enum_ty, tc) else {
            warn!("adt_to_clif arena: cannot compute enum layout for variant_get");
            return false;
        };

        let Some(variant_layout) = find_variant_layout(&enum_layout, tag) else {
            warn!("adt_to_clif arena: unknown variant tag {:?}", tag);
            return false;
        };

        if field_idx >= variant_layout.field_offsets.len() {
            return false;
        }

        let loc = ctx.op(op).location;
        let Some(total_offset) = enum_layout
            .fields_offset
            .checked_add(variant_layout.field_offsets[field_idx])
        else {
            warn!("adt_to_clif arena: variant field offset overflow");
            return false;
        };
        let Ok(offset) = i32::try_from(total_offset) else {
            warn!("adt_to_clif arena: variant field offset exceeds i32");
            return false;
        };
        let ref_val = variant_get.r#ref(ctx);

        // Determine the load type from the enum type definition.
        // The field was stored with its native type, so we must load with the
        // same type rather than the type-erased result type (which may be
        // tribute_rt.any instead of core.ptr).
        let load_ty = get_enum_variants_arena(ctx, enum_ty)
            .and_then(|variants| {
                variants
                    .iter()
                    .find(|(name, _)| *name == tag)
                    .and_then(|(_, fields)| fields.get(field_idx).copied())
            })
            .map(|field_ty| {
                // Convert the field type to native (e.g., tribute_rt.any -> core.ptr).
                tc.convert_type_or_identity(ctx, field_ty)
            })
            .or_else(|| {
                let result_types = ctx.op_result_types(op);
                let result_ty = result_types.first().copied()?;
                Some(tc.convert_type_or_identity(ctx, result_ty))
            });

        let Some(load_ty) = load_ty else {
            warn!("adt_to_clif arena: variant_get has no result type");
            return false;
        };

        let load_op = arena_clif::load(ctx, loc, ref_val, load_ty, offset);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct RefNullPattern;

impl ArenaRewritePattern for RefNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if arena_adt::RefNull::from_op(ctx, op).is_err() {
            return false;
        }
        let loc = ctx.op(op).location;
        let ptr_ty = intern_ptr_type(ctx);
        let iconst_op = arena_clif::iconst(ctx, loc, ptr_ty, 0);
        rewriter.replace_op(iconst_op.op_ref());
        true
    }
}

struct RefCastPattern;

impl ArenaRewritePattern for RefCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_cast) = arena_adt::RefCast::from_op(ctx, op) else {
            return false;
        };
        let ref_val = ref_cast.r#ref(ctx);
        rewriter.erase_op(vec![ref_val]);
        true
    }
}

struct RefIsNullPattern;

impl ArenaRewritePattern for RefIsNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_is_null) = arena_adt::RefIsNull::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = intern_ptr_type(ctx);
        let i1_ty = intern_i1_type(ctx);
        let ref_val = ref_is_null.r#ref(ctx);

        let null_op = arena_clif::iconst(ctx, loc, ptr_ty, 0);
        let icmp_op = arena_clif::icmp(
            ctx,
            loc,
            ref_val,
            null_op.result(ctx),
            i1_ty,
            Symbol::new("eq"),
        );
        rewriter.insert_op(null_op.op_ref());
        rewriter.replace_op(icmp_op.op_ref());
        true
    }
}

#[cfg(test)]
mod tests {
    use trunk_ir::arena::context::IrContext;
    use trunk_ir::arena::parser::parse_test_module;
    use trunk_ir::arena::printer::print_module;
    use trunk_ir::arena::rewrite::ArenaTypeConverter;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = ArenaTypeConverter::new();
        super::lower(&mut ctx, module, type_converter);
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_struct_get_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i32 {
    %0 = clif.iconst {value = 0} : core.ptr
    %1 = adt.struct_get %0 {field = 1, type = adt.struct(core.i32, core.i32) {fields = [[@x, core.i32], [@y, core.i32]], name = @Point}} : core.i32
    func.return %1
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_struct_set_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.nil {
    %0 = clif.iconst {value = 0} : core.ptr
    %1 = clif.iconst {value = 42} : core.i32
    adt.struct_set %0, %1 {field = 0, type = adt.struct(core.i32, core.i32) {fields = [[@x, core.i32], [@y, core.i32]], name = @Point}}
    func.return
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_ref_null_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %0 = adt.ref_null {type = adt.struct() {name = @Env, fields = [@x]}} : core.ptr
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_ref_cast_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %0 = clif.iconst {value = 100} : core.ptr
    %1 = adt.ref_cast %0 {type = adt.struct() {name = @Env, fields = [@x]}} : core.ptr
    func.return %1
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_ref_is_null_to_clif() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i1 {
    %0 = clif.iconst {value = 42} : core.ptr
    %1 = adt.ref_is_null %0 : core.i1
    func.return %1
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }
}
