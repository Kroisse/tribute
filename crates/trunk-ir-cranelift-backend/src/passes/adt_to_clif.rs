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
    compute_enum_layout, compute_struct_layout, find_variant_layout, get_enum_variants,
};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;

/// Lower ADT operations to clif dialect.
///
/// This is a partial lowering: struct access and reference operations are converted.
/// Other ADT operations (struct_new, variant, array) pass through unchanged.
///
/// The `type_converter` parameter is used to determine field sizes for
/// layout computation.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .add_pattern(VariantIsPattern)
        .add_pattern(VariantCastPattern)
        .add_pattern(VariantGetPattern)
        .add_pattern(RefNullPattern)
        .add_pattern(RefCastPattern)
        .add_pattern(RefIsNullPattern)
        .with_target(adt_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "adt-to-clif")?;
    Ok(())
}

fn adt_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("adt")
        .legal_op("adt", "struct_new")
        .legal_op("adt", "variant_new")
        .legal_op("adt", "array_new")
        .legal_op("adt", "array_get")
        .legal_op("adt", "array_set")
        .legal_op("adt", "array_len")
        .legal_op("adt", "string_const")
        .legal_op("adt", "bytes_const")
}

fn intern_i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

fn intern_i1_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build())
}

struct StructGetPattern;

impl RewritePattern for StructGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_get) = adt::StructGet::from_op(ctx, op) else {
            return false;
        };

        let struct_ty = struct_get.r#type(ctx);
        let field_idx = struct_get.field(ctx) as usize;
        let tc = rewriter.type_converter();

        let Some(layout) = compute_struct_layout(ctx, struct_ty, tc) else {
            warn!("adt_to_clif arena: cannot compute layout for struct_get");
            return false;
        };

        if field_idx >= layout.field_offsets.len() {
            warn!("adt_to_clif arena: field index {} out of bounds", field_idx);
            return false;
        }

        let loc = ctx.op(op).location;
        let offset =
            i32::try_from(layout.field_offsets[field_idx]).expect("field offset exceeds i32");
        let ref_val = struct_get.r#ref(ctx);

        // Convert result type through type converter (e.g. tribute_rt.any -> core.ptr)
        let result_types = ctx.op_result_types(op);
        let Some(result_ty) = result_types.first().copied() else {
            warn!("adt_to_clif arena: struct_get has no result type");
            return false;
        };
        let result_ty = tc.convert_type_or_identity(ctx, result_ty);

        let load_op = clif::load(ctx, loc, ref_val, result_ty, offset);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct StructSetPattern;

impl RewritePattern for StructSetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_set) = adt::StructSet::from_op(ctx, op) else {
            return false;
        };

        let struct_ty = struct_set.r#type(ctx);
        let field_idx = struct_set.field(ctx) as usize;
        let tc = rewriter.type_converter();

        let Some(layout) = compute_struct_layout(ctx, struct_ty, tc) else {
            warn!("adt_to_clif arena: cannot compute layout for struct_set");
            return false;
        };

        if field_idx >= layout.field_offsets.len() {
            return false;
        }

        let loc = ctx.op(op).location;
        let offset =
            i32::try_from(layout.field_offsets[field_idx]).expect("field offset exceeds i32");
        let ref_val = struct_set.r#ref(ctx);
        let value_val = struct_set.value(ctx);

        let store_op = clif::store(ctx, loc, value_val, ref_val, offset);
        rewriter.replace_op(store_op.op_ref());
        true
    }
}

struct VariantIsPattern;

impl RewritePattern for VariantIsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_is) = adt::VariantIs::from_op(ctx, op) else {
            return false;
        };

        let enum_ty = variant_is.r#type(ctx);
        let tag = variant_is.tag(ctx);
        let tc = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout(ctx, enum_ty, tc) else {
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
        let tag_load = clif::load(ctx, loc, ref_val, i32_ty, 0);
        let tag_val = tag_load.result(ctx);

        // Compare with expected discriminant
        let expected = clif::iconst(ctx, loc, i32_ty, variant_layout.tag_value as i64);
        let cmp_op = clif::icmp(
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

impl RewritePattern for VariantCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_cast) = adt::VariantCast::from_op(ctx, op) else {
            return false;
        };
        let ref_val = variant_cast.r#ref(ctx);
        rewriter.erase_op(vec![ref_val]);
        true
    }
}

struct VariantGetPattern;

impl RewritePattern for VariantGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_get) = adt::VariantGet::from_op(ctx, op) else {
            return false;
        };

        let enum_ty = variant_get.r#type(ctx);
        let tag = variant_get.tag(ctx);
        let field_idx = variant_get.field(ctx) as usize;
        let tc = rewriter.type_converter();

        let Some(enum_layout) = compute_enum_layout(ctx, enum_ty, tc) else {
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
        let offset =
            i32::try_from(enum_layout.fields_offset + variant_layout.field_offsets[field_idx])
                .expect("field offset exceeds i32");
        let ref_val = variant_get.r#ref(ctx);

        // Determine the load type from the enum type definition.
        // The field was stored with its native type, so we must load with the
        // same type rather than the type-erased result type (which may be
        // tribute_rt.any instead of core.ptr).
        let load_ty = get_enum_variants(ctx, enum_ty)
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

        let load_op = clif::load(ctx, loc, ref_val, load_ty, offset);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct RefNullPattern;

impl RewritePattern for RefNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if adt::RefNull::from_op(ctx, op).is_err() {
            return false;
        }
        let loc = ctx.op(op).location;
        let ptr_ty = core::ptr(ctx).as_type_ref();
        let iconst_op = clif::iconst(ctx, loc, ptr_ty, 0);
        rewriter.replace_op(iconst_op.op_ref());
        true
    }
}

struct RefCastPattern;

impl RewritePattern for RefCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_cast) = adt::RefCast::from_op(ctx, op) else {
            return false;
        };
        let ref_val = ref_cast.r#ref(ctx);
        rewriter.erase_op(vec![ref_val]);
        true
    }
}

struct RefIsNullPattern;

impl RewritePattern for RefIsNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_is_null) = adt::RefIsNull::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = core::ptr(ctx).as_type_ref();
        let Some(result_ty) = ctx.op_result_types(op).first().copied() else {
            return false;
        };
        let result_ty = rewriter
            .type_converter()
            .convert_type_or_identity(ctx, result_ty);
        let result_data = ctx.types.get(result_ty);
        let can_hold_i8 = result_data.dialect == Symbol::new("core")
            && matches!(
                result_data.name.to_string().as_str(),
                "i8" | "i16" | "i32" | "i64"
            );
        if !can_hold_i8 {
            return false;
        }
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
        let ref_val = ref_is_null.r#ref(ctx);

        let null_op = clif::iconst(ctx, loc, ptr_ty, 0);
        let icmp_op = clif::icmp(
            ctx,
            loc,
            ref_val,
            null_op.result(ctx),
            i8_ty,
            Symbol::new("eq"),
        );
        rewriter.insert_op(null_op.op_ref());
        if result_ty == i8_ty {
            rewriter.replace_op(icmp_op.op_ref());
        } else {
            rewriter.insert_op(icmp_op.op_ref());
            let extended = clif::uextend(ctx, loc, icmp_op.result(ctx), result_ty);
            rewriter.replace_op(extended.op_ref());
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::TypeConverter;

    fn run_pass_result(ir: &str) -> Result<String, trunk_ir::rewrite::ConversionError> {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = TypeConverter::new();
        super::lower(&mut ctx, module, type_converter)?;
        Ok(print_module(&ctx, module.op()))
    }

    fn run_pass(ir: &str) -> String {
        run_pass_result(ir).unwrap()
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
  func.func @test_fn() -> core.i32 {
    %0 = clif.iconst {value = 42} : core.ptr
    %1 = adt.ref_is_null %0 : core.i32
    func.return %1
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_ref_is_null_i8_needs_no_extension() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i8 {
    %0 = clif.iconst {value = 42} : core.ptr
    %1 = adt.ref_is_null %0 : core.i8
    func.return %1
  }
}"#,
        );
        assert!(result.contains("clif.icmp"));
        assert!(!result.contains("clif.uextend"));
    }

    #[test]
    fn test_ref_is_null_rejects_unlowered_i1_result() {
        let result = run_pass_result(
            r#"core.module @test {
  func.func @test_fn() -> core.i1 {
    %0 = clif.iconst {value = 42} : core.ptr
    %1 = adt.ref_is_null %0 : core.i1
    func.return %1
  }
}"#,
        );
        assert!(result.is_err());
    }
}
