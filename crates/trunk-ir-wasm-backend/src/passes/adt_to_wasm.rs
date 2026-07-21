//! Lower adt dialect operations to wasm dialect (arena IR).
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
//! GC operations first lower to `wasm_gc`, which preserves semantic `TypeRef`
//! identity. A module-wide pass assigns the explicit indices required by
//! indexed `wasm` operations before emission.

use tracing::warn;
use trunk_ir::Symbol;
use trunk_ir::adt_layout::get_enum_variants;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc as wasm_gc_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};

/// Prefer a substituted operand type only when it still carries ADT identity.
/// Erased representations such as `anyref` cannot identify the enum whose
/// variant is being tested or cast.
fn resolved_enum_type(ctx: &IrContext, operand_ty: TypeRef, attr_ty: TypeRef) -> TypeRef {
    let operand = ctx.types.get(operand_ty);
    if operand.dialect == Symbol::new("adt")
        && (operand.name == Symbol::new("enum") || operand.name == Symbol::new("typeref"))
    {
        operand_ty
    } else {
        attr_ty
    }
}

/// Lower adt dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(StructNewPattern)
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .add_pattern(VariantNewPattern)
        .add_pattern(VariantIsPattern)
        .add_pattern(VariantCastPattern)
        .add_pattern(VariantGetPattern)
        .add_pattern(ArrayNewPattern)
        .add_pattern(ArrayGetPattern)
        .add_pattern(ArraySetPattern)
        .add_pattern(ArrayLenPattern)
        .add_pattern(RefNullPattern)
        .add_pattern(RefIsNullPattern)
        .add_pattern(RefCastPattern);
    applicator.apply_partial(ctx, module);
}

/// Pattern for `adt.struct_new` -> `wasm.struct_new`
struct StructNewPattern;

impl RewritePattern for StructNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_new) = adt::StructNew::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let fields: Vec<_> = struct_new.fields(ctx).to_vec();
        let result_ty = struct_new.result_ty(ctx);

        // Keep type attribute, emit will convert to type_idx
        // Note: Result type is preserved as-is; emit phase uses type_to_field_type
        // for wasm type conversion.
        let new_op =
            wasm_gc_dialect::struct_new(ctx, loc, fields, result_ty, struct_new.r#type(ctx));
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.struct_get` -> `wasm.struct_get`
///
/// Type casting from abstract types (structref/anyref) to concrete struct types
/// is handled by the emit stage in `struct_handlers.rs`, not here.
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

        let loc = ctx.op(op).location;
        let ref_val = struct_get.r#ref(ctx);
        let result_ty = struct_get.result_ty(ctx);
        let field_idx = struct_get.field(ctx);

        // Build wasm.struct_get: just change dialect/name
        // field attribute is already u32, emit will read it directly
        let new_op = wasm_gc_dialect::struct_get(
            ctx,
            loc,
            ref_val,
            result_ty,
            struct_get.r#type(ctx),
            field_idx,
        );
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.struct_set` -> `wasm.struct_set`
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

        let loc = ctx.op(op).location;
        let ref_val = struct_set.r#ref(ctx);
        let value = struct_set.value(ctx);
        let field_idx = struct_set.field(ctx);

        // Build wasm.struct_set: just change dialect/name
        // field attribute is already u32, emit will read it directly
        let new_op = wasm_gc_dialect::struct_set(
            ctx,
            loc,
            ref_val,
            value,
            struct_set.r#type(ctx),
            field_idx,
        );
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_new` -> `wasm.struct_new`
///
/// With WasmGC subtyping, variants are represented as separate struct types
/// without an explicit tag field. The type itself serves as the discriminant.
struct VariantNewPattern;

impl RewritePattern for VariantNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_new) = adt::VariantNew::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let tag_sym = variant_new.tag(ctx);
        let base_type = variant_new.r#type(ctx);
        let fields: Vec<_> = variant_new.fields(ctx).to_vec();

        // Create variant-specific type: Expr + Add -> Expr$Add
        let variant_type = make_variant_type(ctx, base_type, tag_sym);

        // Create wasm_gc.struct_new with variant-specific type (no tag field).
        let new_op = wasm_gc_dialect::struct_new(ctx, loc, fields, variant_type, variant_type);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Create a variant-specific type by combining base type name with variant tag.
/// e.g., base type `adt.Expr` + tag `Add` -> `adt.Expr$Add`
///
/// The resulting type carries attributes for variant detection:
/// - `is_variant = true` - marks this as a variant instance type
/// - `variant_tag = Symbol` - the variant tag (e.g., `Add`, `Num`)
/// - `base_enum = Type` - the base enum type
fn make_variant_type(ctx: &mut IrContext, base_type: TypeRef, tag: Symbol) -> TypeRef {
    let base_data = ctx.types.get(base_type);
    let dialect = base_data.dialect;

    // For adt.typeref types, extract the actual type name from the name attribute
    // Use full path to avoid collisions (e.g., mod_a::Expr$Add vs mod_b::Expr$Add)
    let is_typeref =
        base_data.dialect == Symbol::new("adt") && base_data.name == Symbol::new("typeref");

    let base_name = if is_typeref {
        // Get the name attribute from the typeref type
        base_data
            .attrs
            .get(&Symbol::new("name"))
            .and_then(|a| match a {
                Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
            .unwrap_or(base_data.name)
    } else {
        base_data.name
    };

    let variant_name = Symbol::from_dynamic(&format!("{base_name}${tag}"));

    // Copy params from base type
    let params: Vec<TypeRef> = base_data.params.to_vec();

    // Add variant type attributes for proper detection (instead of name-based heuristics)
    let builder = TypeDataBuilder::new(dialect, variant_name)
        .params(params)
        .attr(Symbol::new("is_variant"), Attribute::Bool(true))
        .attr(Symbol::new("base_enum"), Attribute::Type(base_type))
        .attr(Symbol::new("variant_tag"), Attribute::Symbol(tag));

    ctx.types.intern(builder.build())
}

/// Pattern for `adt.variant_is` -> `wasm.ref_test`
///
/// Tests if a variant reference is of a specific variant type.
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

        let loc = ctx.op(op).location;
        let tag = variant_is.tag(ctx);
        let ref_val = variant_is.r#ref(ctx);
        let result_ty = variant_is.result_ty(ctx);

        // Prefer a substituted operand type only while it retains ADT identity.
        let operand_ty = ctx.value_ty(ref_val);
        let enum_type = resolved_enum_type(ctx, operand_ty, variant_is.r#type(ctx));

        // Create variant-specific type for the ref.test
        let variant_type = make_variant_type(ctx, enum_type, tag);

        // Create wasm.ref_test with variant-specific type
        let new_op = wasm_gc_dialect::ref_test(ctx, loc, ref_val, result_ty, variant_type);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_cast` -> `wasm.ref_cast`
///
/// Casts a variant reference to a specific variant type after pattern matching.
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

        let loc = ctx.op(op).location;
        let tag = variant_cast.tag(ctx);
        let ref_val = variant_cast.r#ref(ctx);

        // Prefer a substituted operand type only while it retains ADT identity.
        let operand_ty = ctx.value_ty(ref_val);
        let attr_type = variant_cast.r#type(ctx);
        let enum_type = resolved_enum_type(ctx, operand_ty, attr_type);

        // Create variant-specific type for the ref.cast
        let variant_type = make_variant_type(ctx, enum_type, tag);

        // Create wasm_gc.ref_cast with variant-specific type.
        let new_op = wasm_gc_dialect::ref_cast(ctx, loc, ref_val, variant_type, variant_type);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_get` -> `wasm.struct_get`
///
/// With WasmGC subtyping, variant structs no longer have a tag field,
/// so field indices are used directly without offset.
/// The type for struct.get comes from the operand (the variant_cast result).
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

        let loc = ctx.op(op).location;
        let ref_val = variant_get.r#ref(ctx);
        let field_idx = variant_get.field(ctx);
        let declared_field_ty = get_enum_variants(ctx, variant_get.r#type(ctx))
            .and_then(|variants| {
                variants
                    .into_iter()
                    .find(|(tag, _)| *tag == variant_get.tag(ctx))
            })
            .and_then(|(_, fields)| fields.get(field_idx as usize).copied());
        // String::Leaf has the canonical core.bytes layout even though frontend
        // pattern extraction is temporarily erased to anyref. Keep other fields
        // on the normal type-converter path.
        let result_ty = declared_field_ty
            .filter(|ty| {
                let data = ctx.types.get(*ty);
                data.dialect == Symbol::new("core") && data.name == Symbol::new("bytes")
            })
            .unwrap_or_else(|| variant_get.result_ty(ctx));
        let variant_type = make_variant_type(ctx, variant_get.r#type(ctx), variant_get.tag(ctx));

        // Infer type from the operand (the cast result has the variant-specific type)
        // field attribute is already u32 and will be used directly
        let new_op =
            wasm_gc_dialect::struct_get(ctx, loc, ref_val, result_ty, variant_type, field_idx);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.array_new` -> `wasm.array_new` or `wasm.array_new_default`
struct ArrayNewPattern;

impl RewritePattern for ArrayNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_new) = adt::ArrayNew::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = array_new.result_ty(ctx);
        let array_ty = array_new.r#type(ctx);
        let operands = array_new.elements(ctx).to_vec();

        match operands.len() {
            0 => {
                warn!("adt.array_new with no operands");
                return false;
            }
            1 => {
                // Only size operand -> array_new_default
                let new_op =
                    wasm_gc_dialect::array_new_default(ctx, loc, operands[0], result_ty, array_ty);
                rewriter.replace_op(new_op.op_ref());
            }
            2 => {
                // size + init value -> array_new
                let new_op = wasm_gc_dialect::array_new(
                    ctx,
                    loc,
                    operands[0],
                    operands[1],
                    result_ty,
                    array_ty,
                );
                rewriter.replace_op(new_op.op_ref());
            }
            n => {
                warn!("adt.array_new with unexpected {n} operands, expected 1 or 2");
                return false;
            }
        }

        true
    }
}

/// Pattern for `adt.array_get` -> `wasm.array_get`
struct ArrayGetPattern;

impl RewritePattern for ArrayGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_get) = adt::ArrayGet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = array_get.r#ref(ctx);
        let index = array_get.index(ctx);
        let result_ty = array_get.result_ty(ctx);

        let array_ty = ctx.value_ty(ref_val);
        let new_op = wasm_gc_dialect::array_get(ctx, loc, ref_val, index, result_ty, array_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.array_set` -> `wasm.array_set`
struct ArraySetPattern;

impl RewritePattern for ArraySetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_set) = adt::ArraySet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = array_set.r#ref(ctx);
        let index = array_set.index(ctx);
        let value = array_set.value(ctx);

        let array_ty = ctx.value_ty(ref_val);
        let new_op = wasm_gc_dialect::array_set(ctx, loc, ref_val, index, value, array_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.array_len` -> `wasm.array_len`
struct ArrayLenPattern;

impl RewritePattern for ArrayLenPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_len) = adt::ArrayLen::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = array_len.r#ref(ctx);
        let result_ty = array_len.result_ty(ctx);

        let new_op = wasm_dialect::array_len(ctx, loc, ref_val, result_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.ref_null` -> `wasm.ref_null`
struct RefNullPattern;

impl RewritePattern for RefNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_null) = adt::RefNull::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = ref_null.result_ty(ctx);

        let adt_type = ref_null.r#type(ctx);

        let new_op = wasm_gc_dialect::ref_null(ctx, loc, result_ty, adt_type);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.ref_is_null` -> `wasm.ref_is_null`
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
        let ref_val = ref_is_null.r#ref(ctx);
        let result_ty = ref_is_null.result_ty(ctx);

        let new_op = wasm_dialect::ref_is_null(ctx, loc, ref_val, result_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.ref_cast` -> `wasm.ref_cast`
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

        let loc = ctx.op(op).location;
        let ref_val = ref_cast.r#ref(ctx);
        let result_ty = ref_cast.result_ty(ctx);

        // Get the target type from the adt type attribute
        let adt_type = ref_cast.r#type(ctx);

        let new_op = wasm_gc_dialect::ref_cast(ctx, loc, ref_val, result_ty, adt_type);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn resolved_enum_type_rejects_erased_operands() {
        let mut ctx = IrContext::new();
        let enum_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("enum")).build());
        let typeref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref")).build());
        let erased_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref")).build());

        assert_eq!(resolved_enum_type(&ctx, enum_ty, typeref_ty), enum_ty);
        assert_eq!(resolved_enum_type(&ctx, typeref_ty, enum_ty), typeref_ty);
        assert_eq!(resolved_enum_type(&ctx, erased_ty, enum_ty), enum_ty);
    }

    #[test]
    fn lowers_struct_variant_array_and_reference_operations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct() {fields = [[@value, core.i32]], name = @S}
  !E = adt.typeref() {name = @E}
  !A = core.array(core.i32)

  wasm.func @main() -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %one = wasm.i32_const {value = 1} : core.i32
    %struct = adt.struct_new %zero {type = !S} : !S
    %field = adt.struct_get %struct {type = !S, field = 0} : core.i32
    adt.struct_set %struct, %field {type = !S, field = 0}

    %variant = adt.variant_new %one {type = !E, tag = @Some} : !E
    %is_some = adt.variant_is %variant {type = !E, tag = @Some} : core.i32
    %cast = adt.variant_cast %variant {type = !E, tag = @Some} : !E
    %payload = adt.variant_get %cast {type = !E, tag = @Some, field = 0} : core.i32

    %empty = adt.array_new {type = !A} : !A
    %default = adt.array_new %one {type = !A} : !A
    %array = adt.array_new %one, %zero {type = !A} : !A
    %invalid = adt.array_new %one, %zero, %one {type = !A} : !A
    %element = adt.array_get %array, %zero : core.i32
    adt.array_set %array, %zero, %element
    %length = adt.array_len %default : core.i32

    %null = adt.ref_null {type = !S} : !S
    %is_null = adt.ref_is_null %null : core.i32
    %ref = adt.ref_cast %null {type = !S} : !S
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let func = module.ops(&ctx)[0];
        let body = ctx.op(func).regions[0];
        let block = ctx.region(body).blocks[0];
        let remaining_adt_ops = ctx
            .block(block)
            .ops
            .iter()
            .filter(|&&op| ctx.op(op).dialect == adt::DIALECT_NAME())
            .count();
        assert_eq!(remaining_adt_ops, 2, "only malformed arrays should remain");
        assert_eq!(
            ctx.block(block)
                .ops
                .iter()
                .filter(|&&op| ctx.op(op).dialect == wasm_gc_dialect::DIALECT_NAME())
                .count(),
            13
        );
    }
}
