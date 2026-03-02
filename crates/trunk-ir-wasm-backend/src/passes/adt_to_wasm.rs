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
//! Type information is preserved as `type` attribute when available from the source operation.
//! For operations where the result type is set explicitly (e.g., variant_new, variant_cast),
//! emit can infer type_idx from result/operand types without the attribute.

use tracing::warn;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::ir::Symbol;

/// Lower adt dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
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

impl ArenaRewritePattern for StructNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_new) = arena_adt::StructNew::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let fields: Vec<_> = struct_new.fields(ctx).to_vec();
        let result_ty = struct_new.result_ty(ctx);

        // Keep type attribute, emit will convert to type_idx
        // Note: Result type is preserved as-is; emit phase uses type_to_field_type
        // for wasm type conversion.
        let new_op = arena_wasm::struct_new(ctx, loc, fields, result_ty, 0);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.struct_get` -> `wasm.struct_get`
///
/// Type casting from abstract types (structref/anyref) to concrete struct types
/// is handled by the emit stage in `struct_handlers.rs`, not here.
struct StructGetPattern;

impl ArenaRewritePattern for StructGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_get) = arena_adt::StructGet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = struct_get.r#ref(ctx);
        let result_ty = struct_get.result_ty(ctx);
        let field_idx = struct_get.field(ctx);

        // Build wasm.struct_get: just change dialect/name
        // field attribute is already u32, emit will read it directly
        let new_op = arena_wasm::struct_get(ctx, loc, ref_val, result_ty, 0, field_idx);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.struct_set` -> `wasm.struct_set`
struct StructSetPattern;

impl ArenaRewritePattern for StructSetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(struct_set) = arena_adt::StructSet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = struct_set.r#ref(ctx);
        let value = struct_set.value(ctx);
        let field_idx = struct_set.field(ctx);

        // Build wasm.struct_set: just change dialect/name
        // field attribute is already u32, emit will read it directly
        let new_op = arena_wasm::struct_set(ctx, loc, ref_val, value, 0, field_idx);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_new` -> `wasm.struct_new`
///
/// With WasmGC subtyping, variants are represented as separate struct types
/// without an explicit tag field. The type itself serves as the discriminant.
struct VariantNewPattern;

impl ArenaRewritePattern for VariantNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_new) = arena_adt::VariantNew::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let tag_sym = variant_new.tag(ctx);
        let base_type = variant_new.r#type(ctx);
        let fields: Vec<_> = variant_new.fields(ctx).to_vec();

        // Create variant-specific type: Expr + Add -> Expr$Add
        let variant_type = make_variant_type(ctx, base_type, tag_sym);

        // Create wasm.struct_new with variant-specific type (no tag field)
        // Result type is the variant-specific type - emit infers type_idx from it
        let new_op = arena_wasm::struct_new(ctx, loc, fields, variant_type, 0);
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
                ArenaAttribute::Symbol(s) => Some(*s),
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
        .attr(Symbol::new("is_variant"), ArenaAttribute::Bool(true))
        .attr(Symbol::new("base_enum"), ArenaAttribute::Type(base_type))
        .attr(Symbol::new("variant_tag"), ArenaAttribute::Symbol(tag));

    ctx.types.intern(builder.build())
}

/// Pattern for `adt.variant_is` -> `wasm.ref_test`
///
/// Tests if a variant reference is of a specific variant type.
struct VariantIsPattern;

impl ArenaRewritePattern for VariantIsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_is) = arena_adt::VariantIs::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let tag = variant_is.tag(ctx);
        let ref_val = variant_is.r#ref(ctx);
        let result_ty = variant_is.result_ty(ctx);

        // Get the enum type - prefer operand type over attribute type
        // (attribute may have unsubstituted type.var, operand has concrete type)
        let operand_ty = ctx.value_ty(ref_val);
        let enum_type = if operand_ty != result_ty {
            // Use operand type if it's different from result (i.e., it's the real enum type)
            operand_ty
        } else {
            variant_is.r#type(ctx)
        };

        // Create variant-specific type for the ref.test
        let variant_type = make_variant_type(ctx, enum_type, tag);

        // Create wasm.ref_test with variant-specific type
        let new_op = arena_wasm::ref_test(ctx, loc, ref_val, result_ty, variant_type, None);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_cast` -> `wasm.ref_cast`
///
/// Casts a variant reference to a specific variant type after pattern matching.
struct VariantCastPattern;

impl ArenaRewritePattern for VariantCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_cast) = arena_adt::VariantCast::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let tag = variant_cast.tag(ctx);
        let ref_val = variant_cast.r#ref(ctx);

        // Get the enum type - prefer operand type over attribute type
        // (attribute may have unsubstituted type.var, operand has concrete type)
        let operand_ty = ctx.value_ty(ref_val);
        let attr_type = variant_cast.r#type(ctx);
        let enum_type = if operand_ty != attr_type {
            operand_ty
        } else {
            attr_type
        };

        // Create variant-specific type for the ref.cast
        let variant_type = make_variant_type(ctx, enum_type, tag);

        // Create wasm.ref_cast with variant-specific type
        // Result type is the variant-specific type - emit infers type_idx from it
        let new_op = arena_wasm::ref_cast(ctx, loc, ref_val, variant_type, variant_type, None);
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

impl ArenaRewritePattern for VariantGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_get) = arena_adt::VariantGet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = variant_get.r#ref(ctx);
        let result_ty = variant_get.result_ty(ctx);
        let field_idx = variant_get.field(ctx);

        // Infer type from the operand (the cast result has the variant-specific type)
        // field attribute is already u32 and will be used directly
        let new_op = arena_wasm::struct_get(ctx, loc, ref_val, result_ty, 0, field_idx);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.array_new` -> `wasm.array_new` or `wasm.array_new_default`
struct ArrayNewPattern;

impl ArenaRewritePattern for ArrayNewPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_new) = arena_adt::ArrayNew::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = array_new.result_ty(ctx);
        let operands = array_new.elements(ctx).to_vec();

        match operands.len() {
            0 => {
                warn!("adt.array_new with no operands");
                return false;
            }
            1 => {
                // Only size operand -> array_new_default
                let new_op = arena_wasm::array_new_default(ctx, loc, operands[0], result_ty, 0);
                rewriter.replace_op(new_op.op_ref());
            }
            2 => {
                // size + init value -> array_new
                let new_op =
                    arena_wasm::array_new(ctx, loc, operands[0], operands[1], result_ty, 0);
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

impl ArenaRewritePattern for ArrayGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_get) = arena_adt::ArrayGet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = array_get.r#ref(ctx);
        let index = array_get.index(ctx);
        let result_ty = array_get.result_ty(ctx);

        let new_op = arena_wasm::array_get(ctx, loc, ref_val, index, result_ty, 0);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.array_set` -> `wasm.array_set`
struct ArraySetPattern;

impl ArenaRewritePattern for ArraySetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_set) = arena_adt::ArraySet::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = array_set.r#ref(ctx);
        let index = array_set.index(ctx);
        let value = array_set.value(ctx);

        let new_op = arena_wasm::array_set(ctx, loc, ref_val, index, value, 0);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.array_len` -> `wasm.array_len`
struct ArrayLenPattern;

impl ArenaRewritePattern for ArrayLenPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(array_len) = arena_adt::ArrayLen::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = array_len.r#ref(ctx);
        let result_ty = array_len.result_ty(ctx);

        let new_op = arena_wasm::array_len(ctx, loc, ref_val, result_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.ref_null` -> `wasm.ref_null`
struct RefNullPattern;

impl ArenaRewritePattern for RefNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_null) = arena_adt::RefNull::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = ref_null.result_ty(ctx);

        // Get the type name for heap_type from the adt type attribute
        let adt_type = ref_null.r#type(ctx);
        let heap_type = ctx.types.get(adt_type).name;

        let new_op = arena_wasm::ref_null(ctx, loc, result_ty, heap_type, None);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.ref_is_null` -> `wasm.ref_is_null`
struct RefIsNullPattern;

impl ArenaRewritePattern for RefIsNullPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_is_null) = arena_adt::RefIsNull::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = ref_is_null.r#ref(ctx);
        let result_ty = ref_is_null.result_ty(ctx);

        let new_op = arena_wasm::ref_is_null(ctx, loc, ref_val, result_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.ref_cast` -> `wasm.ref_cast`
struct RefCastPattern;

impl ArenaRewritePattern for RefCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ref_cast) = arena_adt::RefCast::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ref_val = ref_cast.r#ref(ctx);
        let result_ty = ref_cast.result_ty(ctx);

        // Get the target type from the adt type attribute
        let adt_type = ref_cast.r#type(ctx);

        let new_op = arena_wasm::ref_cast(ctx, loc, ref_val, result_ty, adt_type, None);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}
