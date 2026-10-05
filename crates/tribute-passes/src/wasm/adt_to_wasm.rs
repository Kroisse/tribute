//! Lower adt dialect operations to wasm dialect (arena IR).
//!
//! This pass converts ADT (Algebraic Data Type) operations to wasm operations.
//!
//! ## Struct Operations
//! - `adt.struct_new` -> `wasm.struct_new`
//! - `adt.struct_get` -> `wasm.struct_get`
//! - `adt.struct_set` -> `wasm.struct_set`
//!
//! ## Runtime type descriptors
//!
//! A user struct or variant value carries its runtime type descriptor
//! number as an `i32` first field, so its source fields start at index 1.
//! The number comes from the module's `tribute_rtti.layout` declarations
//! ([`super::descriptors::declare`]), which this pass erases. Builtin layouts
//! have no descriptor field.
//!
//! ## Variant Operations
//!
//! A variant's type is the structural `wasm_gc.struct` of the descriptor
//! field followed by the variant's own fields
//! ([`super::struct_layouts::variant_type`]), so variants of equal fields share
//! one type. A variant is told apart by its descriptor number, not by its
//! type: every user struct and variant type is a subtype of the `Described`
//! type, which holds the descriptor field alone.
//!
//! - `adt.variant_new` -> `wasm.struct_new` of the variant's struct type
//! - `adt.variant_is` -> `wasm.ref_cast` to the `Described` type, a read of
//!   its descriptor field, and a comparison with the variant's number
//! - `adt.variant_cast` -> `wasm.ref_cast` (casts to specific variant type)
//! - `adt.variant_get` -> `wasm.struct_get` (field access after the descriptor field)
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
//! GC operations first lower to `wasm_gc`, which identifies GC types by
//! `TypeRef`. Struct operations keep their user `adt.struct` layout until
//! [`super::struct_layouts::convert`] replaces it with its structural type. A
//! module-wide pass assigns the explicit indices required by indexed `wasm`
//! operations before emission.

use tracing::warn;
use tribute_ir::dialect::adt;
use tribute_ir::dialect::adt::layout::get_enum_variants;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core::{self, IntegerLike};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc as wasm_gc_dialect;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;
use trunk_ir::{StringRef, Symbol};

use rustc_hash::FxHashMap as HashMap;

use tribute_ir::dialect::tribute_rtti;
use tribute_ir::runtime_layout::{self, has_runtime_layout};

use super::descriptors::has_descriptor_field;
use super::struct_layouts::variant_type;

/// The logical variant operation's `type` attribute is its exact enum-layout
/// identity. Operand types may be an equivalent `adt.typeref` or already have
/// an erased target representation, neither of which may choose a distinct
/// WasmGC nominal variant type.
fn canonical_enum_type(ctx: &IrContext, attr_ty: TypeRef) -> Option<TypeRef> {
    get_enum_variants(ctx, attr_ty).map(|_| attr_ty)
}

/// Resolve a logical ADT reference through its exact type alias to an enum
/// layout. A typeref's `name` attribute is its nominal declaration link; do
/// not fall back to structural or same-name equivalence here.
fn canonical_typeref_enum_type(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    let data = ctx.get_type(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("typeref") {
        return None;
    }
    let name = data.attrs.get_str(ctx, "name")?;
    let enum_ty = ctx.type_alias_by_text(name)?;
    canonical_enum_type(ctx, enum_ty)
}

/// Convert a logical ADT reference that remains in enum attributes to its
/// Wasm physical representation.
fn physical_variant_field_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    let data = ctx.get_type(ty);
    if data.dialect == Symbol::new("adt") && data.name == Symbol::new("typeref") {
        return ctx.intern_type(TypeDataBuilder::new("wasm", "structref").build());
    }
    ty
}

/// The declared number of each user allocation descriptor.
type DescriptorNumbers = HashMap<(TypeRef, Option<StringRef>), u32>;

/// The field index of source field `field` in a value of layout `ty`.
fn physical_field(ctx: &IrContext, ty: TypeRef, field: u32) -> u32 {
    field + u32::from(has_descriptor_field(ctx, ty))
}

/// The descriptor field operand for a new value of the descriptor
/// `(ty, tag)`, inserted before the allocation, or `None` when the module does
/// not declare that descriptor.
fn descriptor_operand(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    numbers: &DescriptorNumbers,
    descriptor: (TypeRef, Option<StringRef>),
    loc: trunk_ir::types::Location,
) -> Option<trunk_ir::refs::ValueRef> {
    let number = *numbers.get(&descriptor)?;
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let constant = wasm_dialect::I32Const::operands()
        .value(number as i32)
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(constant.op_ref());
    Some(constant.result(ctx))
}

/// Lower adt dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    let numbers = tribute_rtti::Layout::declared_indices(ctx, module);
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(StructNewPattern {
            numbers: numbers.clone(),
        })
        .add_pattern(StructGetPattern)
        .add_pattern(StructSetPattern)
        .add_pattern(VariantNewPattern {
            numbers: numbers.clone(),
        })
        .add_pattern(VariantIsPattern { numbers })
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
    for layout in tribute_rtti::Layout::declared(ctx, module) {
        trunk_ir::rewrite::erase_op(ctx, layout.op_ref());
    }
}

/// Pattern for `adt.struct_new` -> `wasm.struct_new`
struct StructNewPattern {
    numbers: DescriptorNumbers,
}

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
        let struct_ty = struct_new.r#type(ctx);
        let mut fields: Vec<_> = struct_new.fields(ctx).to_vec();
        if has_descriptor_field(ctx, struct_ty) {
            let Some(descriptor) =
                descriptor_operand(ctx, rewriter, &self.numbers, (struct_ty, None), loc)
            else {
                return false;
            };
            fields.insert(0, descriptor);
        }
        let result_ty = struct_new.result_ty(ctx);

        // Keep type attribute, emit will convert to type_idx
        // Note: Result type is preserved as-is; emit phase uses type_to_field_type
        // for wasm type conversion.
        let new_op = wasm_gc_dialect::StructNew::operands(fields)
            .r#type(struct_new.r#type(ctx))
            .results(result_ty)
            .build(ctx, loc);
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
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let field_idx = physical_field(ctx, struct_get.r#type(ctx), struct_get.field(ctx));

        // Build wasm.struct_get with the converted field result type.
        // field attribute is already u32, emit will read it directly
        let new_op = wasm_gc_dialect::StructGet::operands(ref_val)
            .r#type(struct_get.r#type(ctx))
            .field_idx(field_idx)
            .results(result_ty)
            .build(ctx, loc);
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
        let field_idx = physical_field(ctx, struct_set.r#type(ctx), struct_set.field(ctx));

        // Build wasm.struct_set: just change dialect/name
        // field attribute is already u32, emit will read it directly
        let new_op = wasm_gc_dialect::StructSet::operands(ref_val, value)
            .r#type(struct_set.r#type(ctx))
            .field_idx(field_idx)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_new` -> `wasm.struct_new`
///
/// Variants are represented as separate struct types without an explicit tag
/// field. The leading field holds the variant's runtime type descriptor, which
/// is its discriminant.
struct VariantNewPattern {
    numbers: DescriptorNumbers,
}

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
        let tag_sym = variant_new.tag_ref(ctx);
        let Some(base_type) = canonical_enum_type(ctx, variant_new.r#type(ctx)) else {
            return false;
        };
        let mut fields: Vec<_> = variant_new.fields(ctx).to_vec();
        let descriptor = (variant_new.r#type(ctx), Some(tag_sym));
        let Some(descriptor) = descriptor_operand(ctx, rewriter, &self.numbers, descriptor, loc)
        else {
            return false;
        };
        fields.insert(0, descriptor);

        let Some(variant_type) = variant_type(ctx, base_type, tag_sym) else {
            return false;
        };

        let new_op = wasm_gc_dialect::StructNew::operands(fields)
            .r#type(variant_type)
            .results(variant_type)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `adt.variant_is` -> a comparison of the value's descriptor.
///
/// The reference is cast to the `Described` type to read its descriptor
/// field, which is compared with the variant's descriptor number. A variant
/// the module never allocates has no number and never matches.
struct VariantIsPattern {
    numbers: DescriptorNumbers,
}

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
        let tag = variant_is.tag_ref(ctx);
        let ref_val = variant_is.r#ref(ctx);
        let result_ty = variant_is.result_ty(ctx);

        let Some(enum_type) = canonical_enum_type(ctx, variant_is.r#type(ctx)) else {
            return false;
        };

        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let Some(&number) = self.numbers.get(&(enum_type, Some(tag))) else {
            let never = wasm_dialect::I32Const::operands()
                .value(0)
                .results(result_ty)
                .build(ctx, loc);
            rewriter.replace_op(never.op_ref());
            return true;
        };

        let described_ty = super::type_converter::described_adt_type(ctx);
        let described = wasm_gc_dialect::RefCast::operands(ref_val)
            .target_type(described_ty)
            .results(described_ty)
            .build(ctx, loc);
        let descriptor = wasm_gc_dialect::StructGet::operands(described.result(ctx))
            .r#type(described_ty)
            .field_idx(0)
            .results(i32_ty)
            .build(ctx, loc);
        let expected = wasm_dialect::I32Const::operands()
            .value(number as i32)
            .results(i32_ty)
            .build(ctx, loc);
        let matches = wasm_dialect::I32Eq::operands(descriptor.result(ctx), expected.result(ctx))
            .results(result_ty)
            .build(ctx, loc);
        rewriter.insert_op(described.op_ref());
        rewriter.insert_op(descriptor.op_ref());
        rewriter.insert_op(expected.op_ref());
        rewriter.replace_op(matches.op_ref());
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
        let tag = variant_cast.tag_ref(ctx);
        let ref_val = variant_cast.r#ref(ctx);

        let Some(enum_type) = canonical_enum_type(ctx, variant_cast.r#type(ctx)) else {
            return false;
        };

        let Some(variant_type) = variant_type(ctx, enum_type, tag) else {
            return false;
        };

        let new_op = wasm_gc_dialect::RefCast::operands(ref_val)
            .target_type(variant_type)
            .results(variant_type)
            .build(ctx, loc);
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
        let tag = variant_get.tag_ref(ctx);
        let Some(enum_type) = canonical_enum_type(ctx, variant_get.r#type(ctx)) else {
            return false;
        };
        let Some(declared_field_ty) = get_enum_variants(ctx, enum_type)
            .and_then(|variants| {
                variants
                    .into_iter()
                    .find(|(variant_tag, _)| *variant_tag == tag)
            })
            .and_then(|(_, fields)| fields.get(field_idx as usize).copied())
        else {
            return false;
        };
        let declared_field_ty = physical_variant_field_type(ctx, declared_field_ty);
        let requested_result_ty = physical_variant_field_type(ctx, variant_get.result_ty(ctx));
        // String::Leaf has the canonical bytes layout even though frontend
        // pattern extraction is temporarily erased to wasm.anyref. All other
        // variant_get results must agree with their declared enum field type.
        let declared_is_bytes = has_runtime_layout(ctx, declared_field_ty, runtime_layout::BYTES);
        let is_bytes_anyref_erasure =
            declared_is_bytes && wasm_dialect::Anyref::matches(ctx, requested_result_ty);
        if requested_result_ty != declared_field_ty && !is_bytes_anyref_erasure {
            return false;
        }
        let result_ty = declared_field_ty;
        let Some(variant_type) = variant_type(ctx, enum_type, tag) else {
            return false;
        };
        // An operand that already has a struct type is a cast to this
        // variant's layout; a logical reference must name this enum.
        let operand_ty = ctx.value_ty(ref_val);
        if wasm_gc_dialect::Struct::matches(ctx, operand_ty) && operand_ty != variant_type {
            return false;
        }
        let operand_data = ctx.get_type(operand_ty);
        if operand_data.dialect == Symbol::new("adt")
            && operand_data.name == Symbol::new("typeref")
            && canonical_typeref_enum_type(ctx, operand_ty) != Some(enum_type)
        {
            return false;
        }

        // The variant's fields follow its descriptor field.
        let new_op = wasm_gc_dialect::StructGet::operands(ref_val)
            .r#type(variant_type)
            .field_idx(field_idx + 1)
            .results(result_ty)
            .build(ctx, loc);
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
                let new_op = wasm_gc_dialect::ArrayNewDefault::operands(operands[0])
                    .r#type(array_ty)
                    .results(result_ty)
                    .build(ctx, loc);
                rewriter.replace_op(new_op.op_ref());
            }
            2 => {
                // size + init value -> array_new
                let new_op = wasm_gc_dialect::ArrayNew::operands(operands[0], operands[1])
                    .r#type(array_ty)
                    .results(result_ty)
                    .build(ctx, loc);
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
        // A packed element has no Wasm value type of its own. It is read
        // into an `i32` whose upper bits are unspecified, so the unsigned
        // read is as good as the signed one; one is fixed here.
        let new_op = if has_packed_elements(ctx, array_ty) {
            wasm_gc_dialect::ArrayGetU::operands(ref_val, index)
                .r#type(array_ty)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else {
            wasm_gc_dialect::ArrayGet::operands(ref_val, index)
                .r#type(array_ty)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        };
        rewriter.replace_op(new_op);
        true
    }
}

/// Whether `array_ty` is a `core.array` of 8- or 16-bit integers, which
/// WasmGC stores packed.
fn has_packed_elements(ctx: &IrContext, array_ty: TypeRef) -> bool {
    core::Array::from_type_ref(ctx, array_ty)
        .is_some_and(|array| matches!(IntegerLike::width(ctx, array.element(ctx)), Some(8 | 16)))
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
        let new_op = wasm_gc_dialect::ArraySet::operands(ref_val, index, value)
            .r#type(array_ty)
            .build(ctx, loc);
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

        let new_op = wasm_dialect::ArrayLen::operands(ref_val)
            .results(result_ty)
            .build(ctx, loc);
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

        let new_op = wasm_gc_dialect::RefNull::operands()
            .target_type(adt_type)
            .results(result_ty)
            .build(ctx, loc);
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

        let new_op = wasm_dialect::RefIsNull::operands(ref_val)
            .results(result_ty)
            .build(ctx, loc);
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

        let new_op = wasm_gc_dialect::RefCast::operands(ref_val)
            .target_type(adt_type)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::dialect::wasm;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir_wasm_backend::gc_types::FIRST_USER_TYPE_IDX;

    /// Declare the module's descriptors and lower it, as the Wasm pipeline does.
    fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
        crate::wasm::descriptors::declare(ctx, module);
        super::lower(ctx, module, type_converter);
    }

    #[test]
    fn canonical_enum_type_requires_an_exact_enum_layout() {
        let mut ctx = IrContext::new();
        let _module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !E = adt.enum<E {}>
  !ERef = adt.typeref<{name = "E"}>
}"#,
        );
        let enum_ty = ctx.type_alias_by_text("E").expect("enum layout");
        let typeref_ty = ctx.type_alias_by_text("ERef").expect("enum reference");

        assert_eq!(canonical_enum_type(&ctx, enum_ty), Some(enum_ty));
        assert_eq!(canonical_enum_type(&ctx, typeref_ty), None);
    }

    #[test]
    fn lowers_struct_variant_array_and_reference_operations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<S(value: core.i32)>
  !E = adt.enum<E { Some(core.i32) }>
  !ERef = adt.typeref<{name = "E"}>
  !A = core.array<core.i32>

  wasm.func @main() -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %one = wasm.i32_const {value = 1} : core.i32
    %struct = adt.struct_new %zero {type = !S} : !S
    %field = adt.struct_get %struct {type = !S, field = 0} : core.i32
    adt.struct_set %struct, %field {type = !S, field = 0}

    %variant = adt.variant_new %one {type = !E, tag = "Some"} : !ERef
    %is_some = adt.variant_is %variant {type = !E, tag = "Some"} : core.i32
    %cast = adt.variant_cast %variant {type = !E, tag = "Some"} : !ERef
    %payload = adt.variant_get %cast {type = !E, tag = "Some", field = 0} : core.i32

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
        let body = ctx.op_region(func, 0).unwrap();
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
            14
        );
        let variant_get = ctx
            .block(block)
            .ops
            .iter()
            .copied()
            .find(|&op| {
                let data = ctx.op(op);
                let Some(ty) = data.attributes.get_type("type") else {
                    return false;
                };
                data.dialect == wasm_gc_dialect::DIALECT_NAME()
                    && data.name == "struct_get"
                    && wasm_gc_dialect::Struct::matches(&ctx, ty)
            })
            .expect("lowered variant_get");
        let variant_ty = ctx
            .op(variant_get)
            .attributes
            .get_type("type")
            .expect("struct_get type must be a Type attribute");
        assert_eq!(variant_ty, ctx.value_ty(ctx.op_operands(variant_get)[0]));
    }

    #[test]
    fn user_values_lead_with_their_descriptor_number() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<S(value: core.i32)>
  !Closure = adt.struct<_closure(table_idx: core.i32, env: wasm.anyref), {layout = "closure"}>
  !E = adt.enum<E { None(), Some(core.i32) }>

  wasm.func @main(%env: wasm.anyref) -> core.nil {
    %one = wasm.i32_const {value = 1} : core.i32
    %first = adt.struct_new %one {type = !S} : !S
    %second = adt.struct_new %one {type = !S} : !S
    %field = adt.struct_get %first {type = !S, field = 0} : core.i32
    adt.struct_set %first, %field {type = !S, field = 0}
    %closure = adt.struct_new %one, %env {type = !Closure} : !Closure
    %table = adt.struct_get %closure {type = !Closure, field = 0} : core.i32
    %some = adt.variant_new %one {type = !E, tag = "Some"} : !E
    %none = adt.variant_new {type = !E, tag = "None"} : !E
    %payload = adt.variant_get %some {type = !E, tag = "Some", field = 0} : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        assert!(
            tribute_rtti::Layout::declared(&ctx, module).is_empty(),
            "lowering consumes the descriptor declarations"
        );
        let func = module.ops(&ctx)[0];
        let block = ctx.region(ctx.op_region(func, 0).unwrap()).blocks[0];
        let ops = ctx.block(block).ops.clone();
        let descriptor_number = |op: OpRef| {
            let first = ctx.op_operands(op).first().copied()?;
            let trunk_ir::refs::ValueDef::OpResult(def, _) = ctx.value_def(first) else {
                return None;
            };
            wasm_dialect::I32Const::from_op(&ctx, def)
                .ok()
                .map(|constant| constant.value(&ctx))
        };
        let news = ops
            .iter()
            .copied()
            .filter(|&op| wasm_gc_dialect::StructNew::matches(&ctx, op))
            .collect::<Vec<_>>();
        let first_user = FIRST_USER_TYPE_IDX as i32;
        assert_eq!(descriptor_number(news[0]), Some(first_user));
        assert_eq!(descriptor_number(news[1]), Some(first_user));
        assert_eq!(ctx.op_operands(news[2]).len(), 2, "builtin closure");
        assert_eq!(descriptor_number(news[3]), Some(first_user + 1));
        assert_eq!(descriptor_number(news[4]), Some(first_user + 2));
        assert_eq!(
            ctx.op_operands(news[4]).len(),
            1,
            "None has only a descriptor"
        );

        let field_indices = ops
            .iter()
            .filter_map(|&op| {
                wasm_gc_dialect::StructGet::from_op(&ctx, op)
                    .map(|get| get.field_idx(&ctx))
                    .or_else(|_| {
                        wasm_gc_dialect::StructSet::from_op(&ctx, op).map(|set| set.field_idx(&ctx))
                    })
                    .ok()
            })
            .collect::<Vec<_>>();
        assert_eq!(field_indices, [1, 1, 0, 1]);
    }

    #[test]
    fn packed_array_reads_use_the_unsigned_get() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @read(%bytes: core.array<core.i8>, %words: core.array<core.i32>) -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %byte = adt.array_get %bytes, %zero : core.i8
    %word = adt.array_get %words, %zero : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = trunk_ir::printer::print_module(&ctx, module.op());
        assert_eq!(output.matches("wasm_gc.array_get_u").count(), 1, "{output}");
        assert_eq!(output.matches("wasm_gc.array_get ").count(), 1, "{output}");
    }

    #[test]
    fn recursive_list_variant_operations_keep_the_definition_layout() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ListRef = adt.typeref<{name = "List"}>
  !List = adt.enum<List { Empty(), Cons(core.i32, !ListRef) }>

  wasm.func @main(%input: !ListRef) -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %empty = adt.variant_new {type = !List, tag = "Empty"} : !ListRef
    %list = adt.variant_new %zero, %empty {type = !List, tag = "Cons"} : !ListRef
    %is_cons = adt.variant_is %input {type = !List, tag = "Cons"} : core.i1
    %cast = adt.variant_cast %input {type = !List, tag = "Cons"} : !ListRef
    %tail = adt.variant_get %cast {type = !List, tag = "Cons", field = 1} : !ListRef
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let variant_types: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .copied()
            .filter_map(|op| {
                let data = ctx.op(op);
                if data.dialect != wasm_gc_dialect::DIALECT_NAME() {
                    return None;
                }
                match data.name.clone() {
                    name if name == Symbol::new("struct_new")
                        || name == Symbol::new("struct_get") =>
                    {
                        data.attributes.get_type("type")
                    }
                    name if name == Symbol::new("ref_test") || name == Symbol::new("ref_cast") => {
                        data.attributes.get_type("target_type")
                    }
                    _ => None,
                }
            })
            .collect();
        let list = ctx.type_alias_by_text("List").expect("list layout");
        let cons_tag = ctx.intern_str("Cons");
        let empty_tag = ctx.intern_str("Empty");
        let cons = variant_type(&mut ctx, list, cons_tag).expect("Cons layout");
        let empty = variant_type(&mut ctx, list, empty_tag).expect("Empty layout");
        let described = crate::wasm::type_converter::described_adt_type(&mut ctx);

        // `variant_is` reads the descriptor through the `Described` type
        // instead of testing for the variant's type.
        assert_eq!(
            variant_types,
            [empty, cons, described, described, cons, cons]
        );
        assert_ne!(empty, cons);
        // The recursive tail is the abstract struct supertype.
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let structref = ctx.intern_type(TypeDataBuilder::new("wasm", "structref").build());
        assert_eq!(
            wasm_gc_dialect::Struct::from_type_ref(&ctx, cons)
                .expect("structural variant")
                .fields(&ctx),
            [i32_ty, i32_ty, structref]
        );

        trunk_ir_wasm_backend::passes::wasm_gc_to_wasm::lower(&mut ctx, module);
        let indexed_variant_ops: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| {
                wasm::StructNew::from_op(&ctx, op)
                    .ok()
                    .map(|op| op.type_idx(&ctx))
                    .or_else(|| {
                        wasm::StructGet::from_op(&ctx, op)
                            .ok()
                            .map(|op| op.type_idx(&ctx))
                    })
                    .or_else(|| {
                        wasm::RefTest::from_op(&ctx, op)
                            .ok()
                            .and_then(|op| op.type_idx(&ctx))
                    })
                    .or_else(|| {
                        wasm::RefCast::from_op(&ctx, op)
                            .ok()
                            .and_then(|op| op.type_idx(&ctx))
                    })
            })
            .collect();
        let described_idx = trunk_ir_wasm_backend::gc_types::DESCRIBED_IDX;
        let [empty_idx, cons_idx, ..] = indexed_variant_ops[..] else {
            panic!("variant allocations")
        };
        assert_ne!(empty_idx, cons_idx);
        assert_eq!(
            indexed_variant_ops,
            [
                empty_idx,
                cons_idx,
                described_idx,
                described_idx,
                cons_idx,
                cons_idx
            ]
        );
    }

    #[test]
    fn variant_is_compares_the_descriptor_and_is_false_for_an_unallocated_variant() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ERef = adt.typeref<{name = "E"}>
  !E = adt.enum<E { None(), Some(core.i32), Other() }>

  wasm.func @main(%input: !ERef) -> core.nil {
    %one = wasm.i32_const {value = 1} : core.i32
    %none = adt.variant_new {type = !E, tag = "None"} : !ERef
    %some = adt.variant_new %one {type = !E, tag = "Some"} : !ERef
    %is_some = adt.variant_is %input {type = !E, tag = "Some"} : core.i32
    %is_other = adt.variant_is %input {type = !E, tag = "Other"} : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!printed.contains("ref_test"), "{printed}");
        assert!(!printed.contains("adt.variant_is"), "{printed}");
        let func = module.ops(&ctx)[0];
        let block = ctx.region(ctx.op_region(func, 0).unwrap()).blocks[0];
        let defining_op = |value| match ctx.value_def(value) {
            trunk_ir::refs::ValueDef::OpResult(op, _) => op,
            other => panic!("expected an operation result, found {other:?}"),
        };
        let comparisons: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| wasm::I32Eq::from_op(&ctx, op).ok())
            .collect();
        let [is_some] = comparisons[..] else {
            panic!("one descriptor comparison: {printed}")
        };

        let descriptor = wasm_gc_dialect::StructGet::from_op(&ctx, defining_op(is_some.lhs(&ctx)))
            .expect("descriptor read");
        assert_eq!(descriptor.field_idx(&ctx), 0);
        assert!(has_runtime_layout(
            &ctx,
            descriptor.r#type(&ctx),
            runtime_layout::DESCRIBED
        ));
        let cast = wasm_gc_dialect::RefCast::from_op(&ctx, defining_op(descriptor.r#ref(&ctx)))
            .expect("cast to the Described type");
        assert_eq!(cast.target_type(&ctx), descriptor.r#type(&ctx));
        // `Some` is the second allocated descriptor of the module.
        let expected = wasm::I32Const::from_op(&ctx, defining_op(is_some.rhs(&ctx)))
            .expect("descriptor number");
        assert_eq!(expected.value(&ctx), (FIRST_USER_TYPE_IDX + 1) as i32);

        // `Other` is never allocated, so its test is the constant false.
        let is_other = ctx.block(block).ops[ctx.block(block).ops.len() - 2];
        let is_other = wasm::I32Const::from_op(&ctx, is_other).expect("constant result");
        assert_eq!(is_other.value(&ctx), 0);
    }

    #[test]
    fn variant_get_with_mismatched_variant_provenance_stays_unlowered() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ARef = adt.typeref<{name = "A"}>
  !BRef = adt.typeref<{name = "B"}>
  !A = adt.enum<A { Some(core.i32), Other(core.f64) }>
  !B = adt.enum<B { Some(core.f64) }>

  wasm.func @main(%from_a_ref: !ARef, %from_b_ref: !BRef, %x: core.f64) -> core.nil {
    %from_matching_typeref = adt.variant_get %from_a_ref {type = !A, tag = "Some", field = 0} : core.i32
    %from_mismatched_typeref = adt.variant_get %from_b_ref {type = !A, tag = "Some", field = 0} : core.i32
    %from_b = adt.variant_new %x {type = !B, tag = "Some"} : !BRef
    %wrong_enum = adt.variant_get %from_b {type = !A, tag = "Some", field = 0} : core.i32
    %from_a_other = adt.variant_new %x {type = !A, tag = "Other"} : !ARef
    %wrong_tag = adt.variant_get %from_a_other {type = !A, tag = "Some", field = 0} : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let function_args = ctx.block_args(block).to_vec();
        let remaining_variant_gets = ctx
            .block(block)
            .ops
            .iter()
            .filter(|&&op| adt::VariantGet::from_op(&ctx, op).is_ok())
            .count();
        assert_eq!(remaining_variant_gets, 3);
        let lowered_variant_gets: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .copied()
            .filter(|&op| wasm_gc_dialect::StructGet::from_op(&ctx, op).is_ok())
            .collect();
        assert_eq!(lowered_variant_gets.len(), 1);
        assert_eq!(
            ctx.op_operands(lowered_variant_gets[0])[0],
            function_args[0],
            "only the matching !ARef operand lowers",
        );
    }

    #[test]
    fn variant_get_requires_declared_result_type_except_bytes_anyref_erasure() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ERef = adt.typeref<{name = "E"}>
  !BoxRef = adt.typeref<{name = "Box"}>
  !NodeRef = adt.typeref<{name = "Node"}>
  !StringRef = adt.typeref<{name = "String"}>
  !E = adt.enum<E { Some(core.i32) }>
  !Box = adt.enum<Box { Next(!NodeRef) }>
  !Node = adt.enum<Node { Node() }>
  !Data = core.array<core.i8, {layout = "bytes_data"}>
  !Bytes = adt.struct<_Bytes(data: !Data, offset: core.i32, len: core.i32), {layout = "bytes"}>
  !String = adt.enum<String { Leaf(!Bytes) }>

  wasm.func @main(%e: !ERef, %box: !BoxRef, %string: !StringRef) -> core.nil {
    %valid = adt.variant_get %e {type = !E, tag = "Some", field = 0} : core.i32
    %invalid = adt.variant_get %e {type = !E, tag = "Some", field = 0} : core.i64
    %node = adt.variant_get %box {type = !Box, tag = "Next", field = 0} : wasm.structref
    %bytes = adt.variant_get %string {type = !String, tag = "Leaf", field = 0} : wasm.anyref
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let lowered_result_types: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| wasm_gc_dialect::StructGet::from_op(&ctx, op).ok())
            .map(|op| op.result_ty(&ctx))
            .collect();
        assert_eq!(lowered_result_types.len(), 3);
        assert!(lowered_result_types.iter().any(|&ty| {
            let data = ctx.get_type(ty);
            data.dialect == Symbol::new("core") && data.name == Symbol::new("i32")
        }));
        assert!(lowered_result_types.iter().any(|&ty| has_runtime_layout(
            &ctx,
            ty,
            runtime_layout::BYTES
        )));
        assert!(lowered_result_types.iter().any(|&ty| {
            let data = ctx.get_type(ty);
            data.dialect == Symbol::new("wasm") && data.name == Symbol::new("structref")
        }));
        assert_eq!(
            ctx.block(block)
                .ops
                .iter()
                .filter(|&&op| adt::VariantGet::from_op(&ctx, op).is_ok())
                .count(),
            1,
        );
    }
}
