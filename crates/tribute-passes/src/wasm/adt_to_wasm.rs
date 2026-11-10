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
//! (`descriptors::declare`), which this pass erases. Builtin layouts
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
//! - `adt.variant_tag` -> the same descriptor read, less the number of the
//!   enum's first variant: an enum numbers its variants consecutively
//! - `adt.variant_get` -> `wasm.ref_cast` to the variant's struct type and
//!   `wasm.struct_get` of the field after the descriptor field
//!
//! Each pattern casts its own operand. After the patterns, a cast that an
//! identical cast of the same value dominates reuses that cast's result
//! (`merge_dominated_casts`), so the field reads of one value under one
//! variant test share one cast.
//!
//! ## Array Operations
//! - `adt.array_new` -> `wasm.array_new`
//! - `adt.array_of` -> `wasm.array_new_fixed`
//! - `adt.array_splice` -> a range check, `wasm.array_new`, and `wasm.array_copy`
//!   of the kept ranges around `wasm.array_set` of the new elements
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
//! `struct_layouts::convert` replaces it with its structural type. A
//! module-wide pass assigns the explicit indices required by indexed `wasm`
//! operations before emission.

use super::type_converter::gc_array_type;
use tribute_ir::dialect::adt;
use tribute_ir::dialect::adt::layout::get_enum_variants;
use trunk_ir::context::{IrContext, OpList, RegionList};
use trunk_ir::dialect::core::{self, IntegerLike};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc as wasm_gc_dialect;
use trunk_ir::op_flag::{IsolatedFromAbove, OpFlag};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;

use rustc_hash::FxHashMap as HashMap;

use tribute_ir::dialect::tribute_rtti;
use tribute_ir::runtime_layout::{self, has_runtime_layout};

use super::descriptors::has_descriptor_field;
use super::struct_layouts::variant_type;

/// The logical variant operation's `type` attribute is its exact enum-layout
/// identity. Operand types may be a row instance of it or already have an
/// erased target representation, neither of which may choose a distinct
/// WasmGC nominal variant type.
fn canonical_enum_type(ctx: &IrContext, attr_ty: TypeRef) -> Option<TypeRef> {
    get_enum_variants(ctx, attr_ty).map(|_| attr_ty)
}

/// Resolve a logical nominal value type to the enum layout it names: the
/// layout a key or row instance denotes. Do not fall back to
/// structural or same-name equivalence here.
fn canonical_reference_enum_type(ctx: &IrContext, ty: TypeRef) -> Option<TypeRef> {
    if !adt::is_nominal(ctx, ty) {
        return None;
    }
    canonical_enum_type(ctx, adt::layout_of(ctx, ty)?)
}

/// Convert a logical ADT reference that remains in enum attributes to its
/// Wasm physical representation.
fn physical_variant_field_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    if adt::is_nominal_value(ctx, ty) {
        return wasm_dialect::Structref::params().intern(ctx).as_type_ref();
    }
    ty
}

/// The declared number of each user allocation descriptor.
type DescriptorNumbers = HashMap<(TypeRef, Option<u32>), u32>;

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
    descriptor: (TypeRef, Option<u32>),
    loc: trunk_ir::types::Location,
) -> Option<trunk_ir::refs::ValueRef> {
    let number = *numbers.get(&descriptor)?;
    let i32_ty = TypeDataBuilder::new("core.i32").intern(ctx);
    let constant = wasm_dialect::I32Const::operands()
        .value(number as i32)
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(constant.op_ref());
    Some(constant.result(ctx))
}

/// Lower `adt.ref_null`, `adt.ref_is_null`, and `adt.ref_cast` to their
/// Wasm operations, the `adt-references-to-wasm` pass.
pub(crate) fn lower_references(ctx: &mut IrContext, module: Module) {
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(RefNullPattern)
        .add_pattern(RefIsNullPattern)
        .add_pattern(RefCastPattern)
        .apply_partial(ctx, module);
}

/// Lower adt dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub(crate) fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
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
        .add_pattern(VariantTagPattern {
            first_variants: first_variant_numbers(ctx, &numbers),
        })
        .add_pattern(VariantIsPattern { numbers })
        .add_pattern(VariantGetPattern)
        .add_pattern(ArrayNewPattern)
        .add_pattern(ArrayOfPattern)
        .add_pattern(ArraySplicePattern)
        .add_pattern(ArrayGetPattern)
        .add_pattern(ArraySetPattern)
        .add_pattern(ArrayLenPattern)
        .add_pattern(adt::EraseNominalDecls);
    applicator.apply_partial(ctx, module);
    for layout in tribute_rtti::Layout::declared(ctx, module) {
        trunk_ir::rewrite::erase_op(ctx, layout.op_ref());
    }
    if let Some(body) = module.body(ctx) {
        merge_dominated_casts(ctx, body, &mut CastScope::default());
    }
}

/// The casts visible at a point of a walk, keyed by operand, target type, and
/// result type, with the keys each enclosing block added.
#[derive(Default)]
struct CastScope {
    casts: HashMap<(ValueRef, TypeRef, TypeRef), ValueRef>,
    added: Vec<(ValueRef, TypeRef, TypeRef)>,
}

/// Replace each `wasm_gc.ref_cast` that an identical cast dominates with that
/// cast's result.
///
/// A cast of the same value to the same type that succeeded earlier proves
/// the later one succeeds with the same reference, so a variant arm that reads
/// several fields of one value casts it once. A cast is visible to the ops
/// after it in its block and to their nested regions, but not to the other
/// blocks of its region, whose order is not dominance, nor to regions isolated
/// from above. No cast moves, so none runs before the test that guards it.
///
/// This is a local step of this lowering; a general CSE of operations whose
/// only side effect is a trap would replace it (Kroisse/tribute#1887).
fn merge_dominated_casts(ctx: &mut IrContext, region: RegionRef, scope: &mut CastScope) {
    let blocks = ctx.region(region).blocks.clone();
    for block in blocks {
        let mark = scope.added.len();
        let ops: OpList = ctx.block(block).ops.clone();
        for op in ops {
            if let Ok(cast) = wasm_gc_dialect::RefCast::from_op(ctx, op) {
                let result = cast.result(ctx);
                let key = (cast.r#ref(ctx), cast.target_type(ctx), ctx.value_ty(result));
                if let Some(&earlier) = scope.casts.get(&key) {
                    ctx.replace_all_uses(result, earlier);
                    trunk_ir::rewrite::erase_op(ctx, op);
                } else {
                    scope.casts.insert(key, result);
                    scope.added.push(key);
                }
                continue;
            }
            let regions: RegionList = ctx.op_regions(op).collect();
            if regions.is_empty() {
                continue;
            }
            if IsolatedFromAbove::is_set(ctx, op) {
                for region in regions {
                    merge_dominated_casts(ctx, region, &mut CastScope::default());
                }
            } else {
                for region in regions {
                    merge_dominated_casts(ctx, region, scope);
                }
            }
        }
        for key in scope.added.drain(mark..) {
            scope.casts.remove(&key);
        }
    }
}

/// Pattern for `adt.struct_new` -> `wasm.struct_new`
struct StructNewPattern {
    numbers: DescriptorNumbers,
}

impl OpRewritePattern for StructNewPattern {
    type Op = adt::StructNew;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        struct_new: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = struct_new.op_ref();
        let loc = ctx.op(op).location;
        let struct_ty = struct_new.layout(ctx);
        let mut fields: Vec<_> = struct_new.fields(ctx).to_vec();
        if has_descriptor_field(ctx, struct_ty) {
            let descriptor =
                descriptor_operand(ctx, rewriter, &self.numbers, (struct_ty, None), loc)?;
            fields.insert(0, descriptor);
        }
        let result_ty = struct_new.result_ty(ctx);

        // Keep type attribute, emit will convert to type_idx
        // Note: Result type is preserved as-is; emit phase uses type_to_field_type
        // for wasm type conversion.
        let new_op = wasm_gc_dialect::StructNew::operands(fields)
            .r#type(struct_ty)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.struct_get` -> `wasm.struct_get`
///
/// Type casting from abstract types (structref/anyref) to concrete struct types
/// is handled by the emit stage in `struct_handlers.rs`, not here.
struct StructGetPattern;

impl OpRewritePattern for StructGetPattern {
    type Op = adt::StructGet;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        struct_get: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = struct_get.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = struct_get.r#ref(ctx);
        let result_ty = rewriter.result_type(ctx, op, 0)?;
        let field_idx = physical_field(ctx, struct_get.r#type(ctx), struct_get.field(ctx));

        // Build wasm.struct_get with the converted field result type.
        // field attribute is already u32, emit will read it directly
        let new_op = wasm_gc_dialect::StructGet::operands(ref_val)
            .r#type(struct_get.r#type(ctx))
            .field_idx(field_idx)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.struct_set` -> `wasm.struct_set`
struct StructSetPattern;

impl OpRewritePattern for StructSetPattern {
    type Op = adt::StructSet;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        struct_set: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = struct_set.op_ref();
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
        Some(())
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

impl OpRewritePattern for VariantNewPattern {
    type Op = adt::VariantNew;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        variant_new: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = variant_new.op_ref();
        let loc = ctx.op(op).location;
        let tag = variant_new.variant(ctx);
        let base_type = canonical_enum_type(ctx, variant_new.layout(ctx))?;
        let mut fields: Vec<_> = variant_new.fields(ctx).to_vec();
        let descriptor = (variant_new.layout(ctx), Some(tag));
        let descriptor = descriptor_operand(ctx, rewriter, &self.numbers, descriptor, loc)?;
        fields.insert(0, descriptor);

        let variant_type = variant_type(ctx, base_type, tag)?;

        let new_op = wasm_gc_dialect::StructNew::operands(fields)
            .r#type(variant_type)
            .results(variant_type)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
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

impl OpRewritePattern for VariantIsPattern {
    type Op = adt::VariantIs;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        variant_is: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = variant_is.op_ref();
        let loc = ctx.op(op).location;
        let tag = variant_is.variant(ctx);
        let ref_val = variant_is.r#ref(ctx);
        let result_ty = variant_is.result_ty(ctx);

        let enum_type = canonical_enum_type(ctx, variant_is.r#type(ctx))?;

        let i32_ty = TypeDataBuilder::new("core.i32").intern(ctx);
        let Some(&number) = self.numbers.get(&(enum_type, Some(tag))) else {
            let never = wasm_dialect::I32Const::operands()
                .value(0)
                .results(result_ty)
                .build(ctx, loc);
            rewriter.replace_op(never.op_ref());
            return Some(());
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
        Some(())
    }
}

/// The descriptor number of each numbered enum's first declared variant,
/// whether or not the module allocates it: the number of a declared variant
/// less its position.
fn first_variant_numbers(ctx: &IrContext, numbers: &DescriptorNumbers) -> HashMap<TypeRef, u32> {
    numbers
        .iter()
        .filter_map(|(&(ty, tag), &number)| {
            adt::Enum::from_type_ref(ctx, ty)?;
            Some((ty, number - tag?))
        })
        .collect()
}

/// Pattern for `adt.variant_tag` -> the value's descriptor number less the
/// number of its enum's first variant.
///
/// An enum the module never allocates has no value, so its tag is constant.
struct VariantTagPattern {
    first_variants: HashMap<TypeRef, u32>,
}

impl OpRewritePattern for VariantTagPattern {
    type Op = adt::VariantTag;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        variant_tag: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let loc = ctx.op(variant_tag.op_ref()).location;
        let enum_type = canonical_enum_type(ctx, variant_tag.r#type(ctx))?;
        let i32_ty = TypeDataBuilder::new("core.i32").intern(ctx);
        let tag = match self.first_variants.get(&enum_type) {
            Some(&first) => {
                let described_ty = super::type_converter::described_adt_type(ctx);
                let described = wasm_gc_dialect::RefCast::operands(variant_tag.r#ref(ctx))
                    .target_type(described_ty)
                    .results(described_ty)
                    .build(ctx, loc);
                let descriptor = wasm_gc_dialect::StructGet::operands(described.result(ctx))
                    .r#type(described_ty)
                    .field_idx(0)
                    .results(i32_ty)
                    .build(ctx, loc);
                let first = wasm_dialect::I32Const::operands()
                    .value(first as i32)
                    .results(i32_ty)
                    .build(ctx, loc);
                let tag = wasm_dialect::I32Sub::operands(descriptor.result(ctx), first.result(ctx))
                    .results(i32_ty)
                    .build(ctx, loc);
                for op in [described.op_ref(), descriptor.op_ref(), first.op_ref()] {
                    rewriter.insert_op(op);
                }
                tag.op_ref()
            }
            None => wasm_dialect::I32Const::operands()
                .value(0)
                .results(i32_ty)
                .build(ctx, loc)
                .op_ref(),
        };
        rewriter.insert_op(tag);
        let index = core::UnrealizedConversionCast::operands(ctx.op_result(tag, 0))
            .results(variant_tag.result_ty(ctx))
            .build(ctx, loc);
        rewriter.replace_op(index.op_ref());
        Some(())
    }
}

/// Pattern for `adt.variant_get` -> `wasm.ref_cast` to the variant's struct
/// type, then `wasm.struct_get` of the field after the descriptor field.
struct VariantGetPattern;

impl OpRewritePattern for VariantGetPattern {
    type Op = adt::VariantGet;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        variant_get: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = variant_get.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = variant_get.r#ref(ctx);
        let field_idx = variant_get.field(ctx);
        let tag = variant_get.variant(ctx);
        let enum_type = canonical_enum_type(ctx, variant_get.r#type(ctx))?;
        let declared_field_ty = *adt::Enum::from_type_ref(ctx, enum_type)?
            .variant_fields(ctx, tag)?
            .get(field_idx as usize)?;
        let declared_field_ty = physical_variant_field_type(ctx, declared_field_ty);
        let requested_result_ty = physical_variant_field_type(ctx, variant_get.result_ty(ctx));
        // String::Leaf has the canonical bytes layout even though frontend
        // pattern extraction is temporarily erased to wasm.anyref. All other
        // variant_get results must agree with their declared enum field type.
        let declared_is_bytes = has_runtime_layout(ctx, declared_field_ty, runtime_layout::BYTES);
        let is_bytes_anyref_erasure =
            declared_is_bytes && wasm_dialect::Anyref::matches(ctx, requested_result_ty);
        rewriter.ensure(
            requested_result_ty == declared_field_ty || is_bytes_anyref_erasure,
            "result type differs from the declared field type",
        )?;
        let result_ty = declared_field_ty;
        let variant_type = variant_type(ctx, enum_type, tag)?;
        // A logical reference must name this enum.
        let operand_ty = ctx.value_ty(ref_val);
        rewriter.ensure(
            !adt::is_nominal_value(ctx, operand_ty)
                || canonical_reference_enum_type(ctx, operand_ty) == Some(enum_type),
            "operand reference names another enum",
        )?;

        // `struct.get` reads a reference of the variant's own struct type. A
        // value built as another variant reaches here only on a path its
        // variant test excludes, so the cast never fails.
        let ref_val = if operand_ty == variant_type {
            ref_val
        } else {
            let cast = wasm_gc_dialect::RefCast::operands(ref_val)
                .target_type(variant_type)
                .results(variant_type)
                .build(ctx, loc);
            rewriter.insert_op(cast.op_ref());
            cast.result(ctx)
        };
        // The variant's fields follow its descriptor field.
        let new_op = wasm_gc_dialect::StructGet::operands(ref_val)
            .r#type(variant_type)
            .field_idx(field_idx + 1)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.array_new` -> `wasm.array_new`
struct ArrayNewPattern;

impl OpRewritePattern for ArrayNewPattern {
    type Op = adt::ArrayNew;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_new: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_new.op_ref();
        let loc = ctx.op(op).location;
        let length = wasm_index(ctx, rewriter, array_new.length(ctx), loc);
        let array_ty = gc_array_type(ctx, array_new.r#type(ctx));
        let initial = gc_element_operand(ctx, rewriter, array_ty, array_new.initial(ctx), loc);
        let new_op = wasm_gc_dialect::ArrayNew::operands(length, initial)
            .r#type(array_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.array_of` -> `wasm.array_new_fixed`
struct ArrayOfPattern;

impl OpRewritePattern for ArrayOfPattern {
    type Op = adt::ArrayOf;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_of: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_of.op_ref();
        let loc = ctx.op(op).location;
        let array_ty = gc_array_type(ctx, array_of.r#type(ctx));
        let mut elements = array_of.elements(ctx).to_vec();
        for element in &mut elements {
            *element = gc_element_operand(ctx, rewriter, array_ty, *element, loc);
        }
        let new_op = wasm_gc_dialect::ArrayNewFixed::operands(elements)
            .r#type(array_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.array_splice` -> a new array built from the kept ranges
///
/// After trapping unless `start <= end <= len`, the result is filled with a
/// value of the element type, `source[0, start)` and `source[end, len)` are
/// copied to its two ends, and the new elements are stored between them. The
/// fill value is the first new element; without one, the result is either
/// empty or holds a `source` element, so `source[0]` fills it.
struct ArraySplicePattern;

impl OpRewritePattern for ArraySplicePattern {
    type Op = adt::ArraySplice;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        splice: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = splice.op_ref();
        let loc = ctx.op(op).location;
        let array_ty = gc_array_type(ctx, splice.r#type(ctx));
        let source = gc_array_operand(ctx, rewriter, splice.source(ctx), loc);
        let mut elements = splice.elements(ctx).to_vec();
        for element in &mut elements {
            *element = gc_element_operand(ctx, rewriter, array_ty, *element, loc);
        }
        let start = wasm_index(ctx, rewriter, splice.start(ctx), loc);
        let end = wasm_index(ctx, rewriter, splice.end(ctx), loc);
        let i32_ty = core::I32::type_ref(ctx);
        let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
        let i32_const = |ctx: &mut IrContext, rewriter: &mut PatternRewriter<'_>, value: i32| {
            let op = wasm_dialect::I32Const::operands()
                .value(value)
                .results(i32_ty)
                .build(ctx, loc);
            rewriter.insert_op(op.op_ref());
            op.result(ctx)
        };

        let len = wasm_dialect::ArrayLen::operands(source).build(ctx, loc);
        rewriter.insert_op(len.op_ref());
        let len = len.result(ctx);
        let reversed = wasm_dialect::I32GtU::operands(start, end)
            .results(i32_ty)
            .build(ctx, loc);
        rewriter.insert_op(reversed.op_ref());
        let past_len = wasm_dialect::I32GtU::operands(end, len)
            .results(i32_ty)
            .build(ctx, loc);
        rewriter.insert_op(past_len.op_ref());
        let out_of_range =
            wasm_dialect::I32Or::operands(reversed.result(ctx), past_len.result(ctx))
                .results(i32_ty)
                .build(ctx, loc);
        rewriter.insert_op(out_of_range.op_ref());
        let trap = super::intrinsic_to_wasm::trap_region(ctx, loc);
        let ok = super::intrinsic_to_wasm::empty_region(ctx, loc);
        let check = wasm_dialect::If::operands(out_of_range.result(ctx))
            .results([nil_ty])
            .regions(trap, ok)
            .build(ctx, loc);
        rewriter.insert_op(check.op_ref());

        let inserted = i32_const(ctx, rewriter, elements.len() as i32);
        let tail = wasm_dialect::I32Sub::operands(len, end)
            .results(i32_ty)
            .build(ctx, loc);
        rewriter.insert_op(tail.op_ref());
        let head_and_inserted = wasm_dialect::I32Add::operands(start, inserted).build(ctx, loc);
        rewriter.insert_op(head_and_inserted.op_ref());
        let result_len =
            wasm_dialect::I32Add::operands(head_and_inserted.result(ctx), tail.result(ctx))
                .build(ctx, loc);
        rewriter.insert_op(result_len.op_ref());
        let result_len = result_len.result(ctx);

        let result = if let Some(&first) = elements.first() {
            let fresh = wasm_gc_dialect::ArrayNew::operands(result_len, first)
                .r#type(array_ty)
                .build(ctx, loc);
            rewriter.insert_op(fresh.op_ref());
            fresh.result(ctx)
        } else {
            let filled = filled_from_source(ctx, loc, array_ty, source, result_len);
            let empty = wasm_gc_dialect::ArrayNewFixed::operands([])
                .r#type(array_ty)
                .build(ctx, loc);
            let empty = yielding_region(ctx, loc, [empty.op_ref()]);
            let fresh = wasm_dialect::If::operands(result_len)
                .results([array_ty])
                .regions(filled, empty)
                .build(ctx, loc);
            rewriter.insert_op(fresh.op_ref());
            fresh.results(ctx)[0]
        };

        let zero = i32_const(ctx, rewriter, 0);
        let copy_head = wasm_gc_dialect::ArrayCopy::operands(result, zero, source, zero, start)
            .dst_type(array_ty)
            .src_type(array_ty)
            .build(ctx, loc);
        rewriter.insert_op(copy_head.op_ref());
        for (offset, &element) in elements.iter().enumerate().skip(1) {
            let offset = i32_const(ctx, rewriter, offset as i32);
            let index = wasm_dialect::I32Add::operands(start, offset).build(ctx, loc);
            rewriter.insert_op(index.op_ref());
            let store = wasm_gc_dialect::ArraySet::operands(result, index.result(ctx), element)
                .r#type(array_ty)
                .build(ctx, loc);
            rewriter.insert_op(store.op_ref());
        }
        let copy_tail = wasm_gc_dialect::ArrayCopy::operands(
            result,
            head_and_inserted.result(ctx),
            source,
            end,
            tail.result(ctx),
        )
        .dst_type(array_ty)
        .src_type(array_ty)
        .build(ctx, loc);
        rewriter.insert_op(copy_tail.op_ref());
        rewriter.erase_op(vec![result]);
        Some(())
    }
}

/// A region yielding a `len`-element array filled with `source[0]`.
fn filled_from_source(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    array_ty: TypeRef,
    source: ValueRef,
    len: ValueRef,
) -> trunk_ir::RegionRef {
    let i32_ty = core::I32::type_ref(ctx);
    let zero = wasm_dialect::I32Const::operands()
        .value(0)
        .results(i32_ty)
        .build(ctx, loc);
    let first = array_read(ctx, loc, array_ty, source, zero.result(ctx));
    let first_value = ctx.op_result(first, 0);
    let fresh = wasm_gc_dialect::ArrayNew::operands(len, first_value)
        .r#type(array_ty)
        .build(ctx, loc);
    yielding_region(ctx, loc, [zero.op_ref(), first, fresh.op_ref()])
}

/// A region running `ops` and yielding the last one's result.
fn yielding_region(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    ops: impl IntoIterator<Item = OpRef>,
) -> trunk_ir::RegionRef {
    let region = super::intrinsic_to_wasm::empty_region(ctx, loc);
    let block = ctx.region(region).blocks[0];
    let mut last = None;
    for op in ops {
        ctx.push_op(block, op);
        last = Some(op);
    }
    let value = ctx.op_result(last.expect("a yielding region has an op"), 0);
    let yield_value = wasm_dialect::Yield::operands(value).build(ctx, loc);
    ctx.push_op(block, yield_value.op_ref());
    region
}

/// A `wasm_gc` read of `array[index]`, unsigned for packed elements.
fn array_read(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    array_ty: TypeRef,
    array: ValueRef,
    index: ValueRef,
) -> OpRef {
    // A packed element has no Wasm value type of its own. It is read
    // into an `i32` whose upper bits are unspecified, so the unsigned
    // read is as good as the signed one; one is fixed here.
    if has_packed_elements(ctx, array_ty) {
        let element_ty = trunk_ir::dialect::wasm_gc::Array::from_type_ref(ctx, array_ty)
            .expect("packed elements belong to a GC array")
            .element(ctx);
        wasm_gc_dialect::ArrayGetU::operands(array, index)
            .r#type(array_ty)
            .results(element_ty)
            .build(ctx, loc)
            .op_ref()
    } else {
        wasm_gc_dialect::ArrayGet::operands(array, index)
            .r#type(array_ty)
            .build(ctx, loc)
            .op_ref()
    }
}

/// Pattern for `adt.array_get` -> `wasm.array_get`
struct ArrayGetPattern;

impl OpRewritePattern for ArrayGetPattern {
    type Op = adt::ArrayGet;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_get: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_get.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = gc_array_operand(ctx, rewriter, array_get.r#ref(ctx), loc);
        let index = wasm_index(ctx, rewriter, array_get.index(ctx), loc);

        let array_ty = gc_array_type(ctx, ctx.value_ty(ref_val));
        let new_op = array_read(ctx, loc, array_ty, ref_val, index);
        rewriter.replace_op(new_op);
        Some(())
    }
}

/// Whether `array_ty`, a GC array, holds 8- or 16-bit integers, which
/// WasmGC stores packed.
fn has_packed_elements(ctx: &IrContext, array_ty: TypeRef) -> bool {
    trunk_ir::dialect::wasm_gc::Array::from_type_ref(ctx, array_ty)
        .is_some_and(|array| matches!(IntegerLike::width(ctx, array.element(ctx)), Some(8 | 16)))
}

/// `array` as its GC array type. A value still of its `adt.array` type, such
/// as a parameter that signature conversion cast back for this operation, is
/// cast to the GC array; cast reconciliation folds the round trip.
fn gc_array_operand(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    array: ValueRef,
    loc: trunk_ir::types::Location,
) -> ValueRef {
    let ty = ctx.value_ty(array);
    if !adt::Array::matches(ctx, ty) {
        return array;
    }
    let gc_ty = gc_array_type(ctx, ty);
    let cast = core::UnrealizedConversionCast::operands(array)
        .results(gc_ty)
        .build(ctx, loc);
    rewriter.insert_op(cast.op_ref());
    cast.result(ctx)
}

/// `element` as the element type of the GC array `array_ty`. An element of
/// another type, such as a marker stored into evidence's `structref`
/// elements, is cast to it; the cast is an upcast once the element converts.
fn gc_element_operand(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    array_ty: TypeRef,
    element: ValueRef,
    loc: trunk_ir::types::Location,
) -> ValueRef {
    let Some(array) = trunk_ir::dialect::wasm_gc::Array::from_type_ref(ctx, array_ty) else {
        return element;
    };
    let element_ty = array.element(ctx);
    if ctx.value_ty(element) == element_ty {
        return element;
    }
    let cast = core::UnrealizedConversionCast::operands(element)
        .results(element_ty)
        .build(ctx, loc);
    rewriter.insert_op(cast.op_ref());
    cast.result(ctx)
}

/// Pattern for `adt.array_set` -> `wasm.array_set`
struct ArraySetPattern;

impl OpRewritePattern for ArraySetPattern {
    type Op = adt::ArraySet;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_set: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_set.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = gc_array_operand(ctx, rewriter, array_set.r#ref(ctx), loc);
        let index = wasm_index(ctx, rewriter, array_set.index(ctx), loc);
        let array_ty = gc_array_type(ctx, ctx.value_ty(ref_val));
        let value = gc_element_operand(ctx, rewriter, array_ty, array_set.value(ctx), loc);
        let new_op = wasm_gc_dialect::ArraySet::operands(ref_val, index, value)
            .r#type(array_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.array_len` -> `wasm.array_len`
struct ArrayLenPattern;

impl OpRewritePattern for ArrayLenPattern {
    type Op = adt::ArrayLen;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_len: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_len.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = gc_array_operand(ctx, rewriter, array_len.r#ref(ctx), loc);
        let result_ty = array_len.result_ty(ctx);

        let len = wasm_dialect::ArrayLen::operands(ref_val).build(ctx, loc);
        rewriter.insert_op(len.op_ref());
        let index = core::UnrealizedConversionCast::operands(len.result(ctx))
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(index.op_ref());
        Some(())
    }
}

/// `index`, a `core.index`, as the `core.i32` that WasmGC array
/// instructions take.
fn wasm_index(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    index: ValueRef,
    loc: trunk_ir::types::Location,
) -> ValueRef {
    let i32_ty = core::I32::type_ref(ctx);
    let cast = core::UnrealizedConversionCast::operands(index)
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(cast.op_ref());
    cast.result(ctx)
}

/// Pattern for `adt.ref_null` -> `wasm.ref_null`
struct RefNullPattern;

impl OpRewritePattern for RefNullPattern {
    type Op = adt::RefNull;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        ref_null: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = ref_null.op_ref();
        let loc = ctx.op(op).location;
        let result_ty = ref_null.result_ty(ctx);

        let new_op = wasm_gc_dialect::RefNull::operands()
            .target_type(result_ty)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.ref_is_null` -> `wasm.ref_is_null`
struct RefIsNullPattern;

impl OpRewritePattern for RefIsNullPattern {
    type Op = adt::RefIsNull;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        ref_is_null: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = ref_is_null.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = ref_is_null.r#ref(ctx);
        let result_ty = ref_is_null.result_ty(ctx);

        let new_op = wasm_dialect::RefIsNull::operands(ref_val)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// Pattern for `adt.ref_cast` -> `wasm.ref_cast`
struct RefCastPattern;

impl OpRewritePattern for RefCastPattern {
    type Op = adt::RefCast;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        ref_cast: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = ref_cast.op_ref();
        let loc = ctx.op(op).location;
        let ref_val = ref_cast.r#ref(ctx);
        let result_ty = ref_cast.result_ty(ctx);

        let new_op = wasm_gc_dialect::RefCast::operands(ref_val)
            .target_type(result_ty)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::dialect::wasm;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir_wasm_backend::gc_types::FIRST_USER_TYPE_IDX;

    /// Declare the module's descriptors and lower it, as the Wasm pipeline does.
    fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
        crate::wasm::descriptors::declare(ctx, module);
        super::lower(ctx, module, type_converter);
    }

    /// The lowered `splice` function's body ops and its returned value.
    fn lower_splice(elements: &str) -> (IrContext, Vec<OpRef>, ValueRef) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !A = adt.array<core.i32>
  wasm.func @splice(%source: !A, %start: core.index, %end: core.index, %x: core.i32, %y: core.i32) -> !A {{
    %result = adt.array_splice %source[%start, %end]({elements}) attributes {{type = !A}} : !A
    wasm.return %result
  }}
}}"#
            ),
        );
        lower(&mut ctx, module, TypeConverter::new());
        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let ops = ctx.block(block).ops.to_vec();
        let ret = *ops.last().unwrap();
        let returned = ctx.op_operands(ret)[0];
        (ctx, ops, returned)
    }

    fn op_names(ctx: &IrContext, ops: &[OpRef]) -> Vec<String> {
        ops.iter()
            .map(|&op| ctx.op(op).name.display(ctx).to_string())
            .collect()
    }

    #[test]
    fn array_splice_fills_with_its_first_element_and_copies_the_kept_ranges() {
        let (ctx, ops, returned) = lower_splice("%x, %y");

        assert_eq!(
            op_names(&ctx, &ops),
            [
                "core.unrealized_conversion_cast",
                "core.unrealized_conversion_cast",
                "core.unrealized_conversion_cast",
                "wasm.array_len",
                "wasm.i32_gt_u",
                "wasm.i32_gt_u",
                "wasm.i32_or",
                "wasm.if",
                "wasm.i32_const",
                "wasm.i32_sub",
                "wasm.i32_add",
                "wasm.i32_add",
                "wasm_gc.array_new",
                "wasm.i32_const",
                "wasm_gc.array_copy",
                "wasm.i32_const",
                "wasm.i32_add",
                "wasm_gc.array_set",
                "wasm_gc.array_copy",
                "wasm.return",
            ]
        );
        let fresh = wasm_gc_dialect::ArrayNew::from_op(&ctx, ops[12]).unwrap();
        assert_eq!(returned, fresh.result(&ctx));
        let block_arg = |index| {
            let func_block = ctx.op(ops[0]).parent_block.unwrap();
            ctx.block_arg(func_block, index)
        };
        assert_eq!(fresh.init(&ctx), block_arg(3));
        let store = wasm_gc_dialect::ArraySet::from_op(&ctx, ops[17]).unwrap();
        assert_eq!(store.value(&ctx), block_arg(4));
        // The source, still of its `adt.array` type here, is cast to its GC
        // array.
        let tail = wasm_gc_dialect::ArrayCopy::from_op(&ctx, ops[18]).unwrap();
        assert_eq!(ctx.op_operands(ops[0])[0], block_arg(0));
        assert_eq!(tail.src(&ctx), ctx.op_result(ops[0], 0));
        assert_eq!(tail.dst_offset(&ctx), ctx.op_result(ops[10], 0));
    }

    #[test]
    fn array_splice_without_elements_fills_a_nonempty_result_from_its_source() {
        let (ctx, ops, returned) = lower_splice("");

        // The first `wasm.if` is the range check.
        let fill = ops
            .iter()
            .rev()
            .find_map(|&op| wasm::If::from_op(&ctx, op).ok())
            .expect("the result is chosen by its length");
        assert_eq!(returned, fill.results(&ctx)[0]);
        let region_ops = |region| {
            let block = ctx.region(region).blocks[0];
            op_names(&ctx, &ctx.block(block).ops)
        };
        assert_eq!(
            region_ops(fill.then_region(&ctx)),
            [
                "wasm.i32_const",
                "wasm_gc.array_get",
                "wasm_gc.array_new",
                "wasm.yield"
            ]
        );
        assert_eq!(
            region_ops(fill.else_region(&ctx)),
            ["wasm_gc.array_new_fixed", "wasm.yield"]
        );
        assert!(!op_names(&ctx, &ops).contains(&"wasm_gc.array_set".to_owned()));
    }

    #[test]
    fn canonical_enum_type_requires_an_exact_enum_layout() {
        let mut ctx = IrContext::new();
        let _module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !E = adt.enum<"E", []>
}"#,
        );
        let enum_ty = ctx.type_alias_by_text("E").expect("enum layout");

        assert_eq!(canonical_enum_type(&ctx, enum_ty), Some(enum_ty));
    }

    #[test]
    fn lowers_struct_variant_array_and_reference_operations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<"S", (core.i32)>
  !E = adt.enum<"E", [(core.i32)]>
  !ERef = adt.enum<"E">
  !A = adt.array<core.i32>

  wasm.func @main() -> core.nil {
    %zero = wasm.i32_const 0 : core.i32
    %one = wasm.i32_const 1 : core.i32
    %struct = adt.struct_new(%zero) : (core.i32) -> !S
    %field = adt.struct_get %struct {type = !S, field = 0} : core.i32
    adt.struct_set %struct, %field {type = !S, field = 0}

    %variant = adt.variant_new 0(%one) : (core.i32) -> !E
    %is_some = adt.variant_is %variant {type = !E, tag = 0} : core.i32
    %payload = adt.variant_get %variant {type = !E, tag = 0, field = 0} : core.i32

    %at = arith.constant 0 : core.index
    %count = arith.constant 2 : core.index
    %empty = adt.array_of {type = !A} : !A
    %filled = adt.array_new %count, %zero {type = !A} : !A
    %array = adt.array_of %one, %zero {type = !A} : !A
    %element = adt.array_get %array, %at : core.i32
    adt.array_set %array, %at, %element
    %length = adt.array_len %filled : core.index

    %null = adt.ref_null : !S
    %is_null = adt.ref_is_null %null : !S
    %ref = adt.ref_cast %null : !S to !S
    wasm.return
  }
}"#,
        );

        lower_references(&mut ctx, module);
        lower(&mut ctx, module, TypeConverter::new());

        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let remaining_adt_ops = ctx
            .block(block)
            .ops
            .iter()
            .filter(|&&op| ctx.op(op).name.dialect(&ctx) == adt::DIALECT_NAME)
            .count();
        assert_eq!(remaining_adt_ops, 0);
        assert_eq!(
            ctx.block(block)
                .ops
                .iter()
                .filter(|&&op| ctx.op(op).name.dialect(&ctx) == wasm_gc_dialect::DIALECT_NAME)
                .count(),
            14
        );
        let variant_get = ctx
            .block(block)
            .ops
            .iter()
            .copied()
            .find(|&op| {
                let Some(ty) = ctx.op_attrs(op).get_type("type") else {
                    return false;
                };
                wasm_gc_dialect::StructGet::matches(&ctx, op)
                    && wasm_gc_dialect::Struct::matches(&ctx, ty)
            })
            .expect("lowered variant_get");
        let variant_ty = ctx
            .op_attrs(variant_get)
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
  !S = adt.struct<"S", (core.i32)>
  !Closure = adt.struct<(tribute_rt.convention_table_ref), {layout = "closure"}>
  !E = adt.enum<"E", [(), (core.i32)]>

  wasm.func @main(%env: wasm.anyref) -> core.nil {
    %one = wasm.i32_const 1 : core.i32
    %first = adt.struct_new(%one) : (core.i32) -> !S
    %second = adt.struct_new(%one) : (core.i32) -> !S
    %field = adt.struct_get %first {type = !S, field = 0} : core.i32
    adt.struct_set %first, %field {type = !S, field = 0}
    %closure = adt.struct_new(%one) : (core.i32) -> !Closure
    %table = adt.struct_get %closure {type = !Closure, field = 0} : core.i32
    %some = adt.variant_new 1(%one) : (core.i32) -> !E
    %none = adt.variant_new 0() : () -> !E
    %payload = adt.variant_get %some {type = !E, tag = 1, field = 0} : core.i32
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
        assert_eq!(ctx.op_operands(news[2]).len(), 1, "builtin closure");
        // `E` numbers `None` before `Some`, though `Some` is allocated first.
        assert_eq!(descriptor_number(news[3]), Some(first_user + 2));
        assert_eq!(descriptor_number(news[4]), Some(first_user + 1));
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
  wasm.func @read(%bytes: adt.array<core.i8>, %words: adt.array<core.i32>) -> core.nil {
    %zero = arith.constant 0 : core.index
    %byte = adt.array_get %bytes, %zero : core.i8
    %word = adt.array_get %words, %zero : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = module.op().display(&ctx).to_string();
        assert_eq!(output.matches("wasm_gc.array_get_u").count(), 1, "{output}");
        assert_eq!(output.matches("wasm_gc.array_get ").count(), 1, "{output}");
    }

    #[test]
    fn recursive_list_variant_operations_keep_the_definition_layout() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ListRef = adt.enum<"List">
  !List = adt.enum<"List", [(), (core.i32, !ListRef)]>

  wasm.func @main(%input: !ListRef) -> core.nil {
    %zero = wasm.i32_const 0 : core.i32
    %empty = adt.variant_new 0() : () -> !ListRef
    %list = adt.variant_new 1(%zero, %empty) : (core.i32, !ListRef) -> !ListRef
    %is_cons = adt.variant_is %input {type = !List, tag = 1} : core.i1
    %tail = adt.variant_get %input {type = !List, tag = 1, field = 1} : !ListRef
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
                use wasm_gc_dialect::{RefCast, RefTest, StructGet, StructNew};
                if StructNew::matches(&ctx, op) || StructGet::matches(&ctx, op) {
                    ctx.op_attrs(op).get_type("type")
                } else if RefTest::matches(&ctx, op) || RefCast::matches(&ctx, op) {
                    ctx.op_attrs(op).get_type("target_type")
                } else {
                    None
                }
            })
            .collect();
        let list = ctx.type_alias_by_text("List").expect("list layout");
        let cons = variant_type(&mut ctx, list, 1).expect("Cons layout");
        let empty = variant_type(&mut ctx, list, 0).expect("Empty layout");
        let described = crate::wasm::type_converter::described_adt_type(&mut ctx);

        // `variant_is` reads the descriptor through the `Described` type
        // instead of testing for the variant's type.
        assert_eq!(
            variant_types,
            [empty, cons, described, described, cons, cons]
        );
        assert_ne!(empty, cons);
        // The recursive tail is the abstract struct supertype.
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let structref = wasm_dialect::Structref::params()
            .intern(&mut ctx)
            .as_type_ref();
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
  !ERef = adt.enum<"E">
  !E = adt.enum<"E", [(), (core.i32), ()]>

  wasm.func @main(%input: !ERef) -> core.nil {
    %one = wasm.i32_const 1 : core.i32
    %none = adt.variant_new 0() : () -> !ERef
    %some = adt.variant_new 1(%one) : (core.i32) -> !ERef
    %is_some = adt.variant_is %input {type = !E, tag = 1} : core.i32
    %is_other = adt.variant_is %input {type = !E, tag = 2} : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let printed = module.op().display(&ctx).to_string();
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
    fn variant_tag_subtracts_the_number_of_the_enums_first_variant() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ERef = adt.enum<"E">
  !E = adt.enum<"E", [(), (core.i32), ()]>
  !FRef = adt.enum<"F">
  !F = adt.enum<"F", [(), ()]>

  wasm.func @main(%input: !ERef, %never: !FRef) -> core.nil {
    %one = wasm.i32_const 1 : core.i32
    %some = adt.variant_new 1(%one) : (core.i32) -> !ERef
    %tag =adt.variant_tag %input {type = !E}
    %unallocated = adt.variant_tag %never {type = !F}
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let printed = module.op().display(&ctx).to_string();
        assert!(!printed.contains("adt.variant_tag"), "{printed}");
        let func = module.ops(&ctx)[0];
        let block = ctx.region(ctx.op_region(func, 0).unwrap()).blocks[0];
        let defining_op = |value| match ctx.value_def(value) {
            trunk_ir::refs::ValueDef::OpResult(op, _) => op,
            other => panic!("expected an operation result, found {other:?}"),
        };
        let casts: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| core::UnrealizedConversionCast::from_op(&ctx, op).ok())
            .collect();
        let [tag, unallocated] = casts[..] else {
            panic!("two tags cast to core.index: {printed}")
        };

        let sub = wasm::I32Sub::from_op(&ctx, defining_op(tag.value(&ctx)))
            .expect("descriptor less the first number");
        let descriptor = wasm_gc_dialect::StructGet::from_op(&ctx, defining_op(sub.lhs(&ctx)))
            .expect("descriptor read");
        assert_eq!(descriptor.field_idx(&ctx), 0);
        // `None` is never allocated, but `E` numbers it first, before `Some`.
        let first =
            wasm::I32Const::from_op(&ctx, defining_op(sub.rhs(&ctx))).expect("first number");
        assert_eq!(first.value(&ctx), FIRST_USER_TYPE_IDX as i32);

        // `F` has no value, so its tag is a constant.
        let constant = wasm::I32Const::from_op(&ctx, defining_op(unallocated.value(&ctx)))
            .expect("constant tag");
        assert_eq!(constant.value(&ctx), 0);
    }

    #[test]
    fn variant_get_casts_its_operand_unless_it_names_another_enum() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ARef = adt.enum<"A">
  !BRef = adt.enum<"B">
  !A = adt.enum<"A", [(core.i32), (core.f64)]>
  !B = adt.enum<"B", [(core.f64)]>

  wasm.func @main(%from_a_ref: !ARef, %from_b_ref: !BRef, %x: core.f64) -> core.nil {
    %from_matching_enum = adt.variant_get %from_a_ref {type = !A, tag = 0, field = 0} : core.i32
    %from_mismatched_enum = adt.variant_get %from_b_ref {type = !A, tag = 0, field = 0} : core.i32
    %from_b = adt.variant_new 0(%x) : (core.f64) -> !BRef
    %wrong_enum = adt.variant_get %from_b {type = !A, tag = 0, field = 0} : core.i32
    %from_a_other = adt.variant_new 1(%x) : (core.f64) -> !ARef
    %wrong_tag = adt.variant_get %from_a_other {type = !A, tag = 0, field = 0} : core.i32
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
        assert_eq!(remaining_variant_gets, 1, "only the !BRef operand stays");
        let casts: Vec<_> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| wasm_gc_dialect::RefCast::from_op(&ctx, op).ok())
            .collect();
        assert_eq!(casts.len(), 3);
        assert_eq!(casts[0].r#ref(&ctx), function_args[0]);
        let a = ctx.type_alias_by_text("A").expect("A layout");
        let some_ty = variant_type(&mut ctx, a, 0).expect("Some layout");
        assert!(casts.iter().all(|cast| cast.target_type(&ctx) == some_ty));
    }

    #[test]
    fn variant_get_requires_declared_result_type_except_bytes_anyref_erasure() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !ERef = adt.enum<"E">
  !BoxRef = adt.enum<"Box">
  !NodeRef = adt.enum<"Node">
  !StringRef = adt.enum<"String">
  !E = adt.enum<"E", [(core.i32)]>
  !Box = adt.enum<"Box", [(!NodeRef)]>
  !Node = adt.enum<"Node", [()]>
  !Data = adt.array<core.i8, {layout = "bytes_data"}>
  !Bytes = adt.struct<(!Data, core.i32, core.i32), {layout = "bytes"}>
  !String = adt.enum<"String", [(!Bytes)]>

  wasm.func @main(%e: !ERef, %box: !BoxRef, %string: !StringRef) -> core.nil {
    %valid = adt.variant_get %e {type = !E, tag = 0, field = 0} : core.i32
    %invalid = adt.variant_get %e {type = !E, tag = 0, field = 0} : core.i64
    %node = adt.variant_get %box {type = !Box, tag = 0, field = 0} : wasm.structref
    %bytes = adt.variant_get %string {type = !String, tag = 0, field = 0} : wasm.anyref
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
            data.name.is(&ctx, "core.i32")
        }));
        assert!(lowered_result_types.iter().any(|&ty| has_runtime_layout(
            &ctx,
            ty,
            runtime_layout::BYTES
        )));
        assert!(lowered_result_types.iter().any(|&ty| {
            let data = ctx.get_type(ty);
            data.name.is(&ctx, "wasm.structref")
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

    /// Lower `body` as the body of `@main(%e: !ERef, %c: core.i32)`, where
    /// variant 0 of `!E` has the fields `(core.i32, core.f64)`, and return the
    /// function's casts and field reads.
    fn lower_variant_reads(
        body: &str,
    ) -> (
        IrContext,
        Vec<wasm_gc_dialect::RefCast>,
        Vec<wasm_gc_dialect::StructGet>,
    ) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !ERef = adt.enum<"E">
  !E = adt.enum<"E", [(core.i32, core.f64), (core.f64)]>

  wasm.func @main(%e: !ERef, %c: core.i32) -> core.nil {{
{body}
  }}
}}"#
            ),
        );
        lower(&mut ctx, module, TypeConverter::new());
        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let mut casts = Vec::new();
        let mut gets = Vec::new();
        let _ = trunk_ir::walk::walk_region::<()>(&ctx, body, &mut |op| {
            if let Ok(cast) = wasm_gc_dialect::RefCast::from_op(&ctx, op) {
                casts.push(cast);
            }
            if let Ok(get) = wasm_gc_dialect::StructGet::from_op(&ctx, op) {
                gets.push(get);
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        (ctx, casts, gets)
    }

    #[test]
    fn variant_field_reads_share_a_dominating_cast() {
        let (ctx, casts, gets) = lower_variant_reads(
            r#"    %x = adt.variant_get %e {type = !E, tag = 0, field = 0} : core.i32
    %y = adt.variant_get %e {type = !E, tag = 0, field = 1} : core.f64
    %r = wasm.if %c -> (core.f64) {
      %z = adt.variant_get %e {type = !E, tag = 0, field = 1} : core.f64
      wasm.yield %z
    } else {
      wasm.yield %y
    }
    wasm.return"#,
        );

        assert_eq!(casts.len(), 1, "one cast serves every read it dominates");
        let cast = casts[0].result(&ctx);
        assert_eq!(gets.len(), 3);
        assert!(gets.iter().all(|get| get.r#ref(&ctx) == cast));
    }

    #[test]
    fn variant_field_reads_in_sibling_regions_cast_separately() {
        let (ctx, casts, gets) = lower_variant_reads(
            r#"    %r = wasm.if %c -> (core.f64) {
      %x = adt.variant_get %e {type = !E, tag = 0, field = 0} : core.i32
      %y = adt.variant_get %e {type = !E, tag = 0, field = 1} : core.f64
      wasm.yield %y
    } else {
      %z = adt.variant_get %e {type = !E, tag = 0, field = 1} : core.f64
      %w = adt.variant_get %e {type = !E, tag = 1, field = 0} : core.f64
      wasm.yield %w
    }
    wasm.return"#,
        );

        // One cast per arm for tag 0, and one for tag 1, which has another type.
        assert_eq!(casts.len(), 3);
        let uses_of = |cast: &wasm_gc_dialect::RefCast| {
            let result = cast.result(&ctx);
            gets.iter().filter(|get| get.r#ref(&ctx) == result).count()
        };
        assert_eq!(casts.iter().map(uses_of).collect::<Vec<_>>(), [2, 1, 1]);
    }

    #[test]
    fn variant_field_reads_in_other_blocks_cast_separately() {
        let (ctx, casts, gets) = lower_variant_reads(
            r#"  ^bb0:
    %x = adt.variant_get %e {type = !E, tag = 0, field = 0} : core.i32
    %y = adt.variant_get %e {type = !E, tag = 0, field = 1} : core.f64
    cf.br [^bb1]
  ^bb1:
    %z = adt.variant_get %e {type = !E, tag = 0, field = 1} : core.f64
    wasm.return"#,
        );

        // Block order is not dominance, so each block casts once.
        assert_eq!(casts.len(), 2);
        assert_eq!(gets.len(), 3);
        assert_eq!(gets[0].r#ref(&ctx), casts[0].result(&ctx));
        assert_eq!(gets[1].r#ref(&ctx), casts[0].result(&ctx));
        assert_eq!(gets[2].r#ref(&ctx), casts[1].result(&ctx));
    }
}
