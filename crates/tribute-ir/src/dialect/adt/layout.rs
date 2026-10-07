//! ADT memory layout computation.
//!
//! Computes field offsets, sizes, and alignment for struct and enum types.
//! Uses natural alignment: each field is aligned to its own size.
//!
//! ## Layout rules
//!
//! - Fields are laid out in declaration order
//! - Each field is naturally aligned (aligned to its own size)
//! - Total struct size is padded to the maximum field alignment
//!
//! ## Enum layout
//!
//! A variant object holds only its own fields, laid out like a struct from
//! offset 0, and is allocated with the size of those fields. Which variant it
//! is comes from the object's runtime type descriptor, not from the payload.
//!
//! ## Size mapping
//!
//! | Type        | Size | Alignment |
//! |-------------|------|-----------|
//! | `core.i8`   | 1    | 1         |
//! | `core.i16`  | 2    | 2         |
//! | `core.i32`  | 4    | 4         |
//! | `core.i64`  | 8    | 8         |
//! | `core.f32`  | 4    | 4         |
//! | `core.f64`  | 8    | 8         |
//! | `core.i1`   | 4    | 4         |
//! | `core.ptr`  | 8    | 8         |
//! | other       | 8    | 8         |

use trunk_ir::context::IrContext;
use trunk_ir::dialect::mem;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::type_converter::TypeConverter;
use trunk_ir::types::StringRef;

/// Memory layout of a struct type.
#[derive(Debug, Clone)]
pub struct StructLayout {
    /// Byte offset of each field.
    pub field_offsets: Vec<u32>,
    /// Total size in bytes (padded to alignment).
    pub total_size: u32,
    /// Maximum alignment of any field.
    pub alignment: u32,
}

/// Get the size and alignment of a native type in bytes.
///
/// After type conversion, all types should be one of the core types.
/// Unknown types default to pointer size (8 bytes) for safety.
pub fn type_size_align(ctx: &IrContext, ty: TypeRef) -> (u32, u32) {
    let data = ctx.get_type(ty);
    if data.dialect != "core" {
        return (8, 8);
    }
    let name = data.name.clone();
    if name == "i8" {
        (1, 1)
    } else if name == "i16" {
        (2, 2)
    } else if name == "i32" || name == "i1" {
        (4, 4)
    } else if name == "i64" {
        (8, 8)
    } else if name == "f32" {
        (4, 4)
    } else {
        // f64, ptr, and any unknown types default to 8-byte size/align
        (8, 8)
    }
}

/// Extract struct fields from an arena TypeRef.
///
/// Returns `None` if the type is not a valid `adt.struct`.
pub fn get_struct_fields(ctx: &IrContext, ty: TypeRef) -> Option<Vec<(StringRef, TypeRef)>> {
    let adt_struct = super::Struct::from_type_ref(ctx, ty)?;
    Some(
        (0..adt_struct.field_count(ctx))
            .map(|index| {
                let name = adt_struct
                    .field_name_ref(ctx, index)
                    .expect("field index is in range");
                let ty = adt_struct
                    .field_type(ctx, index)
                    .expect("field index is in range");
                (name, ty)
            })
            .collect(),
    )
}

/// Extract enum variants from an arena TypeRef.
///
/// Returns `None` if the type is not a valid `adt.enum`.
pub fn get_enum_variants(ctx: &IrContext, ty: TypeRef) -> Option<Vec<(StringRef, Vec<TypeRef>)>> {
    let adt_enum = super::Enum::from_type_ref(ctx, ty)?;
    Some(
        adt_enum
            .variants(ctx)
            .map(|(name, fields)| (name, fields.to_vec()))
            .collect(),
    )
}

/// Lay out fields of the given native types in order, each at its natural
/// alignment.
fn natural_layout(ctx: &IrContext, fields: impl IntoIterator<Item = TypeRef>) -> StructLayout {
    let mut offset: u32 = 0;
    let mut max_align: u32 = 1;
    let mut field_offsets = Vec::new();

    for native_ty in fields {
        let (size, align) = type_size_align(ctx, native_ty);

        offset = (offset + align - 1) & !(align - 1);
        field_offsets.push(offset);
        offset += size;
        max_align = max_align.max(align);
    }

    let total_size = (offset + max_align - 1) & !(max_align - 1);

    StructLayout {
        field_offsets,
        total_size,
        alignment: max_align,
    }
}

/// Compute the memory layout for an `adt.struct` type.
///
/// Uses the `TypeConverter` to determine the native size of each field type.
/// Returns `None` if the type is not an `adt.struct` or fields cannot be extracted.
pub fn compute_struct_layout(
    ctx: &IrContext,
    struct_ty: TypeRef,
    type_converter: &TypeConverter,
) -> Option<StructLayout> {
    let fields = get_struct_fields(ctx, struct_ty)?;
    Some(natural_layout(
        ctx,
        fields
            .into_iter()
            .map(|(_name, field_ty)| type_converter.convert_type_or_identity(ctx, field_ty)),
    ))
}

/// Compute the memory layout of a `mem.struct` type.
///
/// Its fields are already target representations, so no type converter is
/// involved. Returns `None` if the type is not a `mem.struct`.
pub fn compute_mem_struct_layout(ctx: &IrContext, struct_ty: TypeRef) -> Option<StructLayout> {
    let fields = mem::Struct::from_type_ref(ctx, struct_ty)?.fields(ctx);
    Some(natural_layout(ctx, fields.iter().copied()))
}

/// Compute the memory layout of the variant `tag` of an `adt.enum` type.
///
/// A variant object holds only its own fields, so its layout is that of a
/// struct with those fields. Uses the `TypeConverter` to determine the native
/// size of each field type. Returns `None` if the type is not an `adt.enum`
/// or has no such variant.
pub fn compute_variant_layout(
    ctx: &IrContext,
    enum_ty: TypeRef,
    tag: StringRef,
    type_converter: &TypeConverter,
) -> Option<StructLayout> {
    let fields = super::Enum::from_type_ref(ctx, enum_ty)?.variant_fields(ctx, tag)?;
    Some(natural_layout(
        ctx,
        fields
            .iter()
            .map(|&field_ty| type_converter.convert_type_or_identity(ctx, field_ty)),
    ))
}
