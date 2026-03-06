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
//! ## Enum layout (tagged union)
//!
//! ```text
//! [Payload]
//!   offset 0: tag (i32)         discriminant (0, 1, 2, ...)
//!   offset 4: <padding 4B>      fixed alignment to 8
//!   offset 8: variant fields    union: sized to max variant
//! ```
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
//! | `core.ptr`  | 8    | 8         |
//! | other       | 8    | 8         |

use crate::Symbol;
use crate::arena::context::IrContext;
use crate::arena::refs::TypeRef;
use crate::arena::rewrite::type_converter::TypeConverter;
use crate::arena::types::Attribute;

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

/// Fixed offset from payload start to variant fields (tag 4B + padding 4B).
pub const ENUM_FIELDS_OFFSET: u32 = 8;

/// Memory layout of an enum type (tagged union).
///
/// All variants share the same allocation, sized to the largest variant.
/// Tag is stored at offset 0 (i32), fields start at offset 8.
#[derive(Debug, Clone)]
pub struct EnumLayout {
    /// Byte offset of the tag field (always 0).
    pub tag_offset: u32,
    /// Size of the tag field in bytes (always 4 = i32).
    pub tag_size: u32,
    /// Byte offset where variant fields begin (always 8).
    pub fields_offset: u32,
    /// Layout for each variant, in declaration order.
    pub variant_layouts: Vec<VariantFieldLayout>,
    /// Total payload size (8 + max variant fields size).
    pub total_size: u32,
    /// Overall alignment.
    pub alignment: u32,
}

/// Layout of a single variant's fields within the union payload.
#[derive(Debug, Clone)]
pub struct VariantFieldLayout {
    /// Variant name.
    pub name: Symbol,
    /// Discriminant value (0, 1, 2, ...).
    pub tag_value: u32,
    /// Field offsets relative to `fields_offset`.
    pub field_offsets: Vec<u32>,
    /// Total size of this variant's fields.
    pub fields_size: u32,
}

/// Get the size and alignment of a native type in bytes.
///
/// After type conversion, all types should be one of the core types.
/// Unknown types default to pointer size (8 bytes) for safety.
pub fn type_size_align(ctx: &IrContext, ty: TypeRef) -> (u32, u32) {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("core") {
        return (8, 8);
    }
    let name = data.name;
    if name == Symbol::new("i8") {
        (1, 1)
    } else if name == Symbol::new("i16") {
        (2, 2)
    } else if name == Symbol::new("i32") || name == Symbol::new("i1") {
        (4, 4)
    } else if name == Symbol::new("i64") {
        (8, 8)
    } else if name == Symbol::new("f32") {
        (4, 4)
    } else {
        // f64, ptr, and any unknown types default to 8-byte size/align
        (8, 8)
    }
}

/// Extract struct fields from an arena TypeRef.
///
/// Returns `None` if the type is not `adt.struct`.
pub fn get_struct_fields(ctx: &IrContext, ty: TypeRef) -> Option<Vec<(Symbol, TypeRef)>> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return None;
    }

    let fields_attr = data.attrs.get(&Symbol::new("fields"))?;
    let Attribute::List(fields) = fields_attr else {
        return None;
    };

    let mut result = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let Attribute::List(pair) = field else {
            panic!("get_struct_fields: field[{i}] expected List, got {field:?}");
        };
        assert!(
            pair.len() >= 2,
            "get_struct_fields: field[{i}] pair too short (len={})",
            pair.len()
        );
        let Attribute::Symbol(name) = &pair[0] else {
            panic!(
                "get_struct_fields: field[{i}] name expected Symbol, got {:?}",
                pair[0]
            );
        };
        let Attribute::Type(field_ty) = &pair[1] else {
            panic!(
                "get_struct_fields: field[{i}] type expected Type, got {:?}",
                pair[1]
            );
        };
        result.push((*name, *field_ty));
    }

    Some(result)
}

/// Extract enum variants from an arena TypeRef.
///
/// Returns `None` if the type is not `adt.enum`.
pub fn get_enum_variants(ctx: &IrContext, ty: TypeRef) -> Option<Vec<(Symbol, Vec<TypeRef>)>> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("enum") {
        return None;
    }

    let variants_attr = data.attrs.get(&Symbol::new("variants"))?;
    let Attribute::List(variants) = variants_attr else {
        return None;
    };

    let mut result = Vec::new();
    for (i, variant) in variants.iter().enumerate() {
        let Attribute::List(pair) = variant else {
            panic!("get_enum_variants: variant[{i}] expected List, got {variant:?}");
        };
        assert!(
            pair.len() >= 2,
            "get_enum_variants: variant[{i}] pair too short (len={})",
            pair.len()
        );
        let Attribute::Symbol(name) = &pair[0] else {
            panic!(
                "get_enum_variants: variant[{i}] name expected Symbol, got {:?}",
                pair[0]
            );
        };
        let Attribute::List(field_types_attr) = &pair[1] else {
            panic!(
                "get_enum_variants: variant[{i}] fields expected List, got {:?}",
                pair[1]
            );
        };

        let field_types: Vec<TypeRef> = field_types_attr
            .iter()
            .enumerate()
            .map(|(j, a)| {
                let Attribute::Type(ty) = a else {
                    panic!("get_enum_variants: variant[{i}] field[{j}] expected Type, got {a:?}");
                };
                *ty
            })
            .collect();

        result.push((*name, field_types));
    }

    Some(result)
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

    let mut offset: u32 = 0;
    let mut max_align: u32 = 1;
    let mut field_offsets = Vec::with_capacity(fields.len());

    for (_name, field_ty) in &fields {
        let native_ty = type_converter.convert_type_or_identity(ctx, *field_ty);
        let (size, align) = type_size_align(ctx, native_ty);

        offset = (offset + align - 1) & !(align - 1);
        field_offsets.push(offset);
        offset += size;
        max_align = max_align.max(align);
    }

    let total_size = (offset + max_align - 1) & !(max_align - 1);

    Some(StructLayout {
        field_offsets,
        total_size,
        alignment: max_align,
    })
}

/// Compute the memory layout for an `adt.enum` type.
///
/// Uses the `TypeConverter` to determine the native size of each field type.
/// Returns `None` if the type is not an `adt.enum` or variants cannot be extracted.
pub fn compute_enum_layout(
    ctx: &IrContext,
    enum_ty: TypeRef,
    type_converter: &TypeConverter,
) -> Option<EnumLayout> {
    let variants = get_enum_variants(ctx, enum_ty)?;

    let mut variant_layouts = Vec::with_capacity(variants.len());
    let mut max_fields_size: u32 = 0;
    let mut max_align: u32 = 8;

    for (tag_value, (variant_name, field_types)) in variants.iter().enumerate() {
        let mut offset: u32 = 0;
        let mut field_offsets = Vec::with_capacity(field_types.len());

        for field_ty in field_types {
            let native_ty = type_converter.convert_type_or_identity(ctx, *field_ty);
            let (size, align) = type_size_align(ctx, native_ty);

            offset = (offset + align - 1) & !(align - 1);
            field_offsets.push(offset);
            offset += size;
            max_align = max_align.max(align);
        }

        let fields_size = (offset + max_align - 1) & !(max_align - 1);
        max_fields_size = max_fields_size.max(fields_size);

        variant_layouts.push(VariantFieldLayout {
            name: *variant_name,
            tag_value: tag_value as u32,
            field_offsets,
            fields_size,
        });
    }

    let total_size = ENUM_FIELDS_OFFSET + max_fields_size;

    Some(EnumLayout {
        tag_offset: 0,
        tag_size: 4,
        fields_offset: ENUM_FIELDS_OFFSET,
        variant_layouts,
        total_size,
        alignment: max_align,
    })
}

/// Find the variant layout for a given tag name.
pub fn find_variant_layout(layout: &EnumLayout, tag: Symbol) -> Option<&VariantFieldLayout> {
    layout.variant_layouts.iter().find(|v| v.name == tag)
}
