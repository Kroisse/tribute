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

use crate::dialect::{adt, core};
use crate::rewrite::TypeConverter;
use crate::{DialectType, Symbol, Type};

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

/// Compute the memory layout for an `adt.struct` type.
///
/// Uses the `TypeConverter` to determine the native size of each field type.
/// Returns `None` if the type is not an `adt.struct` or fields cannot be extracted.
pub fn compute_struct_layout<'db>(
    db: &'db dyn salsa::Database,
    struct_ty: Type<'db>,
    type_converter: &TypeConverter,
) -> Option<StructLayout> {
    let fields = adt::get_struct_fields(db, struct_ty)?;

    let mut offset: u32 = 0;
    let mut max_align: u32 = 1;
    let mut field_offsets = Vec::with_capacity(fields.len());

    for (_name, field_ty) in &fields {
        // Convert the field type to its native representation
        let native_ty = type_converter
            .convert_type(db, *field_ty)
            .unwrap_or(*field_ty);
        let (size, align) = type_size_align(db, native_ty);

        // Align offset
        offset = (offset + align - 1) & !(align - 1);
        field_offsets.push(offset);
        offset += size;
        max_align = max_align.max(align);
    }

    // Pad total size to alignment
    let total_size = (offset + max_align - 1) & !(max_align - 1);

    Some(StructLayout {
        field_offsets,
        total_size,
        alignment: max_align,
    })
}

// =============================================================================
// Enum Layout
// =============================================================================

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

/// Compute the memory layout for an `adt.enum` type.
///
/// Uses the `TypeConverter` to determine the native size of each field type.
/// Returns `None` if the type is not an `adt.enum` or variants cannot be extracted.
pub fn compute_enum_layout<'db>(
    db: &'db dyn salsa::Database,
    enum_ty: Type<'db>,
    type_converter: &TypeConverter,
) -> Option<EnumLayout> {
    let variants = adt::get_enum_variants(db, enum_ty)?;

    let mut variant_layouts = Vec::with_capacity(variants.len());
    let mut max_fields_size: u32 = 0;
    let mut max_align: u32 = 8; // minimum alignment = 8 (pointer-aligned)

    for (tag_value, (variant_name, field_types)) in variants.iter().enumerate() {
        let mut offset: u32 = 0;
        let mut field_offsets = Vec::with_capacity(field_types.len());

        for field_ty in field_types {
            let native_ty = type_converter
                .convert_type(db, *field_ty)
                .unwrap_or(*field_ty);
            let (size, align) = type_size_align(db, native_ty);

            // Align offset
            offset = (offset + align - 1) & !(align - 1);
            field_offsets.push(offset);
            offset += size;
            max_align = max_align.max(align);
        }

        // Pad to alignment
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
pub fn find_variant_layout<'a>(
    layout: &'a EnumLayout,
    tag: Symbol,
) -> Option<&'a VariantFieldLayout> {
    layout.variant_layouts.iter().find(|v| v.name == tag)
}

/// Get the size and alignment of a native type in bytes.
///
/// After type conversion, all types should be one of the core types.
/// Unknown types default to pointer size (8 bytes) for safety.
pub fn type_size_align(db: &dyn salsa::Database, ty: Type<'_>) -> (u32, u32) {
    if core::I8::from_type(db, ty).is_some() {
        return (1, 1);
    }
    if core::I16::from_type(db, ty).is_some() {
        return (2, 2);
    }
    if core::I32::from_type(db, ty).is_some() {
        return (4, 4);
    }
    if core::I64::from_type(db, ty).is_some() {
        return (8, 8);
    }
    if core::F32::from_type(db, ty).is_some() {
        return (4, 4);
    }
    if core::F64::from_type(db, ty).is_some() {
        return (8, 8);
    }
    if core::Ptr::from_type(db, ty).is_some() {
        return (8, 8);
    }
    // Default: treat as pointer (8 bytes on 64-bit)
    (8, 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbol;
    use salsa_test_macros::salsa_test;

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

    #[salsa_test]
    fn test_empty_struct_layout(db: &salsa::DatabaseImpl) {
        let struct_ty = adt::struct_type(db, Symbol::new("Empty"), vec![]);
        let layout = compute_struct_layout(db, struct_ty, &test_converter()).unwrap();

        assert!(layout.field_offsets.is_empty());
        assert_eq!(layout.total_size, 0);
        assert_eq!(layout.alignment, 1);
    }

    #[salsa_test]
    fn test_single_i32_field(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let struct_ty =
            adt::struct_type(db, Symbol::new("Wrapper"), vec![(Symbol::new("x"), i32_ty)]);
        let layout = compute_struct_layout(db, struct_ty, &test_converter()).unwrap();

        assert_eq!(layout.field_offsets, vec![0]);
        assert_eq!(layout.total_size, 4);
        assert_eq!(layout.alignment, 4);
    }

    #[salsa_test]
    fn test_two_i32_fields(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );
        let layout = compute_struct_layout(db, struct_ty, &test_converter()).unwrap();

        assert_eq!(layout.field_offsets, vec![0, 4]);
        assert_eq!(layout.total_size, 8);
        assert_eq!(layout.alignment, 4);
    }

    #[salsa_test]
    fn test_mixed_types_with_padding(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let i64_ty = core::I64::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Mixed"),
            vec![(Symbol::new("a"), i32_ty), (Symbol::new("b"), i64_ty)],
        );
        let layout = compute_struct_layout(db, struct_ty, &test_converter()).unwrap();

        assert_eq!(layout.field_offsets, vec![0, 8]);
        assert_eq!(layout.total_size, 16);
        assert_eq!(layout.alignment, 8);
    }

    #[salsa_test]
    fn test_ptr_fields(db: &salsa::DatabaseImpl) {
        let ptr_ty = core::Ptr::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Closure"),
            vec![
                (Symbol::new("func"), ptr_ty),
                (Symbol::new("env"), ptr_ty),
                (Symbol::new("tag"), i32_ty),
            ],
        );
        let layout = compute_struct_layout(db, struct_ty, &test_converter()).unwrap();

        assert_eq!(layout.field_offsets, vec![0, 8, 16]);
        assert_eq!(layout.total_size, 24);
        assert_eq!(layout.alignment, 8);
    }

    #[salsa_test]
    fn test_not_struct_returns_none(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        assert!(compute_struct_layout(db, i32_ty, &test_converter()).is_none());
    }

    // === Enum layout tests ===

    #[salsa_test]
    fn test_enum_empty_variants(db: &salsa::DatabaseImpl) {
        // enum Color { Red, Green, Blue }
        let enum_ty = adt::enum_type(
            db,
            Symbol::new("Color"),
            vec![
                (Symbol::new("Red"), vec![]),
                (Symbol::new("Green"), vec![]),
                (Symbol::new("Blue"), vec![]),
            ],
        );
        let layout = compute_enum_layout(db, enum_ty, &test_converter()).unwrap();

        assert_eq!(layout.tag_offset, 0);
        assert_eq!(layout.tag_size, 4);
        assert_eq!(layout.fields_offset, ENUM_FIELDS_OFFSET);
        assert_eq!(layout.variant_layouts.len(), 3);
        assert_eq!(layout.variant_layouts[0].tag_value, 0);
        assert_eq!(layout.variant_layouts[1].tag_value, 1);
        assert_eq!(layout.variant_layouts[2].tag_value, 2);
        // All variants have no fields
        assert!(layout.variant_layouts[0].field_offsets.is_empty());
        // total_size = 8 (tag + padding) + 0 (no fields)
        assert_eq!(layout.total_size, 8);
    }

    #[salsa_test]
    fn test_enum_with_fields(db: &salsa::DatabaseImpl) {
        // enum Shape { Circle(i32), Square(i32) }
        let i32_ty = core::I32::new(db).as_type();
        let enum_ty = adt::enum_type(
            db,
            Symbol::new("Shape"),
            vec![
                (Symbol::new("Circle"), vec![i32_ty]),
                (Symbol::new("Square"), vec![i32_ty]),
            ],
        );
        let layout = compute_enum_layout(db, enum_ty, &test_converter()).unwrap();

        assert_eq!(layout.fields_offset, ENUM_FIELDS_OFFSET);
        // Each variant has one i32 field at offset 0 (relative to fields_offset)
        assert_eq!(layout.variant_layouts[0].field_offsets, vec![0]);
        assert_eq!(layout.variant_layouts[1].field_offsets, vec![0]);
        // total_size = 8 (tag + padding) + 8 (padded i32)
        assert_eq!(layout.total_size, 16);
    }

    #[salsa_test]
    fn test_enum_mixed_variants(db: &salsa::DatabaseImpl) {
        // enum Expr { Lit(i32), Add(ptr, ptr) }
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();
        let enum_ty = adt::enum_type(
            db,
            Symbol::new("Expr"),
            vec![
                (Symbol::new("Lit"), vec![i32_ty]),
                (Symbol::new("Add"), vec![ptr_ty, ptr_ty]),
            ],
        );
        let layout = compute_enum_layout(db, enum_ty, &test_converter()).unwrap();

        // Lit variant: one i32 field at offset 0
        assert_eq!(layout.variant_layouts[0].field_offsets, vec![0]);
        // Add variant: two ptr fields at offsets 0, 8
        assert_eq!(layout.variant_layouts[1].field_offsets, vec![0, 8]);
        // max variant size = 16 (two ptrs), total = 8 + 16 = 24
        assert_eq!(layout.total_size, 24);
    }

    #[salsa_test]
    fn test_find_variant_layout(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let enum_ty = adt::enum_type(
            db,
            Symbol::new("Shape"),
            vec![
                (Symbol::new("Circle"), vec![i32_ty]),
                (Symbol::new("Square"), vec![i32_ty]),
            ],
        );
        let layout = compute_enum_layout(db, enum_ty, &test_converter()).unwrap();

        let circle = find_variant_layout(&layout, Symbol::new("Circle")).unwrap();
        assert_eq!(circle.tag_value, 0);

        let square = find_variant_layout(&layout, Symbol::new("Square")).unwrap();
        assert_eq!(square.tag_value, 1);

        assert!(find_variant_layout(&layout, Symbol::new("Triangle")).is_none());
    }

    #[salsa_test]
    fn test_not_enum_returns_none(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        assert!(compute_enum_layout(db, i32_ty, &test_converter()).is_none());
    }
}
