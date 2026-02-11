//! ADT memory layout computation for the Cranelift native backend.
//!
//! Computes field offsets, sizes, and alignment for struct types.
//! Uses natural alignment: each field is aligned to its own size.
//!
//! ## Layout rules
//!
//! - Fields are laid out in declaration order
//! - Each field is naturally aligned (aligned to its own size)
//! - Total struct size is padded to the maximum field alignment
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

use trunk_ir::dialect::{adt, core};
use trunk_ir::rewrite::TypeConverter;
use trunk_ir::{DialectType, Type};

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
    let total_size = if max_align > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        offset
    };

    Some(StructLayout {
        field_offsets,
        total_size,
        alignment: max_align,
    })
}

/// Get the size and alignment of a native type in bytes.
///
/// After type conversion, all types should be one of the core types.
/// Unknown types default to pointer size (8 bytes) for safety.
fn type_size_align(db: &dyn salsa::Database, ty: Type<'_>) -> (u32, u32) {
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
    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

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
        // struct { a: i32, b: i64 }
        // a at offset 0 (4 bytes), then 4 bytes padding, b at offset 8
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
        // struct { func: ptr, env: ptr, tag: i32 }
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
        assert_eq!(layout.total_size, 24); // 20 bytes + 4 padding to align to 8
        assert_eq!(layout.alignment, 8);
    }

    #[salsa_test]
    fn test_not_struct_returns_none(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        assert!(compute_struct_layout(db, i32_ty, &test_converter()).is_none());
    }
}
