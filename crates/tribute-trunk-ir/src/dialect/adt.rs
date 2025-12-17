//! Algebraic Data Type dialect operations.
//!
//! Target-independent operations for structs, variants (enums), arrays, and references.

use crate::dialect;

dialect! {
    mod adt {
        // === Struct (Product Type) ===

        /// `adt.struct_new` operation: creates a struct instance.
        #[attr(r#type: Type)]
        fn struct_new(#[rest] fields) -> result;

        /// `adt.struct_get` operation: reads a field from a struct.
        /// Field can be a name (String) or index (IntBits).
        #[attr(field: any)]
        fn struct_get(r#ref) -> result;

        /// `adt.struct_set` operation: writes a field in a struct.
        /// Field can be a name (String) or index (IntBits).
        #[attr(field: any)]
        fn struct_set(r#ref, value);

        // === Variant (Sum Type) ===

        /// `adt.variant_new` operation: creates a variant instance.
        #[attr(r#type: Type, tag: Symbol)]
        fn variant_new(#[rest] fields) -> result;

        /// `adt.variant_tag` operation: reads the tag from a variant.
        fn variant_tag(r#ref) -> result;

        /// `adt.variant_get` operation: reads a field from a variant.
        /// Field can be a name (String) or index (IntBits).
        #[attr(field: any)]
        fn variant_get(r#ref) -> result;

        // === Array ===

        /// `adt.array_new` operation: creates an array.
        /// First operand is size, optional second is init value.
        #[attr(r#type: Type)]
        fn array_new(#[rest] operands) -> result;

        /// `adt.array_get` operation: reads an element from an array.
        fn array_get(r#ref, index) -> result;

        /// `adt.array_set` operation: writes an element to an array.
        fn array_set(r#ref, index, value);

        /// `adt.array_len` operation: returns the length of an array.
        fn array_len(r#ref) -> result;

        // === Reference ===

        /// `adt.ref_null` operation: creates a null reference.
        #[attr(r#type: Type)]
        fn ref_null() -> result;

        /// `adt.ref_is_null` operation: checks if a reference is null.
        fn ref_is_null(r#ref) -> result;

        /// `adt.ref_cast` operation: casts a reference type (runtime checked).
        #[attr(r#type: Type)]
        fn ref_cast(r#ref) -> result;

        // === Literals ===

        /// `adt.string_const` operation: string constant.
        #[attr(value: String)]
        fn string_const() -> result;

        /// `adt.bytes_const` operation: byte array constant.
        #[attr(value: any)]
        fn bytes_const() -> result;
    }
}
