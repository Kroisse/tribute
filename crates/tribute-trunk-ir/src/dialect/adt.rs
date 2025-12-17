//! Algebraic Data Type dialect operations.
//!
//! Target-independent operations for structs, variants (enums), arrays, and references.

use crate::dialect;

dialect! {
    adt {
        // === Struct (Product Type) ===

        /// `adt.struct_new` operation: creates a struct instance.
        op struct_new[r#type](..fields) -> result;

        /// `adt.struct_get` operation: reads a field from a struct.
        op struct_get[field](r#ref) -> result;

        /// `adt.struct_set` operation: writes a field in a struct.
        op struct_set[field](r#ref, value);

        // === Variant (Sum Type) ===

        /// `adt.variant_new` operation: creates a variant instance.
        op variant_new[r#type, tag](..fields) -> result;

        /// `adt.variant_tag` operation: reads the tag from a variant.
        op variant_tag(r#ref) -> result;

        /// `adt.variant_get` operation: reads a field from a variant.
        op variant_get[field](r#ref) -> result;

        // === Array ===

        /// `adt.array_new` operation: creates an array.
        /// First operand is size, optional second is init value.
        op array_new[r#type](..operands) -> result;

        /// `adt.array_get` operation: reads an element from an array.
        op array_get(r#ref, index) -> result;

        /// `adt.array_set` operation: writes an element to an array.
        op array_set(r#ref, index, value);

        /// `adt.array_len` operation: returns the length of an array.
        op array_len(r#ref) -> result;

        // === Reference ===

        /// `adt.ref_null` operation: creates a null reference.
        op ref_null[r#type]() -> result;

        /// `adt.ref_is_null` operation: checks if a reference is null.
        op ref_is_null(r#ref) -> result;

        /// `adt.ref_cast` operation: casts a reference type (runtime checked).
        op ref_cast[r#type](r#ref) -> result;

        // === Literals ===

        /// `adt.string_const` operation: string constant.
        op string_const[value]() -> result;

        /// `adt.bytes_const` operation: byte array constant.
        op bytes_const[value]() -> result;
    }
}
