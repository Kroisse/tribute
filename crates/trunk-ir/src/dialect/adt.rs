//! Algebraic Data Type dialect operations.
//!
//! Target-independent operations for structs, variants (enums), arrays, and references.

use std::fmt::Write;

use crate::{dialect, op_interface};
use crate::type_interface::Printable;

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

// === Pure trait implementations ===
// Struct operations: new and get are pure (set modifies)
// Variant operations: new, tag, and get are pure
// Array operations: new, get, and len are pure (set modifies)
// Reference operations: all are pure
// Literal operations: all are pure

impl<'db> op_interface::Pure for StructNew<'db> {}
impl<'db> op_interface::Pure for StructGet<'db> {}

impl<'db> op_interface::Pure for VariantNew<'db> {}
impl<'db> op_interface::Pure for VariantTag<'db> {}
impl<'db> op_interface::Pure for VariantGet<'db> {}

impl<'db> op_interface::Pure for ArrayNew<'db> {}
impl<'db> op_interface::Pure for ArrayGet<'db> {}
impl<'db> op_interface::Pure for ArrayLen<'db> {}

impl<'db> op_interface::Pure for RefNull<'db> {}
impl<'db> op_interface::Pure for RefIsNull<'db> {}
impl<'db> op_interface::Pure for RefCast<'db> {}

impl<'db> op_interface::Pure for StringConst<'db> {}
impl<'db> op_interface::Pure for BytesConst<'db> {}

// Register pure operations for runtime lookup
inventory::submit! { op_interface::PureOps::register("adt", "struct_new") }
inventory::submit! { op_interface::PureOps::register("adt", "struct_get") }

inventory::submit! { op_interface::PureOps::register("adt", "variant_new") }
inventory::submit! { op_interface::PureOps::register("adt", "variant_tag") }
inventory::submit! { op_interface::PureOps::register("adt", "variant_get") }

inventory::submit! { op_interface::PureOps::register("adt", "array_new") }
inventory::submit! { op_interface::PureOps::register("adt", "array_get") }
inventory::submit! { op_interface::PureOps::register("adt", "array_len") }

inventory::submit! { op_interface::PureOps::register("adt", "ref_null") }
inventory::submit! { op_interface::PureOps::register("adt", "ref_is_null") }
inventory::submit! { op_interface::PureOps::register("adt", "ref_cast") }

inventory::submit! { op_interface::PureOps::register("adt", "string_const") }
inventory::submit! { op_interface::PureOps::register("adt", "bytes_const") }

// === Printable interface registrations ==="

// adt.* types -> "Name" or "Name(params...)" with capitalized first letter
// This uses Prefix("") as a catch-all for any adt type
inventory::submit! {
    Printable::implement_prefix("adt", "", |db, ty, f| {
        let params = ty.params(db);

        // Capitalize first letter
        let name = ty.name(db).to_string();
        let mut chars = name.chars();
        if let Some(c) = chars.next() {
            for ch in c.to_uppercase() {
                f.write_char(ch)?;
            }
            f.write_str(chars.as_str())?;
        }

        if !params.is_empty() {
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                Printable::print_type(db, p, f)?;
            }
            f.write_char(')')?;
        }

        Ok(())
    })
}
