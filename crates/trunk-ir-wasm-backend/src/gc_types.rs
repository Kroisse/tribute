//! GC Type definitions and constants for WebAssembly GC types.
//!
//! This module provides type index constants and builtin type definitions for WasmGC.
//! The actual type collection from IR operations is handled by `emit::gc_types_collection`.
//!
//! ## Type Index Layout
//!
//! ```text
//! Index 0: BoxedF64 - Float wrapper for polymorphic contexts
//! Index 1: ClosureStruct - struct { i32 } (table index), the closure base
//! Index 2: Described - struct { descriptor: i32 }, supertype of user structs and variants
//! Index 3+: User-defined types (structs, arrays, variants, closure capture layouts, etc.)
//! ```

use wasm_encoder::{FieldType, StorageType, ValType};

/// Type index for BoxedF64 (Float wrapper for polymorphic contexts).
/// This is always index 0 in the GC type section.
pub const BOXED_F64_IDX: u32 = 0;

/// Type index for ClosureStruct (struct { i32 }), the closure base.
/// This is always index 1 in the GC type section.
/// Every closure value has this type; a closure with captures is a user-indexed
/// subtype that adds them after the table index.
pub const CLOSURE_STRUCT_IDX: u32 = 1;

/// Type index for Described (struct { descriptor: i32 }). Only user structs and
/// variants are its subtypes.
/// This is always index 2 in the GC type section.
/// Every user struct and variant type declares it as its `supertype`, so an
/// erased reference can be cast to it to read the runtime type descriptor.
pub const DESCRIBED_IDX: u32 = 2;

/// Runtime layout identifier (the `layout` type attribute) of the builtin
/// boxed float at [`BOXED_F64_IDX`].
pub const BOXED_F64_LAYOUT: &str = "boxed_f64";

/// Runtime layout identifier of the builtin
/// closure struct at [`CLOSURE_STRUCT_IDX`].
pub const CLOSURE_LAYOUT: &str = "closure";

/// Runtime layout identifier of the builtin `Described` supertype at
/// [`DESCRIBED_IDX`].
pub const DESCRIBED_LAYOUT: &str = "described";

/// The builtin GC type index of a type carrying a runtime layout identifier.
pub fn builtin_layout_idx(layout: &str) -> Option<u32> {
    if layout == BOXED_F64_LAYOUT {
        Some(BOXED_F64_IDX)
    } else if layout == CLOSURE_LAYOUT {
        Some(CLOSURE_STRUCT_IDX)
    } else if layout == DESCRIBED_LAYOUT {
        Some(DESCRIBED_IDX)
    } else {
        None
    }
}

/// First type index available for user-defined types.
pub const FIRST_USER_TYPE_IDX: u32 = 3;

/// Closure base field count: the code field (table index) alone. Closure
/// capture layouts are user-indexed subtypes that add their captures.
pub const CLOSURE_FIELD_COUNT: usize = 1;

/// Definition of a GC type (struct or array).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GcTypeDef {
    /// Struct type with ordered field types.
    Struct(Vec<FieldType>),
    /// Array type with element type.
    Array(FieldType),
}

impl GcTypeDef {
    /// Create a struct type definition with the given field types.
    pub fn struct_type(fields: Vec<FieldType>) -> Self {
        GcTypeDef::Struct(fields)
    }

    /// Create an array type definition with the given element type.
    pub fn array_type(element: FieldType) -> Self {
        GcTypeDef::Array(element)
    }

    /// Returns the number of fields if this is a struct type.
    pub fn field_count(&self) -> Option<usize> {
        match self {
            GcTypeDef::Struct(fields) => Some(fields.len()),
            GcTypeDef::Array(_) => None,
        }
    }
}

/// Returns the builtin type definitions.
///
/// These must be prepended to the user-defined types when emitting.
/// Indices: BoxedF64(0), ClosureStruct(1), Described(2).
pub fn builtin_types() -> Vec<GcTypeDef> {
    vec![
        // Index 0: BoxedF64 - struct { value: f64 }
        GcTypeDef::Struct(vec![FieldType {
            element_type: StorageType::Val(ValType::F64),
            mutable: false,
        }]),
        // Index 1: ClosureStruct - struct { code: i32 }, the closure base
        GcTypeDef::Struct(vec![FieldType {
            element_type: StorageType::Val(ValType::I32),
            mutable: false,
        }]),
        // Index 2: Described - struct { descriptor: i32 }
        GcTypeDef::Struct(vec![descriptor_field()]),
    ]
}

/// The runtime type descriptor field that starts [`DESCRIBED_IDX`] and each of
/// its subtypes.
pub fn descriptor_field() -> FieldType {
    FieldType {
        element_type: StorageType::Val(ValType::I32),
        mutable: false,
    }
}

/// Whether the builtin GC type at `index` has struct layout.
///
/// The physical reference relation uses this to tell struct-layout builtins
/// (assignable to `wasm.structref`) from other types. Every builtin is a
/// struct; `builtin_struct_indices_match_the_layout` pins the classification
/// against [`builtin_types`], so adding an array builtin cannot silently
/// change a physical classification.
pub(crate) fn is_builtin_struct_index(index: u32) -> bool {
    index < FIRST_USER_TYPE_IDX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_struct_indices_match_the_layout() {
        assert_eq!(
            [
                BOXED_F64_IDX,
                CLOSURE_STRUCT_IDX,
                DESCRIBED_IDX,
                FIRST_USER_TYPE_IDX
            ],
            [0, 1, 2, 3],
        );
        for (index, definition) in builtin_types().iter().enumerate() {
            let index = index as u32;
            assert_eq!(
                is_builtin_struct_index(index),
                matches!(definition, GcTypeDef::Struct(_)),
                "builtin index {index} classification"
            );
        }
        assert!(!is_builtin_struct_index(FIRST_USER_TYPE_IDX));
    }

    #[test]
    fn test_builtin_types() {
        let builtins = builtin_types();
        assert_eq!(builtins.len(), 3);

        // BoxedF64
        assert!(matches!(&builtins[0], GcTypeDef::Struct(fields) if fields.len() == 1));
        // ClosureStruct
        assert!(matches!(&builtins[1], GcTypeDef::Struct(fields) if fields.len() == 1));
        // Described (the descriptor field alone)
        assert!(
            matches!(&builtins[2], GcTypeDef::Struct(fields) if fields[..] == [descriptor_field()])
        );
    }

    #[test]
    fn test_builtin_struct_field_layouts() {
        let builtins = builtin_types();
        let i32 = StorageType::Val(ValType::I32);
        let f64 = StorageType::Val(ValType::F64);
        assert_eq!(CLOSURE_FIELD_COUNT, 1, "ClosureStruct field count");
        let cases: [(&str, u32, &[StorageType]); 2] = [
            // { code: function table index }
            ("ClosureStruct", CLOSURE_STRUCT_IDX, &[i32]),
            ("BoxedF64", BOXED_F64_IDX, &[f64]),
        ];
        for (name, index, expected) in cases {
            let GcTypeDef::Struct(fields) = &builtins[index as usize] else {
                panic!("{name} (index {index}) should be a struct type");
            };
            let element_types: Vec<_> = fields.iter().map(|field| field.element_type).collect();
            assert_eq!(element_types, expected, "{name} field types");
        }
    }

    #[test]
    fn test_first_user_type_idx_follows_builtins() {
        let builtins = builtin_types();
        assert_eq!(
            FIRST_USER_TYPE_IDX as usize,
            builtins.len(),
            "FIRST_USER_TYPE_IDX should equal the number of builtin types"
        );
    }
}
