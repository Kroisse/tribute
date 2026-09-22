//! GC Type definitions and constants for WebAssembly GC types.
//!
//! This module provides type index constants and builtin type definitions for WasmGC.
//! The actual type collection from IR operations is handled by `emit::gc_types_collection`.
//!
//! ## Type Index Layout
//!
//! ```text
//! Index 0: BoxedF64 - Float wrapper for polymorphic contexts
//! Index 1: BytesArray - array i8 backing storage for Bytes
//! Index 2: BytesStruct - struct { data: ref BytesArray, offset: i32, len: i32 }
//! Index 3: ClosureStruct - struct { i32, anyref } (table index + env)
//! Index 4: Marker - struct { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref } (evidence)
//! Index 5: Evidence - array (ref Marker) (evidence array)
//! Index 6+: User-defined types (structs, arrays, variants, closures, etc.)
//! ```

use wasm_encoder::{FieldType, HeapType, RefType, StorageType, ValType};

/// Type index for BoxedF64 (Float wrapper for polymorphic contexts).
/// This is always index 0 in the GC type section.
pub const BOXED_F64_IDX: u32 = 0;

/// Type index for BytesArray (array i8) - backing storage for Bytes.
/// This is always index 1 in the GC type section.
pub const BYTES_ARRAY_IDX: u32 = 1;

/// Type index for BytesStruct (struct { data: ref BytesArray, offset: i32, len: i32 }).
/// This is always index 2 in the GC type section.
pub const BYTES_STRUCT_IDX: u32 = 2;

/// Type index for ClosureStruct (struct { i32, anyref }).
/// This is always index 3 in the GC type section.
/// All closures share this uniform representation: (table_idx: i32, env: anyref).
pub const CLOSURE_STRUCT_IDX: u32 = 3;

/// Type index for Marker (struct { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref }).
/// This is always index 4 in the GC type section.
/// Used for evidence-based handler dispatch in the ability system.
pub const MARKER_IDX: u32 = 4;

/// Type index for Evidence (array (ref Marker)).
/// This is always index 5 in the GC type section.
/// Evidence is a sorted array of markers for ability handler lookup.
pub const EVIDENCE_IDX: u32 = 5;

/// First type index available for user-defined types.
pub const FIRST_USER_TYPE_IDX: u32 = 6;

/// Closure struct field count.
/// Closure structs always have 2 fields: (table_idx: i32, env: anyref)
pub const CLOSURE_FIELD_COUNT: usize = 2;

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
/// Indices: BoxedF64(0), BytesArray(1), BytesStruct(2), ClosureStruct(3),
///          Marker(4), Evidence(5).
pub fn builtin_types() -> Vec<GcTypeDef> {
    vec![
        // Index 0: BoxedF64 - struct { value: f64 }
        GcTypeDef::Struct(vec![FieldType {
            element_type: StorageType::Val(ValType::F64),
            mutable: false,
        }]),
        // Index 1: BytesArray - array i8
        GcTypeDef::Array(FieldType {
            element_type: StorageType::I8,
            mutable: true,
        }),
        // Index 2: BytesStruct - struct { data: ref BytesArray, offset: i32, len: i32 }
        GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::Ref(wasm_encoder::RefType {
                    nullable: false,
                    heap_type: wasm_encoder::HeapType::Concrete(BYTES_ARRAY_IDX),
                })),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
        ]),
        // Index 3: ClosureStruct - struct { func_idx: i32, env: anyref }
        GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                mutable: false,
            },
        ]),
        // Index 4: Marker - struct { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref }
        GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                mutable: false,
            },
        ]),
        // Index 5: Evidence - array (ref null Marker)
        GcTypeDef::Array(FieldType {
            element_type: StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Concrete(MARKER_IDX),
            })),
            mutable: true,
        }),
    ]
}

/// Whether the builtin GC type at `index` has struct layout.
///
/// The physical reference relation uses this to tell struct-layout builtins
/// (assignable to `wasm.structref`) from array-layout builtins (assignable to
/// `wasm.arrayref`). `builtin_struct_indices_match_the_layout` pins the
/// classification against [`builtin_types`], so reindexing the builtin layout
/// cannot silently change a physical classification.
pub(crate) fn is_builtin_struct_index(index: u32) -> bool {
    index < FIRST_USER_TYPE_IDX && !matches!(index, BYTES_ARRAY_IDX | EVIDENCE_IDX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_struct_indices_match_the_layout() {
        assert_eq!(
            [
                BOXED_F64_IDX,
                BYTES_ARRAY_IDX,
                BYTES_STRUCT_IDX,
                CLOSURE_STRUCT_IDX,
                MARKER_IDX,
                EVIDENCE_IDX,
                FIRST_USER_TYPE_IDX
            ],
            [0, 1, 2, 3, 4, 5, 6],
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
        assert_eq!(builtins.len(), 6);

        // BoxedF64
        assert!(matches!(&builtins[0], GcTypeDef::Struct(fields) if fields.len() == 1));
        // BytesArray
        assert!(matches!(&builtins[1], GcTypeDef::Array(_)));
        // BytesStruct
        assert!(matches!(&builtins[2], GcTypeDef::Struct(fields) if fields.len() == 3));
        // ClosureStruct
        assert!(matches!(&builtins[3], GcTypeDef::Struct(fields) if fields.len() == 2));
        // Marker (4 fields: ability_id, prompt_tag, tr_dispatch_fn, handler_dispatch)
        assert!(matches!(&builtins[4], GcTypeDef::Struct(fields) if fields.len() == 4));
        // Evidence (array of Marker refs)
        assert!(matches!(&builtins[5], GcTypeDef::Array(_)));
    }

    #[test]
    fn test_evidence_array_element_is_nullable() {
        // Evidence array elements must be nullable to allow array.new_default initialization
        let builtins = builtin_types();
        let evidence_def = &builtins[EVIDENCE_IDX as usize];

        match evidence_def {
            GcTypeDef::Array(field_type) => {
                match field_type.element_type {
                    StorageType::Val(ValType::Ref(ref_type)) => {
                        assert!(
                            ref_type.nullable,
                            "Evidence array elements must be nullable for array.new_default"
                        );
                        // Also verify it references MARKER_IDX
                        assert!(
                            matches!(ref_type.heap_type, HeapType::Concrete(MARKER_IDX)),
                            "Evidence array should contain Marker references"
                        );
                    }
                    _ => panic!("Evidence array element should be a reference type"),
                }
            }
            _ => panic!("Evidence (index 5) should be an array type"),
        }
    }

    #[test]
    fn test_marker_struct_layout() {
        // Marker struct: { ability_id: i32, prompt_tag: i32, tr_dispatch_fn: anyref, handler_dispatch: anyref }
        let builtins = builtin_types();
        let marker_def = &builtins[MARKER_IDX as usize];

        match marker_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), 4, "Marker should have 4 fields");
                assert!(
                    matches!(fields[0].element_type, StorageType::Val(ValType::I32)),
                    "Field 0 (ability_id) should be i32"
                );
                assert!(
                    matches!(fields[1].element_type, StorageType::Val(ValType::I32)),
                    "Field 1 (prompt_tag) should be i32"
                );
                assert!(
                    matches!(
                        fields[2].element_type,
                        StorageType::Val(ValType::Ref(RefType::ANYREF))
                    ),
                    "Field 2 (tr_dispatch_fn) should be anyref"
                );
                assert!(
                    matches!(
                        fields[3].element_type,
                        StorageType::Val(ValType::Ref(RefType::ANYREF))
                    ),
                    "Field 3 (handler_dispatch) should be anyref"
                );
            }
            _ => panic!("Marker (index 4) should be a struct type"),
        }
    }

    #[test]
    fn test_closure_struct_layout() {
        // ClosureStruct: { func_idx: i32, env: anyref }
        let builtins = builtin_types();
        let closure_def = &builtins[CLOSURE_STRUCT_IDX as usize];

        match closure_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), CLOSURE_FIELD_COUNT);
                // Field 0: i32 (function table index)
                assert!(matches!(
                    fields[0].element_type,
                    StorageType::Val(ValType::I32)
                ));
                // Field 1: anyref
                assert!(matches!(
                    fields[1].element_type,
                    StorageType::Val(ValType::Ref(RefType::ANYREF))
                ));
            }
            _ => panic!("Expected struct type for closure"),
        }
    }

    #[test]
    fn test_boxed_f64_has_single_f64_field() {
        let builtins = builtin_types();
        let boxed_def = &builtins[BOXED_F64_IDX as usize];

        match boxed_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), 1, "BoxedF64 should have 1 field");
                assert!(
                    matches!(fields[0].element_type, StorageType::Val(ValType::F64)),
                    "BoxedF64 field should be f64"
                );
            }
            _ => panic!("BoxedF64 should be a struct type"),
        }
    }

    #[test]
    fn test_bytes_array_is_mutable_i8() {
        let builtins = builtin_types();
        let ba_def = &builtins[BYTES_ARRAY_IDX as usize];

        match ba_def {
            GcTypeDef::Array(field) => {
                assert!(matches!(field.element_type, StorageType::I8));
                assert!(field.mutable, "BytesArray should be mutable");
            }
            _ => panic!("BytesArray should be an array type"),
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
