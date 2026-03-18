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
//! Index 3: Step - struct { tag: i32, value: anyref, prompt: i32, op_idx: i32 } (trampoline)
//! Index 4: ClosureStruct - struct { i32, anyref } (table index + env)
//! Index 5: Marker - struct { ability_id: i32, prompt_tag: i32, op_table_index: i32 } (evidence)
//! Index 6: Evidence - array (ref Marker) (evidence array)
//! Index 7: Continuation - struct { func_idx: i32, env: anyref, prompt_tag: i32, state: anyref } (continuation)
//! Index 8: ResumeWrapper - struct { state: anyref, resume_value: anyref } (resume wrapper)
//! Index 9+: User-defined types (structs, arrays, variants, closures, etc.)
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

/// Type index for Step (struct { tag: i32, value: anyref, prompt: i32, op_idx: i32 }).
/// This is always index 3 in the GC type section.
/// Used for trampoline-based effect system in WasmGC backend (without stack switching).
pub const STEP_IDX: u32 = 3;

/// Type index for ClosureStruct (struct { i32, anyref }).
/// This is always index 4 in the GC type section.
/// All closures share this uniform representation: (table_idx: i32, env: anyref).
pub const CLOSURE_STRUCT_IDX: u32 = 4;

/// Type index for Marker (struct { ability_id: i32, prompt_tag: i32, op_table_index: i32 }).
/// This is always index 5 in the GC type section.
/// Used for evidence-based handler dispatch in the ability system.
pub const MARKER_IDX: u32 = 5;

/// Type index for Evidence (array (ref Marker)).
/// This is always index 6 in the GC type section.
/// Evidence is a sorted array of markers for ability handler lookup.
pub const EVIDENCE_IDX: u32 = 6;

/// Type index for Continuation (struct { func_idx: i32, env: anyref, prompt_tag: i32, state: anyref }).
/// This is always index 7 in the GC type section.
/// Used for one-shot continuations in the trampoline effect system.
pub const CONTINUATION_IDX: u32 = 7;

/// Type index for ResumeWrapper (struct { state: anyref, resume_value: anyref }).
/// This is always index 8 in the GC type section.
/// Packages captured state and resume value for continuation resume functions.
pub const RESUME_WRAPPER_IDX: u32 = 8;

/// First type index available for user-defined types.
pub const FIRST_USER_TYPE_IDX: u32 = 9;

/// Closure struct field count.
/// Closure structs always have 2 fields: (table_idx: i32, env: anyref)
pub const CLOSURE_FIELD_COUNT: usize = 2;

/// Step struct field count.
/// Step structs have 4 fields: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
pub const STEP_FIELD_COUNT: usize = 4;

/// Tag value for Done (successful completion with result value).
pub const STEP_TAG_DONE: i32 = 0;

/// Tag value for Shift (suspended with continuation, needs handler dispatch).
pub const STEP_TAG_SHIFT: i32 = 1;

trunk_ir::symbols! {
    /// Attribute name for type information on wasm operations.
    ATTR_TYPE => "type",
    /// Attribute name for explicit type index.
    ATTR_TYPE_IDX => "type_idx",
    /// Attribute name for field index on struct operations.
    ATTR_FIELD_IDX => "field_idx",
    /// Attribute name for field count (used with placeholder types).
    ATTR_FIELD_COUNT => "field_count",
}

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
/// Indices: BoxedF64(0), BytesArray(1), BytesStruct(2), Step(3), ClosureStruct(4),
///          Marker(5), Evidence(6), Continuation(7), ResumeWrapper(8)
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
        // Index 3: Step - struct { tag: i32, value: anyref, prompt: i32, op_idx: i32 }
        GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
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
        // Index 4: ClosureStruct - struct { func_idx: i32, env: anyref }
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
        // Index 5: Marker - struct { ability_id: i32, prompt_tag: i32, op_table_index: i32, handler_dispatch: i32 }
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
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
        ]),
        // Index 6: Evidence - array (ref null Marker)
        GcTypeDef::Array(FieldType {
            element_type: StorageType::Val(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Concrete(MARKER_IDX),
            })),
            mutable: true,
        }),
        // Index 7: Continuation - struct { resume_fn: i32, state: anyref, tag: i32, shift_value: anyref }
        GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::I32),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
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
        ]),
        // Index 8: ResumeWrapper - struct { state: anyref, resume_value: anyref }
        GcTypeDef::Struct(vec![
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                mutable: false,
            },
            FieldType {
                element_type: StorageType::Val(ValType::Ref(RefType::ANYREF)),
                mutable: false,
            },
        ]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_types() {
        let builtins = builtin_types();
        assert_eq!(builtins.len(), 9);

        // BoxedF64
        assert!(matches!(&builtins[0], GcTypeDef::Struct(fields) if fields.len() == 1));
        // BytesArray
        assert!(matches!(&builtins[1], GcTypeDef::Array(_)));
        // BytesStruct
        assert!(matches!(&builtins[2], GcTypeDef::Struct(fields) if fields.len() == 3));
        // Step (4 fields: tag, value, prompt, op_idx)
        assert!(matches!(&builtins[3], GcTypeDef::Struct(fields) if fields.len() == 4));
        // ClosureStruct
        assert!(matches!(&builtins[4], GcTypeDef::Struct(fields) if fields.len() == 2));
        // Marker (3 fields: ability_id, prompt_tag, op_table_index)
        assert!(matches!(&builtins[5], GcTypeDef::Struct(fields) if fields.len() == 4));
        // Evidence (array of Marker refs)
        assert!(matches!(&builtins[6], GcTypeDef::Array(_)));
        // Continuation (4 fields: resume_fn, state, tag, shift_value)
        assert!(matches!(&builtins[7], GcTypeDef::Struct(fields) if fields.len() == 4));
        // ResumeWrapper (2 fields: state, resume_value)
        assert!(matches!(&builtins[8], GcTypeDef::Struct(fields) if fields.len() == 2));
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
            _ => panic!("Evidence (index 6) should be an array type"),
        }
    }

    #[test]
    fn test_continuation_struct_field_layout() {
        // Verify the Continuation GC type (index 7) matches the canonical layout:
        // { resume_fn: i32, state: anyref, tag: i32, shift_value: anyref }
        let builtins = builtin_types();
        let cont_def = &builtins[CONTINUATION_IDX as usize];

        match cont_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), 4, "Continuation should have 4 fields");
                // Field 0: resume_fn (i32)
                assert!(
                    matches!(fields[0].element_type, StorageType::Val(ValType::I32)),
                    "Field 0 (resume_fn) should be i32"
                );
                // Field 1: state (anyref)
                assert!(
                    matches!(
                        fields[1].element_type,
                        StorageType::Val(ValType::Ref(RefType::ANYREF))
                    ),
                    "Field 1 (state) should be anyref"
                );
                // Field 2: tag (i32)
                assert!(
                    matches!(fields[2].element_type, StorageType::Val(ValType::I32)),
                    "Field 2 (tag) should be i32"
                );
                // Field 3: shift_value (anyref)
                assert!(
                    matches!(
                        fields[3].element_type,
                        StorageType::Val(ValType::Ref(RefType::ANYREF))
                    ),
                    "Field 3 (shift_value) should be anyref"
                );
            }
            _ => panic!("Continuation (index 7) should be a struct type"),
        }
    }

    #[test]
    fn test_marker_struct_has_four_i32_fields() {
        // Marker struct: { ability_id: i32, prompt_tag: i32, op_table_index: i32, handler_dispatch: i32 }
        let builtins = builtin_types();
        let marker_def = &builtins[MARKER_IDX as usize];

        match marker_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), 4, "Marker should have 4 fields");
                for (i, field) in fields.iter().enumerate() {
                    assert!(
                        matches!(field.element_type, StorageType::Val(ValType::I32)),
                        "Marker field {} should be i32",
                        i
                    );
                }
            }
            _ => panic!("Marker (index 5) should be a struct type"),
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
    fn test_step_struct_layout() {
        // Step: { tag: i32, value: anyref, prompt: i32, op_idx: i32 }
        let builtins = builtin_types();
        let step_def = &builtins[STEP_IDX as usize];

        match step_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), STEP_FIELD_COUNT);
                assert!(matches!(
                    fields[0].element_type,
                    StorageType::Val(ValType::I32)
                ));
                assert!(matches!(
                    fields[1].element_type,
                    StorageType::Val(ValType::Ref(RefType::ANYREF))
                ));
                assert!(matches!(
                    fields[2].element_type,
                    StorageType::Val(ValType::I32)
                ));
                assert!(matches!(
                    fields[3].element_type,
                    StorageType::Val(ValType::I32)
                ));
            }
            _ => panic!("Expected struct type for step"),
        }
    }

    #[test]
    fn test_resume_wrapper_struct_layout() {
        // ResumeWrapper: { state: anyref, resume_value: anyref }
        let builtins = builtin_types();
        let rw_def = &builtins[RESUME_WRAPPER_IDX as usize];

        match rw_def {
            GcTypeDef::Struct(fields) => {
                assert_eq!(fields.len(), 2, "ResumeWrapper should have 2 fields");
                assert!(
                    matches!(
                        fields[0].element_type,
                        StorageType::Val(ValType::Ref(RefType::ANYREF))
                    ),
                    "Field 0 (state) should be anyref"
                );
                assert!(
                    matches!(
                        fields[1].element_type,
                        StorageType::Val(ValType::Ref(RefType::ANYREF))
                    ),
                    "Field 1 (resume_value) should be anyref"
                );
            }
            _ => panic!("Expected struct type for ResumeWrapper"),
        }
    }

    #[test]
    fn test_step_and_closure_different_indices() {
        // Step (builtin at STEP_IDX=3) and Closure (builtin at CLOSURE_STRUCT_IDX=4)
        // must have different indices.
        assert_ne!(STEP_IDX, CLOSURE_STRUCT_IDX);
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
