//! Helper functions for wasm backend emission.
//!
//! This module contains type conversion and utility functions shared across
//! the emit module.

use std::collections::HashMap;

use trunk_ir::dialect::{adt, cont, core, trampoline, wasm};
use trunk_ir::{Attribute, Attrs, BlockId, DialectType, Symbol, Type, Value, ValueDef};
use wasm_encoder::{AbstractHeapType, HeapType, RefType, ValType};

use crate::errors::CompilationErrorKind;
use crate::gc_types::{ATTR_TYPE, ATTR_TYPE_IDX, BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX, STEP_IDX};
// Re-export is_closure_struct_type for use by handlers via emit.rs
pub(crate) use crate::gc_types::is_closure_struct_type;
use crate::{CompilationError, CompilationResult};

// ============================================================================
// Value type helpers
// ============================================================================

/// Get the type of a value from its definition.
pub(crate) fn value_type<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    block_arg_types: &HashMap<(BlockId, usize), Type<'db>>,
) -> Option<Type<'db>> {
    match value.def(db) {
        ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        ValueDef::BlockArg(block_id) => block_arg_types.get(&(block_id, value.index(db))).copied(),
    }
}

// ============================================================================
// Type predicates
// ============================================================================

/// Check if a type is the nil type (core.nil).
pub(crate) fn is_nil_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    core::Nil::from_type(db, ty).is_some()
}

// Note: is_closure_struct_type is imported from gc_types

/// Check if a type is the Step type (for trampoline-based effect system).
/// Step is an ADT struct with name "_Step".
pub(crate) fn is_step_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    ty == crate::gc_types::step_marker_type(db)
}

/// Check if a type should be normalized to anyref in polymorphic contexts.
///
/// This is used during call_indirect type signature construction where
/// types that need boxing at runtime should be normalized to anyref.
///
/// After normalize_primitive_types pass, tribute_rt types are already converted:
/// - tribute_rt.int/nat/bool → core.i32
/// - tribute_rt.float → core.f64
/// - tribute_rt.any → wasm.anyref
///
/// So we only need to check for the anyref type itself.
pub(crate) fn should_normalize_to_anyref<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    // wasm.anyref is already the normalized form - this is what we expect after
    // normalize_primitive_types converts tribute_rt.any → wasm.anyref
    wasm::Anyref::from_type(db, ty).is_some()
    // Note: core::Nil is NOT normalized to anyref. Nil uses (ref null none) which is
    // a subtype of anyref, so it can be passed where anyref is expected without boxing.
}

// ============================================================================
// Type conversion
// ============================================================================

/// Convert an IR type to a WebAssembly value type.
pub(crate) fn type_to_valtype<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
) -> CompilationResult<ValType> {
    if core::I32::from_type(db, ty).is_some() || core::I1::from_type(db, ty).is_some() {
        Ok(ValType::I32)
    } else if core::I64::from_type(db, ty).is_some() {
        Ok(ValType::I64)
    } else if core::F32::from_type(db, ty).is_some() {
        Ok(ValType::F32)
    } else if core::F64::from_type(db, ty).is_some() {
        Ok(ValType::F64)
    } else if core::Bytes::from_type(db, ty).is_some() {
        // Bytes uses WasmGC struct representation
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(BYTES_STRUCT_IDX),
        }))
    } else if ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == Symbol::new("ptr") {
        // ptr still uses linear memory (i32 pointer)
        Ok(ValType::I32)
    } else if ty.dialect(db) == wasm::DIALECT_NAME() {
        // WASM dialect types (e.g., wasm.structref for continuation frames)
        // IMPORTANT: Must check BEFORE type_idx_by_type.get() to avoid returning
        // incorrect concrete type_idx for placeholder types like wasm.structref
        let name = ty.name(db);
        if name == Symbol::new("structref") {
            Ok(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Struct,
                },
            }))
        } else if name == Symbol::new("funcref") {
            Ok(ValType::Ref(RefType::FUNCREF))
        } else if name == Symbol::new("anyref") {
            Ok(ValType::Ref(RefType::ANYREF))
        } else if name == Symbol::new("i31ref") {
            // i31ref is the nullable form (ref null i31)
            // This matches the output of ref.cast i31ref
            Ok(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::I31,
                },
            }))
        } else if name == Symbol::new("step") {
            // Step is a builtin GC struct type for trampoline-based effect system
            // Always uses fixed type index STEP_IDX (3)
            Ok(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Concrete(STEP_IDX),
            }))
        } else {
            Err(CompilationError::type_error(format!(
                "unsupported wasm type: wasm.{}",
                name
            )))
        }
    } else if core::Func::from_type(db, ty).is_some() {
        // Function types map to funcref for call_indirect operations.
        // The function signature is preserved in the IR and registered
        // in the type section by collect_call_indirect_types.
        Ok(ValType::Ref(RefType::FUNCREF))
    } else if is_closure_struct_type(db, ty) {
        // ADT struct named "_closure" maps to builtin CLOSURE_STRUCT_IDX.
        // Note: closure::Closure types are converted to adt.struct(name="_closure")
        // by TypeConverter before emit, so we only check for the ADT form here.
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(CLOSURE_STRUCT_IDX),
        }))
    } else if cont::Continuation::from_type(db, ty).is_some() {
        // Continuation types are represented as GC structs at runtime
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }))
    } else if trampoline::Step::from_type(db, ty).is_some() {
        // trampoline.step is the same as wasm.step - uses STEP_IDX
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(STEP_IDX),
        }))
    } else if trampoline::Continuation::from_type(db, ty).is_some() {
        // trampoline.continuation is represented as GC struct at runtime
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }))
    } else if trampoline::State::from_type(db, ty).is_some() {
        // trampoline.state is represented as GC struct at runtime
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }))
    } else if trampoline::ResumeWrapper::from_type(db, ty).is_some() {
        // trampoline.resume_wrapper is represented as GC struct at runtime
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }))
    } else if let Some(&type_idx) = type_idx_by_type.get(&ty) {
        // ADT types (structs, variants) - use concrete GC type reference
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }))
    } else if ty.dialect(db) == adt::DIALECT_NAME() {
        // ADT base types (e.g., adt.Expr) without specific variant type_idx
        // These represent "any variant of this enum" and use anyref
        Ok(ValType::Ref(RefType::ANYREF))
    } else if core::Nil::from_type(db, ty).is_some() {
        // Nil type - use (ref null none) for empty environments
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            },
        }))
    } else {
        Err(CompilationError::type_error(format!(
            "unsupported wasm value type: {}.{}",
            ty.dialect(db),
            ty.name(db)
        )))
    }
}

/// Convert an IR return type to WebAssembly result types.
/// Returns an empty vector for nil types (void functions).
pub(crate) fn result_types<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
) -> CompilationResult<Vec<ValType>> {
    if is_nil_type(db, ty) {
        Ok(Vec::new())
    } else {
        Ok(vec![type_to_valtype(db, ty, type_idx_by_type)?])
    }
}

// ============================================================================
// Heap type helpers
// ============================================================================

/// Extract a heap type from operation attributes.
///
/// This function handles three formats:
/// - IntBits: concrete type index
/// - Symbol: abstract heap type name (e.g., "any", "func", "struct")
/// - Type: wasm dialect type (e.g., wasm.i31ref, wasm.step)
pub(crate) fn attr_heap_type<'db>(
    db: &'db dyn salsa::Database,
    attrs: &Attrs<'db>,
    key: Symbol,
) -> CompilationResult<HeapType> {
    match attrs.get(&key) {
        Some(Attribute::IntBits(bits)) => Ok(HeapType::Concrete(*bits as u32)),
        Some(Attribute::Symbol(sym)) => {
            // Handle abstract heap types specified by name
            sym.with_str(symbol_to_abstract_heap_type)
        }
        Some(Attribute::Type(ty)) => {
            // Handle wasm abstract heap types like wasm.i31ref, wasm.anyref, etc.
            if ty.dialect(db) == Symbol::new("wasm") {
                let name = ty.name(db);
                // Handle step as a concrete builtin type (STEP_IDX = 3)
                if name == Symbol::new("step") {
                    return Ok(HeapType::Concrete(STEP_IDX));
                }
                name.with_str(symbol_to_abstract_heap_type)
            } else {
                Err(CompilationError::from(
                    CompilationErrorKind::MissingAttribute("non-wasm type for heap_type"),
                ))
            }
        }
        _ => Err(CompilationError::from(
            CompilationErrorKind::MissingAttribute("heap_type"),
        )),
    }
}

/// Convert a type name string to an abstract heap type.
pub(crate) fn symbol_to_abstract_heap_type(name: &str) -> CompilationResult<HeapType> {
    match name {
        "any" | "anyref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Any,
        }),
        "func" | "funcref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Func,
        }),
        "extern" | "externref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Extern,
        }),
        "none" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::None,
        }),
        "struct" | "structref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Struct,
        }),
        "array" | "arrayref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Array,
        }),
        "i31" | "i31ref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::I31,
        }),
        "eq" | "eqref" => Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Eq,
        }),
        _ => Err(CompilationError::from(
            CompilationErrorKind::MissingAttribute("unknown abstract heap type"),
        )),
    }
}

// ============================================================================
// Type index helpers
// ============================================================================

/// Get type_idx from attributes or inferred type.
///
/// Priority: type_idx attr > type attr > inferred_type (from result/operand)
pub(crate) fn get_type_idx_from_attrs<'db>(
    db: &'db dyn salsa::Database,
    attrs: &Attrs<'db>,
    inferred_type: Option<Type<'db>>,
    type_idx_by_type: &HashMap<Type<'db>, u32>,
) -> Option<u32> {
    // First try type_idx attribute
    if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
        return Some(*idx as u32);
    }
    // Fall back to type attribute (legacy, will be removed)
    if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
        if is_closure_struct_type(db, *ty) {
            return Some(CLOSURE_STRUCT_IDX);
        }
        return type_idx_by_type.get(ty).copied();
    }
    // Fall back to inferred type
    if let Some(ty) = inferred_type {
        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
        if is_closure_struct_type(db, ty) {
            return Some(CLOSURE_STRUCT_IDX);
        }
        return type_idx_by_type.get(&ty).copied();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;

    // ========================================
    // Test: type_to_valtype primitives
    // ========================================

    #[salsa_test]
    fn test_type_to_valtype_i32(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let result = type_to_valtype(db, i32_ty, &HashMap::new());
        assert!(matches!(result, Ok(ValType::I32)));
    }

    #[salsa_test]
    fn test_type_to_valtype_i64(db: &salsa::DatabaseImpl) {
        let i64_ty = core::I64::new(db).as_type();
        let result = type_to_valtype(db, i64_ty, &HashMap::new());
        assert!(matches!(result, Ok(ValType::I64)));
    }

    #[salsa_test]
    fn test_type_to_valtype_f32(db: &salsa::DatabaseImpl) {
        let f32_ty = core::F32::new(db).as_type();
        let result = type_to_valtype(db, f32_ty, &HashMap::new());
        assert!(matches!(result, Ok(ValType::F32)));
    }

    #[salsa_test]
    fn test_type_to_valtype_f64(db: &salsa::DatabaseImpl) {
        let f64_ty = core::F64::new(db).as_type();
        let result = type_to_valtype(db, f64_ty, &HashMap::new());
        assert!(matches!(result, Ok(ValType::F64)));
    }

    #[salsa_test]
    fn test_type_to_valtype_nil(db: &salsa::DatabaseImpl) {
        let nil_ty = core::Nil::new(db).as_type();
        let result = type_to_valtype(db, nil_ty, &HashMap::new());
        assert!(result.is_ok());
        let val_type = result.unwrap();
        assert!(matches!(
            val_type,
            ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::None,
                },
            })
        ));
    }

    // ========================================
    // Test: is_nil_type
    // ========================================

    #[salsa_test]
    fn test_is_nil_type_true(db: &salsa::DatabaseImpl) {
        let nil_ty = core::Nil::new(db).as_type();
        assert!(is_nil_type(db, nil_ty));
    }

    #[salsa_test]
    fn test_is_nil_type_false_for_i32(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        assert!(!is_nil_type(db, i32_ty));
    }

    // ========================================
    // Test: symbol_to_abstract_heap_type
    // ========================================

    #[test]
    fn test_symbol_to_abstract_heap_type_any() {
        let result = symbol_to_abstract_heap_type("any");
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Any,
            }
        ));
    }

    #[test]
    fn test_symbol_to_abstract_heap_type_anyref() {
        let result = symbol_to_abstract_heap_type("anyref");
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Any,
            }
        ));
    }

    #[test]
    fn test_symbol_to_abstract_heap_type_func() {
        let result = symbol_to_abstract_heap_type("func");
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Func,
            }
        ));
    }

    #[test]
    fn test_symbol_to_abstract_heap_type_struct() {
        let result = symbol_to_abstract_heap_type("struct");
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            }
        ));
    }

    #[test]
    fn test_symbol_to_abstract_heap_type_i31() {
        let result = symbol_to_abstract_heap_type("i31");
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::I31,
            }
        ));
    }

    #[test]
    fn test_symbol_to_abstract_heap_type_unknown() {
        let result = symbol_to_abstract_heap_type("unknown");
        assert!(result.is_err());
    }

    // ========================================
    // Test: result_types
    // ========================================

    #[salsa_test]
    fn test_result_types_nil_returns_empty(db: &salsa::DatabaseImpl) {
        let nil_ty = core::Nil::new(db).as_type();
        let result = result_types(db, nil_ty, &HashMap::new());
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[salsa_test]
    fn test_result_types_i32_returns_i32(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let result = result_types(db, i32_ty, &HashMap::new());
        assert!(result.is_ok());
        let types = result.unwrap();
        assert_eq!(types.len(), 1);
        assert!(matches!(types[0], ValType::I32));
    }

    // ========================================
    // Test: get_type_idx_from_attrs
    // ========================================

    #[salsa_test]
    fn test_get_type_idx_from_attrs_with_type_idx(db: &salsa::DatabaseImpl) {
        let mut attrs = trunk_ir::Attrs::new();
        attrs.insert(ATTR_TYPE_IDX(), Attribute::IntBits(42));
        let result = get_type_idx_from_attrs(db, &attrs, None, &HashMap::new());
        assert_eq!(result, Some(42));
    }

    #[salsa_test]
    fn test_get_type_idx_from_attrs_with_type_map(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let mut type_map = HashMap::new();
        type_map.insert(i32_ty, 100u32);

        let mut attrs = trunk_ir::Attrs::new();
        attrs.insert(ATTR_TYPE(), Attribute::Type(i32_ty));

        let result = get_type_idx_from_attrs(db, &attrs, None, &type_map);
        assert_eq!(result, Some(100));
    }

    #[salsa_test]
    fn test_get_type_idx_from_attrs_inferred(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let mut type_map = HashMap::new();
        type_map.insert(i32_ty, 50u32);

        let attrs = trunk_ir::Attrs::new();
        let result = get_type_idx_from_attrs(db, &attrs, Some(i32_ty), &type_map);
        assert_eq!(result, Some(50));
    }

    #[salsa_test]
    fn test_get_type_idx_from_attrs_none_when_not_found(db: &salsa::DatabaseImpl) {
        let attrs = trunk_ir::Attrs::new();
        let result = get_type_idx_from_attrs(db, &attrs, None, &HashMap::new());
        assert_eq!(result, None);
    }
}
