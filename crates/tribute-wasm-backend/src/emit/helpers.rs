//! Helper functions for wasm backend emission.
//!
//! This module contains type conversion and utility functions shared across
//! the emit module.

use std::collections::HashMap;

use tribute_ir::dialect::{ability, adt, closure, tribute, tribute_rt};
use trunk_ir::dialect::{cont, core, wasm};
use trunk_ir::{Attribute, Attrs, BlockId, DialectType, Symbol, Type, Value, ValueDef};
use wasm_encoder::{AbstractHeapType, HeapType, RefType, ValType};

use crate::errors::CompilationErrorKind;
use crate::gc_types::{ATTR_TYPE, ATTR_TYPE_IDX, BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX, STEP_IDX};
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

/// Check if a type is a closure struct (adt.struct with name "_closure").
/// Closure structs contain (funcref, anyref) and are used for call_indirect.
pub(crate) fn is_closure_struct_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    // Check if it's an adt.struct type
    if ty.dialect(db) != adt::DIALECT_NAME() {
        return false;
    }
    if ty.name(db) != Symbol::new("struct") {
        return false;
    }
    // Check if the struct name is "_closure"
    ty.attrs(db).get(&Symbol::new("name")).is_some_and(|attr| {
        if let Attribute::Symbol(name) = attr {
            name.with_str(|s| s == "_closure")
        } else {
            false
        }
    })
}

/// Check if a type is the Step type (for trampoline-based effect system).
pub(crate) fn is_step_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    wasm::Step::from_type(db, ty).is_some()
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
    if core::I32::from_type(db, ty).is_some()
        || core::I1::from_type(db, ty).is_some()
        || tribute_rt::is_int(db, ty)
        || tribute_rt::is_nat(db, ty)
        || tribute_rt::is_bool(db, ty)
    {
        // tribute_rt.int/nat/bool are represented as i32 in WebAssembly
        Ok(ValType::I32)
    } else if core::I64::from_type(db, ty).is_some() {
        Ok(ValType::I64)
    } else if core::F32::from_type(db, ty).is_some() {
        Ok(ValType::F32)
    } else if core::F64::from_type(db, ty).is_some() || tribute_rt::is_float(db, ty) {
        Ok(ValType::F64)
    } else if core::Bytes::from_type(db, ty).is_some() {
        // Bytes uses WasmGC struct representation
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(BYTES_STRUCT_IDX),
        }))
    } else if core::String::from_type(db, ty).is_some()
        || (ty.dialect(db) == core::DIALECT_NAME() && ty.name(db) == Symbol::new("ptr"))
    {
        // String and ptr still use linear memory (i32 pointer)
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
    } else if closure::Closure::from_type(db, ty).is_some() {
        // Closure types map to the builtin CLOSURE_STRUCT_IDX which has
        // (funcref, anyref) fields for uniform closure representation.
        // IMPORTANT: Check this BEFORE type_idx_by_type.get() to ensure all
        // closure::Closure types use the builtin CLOSURE_STRUCT_IDX (4).
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(CLOSURE_STRUCT_IDX),
        }))
    } else if is_closure_struct_type(db, ty) {
        // ADT struct named "_closure" maps to builtin CLOSURE_STRUCT_IDX.
        // IMPORTANT: Check this BEFORE type_idx_by_type.get() to ensure
        // _closure structs use the correct builtin type.
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
    } else if let Some(&type_idx) = type_idx_by_type.get(&ty) {
        // ADT types (structs, variants) - use concrete GC type reference
        // Check this BEFORE tribute::is_type_var to handle struct types with type_idx
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }))
    } else if tribute::is_type_var(db, ty) {
        // Generic type variables use anyref (uniform representation)
        // Values must be boxed when passed to generic functions
        Ok(ValType::Ref(RefType::ANYREF))
    } else if tribute::is_unresolved_type(db, ty) {
        // Unresolved type references (tribute.type) - map known primitives
        if let Some(Attribute::Symbol(name_sym)) = ty.get_attr(db, Symbol::new("name")) {
            let result = name_sym.with_str(|name_str| match name_str {
                "Int" | "Nat" | "Bool" => Some(ValType::I32),
                "Float" => Some(ValType::F64),
                "String" => Some(ValType::Ref(RefType::ANYREF)),
                _ => None, // User-defined type
            });
            if let Some(val_type) = result {
                return Ok(val_type);
            }
        }
        // User-defined types use anyref
        Ok(ValType::Ref(RefType::ANYREF))
    } else if ty.dialect(db) == adt::DIALECT_NAME() {
        // ADT base types (e.g., adt.Expr) without specific variant type_idx
        // These represent "any variant of this enum" and use anyref
        Ok(ValType::Ref(RefType::ANYREF))
    } else if ability::EvidencePtr::from_type(db, ty).is_some() {
        // Evidence pointer for ability system - use anyref as runtime handle
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
