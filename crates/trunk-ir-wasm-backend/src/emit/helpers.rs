//! Helper functions for wasm backend emission.
//!
//! This module contains type conversion and utility functions shared across
//! the emit module.

use std::collections::{BTreeMap, HashMap};

use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Attribute;
use wasm_encoder::{AbstractHeapType, HeapType, RefType, ValType};

use crate::errors::CompilationErrorKind;
use crate::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX, STEP_IDX};
use crate::{CompilationError, CompilationResult};

// ============================================================================
// Type checking helpers (arena)
// ============================================================================

/// Check if a TypeRef matches a specific dialect and name.
pub(crate) fn is_type(
    ctx: &IrContext,
    ty: TypeRef,
    dialect: &'static str,
    name: &'static str,
) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

// ============================================================================
// Value type helpers
// ============================================================================

/// Get the type of a value from its definition (arena version).
pub(crate) fn value_type(ctx: &IrContext, value: ValueRef) -> TypeRef {
    ctx.value_ty(value)
}

// ============================================================================
// Type predicates
// ============================================================================

/// Check if a type is the nil type (core.nil).
pub(crate) fn is_nil_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, "core", "nil")
}

/// Check if a type is a closure struct type (adt.struct with name "_closure").
pub(crate) fn is_closure_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_named_adt_struct(ctx, ty, "_closure")
}

/// Check if a type is an adt.struct with the given name.
fn is_named_adt_struct(ctx: &IrContext, ty: TypeRef, expected_name: &'static str) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    match data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(name)) => name.with_str(|s| s == expected_name),
        _ => false,
    }
}

/// Check if a type is the Step type (for trampoline-based effect system).
pub(crate) fn is_step_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_named_adt_struct(ctx, ty, "_Step")
}

/// Check if a type should be normalized to anyref in polymorphic contexts.
pub(crate) fn should_normalize_to_anyref(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, "wasm", "anyref")
}

// ============================================================================
// Type conversion
// ============================================================================

/// Get the params and return type of a core.func TypeRef.
/// Returns (param_types, return_type) or None if not a core.func.
pub(crate) fn func_type_parts(ctx: &IrContext, ty: TypeRef) -> Option<(&[TypeRef], TypeRef)> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return None;
    }
    if data.params.is_empty() {
        return None;
    }
    let (ret, params) = data.params.split_first()?;
    Some((params, *ret))
}

/// Convert an IR type to a WebAssembly value type.
pub(crate) fn type_to_valtype(
    ctx: &IrContext,
    ty: TypeRef,
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> CompilationResult<ValType> {
    if is_type(ctx, ty, "core", "i32") || is_type(ctx, ty, "core", "i1") {
        Ok(ValType::I32)
    } else if is_type(ctx, ty, "core", "i64") {
        Ok(ValType::I64)
    } else if is_type(ctx, ty, "core", "f32") {
        Ok(ValType::F32)
    } else if is_type(ctx, ty, "core", "f64") {
        Ok(ValType::F64)
    } else if is_type(ctx, ty, "core", "bytes") {
        Ok(ValType::Ref(RefType {
            nullable: false,
            heap_type: HeapType::Concrete(BYTES_STRUCT_IDX),
        }))
    } else if is_bytes_array_ref(ctx, ty) {
        let nullable = matches!(
            ctx.types.get(ty).attrs.get(&Symbol::new("nullable")),
            Some(Attribute::Bool(true))
        );
        Ok(ValType::Ref(RefType {
            nullable,
            heap_type: HeapType::Concrete(BYTES_ARRAY_IDX),
        }))
    } else if is_type(ctx, ty, "core", "ptr") {
        Ok(ValType::I32)
    } else if let Some(&type_idx) = type_idx_by_type.get(&ty) {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }))
    } else if ctx.types.get(ty).dialect == Symbol::new("wasm") {
        let name = ctx.types.get(ty).name;
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
            Ok(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::I31,
                },
            }))
        } else if name == Symbol::new("arrayref") {
            Ok(ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Array,
                },
            }))
        } else {
            Err(CompilationError::type_error(format!(
                "unsupported wasm type: wasm.{}",
                name
            )))
        }
    } else if is_type(ctx, ty, "core", "array") {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Array,
            },
        }))
    } else if is_type(ctx, ty, "core", "func") {
        Ok(ValType::Ref(RefType::FUNCREF))
    } else if is_closure_struct_type(ctx, ty) {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(CLOSURE_STRUCT_IDX),
        }))
    } else if ctx.types.get(ty).dialect == Symbol::new("adt") {
        Ok(ValType::Ref(RefType::ANYREF))
    } else if is_nil_type(ctx, ty) {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            },
        }))
    } else {
        let data = ctx.types.get(ty);
        Err(CompilationError::type_error(format!(
            "unsupported wasm value type: {}.{}",
            data.dialect, data.name
        )))
    }
}

fn is_bytes_array_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let reference = ctx.types.get(ty);
    if reference.dialect != Symbol::new("core")
        || reference.name != Symbol::new("ref")
        || reference.params.len() != 1
    {
        return false;
    }
    let array = ctx.types.get(reference.params[0]);
    array.dialect == Symbol::new("core")
        && array.name == Symbol::new("array")
        && array.params.len() == 1
        && is_type(ctx, array.params[0], "core", "i8")
}

/// Convert an IR return type to WebAssembly result types.
/// Returns an empty vector for nil types (void functions).
pub(crate) fn result_types(
    ctx: &IrContext,
    ty: TypeRef,
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> CompilationResult<Vec<ValType>> {
    if is_nil_type(ctx, ty) {
        Ok(Vec::new())
    } else {
        Ok(vec![type_to_valtype(ctx, ty, type_idx_by_type)?])
    }
}

// ============================================================================
// Heap type helpers
// ============================================================================

/// Extract a heap type from operation attributes.
pub(crate) fn attr_heap_type(
    ctx: &IrContext,
    attrs: &std::collections::BTreeMap<Symbol, Attribute>,
    key: Symbol,
) -> CompilationResult<HeapType> {
    match attrs.get(&key) {
        Some(Attribute::Int(bits)) => {
            let idx = u32::try_from(*bits).map_err(|_| {
                CompilationError::invalid_attribute(format!(
                    "heap type index {} out of u32 range",
                    bits
                ))
            })?;
            Ok(HeapType::Concrete(idx))
        }
        Some(Attribute::Symbol(sym)) => sym.with_str(symbol_to_abstract_heap_type),
        Some(Attribute::Type(ty)) => {
            let data = ctx.types.get(*ty);
            if data.dialect == Symbol::new("wasm") {
                let name = data.name;
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
// Attribute extraction helpers
// ============================================================================

/// Get attribute value as u32 (checked conversion).
///
/// Distinguishes three cases:
/// - Key absent → `missing_attribute` error
/// - Key present but wrong variant → `invalid_attribute` error
/// - Key present and Int → checked u32 conversion
pub(crate) fn attr_u32(attrs: &BTreeMap<Symbol, Attribute>, key: Symbol) -> CompilationResult<u32> {
    match attrs.get(&key) {
        Some(Attribute::Int(bits)) => u32::try_from(*bits).map_err(|_| {
            CompilationError::invalid_attribute(format!(
                "attribute '{}' value {} out of u32 range",
                key, bits
            ))
        }),
        Some(other) => Err(CompilationError::invalid_attribute(format!(
            "attribute '{}' expected Int, got {:?}",
            key, other
        ))),
        None => Err(CompilationError::missing_attribute("u32")),
    }
}
