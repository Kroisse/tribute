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
use crate::gc_types::{BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX, STEP_IDX};
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

/// Check if a type is a continuation struct (adt.struct with name "_Continuation").
pub(crate) fn is_continuation_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_named_adt_struct(ctx, ty, "_Continuation")
}

/// Check if a type is a resume wrapper struct (adt.struct with name "_ResumeWrapper").
pub(crate) fn is_resume_wrapper_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_named_adt_struct(ctx, ty, "_ResumeWrapper")
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
    let (params, ret) = data.params.split_at(data.params.len() - 1);
    Some((params, ret[0]))
}

/// Convert an IR type to a WebAssembly value type.
pub(crate) fn type_to_valtype(
    ctx: &IrContext,
    ty: TypeRef,
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> CompilationResult<ValType> {
    if is_type(ctx, ty, "core", "i32")
        || is_type(ctx, ty, "core", "i1")
        || is_type(ctx, ty, "cont", "prompt_tag")
    {
        Ok(ValType::I32)
    } else if is_type(ctx, ty, "core", "i64") {
        Ok(ValType::I64)
    } else if is_type(ctx, ty, "core", "f32") {
        Ok(ValType::F32)
    } else if is_type(ctx, ty, "core", "f64") {
        Ok(ValType::F64)
    } else if is_type(ctx, ty, "core", "bytes") {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(BYTES_STRUCT_IDX),
        }))
    } else if is_type(ctx, ty, "core", "string") || is_type(ctx, ty, "core", "ptr") {
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
    } else if is_type(ctx, ty, "cont", "continuation") {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }))
    } else if is_type(ctx, ty, "trampoline", "step") {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(STEP_IDX),
        }))
    } else if is_type(ctx, ty, "trampoline", "continuation")
        || is_type(ctx, ty, "trampoline", "state")
        || is_type(ctx, ty, "trampoline", "resume_wrapper")
    {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
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
// Type index helpers
// ============================================================================

/// Get type_idx from attributes or inferred type.
///
/// Priority: type_idx attr > type attr > inferred_type (from result/operand)
pub(crate) fn get_type_idx_from_attrs(
    ctx: &IrContext,
    attrs: &std::collections::BTreeMap<Symbol, Attribute>,
    inferred_type: Option<TypeRef>,
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> Option<u32> {
    // First try type_idx attribute
    match attrs.get(&Symbol::new("type_idx")) {
        Some(Attribute::Int(idx)) => {
            return Some(u32::try_from(*idx).expect("type_idx attribute value out of u32 range"));
        }
        Some(_) => {
            // type_idx present but wrong variant — this is an invariant violation
            panic!("type_idx attribute has unexpected variant (expected Int)");
        }
        None => {} // not present, continue to fallback
    }
    // Fall back to type attribute
    if let Some(Attribute::Type(ty)) = attrs.get(&Symbol::new("type")) {
        if is_closure_struct_type(ctx, *ty) {
            return Some(CLOSURE_STRUCT_IDX);
        }
        return type_idx_by_type.get(ty).copied();
    }
    // Fall back to inferred type
    if let Some(ty) = inferred_type {
        if is_closure_struct_type(ctx, ty) {
            return Some(CLOSURE_STRUCT_IDX);
        }
        return type_idx_by_type.get(&ty).copied();
    }
    None
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

/// Get field index from attributes, trying both `field_idx` and `field` attribute names.
///
/// Only falls back to `field` when `field_idx` is missing. If `field_idx` is present
/// but has a wrong variant or out-of-range value, that error is propagated immediately.
pub(crate) fn attr_field_idx(attrs: &BTreeMap<Symbol, Attribute>) -> CompilationResult<u32> {
    match attr_u32(attrs, Symbol::new("field_idx")) {
        Ok(v) => Ok(v),
        Err(e) if e.is_missing_attribute() => attr_u32(attrs, Symbol::new("field")),
        Err(e) => Err(e),
    }
}
