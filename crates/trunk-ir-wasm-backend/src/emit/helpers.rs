//! Helper functions for wasm backend emission.
//!
//! This module contains type conversion and utility functions shared across
//! the emit module.

use std::collections::HashMap;

use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm;
use trunk_ir::op_interface::IndirectCallLikeOps;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, AttributeMap};
use wasm_encoder::{AbstractHeapType, HeapType, RefType, ValType};

use crate::errors::CompilationErrorKind;
use crate::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX};
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
    let data = ctx.types().get(ty);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

/// Intern an `adt.struct` type with the given name attribute.
pub(crate) fn intern_named_adt_struct(ctx: &mut IrContext, name: &'static str) -> TypeRef {
    let mut attrs = AttributeMap::new();
    attrs.insert(Symbol::new("name"), Attribute::Symbol(Symbol::new(name)));
    ctx.intern_type(trunk_ir::types::TypeData {
        dialect: Symbol::new("adt"),
        name: Symbol::new("struct"),
        params: Default::default(),
        attrs,
    })
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
    let data = ctx.types().get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return false;
    }
    data.attrs
        .get_symbol("name")
        .is_some_and(|name| name == expected_name)
}

// ============================================================================
// Type conversion
// ============================================================================

/// Get the ordered params and results of a target-owned `wasm.func_sig`.
pub(crate) fn func_type_parts(ctx: &IrContext, ty: TypeRef) -> Option<(&[TypeRef], &[TypeRef])> {
    let function = wasm::FuncSig::from_type_ref(ctx, ty)?;
    Some((function.inputs(ctx), function.results(ctx)))
}

/// Whether this IR type is registered by the backend as a concrete WasmGC
/// struct reference, and is therefore assignable to the abstract `wasm.structref`
/// without a runtime cast.
///
/// Registration follows the same structure the rest of the backend uses: builtin
/// layouts at their reserved indices (`core.bytes`, `_closure`,
/// `_Marker`, ...) and the ADT types that
/// `emit::gc_types_collection::normalize_type_for_gc` physicalizes as the
/// abstract struct supertype (`adt.typeref` and concrete variant instances
/// carrying `base_enum`). An ADT spelling without that registration evidence
/// proves nothing and stays rejected.
fn is_registered_gc_struct_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    if let Some(index) = crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty) {
        return crate::gc_types::is_builtin_struct_index(index);
    }
    let data = ctx.types().get(ty);
    data.dialect == Symbol::new("adt")
        && (data.name == Symbol::new("typeref") || data.attrs.get_type("base_enum").is_some())
}

/// Whether this IR type is registered by the backend as a concrete WasmGC array
/// reference, and is therefore assignable to the abstract `wasm.arrayref`
/// without a runtime cast. Only builtin array layouts (Bytes backing arrays and
/// the Evidence array) qualify; `core.array` spellings are handled by the
/// abstract array rule and never acquire a concrete index on their own.
fn is_registered_gc_array_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty)
        .is_some_and(|index| !crate::gc_types::is_builtin_struct_index(index))
}

/// Whether an argument can satisfy an indirect-tail parameter after the Wasm
/// backend's physical type mapping.
///
/// This is the narrow physical assignability relation shared by argument,
/// result, exact indirect/tail signature, and CPS dispatch payload checks. It
/// only accepts widenings that emission performs without a runtime cast.
pub fn is_wasm_physical_argument_assignable(
    ctx: &IrContext,
    argument: TypeRef,
    parameter: TypeRef,
) -> bool {
    if argument == parameter {
        return true;
    }

    // `core.i1` is represented by an i32 in the Wasm value space.
    if is_type(ctx, argument, "core", "i1") && is_type(ctx, parameter, "core", "i32") {
        return true;
    }

    let argument_is_typeref = is_type(ctx, argument, "adt", "typeref");
    let argument_is_structref = is_type(ctx, argument, "wasm", "structref");
    let parameter_is_structref = is_type(ctx, parameter, "wasm", "structref");
    let parameter_is_arrayref = is_type(ctx, parameter, "wasm", "arrayref");
    let parameter_is_anyref = is_type(ctx, parameter, "wasm", "anyref");

    // `adt.typeref` is emitted as the abstract Wasm `structref` type.
    if argument_is_typeref && (parameter_is_structref || parameter_is_anyref) {
        return true;
    }

    // Registered concrete GC references widen to the abstract heap type the
    // emission chose for the slot. Reaching a concrete type from an abstract one
    // still requires `wasm.ref_cast`, so the reverse directions remain false.
    if parameter_is_structref && is_registered_gc_struct_reference(ctx, argument) {
        return true;
    }
    if parameter_is_arrayref && is_registered_gc_array_reference(ctx, argument) {
        return true;
    }
    // Every registered reference denotes a struct or array, and `anyref` is their
    // common supertype, so the same registration evidence satisfies an `anyref`
    // slot. `core.array` spellings are emitted as the abstract array reference.
    if parameter_is_anyref
        && (is_registered_gc_struct_reference(ctx, argument)
            || is_registered_gc_array_reference(ctx, argument)
            || is_type(ctx, argument, "core", "array"))
    {
        return true;
    }

    // An unregistered `adt.struct` spelling is emitted as the erased `anyref`
    // reference, so it satisfies an `anyref` slot. A variant-marked type is not
    // a second spelling of that erasure: it must carry registration evidence
    // (`base_enum`), which the registered-struct rule above already accepts, and
    // a variant instance without it is malformed rather than erased.
    if parameter_is_anyref && is_type(ctx, argument, "adt", "struct") {
        return true;
    }

    let argument_is_core_array = is_type(ctx, argument, "core", "array");
    if argument_is_core_array && parameter_is_arrayref {
        return true;
    }

    // WasmGC abstract-reference upcasts. The reverse directions are checked
    // downcasts and therefore intentionally remain false here.
    let argument_is_wasm_gc_ref = is_type(ctx, argument, "wasm", "i31ref")
        || argument_is_structref
        || is_type(ctx, argument, "wasm", "arrayref");
    argument_is_wasm_gc_ref && parameter_is_anyref
}

/// Read and validate the exact physical function type required by an ordinary
/// indirect call. Erased operands cannot reconstruct the callable contract.
pub(crate) fn exact_call_indirect_signature(
    ctx: &IrContext,
    op: OpRef,
) -> CompilationResult<TypeRef> {
    let signature = IndirectCallLikeOps::exact_signature(ctx, op)
        .ok_or_else(|| CompilationError::invalid_module("wasm.call_indirect lacks signature"))?;
    exact_call_indirect_signature_with(ctx, op, signature)
}

/// Validate an exact ordinary indirect-call signature without mutating the
/// operation that carries it.
pub(crate) fn exact_call_indirect_signature_with(
    ctx: &IrContext,
    op: OpRef,
    signature: TypeRef,
) -> CompilationResult<TypeRef> {
    let results = ctx.op_result_types(op).to_vec();
    exact_call_indirect_signature_with_results(ctx, op, signature, &results)
}

/// Validate an exact ordinary indirect-call signature against an explicit result
/// list.
///
/// The lowering boundary holds a candidate replacement whose results are already
/// converted to target types, so the physical check must compare that candidate
/// list rather than the still-unconverted operation.
pub(crate) fn exact_call_indirect_signature_with_results(
    ctx: &IrContext,
    op: OpRef,
    signature: TypeRef,
    results: &[TypeRef],
) -> CompilationResult<TypeRef> {
    let (params, signature_results) = func_type_parts(ctx, signature).ok_or_else(|| {
        CompilationError::invalid_module("wasm.call_indirect signature must be wasm.func_sig")
    })?;
    let Some(_table_index) = IndirectCallLikeOps::callee(ctx, op) else {
        return Err(CompilationError::invalid_module(
            "wasm.call_indirect requires a table index operand",
        ));
    };
    let Some(args) = IndirectCallLikeOps::arguments(ctx, op) else {
        return Err(CompilationError::invalid_module(
            "wasm.call_indirect has malformed operands",
        ));
    };
    // Unit result slots produce no Wasm stack result. A `[nil]` signature
    // therefore permits a resultless call; nil operands remain nullable refs.
    let results_match = results == signature_results
        || (results.is_empty()
            && matches!(signature_results, [result] if is_nil_type(ctx, *result)));
    if params.len() != args.len()
        || params.iter().zip(args).any(|(param, arg)| {
            !is_wasm_physical_argument_assignable(ctx, value_type(ctx, *arg), *param)
        })
        || !results_match
    {
        return Err(CompilationError::invalid_module(
            "wasm.call_indirect operands or result do not match its exact signature",
        ));
    }
    Ok(signature)
}

/// Read and validate the exact physical function type required by an indirect
/// proper tail transfer. This backend deliberately does not infer the type
/// from the runtime table index and operands.
pub(crate) fn exact_return_call_indirect_signature(
    ctx: &IrContext,
    op: OpRef,
) -> CompilationResult<TypeRef> {
    let signature = IndirectCallLikeOps::exact_signature(ctx, op).ok_or_else(|| {
        CompilationError::invalid_module("wasm.return_call_indirect lacks signature")
    })?;
    exact_return_call_indirect_signature_with(ctx, op, signature)
}

/// Validate an exact callable signature against an indirect proper tail
/// transfer without reading or mutating its attribute map.
pub(crate) fn exact_return_call_indirect_signature_with(
    ctx: &IrContext,
    op: OpRef,
    signature: TypeRef,
) -> CompilationResult<TypeRef> {
    let (params, results) = func_type_parts(ctx, signature).ok_or_else(|| {
        CompilationError::invalid_module(
            "wasm.return_call_indirect signature must be wasm.func_sig",
        )
    })?;
    let cps = ctx.op(op).attributes.get("tribute.calling_convention")
        == Some(&trunk_ir::types::Attribute::Int(2));
    if !results.is_empty() && (cps || !matches!(results, [result] if is_nil_type(ctx, *result))) {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect signature must have an empty result",
        ));
    }
    let Some(table_index) = IndirectCallLikeOps::callee(ctx, op) else {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect requires a table index operand",
        ));
    };
    let Some(args) = IndirectCallLikeOps::arguments(ctx, op) else {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect has malformed operands",
        ));
    };
    if !is_type(ctx, value_type(ctx, table_index), "core", "i32") {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect first operand must be an i32 table index",
        ));
    }
    if params.len() != args.len()
        || params.iter().zip(args).any(|(param, arg)| {
            !is_wasm_physical_argument_assignable(ctx, value_type(ctx, *arg), *param)
        })
    {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect operands do not match its exact signature",
        ));
    }
    Ok(signature)
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
        let nullable = ctx.types().get(ty).attrs.get_bool("nullable") == Some(true);
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
    } else if is_type(ctx, ty, "wasm", "func_sig") {
        Ok(ValType::Ref(RefType::FUNCREF))
    } else if ctx.types().get(ty).dialect == Symbol::new("wasm") {
        let name = ctx.types().get(ty).name;
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
    } else if is_closure_struct_type(ctx, ty) {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(CLOSURE_STRUCT_IDX),
        }))
    } else if is_type(ctx, ty, "adt", "typeref") {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            },
        }))
    } else if ctx.types().get(ty).dialect == Symbol::new("adt") {
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
        let data = ctx.types().get(ty);
        Err(CompilationError::type_error(format!(
            "unsupported wasm value type: {}.{}",
            data.dialect, data.name
        )))
    }
}

fn is_bytes_array_ref(ctx: &IrContext, ty: TypeRef) -> bool {
    let reference = ctx.types().get(ty);
    if reference.dialect != Symbol::new("core")
        || reference.name != Symbol::new("ref")
        || reference.params.len() != 1
    {
        return false;
    }
    let array = ctx.types().get(reference.params[0]);
    array.dialect == Symbol::new("core")
        && array.name == Symbol::new("array")
        && array.params.len() == 1
        && is_type(ctx, array.params[0], "core", "i8")
}

/// Convert an ordered Wasm signature result list to machine result slots.
///
/// Result `core.nil` remains the established omitted target slot rule. Nil in
/// value or input positions is separately lowered as a nullable reference.
pub(crate) fn signature_result_types(
    ctx: &IrContext,
    results: &[TypeRef],
    type_idx_by_type: &HashMap<TypeRef, u32>,
) -> CompilationResult<Vec<ValType>> {
    results
        .iter()
        .filter(|ty| !is_nil_type(ctx, **ty))
        .map(|ty| type_to_valtype(ctx, *ty, type_idx_by_type))
        .collect()
}

// ============================================================================
// Heap type helpers
// ============================================================================

/// Extract a heap type from operation attributes.
pub(crate) fn attr_heap_type(
    ctx: &IrContext,
    attrs: &AttributeMap,
    key: Symbol,
) -> CompilationResult<HeapType> {
    match attrs.get(key) {
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
            let data = ctx.types().get(*ty);
            if data.dialect == Symbol::new("wasm") {
                let name = data.name;
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
pub(crate) fn attr_u32(attrs: &AttributeMap, key: Symbol) -> CompilationResult<u32> {
    match attrs.get(key) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::types::TypeDataBuilder;

    #[test]
    fn core_array_uses_nullable_abstract_array_value_type() {
        let mut ctx = IrContext::new();
        let i32_ty =
            ctx.intern_type(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let array_ty = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
                .param(i32_ty)
                .build(),
        );

        assert_eq!(
            type_to_valtype(&ctx, array_ty, &HashMap::new()).expect("core.array is supported"),
            ValType::Ref(RefType {
                nullable: true,
                heap_type: HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Array,
                },
            })
        );
    }

    #[test]
    fn func_sig_uses_nullable_funcref_value_type() {
        let mut ctx = IrContext::new();
        let i32_ty =
            ctx.intern_type(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let signature = wasm::func_sig(&mut ctx, [i32_ty], [i32_ty]).as_type_ref();

        assert_eq!(
            type_to_valtype(&ctx, signature, &HashMap::new())
                .expect("func.func_sig is a supported Wasm value type"),
            ValType::Ref(RefType::FUNCREF)
        );
    }
}
