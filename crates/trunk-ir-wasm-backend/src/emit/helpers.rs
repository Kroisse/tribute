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

use crate::assignability::is_wasm_physical_argument_assignable;
use crate::errors::CompilationErrorKind;
use crate::gc_types::{
    BYTES_ARRAY_IDX, BYTES_DATA_LAYOUT, BYTES_LAYOUT, BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX,
};
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
    let data = ctx.get_type(ty);
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

/// Whether a type carries the runtime layout identifier `layout`.
pub(crate) fn has_layout(ctx: &IrContext, ty: TypeRef, layout: &'static str) -> bool {
    ctx.get_type(ty)
        .attrs
        .get_str(ctx, trunk_ir::types::LAYOUT_ATTR)
        == Some(layout)
}

/// The builtin GC type index a type's runtime layout identifier selects.
pub(crate) fn builtin_layout_type_idx(ctx: &IrContext, ty: TypeRef) -> Option<u32> {
    let layout = ctx
        .get_type(ty)
        .attrs
        .get_str(ctx, trunk_ir::types::LAYOUT_ATTR)?;
    crate::gc_types::builtin_layout_idx(layout)
}

/// Check if a type is the builtin closure struct, identified by its runtime
/// layout attribute.
pub(crate) fn is_closure_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, "adt", "struct") && has_layout(ctx, ty, crate::gc_types::CLOSURE_LAYOUT)
}

/// The canonical key standing for every type of one builtin runtime layout.
pub(crate) fn intern_layout_key(ctx: &mut IrContext, layout: &'static str) -> TypeRef {
    let mut attrs = AttributeMap::new();
    attrs.insert(trunk_ir::types::LAYOUT_ATTR, ctx.string_attr(layout));
    ctx.intern_type(trunk_ir::types::TypeData {
        dialect: Symbol::new("adt"),
        name: Symbol::new("struct"),
        params: Default::default(),
        attrs,
    })
}

// ============================================================================
// Type conversion
// ============================================================================

/// Get the ordered params and results of a target-owned `wasm.func_sig`.
pub(crate) fn func_type_parts(ctx: &IrContext, ty: TypeRef) -> Option<(&[TypeRef], &[TypeRef])> {
    let function = wasm::FuncSig::from_type_ref(ctx, ty)?;
    Some((function.inputs(ctx), function.results(ctx)))
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
    // Tail legality is the caller/callee result-list agreement checked by the
    // emission validator; this helper validates only the exact call shape.
    let (params, _results) = func_type_parts(ctx, signature).ok_or_else(|| {
        CompilationError::invalid_module(
            "wasm.return_call_indirect signature must be wasm.func_sig",
        )
    })?;
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
    if is_type(ctx, ty, "core", "i32")
        || is_type(ctx, ty, "core", "i1")
        || is_type(ctx, ty, "core", "i8")
        || is_type(ctx, ty, "core", "i16")
    {
        // Narrow integers live in an `i32` with unspecified upper bits.
        Ok(ValType::I32)
    } else if is_type(ctx, ty, "core", "i64") {
        Ok(ValType::I64)
    } else if is_type(ctx, ty, "core", "f32") {
        Ok(ValType::F32)
    } else if is_type(ctx, ty, "core", "f64") {
        Ok(ValType::F64)
    } else if has_layout(ctx, ty, BYTES_LAYOUT) {
        // A Bytes value always exists; its struct is never null.
        Ok(ValType::Ref(RefType {
            nullable: false,
            heap_type: HeapType::Concrete(BYTES_STRUCT_IDX),
        }))
    } else if has_layout(ctx, ty, BYTES_DATA_LAYOUT) {
        // The Bytes struct's backing array field is non-nullable.
        Ok(ValType::Ref(RefType {
            nullable: false,
            heap_type: HeapType::Concrete(BYTES_ARRAY_IDX),
        }))
    } else if is_type(ctx, ty, "core", "ptr") {
        Ok(ValType::I32)
    } else if let Some(type_idx) = builtin_layout_type_idx(ctx, ty) {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }))
    } else if let Some(&type_idx) = type_idx_by_type.get(&ty) {
        Ok(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        }))
    } else if is_type(ctx, ty, "wasm", "func_sig") {
        Ok(ValType::Ref(RefType::FUNCREF))
    } else if ctx.get_type(ty).dialect == Symbol::new("wasm") {
        let name = ctx.get_type(ty).name.clone();
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
    } else if ctx.get_type(ty).dialect == Symbol::new("adt") {
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
        let data = ctx.get_type(ty);
        Err(CompilationError::type_error(format!(
            "unsupported wasm value type: {}.{}",
            data.dialect, data.name
        )))
    }
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
        Some(Attribute::String(name)) => symbol_to_abstract_heap_type(ctx.str(*name)),
        Some(Attribute::Type(ty)) => {
            let data = ctx.get_type(*ty);
            if data.dialect == Symbol::new("wasm") {
                let name = data.name.clone();
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::types::TypeDataBuilder;

    #[test]
    fn builtin_closure_is_identified_by_layout_not_name() {
        let mut ctx = IrContext::new();
        let name_attr = ctx.string_attr("_closure");
        let named = ctx.intern_type(
            TypeDataBuilder::new("adt", "struct")
                .attr("name", name_attr)
                .build(),
        );
        let closure_layout = ctx.string_attr(crate::gc_types::CLOSURE_LAYOUT);
        let name_attr = ctx.string_attr("Other");
        let layout = ctx.intern_type(
            TypeDataBuilder::new("adt", "struct")
                .attr("name", name_attr)
                .attr(trunk_ir::types::LAYOUT_ATTR, closure_layout)
                .build(),
        );

        assert!(!is_closure_struct_type(&ctx, named));
        assert!(is_closure_struct_type(&ctx, layout));
    }

    #[test]
    fn core_array_uses_nullable_abstract_array_value_type() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let array_ty = ctx.intern_type(TypeDataBuilder::new("core", "array").param(i32_ty).build());

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
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let signature = wasm::func_sig(&mut ctx, [i32_ty], [i32_ty]).as_type_ref();

        assert_eq!(
            type_to_valtype(&ctx, signature, &HashMap::new())
                .expect("func.func_sig is a supported Wasm value type"),
            ValType::Ref(RefType::FUNCREF)
        );
    }
}
