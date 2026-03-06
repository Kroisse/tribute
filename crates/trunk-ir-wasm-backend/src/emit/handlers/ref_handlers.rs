//! Reference operation handlers for wasm backend.
//!
//! This module handles WebAssembly reference operations like ref.null, ref.func,
//! ref.cast, and ref.test.

use std::collections::BTreeMap;

use trunk_ir::Symbol;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::types::Attribute;
use wasm_encoder::{AbstractHeapType, Function, HeapType, Instruction};

use crate::gc_types::ATTR_FIELD_COUNT;
use crate::{CompilationError, CompilationResult};

use super::super::helpers::{
    self, attr_heap_type, get_type_idx_from_attrs, symbol_to_abstract_heap_type,
};
use super::super::value_emission::emit_operands;
use super::super::{
    ATTR_HEAP_TYPE, ATTR_TARGET_TYPE, FunctionEmitContext, ModuleInfo, set_result_local,
};

/// Handle ref.null operation
pub(crate) fn handle_ref_null(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let attrs = &ctx.op(op).attributes;
    // Infer type from result type
    let inferred_type = ctx.op_result_types(op).first().copied();

    // Try heap_type attribute first. If present but invalid, propagate error.
    let heap_type = if attrs.get(&ATTR_HEAP_TYPE()).is_some() {
        attr_heap_type(ctx, attrs, ATTR_HEAP_TYPE())?
    } else {
        // Attribute not present - fall back to type inference, then attr_heap_type on
        // the target_type attribute (which may contain abstract wasm types like anyref)
        get_type_idx_from_attrs(ctx, attrs, inferred_type, &module_info.type_idx_by_type)
            .map(HeapType::Concrete)
            .or_else(|| attr_heap_type(ctx, attrs, ATTR_TARGET_TYPE()).ok())
            .or_else(|| inferred_heap_type_from_result(ctx, inferred_type))
            .ok_or_else(|| CompilationError::missing_attribute("heap_type or type"))?
    };

    function.instruction(&Instruction::RefNull(heap_type));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle ref.func operation
pub(crate) fn handle_ref_func(
    ctx: &IrContext,
    ref_func_op: arena_wasm::RefFunc,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let func_name = ref_func_op.func_name(ctx);
    let func_idx = resolve_callee(func_name, module_info)?;
    function.instruction(&Instruction::RefFunc(func_idx));
    set_result_local(ctx, ref_func_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle ref.cast operation
pub(crate) fn handle_ref_cast(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    let attrs = &ctx.op(op).attributes;
    // Infer type from result type (the target type it casts to)
    let inferred_type = ctx.op_result_types(op).first().copied();

    // Check if this is a placeholder struct type (wasm.structref with field_count)
    // If so, use the concrete type index from the placeholder map
    let heap_type = if let Some(Attribute::Type(target_ty)) = attrs.get(&ATTR_TARGET_TYPE()) {
        if helpers::is_type(ctx, *target_ty, "wasm", "structref") {
            // structref placeholder - try to find concrete type via type attribute first
            // The `type` attribute contains the actual adt.struct/typeref type
            if let Some(Attribute::Type(concrete_ty)) = attrs.get(&Symbol::new("type")) {
                if let Some(&type_idx) = module_info.type_idx_by_type.get(concrete_ty) {
                    let data = ctx.types.get(*concrete_ty);
                    tracing::debug!(
                        "ref_cast: using type attr for concrete type_idx={} ({}.{})",
                        type_idx,
                        data.dialect,
                        data.name
                    );
                    HeapType::Concrete(type_idx)
                } else {
                    resolve_placeholder_structref(attrs, *target_ty, module_info)?
                }
            } else {
                resolve_placeholder_structref(attrs, *target_ty, module_info)?
            }
        } else {
            // Non-placeholder type - try registry first, then attr_heap_type, then inferred type
            module_info
                .type_idx_by_type
                .get(target_ty)
                .map(|&idx| HeapType::Concrete(idx))
                .or_else(|| attr_heap_type(ctx, attrs, ATTR_TARGET_TYPE()).ok())
                .or_else(|| {
                    // Fall back to using inferred type
                    get_type_idx_from_attrs(
                        ctx,
                        attrs,
                        inferred_type,
                        &module_info.type_idx_by_type,
                    )
                    .map(HeapType::Concrete)
                })
                .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?
        }
    } else {
        attr_heap_type(ctx, attrs, ATTR_TARGET_TYPE())
            .ok()
            .or_else(|| {
                get_type_idx_from_attrs(ctx, attrs, inferred_type, &module_info.type_idx_by_type)
                    .map(HeapType::Concrete)
            })
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?
    };

    tracing::debug!(
        "ref_cast: emitting RefCastNullable with heap_type={:?}",
        heap_type
    );
    function.instruction(&Instruction::RefCastNullable(heap_type));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle ref.test operation
pub(crate) fn handle_ref_test(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    let attrs = &ctx.op(op).attributes;
    // ref_test result is i32, target type must be in attribute (can't infer)
    // Try attr_heap_type first, then fall back to type-index lookup (mirroring handle_ref_cast)
    // Pass target_type as inferred_type so get_type_idx_from_attrs can find it
    let target_type_ref = attrs.get(&ATTR_TARGET_TYPE()).and_then(|a| {
        if let Attribute::Type(ty) = a {
            Some(*ty)
        } else {
            None
        }
    });
    let heap_type = attr_heap_type(ctx, attrs, ATTR_TARGET_TYPE())
        .ok()
        .or_else(|| {
            get_type_idx_from_attrs(ctx, attrs, target_type_ref, &module_info.type_idx_by_type)
                .map(HeapType::Concrete)
        })
        .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?;

    function.instruction(&Instruction::RefTestNullable(heap_type));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Resolve a placeholder structref to a HeapType by looking up the field_count
/// in the placeholder_struct_type_idx map.
///
/// Falls back to abstract structref if the placeholder is not found in the map.
fn resolve_placeholder_structref(
    attrs: &BTreeMap<Symbol, Attribute>,
    target_ty: TypeRef,
    module_info: &ModuleInfo,
) -> CompilationResult<HeapType> {
    if let Some(Attribute::IntBits(fc)) = attrs.get(&ATTR_FIELD_COUNT()) {
        let field_count = usize::try_from(*fc).map_err(|_| {
            CompilationError::invalid_attribute(format!(
                "ref_cast: field_count value {} out of usize range",
                fc
            ))
        })?;
        if let Some(&type_idx) = module_info
            .placeholder_struct_type_idx
            .get(&(target_ty, field_count))
        {
            tracing::debug!(
                "ref_cast: found placeholder type_idx={} for field_count={}",
                type_idx,
                field_count
            );
            Ok(HeapType::Concrete(type_idx))
        } else {
            tracing::debug!(
                "ref_cast: placeholder lookup FAILED for field_count={}, falling back to abstract structref",
                field_count
            );
            Ok(HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct,
            })
        }
    } else {
        // No field_count - fall back to abstract structref
        Ok(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Struct,
        })
    }
}

/// Resolve a callee symbol to a function index.
fn resolve_callee(path: Symbol, module_info: &ModuleInfo) -> CompilationResult<u32> {
    module_info
        .func_indices
        .get(&path)
        .copied()
        .ok_or_else(|| CompilationError::function_not_found(&path.to_string()))
}

/// Try to infer a HeapType from the result type of an operation.
///
/// If the result type is an abstract WASM type (e.g., wasm.anyref, wasm.i31ref),
/// returns the corresponding abstract HeapType. Returns None for non-wasm or
/// concrete types.
fn inferred_heap_type_from_result(ctx: &IrContext, result_ty: Option<TypeRef>) -> Option<HeapType> {
    let ty = result_ty?;
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("wasm") {
        return None;
    }
    data.name.with_str(symbol_to_abstract_heap_type).ok()
}
