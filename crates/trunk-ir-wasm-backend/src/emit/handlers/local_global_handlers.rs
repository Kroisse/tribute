//! Local and global variable handlers for wasm backend.
//!
//! This module handles WebAssembly local and global variable operations:
//! - wasm.local_get, wasm.local_set, wasm.local_tee
//! - wasm.global_get, wasm.global_set

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;

use super::super::ModuleInfo;
use super::super::helpers;

/// Handle wasm.local_get operation
pub(crate) fn handle_local_get(
    ctx: &IrContext,
    local_op: wasm_dialect::LocalGet,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::LocalGet(local_op.index(ctx)));
    Ok(())
}

/// Handle wasm.local_set operation
pub(crate) fn handle_local_set(
    ctx: &IrContext,
    local_op: wasm_dialect::LocalSet,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::LocalSet(local_op.index(ctx)));
    Ok(())
}

/// Handle wasm.local_tee operation
pub(crate) fn handle_local_tee(
    ctx: &IrContext,
    local_op: wasm_dialect::LocalTee,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::LocalTee(local_op.index(ctx)));
    Ok(())
}

/// Handle wasm.global_get operation
///
/// When reading from an anyref global, the result may need to be cast to the
/// specific type expected by the IR result type. This happens for globals like
/// $yield_cont which store different continuation types as anyref.
pub(crate) fn handle_global_get(
    ctx: &IrContext,
    global_op: wasm_dialect::GlobalGet,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = global_op.op_ref();
    let index = global_op.index(ctx);
    function.instruction(&Instruction::GlobalGet(index));

    // Check if the global's actual type is anyref but the IR result type is more specific.
    // If so, insert a ref.cast to narrow the type.
    if let Some(global_def) = module_info.globals.get(index as usize)
        && is_anyref_valtype(&global_def.valtype)
    {
        let result_types = ctx.op_result_types(op);
        if let Some(&result_ty) = result_types.first()
            && let Ok(result_valtype) =
                helpers::type_to_valtype(ctx, result_ty, &module_info.type_idx_by_type)
            && let Some(heap_type) = concrete_heap_type(&result_valtype)
        {
            function.instruction(&Instruction::RefCastNullable(heap_type));
        }
    }
    Ok(())
}

/// Check if a ValType is anyref.
fn is_anyref_valtype(vt: &wasm_encoder::ValType) -> bool {
    matches!(
        vt,
        wasm_encoder::ValType::Ref(wasm_encoder::RefType {
            nullable: true,
            heap_type: wasm_encoder::HeapType::Abstract {
                shared: false,
                ty: wasm_encoder::AbstractHeapType::Any
            }
        })
    )
}

/// Extract a concrete HeapType from a ValType, if present.
fn concrete_heap_type(vt: &wasm_encoder::ValType) -> Option<wasm_encoder::HeapType> {
    match vt {
        wasm_encoder::ValType::Ref(rt) => match &rt.heap_type {
            ht @ wasm_encoder::HeapType::Concrete(_) => Some(*ht),
            _ => None,
        },
        _ => None,
    }
}

/// Handle wasm.global_set operation
pub(crate) fn handle_global_set(
    ctx: &IrContext,
    global_op: wasm_dialect::GlobalSet,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::GlobalSet(global_op.index(ctx)));
    Ok(())
}
