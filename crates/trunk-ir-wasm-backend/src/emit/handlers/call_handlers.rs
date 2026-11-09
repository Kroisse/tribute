//! Call operation handlers for wasm backend.
//!
//! This module handles WebAssembly function call operations:
//! - wasm.call (direct function call)
//! - wasm.call_indirect (indirect function call via i32 table index)
//! - wasm.return_call / wasm.return_call_indirect (tail calls)

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::refs::OpRef;
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::helpers::{self, attr_u32};
use super::super::{ModuleInfo, resolve_callee};

/// Handle wasm.call operation
pub(crate) fn handle_call(
    ctx: &IrContext,
    call_op: wasm_dialect::Call,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let target = resolve_callee(call_op.callee(ctx), module_info)?;
    function.instruction(&Instruction::Call(target));
    Ok(())
}

/// Handle wasm.call_indirect operation
pub(crate) fn handle_call_indirect(
    ctx: &IrContext,
    op: OpRef,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    // All indirect calls use table-based call_indirect (no call_ref).
    let signature = helpers::exact_call_indirect_signature(ctx, op)?;
    let type_index = module_info
        .type_idx_by_type
        .get(&signature)
        .copied()
        .ok_or_else(|| {
            CompilationError::invalid_module(
                "wasm.call_indirect function type not registered in type section",
            )
        })?;
    let attrs = &ctx.op(op).attributes;
    let table_index = attrs
        .view(ctx)
        .get("table")
        .map(|_| attr_u32(ctx, attrs, "table"))
        .transpose()?
        .unwrap_or(0);

    function.instruction(&Instruction::CallIndirect {
        type_index,
        table_index,
    });
    Ok(())
}

/// Handle wasm.return_call operation (tail call)
pub(crate) fn handle_return_call(
    ctx: &IrContext,
    return_call_op: wasm_dialect::ReturnCall,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let target = resolve_callee(return_call_op.callee(ctx), module_info)?;
    function.instruction(&Instruction::ReturnCall(target));
    Ok(())
}

/// Handle `wasm.return_call_indirect` using its exact physical signature.
pub(crate) fn handle_return_call_indirect(
    ctx: &IrContext,
    op: OpRef,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let signature = helpers::exact_return_call_indirect_signature(ctx, op)?;
    let type_index = module_info
        .type_idx_by_type
        .get(&signature)
        .copied()
        .ok_or_else(|| {
            CompilationError::invalid_module(
                "wasm.return_call_indirect function type not registered in type section",
            )
        })?;
    let table_index = ctx
        .op_attrs(op)
        .get("table")
        .map(|_| attr_u32(ctx, &ctx.op(op).attributes, "table"))
        .transpose()?
        .unwrap_or(0);

    function.instruction(&Instruction::ReturnCallIndirect {
        type_index,
        table_index,
    });
    Ok(())
}
