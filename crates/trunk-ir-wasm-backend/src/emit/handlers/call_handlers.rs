//! Call operation handlers for wasm backend.
//!
//! This module handles WebAssembly function call operations:
//! - wasm.call (direct function call)
//! - wasm.call_indirect (indirect function call via i32 table index)
//! - wasm.return_call / wasm.return_call_indirect (tail calls)

use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::op_interface::IndirectCallLikeOps;
use trunk_ir::refs::OpRef;
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::helpers;
use super::super::value_emission::{emit_operands, emit_value};
use super::super::{FunctionEmitContext, ModuleInfo, resolve_callee, set_result_local};

/// Handle wasm.call operation
pub(crate) fn handle_call(
    ctx: &IrContext,
    call_op: wasm_dialect::Call,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = call_op.op_ref();
    let operands = ctx.op_operands(op);
    let callee = call_op.callee(ctx);
    let target = resolve_callee(callee, module_info)?;

    // Boxing/unboxing for generic calls is now handled by the boxing pass
    // (tribute-passes/src/boxing.rs) which inserts explicit tribute_rt.box_*/unbox_* ops.
    // These are lowered to wasm instructions by tribute_rt_to_wasm.rs.
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::Call(target));

    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle wasm.call_indirect operation
pub(crate) fn handle_call_indirect(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    // wasm.call_indirect: indirect function call via i32 table index
    // All indirect calls use table-based call_indirect (no call_ref).
    // Operands: [table_idx, arg1, arg2, ..., argN]
    // WebAssembly expects: [arg1, arg2, ..., argN, table_idx]

    // The callee (i32 table index) is the FIRST operand, followed by args.
    let first_operand = IndirectCallLikeOps::callee(ctx, op).ok_or_else(|| {
        CompilationError::invalid_module("wasm.call_indirect requires a table index operand")
    })?;
    let args = IndirectCallLikeOps::arguments(ctx, op).ok_or_else(|| {
        CompilationError::invalid_module("wasm.call_indirect has malformed operands")
    })?;
    let first_operand_ty = helpers::value_type(ctx, first_operand);

    // All call_indirect operations must use i32 table index
    if !helpers::is_type(ctx, first_operand_ty, "core", "i32") {
        let data = ctx.types().get(first_operand_ty);
        return Err(CompilationError::invalid_module(format!(
            "call_indirect first operand must be i32 table index, got {}.{}",
            data.dialect, data.name
        )));
    }

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
        .get("table")
        .map(|_| attr_u32(attrs, Symbol::new("table")))
        .transpose()?
        .unwrap_or(0);

    for &arg in args {
        emit_value(ctx, arg, emit_ctx, function)?;
    }
    emit_value(ctx, first_operand, emit_ctx, function)?;
    function.instruction(&Instruction::CallIndirect {
        type_index,
        table_index,
    });
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle wasm.return_call operation (tail call)
pub(crate) fn handle_return_call(
    ctx: &IrContext,
    return_call_op: wasm_dialect::ReturnCall,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(return_call_op.op_ref());
    let callee = return_call_op.callee(ctx);
    let target = resolve_callee(callee, module_info)?;

    // Boxing for generic calls is now handled by the boxing pass
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ReturnCall(target));
    Ok(())
}

/// Handle `wasm.return_call_indirect` using its exact physical signature.
pub(crate) fn handle_return_call_indirect(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
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
        .op(op)
        .attributes
        .get("table")
        .map(|_| attr_u32(&ctx.op(op).attributes, Symbol::new("table")))
        .transpose()?
        .unwrap_or(0);
    let table_index_value = IndirectCallLikeOps::callee(ctx, op).ok_or_else(|| {
        CompilationError::invalid_module("wasm.return_call_indirect requires a table index operand")
    })?;
    let args = IndirectCallLikeOps::arguments(ctx, op).ok_or_else(|| {
        CompilationError::invalid_module("wasm.return_call_indirect has malformed operands")
    })?;

    // WebAssembly evaluates arguments before the table index.
    for &arg in args {
        emit_value(ctx, arg, emit_ctx, function)?;
    }
    emit_value(ctx, table_index_value, emit_ctx, function)?;
    function.instruction(&Instruction::ReturnCallIndirect {
        type_index,
        table_index,
    });
    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

use super::super::helpers::attr_u32;
