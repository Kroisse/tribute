//! Constant value handlers for wasm backend.
//!
//! This module handles the emission of constant values (i32, i64, f32, f64).

use trunk_ir::arena::IrContext;
use trunk_ir::arena::dialect::wasm as wasm_dialect;
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;

use super::super::{FunctionEmitContext, set_result_local};

/// Handle i32.const operation
pub(crate) fn handle_i32_const(
    ctx: &IrContext,
    const_op: wasm_dialect::I32Const,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let value = const_op.value(ctx);
    function.instruction(&Instruction::I32Const(value));
    set_result_local(ctx, const_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle i64.const operation
pub(crate) fn handle_i64_const(
    ctx: &IrContext,
    const_op: wasm_dialect::I64Const,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let value = const_op.value(ctx);
    function.instruction(&Instruction::I64Const(value));
    set_result_local(ctx, const_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle f32.const operation
pub(crate) fn handle_f32_const(
    ctx: &IrContext,
    const_op: wasm_dialect::F32Const,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let value = const_op.value(ctx);
    function.instruction(&Instruction::F32Const(value.into()));
    set_result_local(ctx, const_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle f64.const operation
pub(crate) fn handle_f64_const(
    ctx: &IrContext,
    const_op: wasm_dialect::F64Const,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let value = const_op.value(ctx);
    function.instruction(&Instruction::F64Const(value.into()));
    set_result_local(ctx, const_op.op_ref(), emit_ctx, function)?;
    Ok(())
}
