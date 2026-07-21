//! Array operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC array operations.

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;

use super::super::value_emission::emit_operands;
use super::super::{FunctionEmitContext, ModuleInfo, set_result_local};

/// Handle array.new operation
pub(crate) fn handle_array_new(
    ctx: &IrContext,
    array_new_op: wasm_dialect::ArrayNew,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_new_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ArrayNew(array_new_op.type_idx(ctx)));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.new_default operation
pub(crate) fn handle_array_new_default(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    let array_new = wasm_dialect::ArrayNewDefault::from_op(ctx, op)
        .expect("handler called for wasm.array_new_default");
    function.instruction(&Instruction::ArrayNewDefault(array_new.type_idx(ctx)));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.get operation
pub(crate) fn handle_array_get(
    ctx: &IrContext,
    array_get_op: wasm_dialect::ArrayGet,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ArrayGet(array_get_op.type_idx(ctx)));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.get_s operation (sign-extending load)
pub(crate) fn handle_array_get_s(
    ctx: &IrContext,
    array_get_s_op: wasm_dialect::ArrayGetS,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_s_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ArrayGetS(array_get_s_op.type_idx(ctx)));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.get_u operation (zero-extending load)
pub(crate) fn handle_array_get_u(
    ctx: &IrContext,
    array_get_u_op: wasm_dialect::ArrayGetU,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_u_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ArrayGetU(array_get_u_op.type_idx(ctx)));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.set operation
pub(crate) fn handle_array_set(
    ctx: &IrContext,
    array_set_op: wasm_dialect::ArraySet,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_set_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ArraySet(array_set_op.type_idx(ctx)));
    Ok(())
}

/// Handle array.copy operation
pub(crate) fn handle_array_copy(
    ctx: &IrContext,
    array_copy_op: wasm_dialect::ArrayCopy,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(array_copy_op.op_ref());
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ArrayCopy {
        array_type_index_dst: array_copy_op.dst_type_idx(ctx),
        array_type_index_src: array_copy_op.src_type_idx(ctx),
    });
    Ok(())
}
