//! Array operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC array operations.

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::refs::{OpRef, TypeRef};
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::helpers;
use super::super::value_emission::emit_operands;
use super::super::{FunctionEmitContext, ModuleInfo, set_result_local};

/// Get type_idx from operation attributes or inferred type.
///
/// Uses the shared helper to ensure consistent type resolution logic.
fn get_type_idx(
    ctx: &IrContext,
    op: OpRef,
    inferred_type: Option<TypeRef>,
    module_info: &ModuleInfo,
) -> CompilationResult<u32> {
    helpers::get_type_idx_from_attrs(
        ctx,
        &ctx.op(op).attributes,
        inferred_type,
        &module_info.type_idx_by_type,
    )
    .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
}

/// Handle array.new operation
pub(crate) fn handle_array_new(
    ctx: &IrContext,
    array_new_op: wasm_dialect::ArrayNew,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_new_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from result type (type_idx attr may not be set during IR generation)
    let inferred_type = ctx.op_result_types(op).first().copied();
    let type_idx = get_type_idx(ctx, op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayNew(type_idx));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.new_default operation
pub(crate) fn handle_array_new_default(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from result type
    let inferred_type = ctx.op_result_types(op).first().copied();
    let type_idx = get_type_idx(ctx, op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayNewDefault(type_idx));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.get operation
pub(crate) fn handle_array_get(
    ctx: &IrContext,
    array_get_op: wasm_dialect::ArrayGet,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands.first().map(|v| helpers::value_type(ctx, *v));
    let type_idx = get_type_idx(ctx, op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayGet(type_idx));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.get_s operation (sign-extending load)
pub(crate) fn handle_array_get_s(
    ctx: &IrContext,
    array_get_s_op: wasm_dialect::ArrayGetS,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_s_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands.first().map(|v| helpers::value_type(ctx, *v));
    let type_idx = get_type_idx(ctx, op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayGetS(type_idx));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.get_u operation (zero-extending load)
pub(crate) fn handle_array_get_u(
    ctx: &IrContext,
    array_get_u_op: wasm_dialect::ArrayGetU,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_u_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands.first().map(|v| helpers::value_type(ctx, *v));
    let type_idx = get_type_idx(ctx, op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayGetU(type_idx));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle array.set operation
pub(crate) fn handle_array_set(
    ctx: &IrContext,
    array_set_op: wasm_dialect::ArraySet,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_set_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands.first().map(|v| helpers::value_type(ctx, *v));
    let type_idx = get_type_idx(ctx, op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArraySet(type_idx));
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
