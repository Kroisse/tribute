//! Array operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC array operations.

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;

use super::super::ModuleInfo;

/// Handle array.new operation
pub(crate) fn handle_array_new(
    ctx: &IrContext,
    array_new_op: wasm_dialect::ArrayNew,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArrayNew(array_new_op.type_idx(ctx)));
    Ok(())
}

/// Handle array.new_default operation
pub(crate) fn handle_array_new_default(
    ctx: &IrContext,
    op: OpRef,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let array_new = wasm_dialect::ArrayNewDefault::from_op(ctx, op)
        .expect("handler called for wasm.array_new_default");
    function.instruction(&Instruction::ArrayNewDefault(array_new.type_idx(ctx)));
    Ok(())
}

/// Handle array.new_data operation: a new array of `size` elements read from
/// a passive data segment at `offset`, both on the stack.
pub(crate) fn handle_array_new_data(
    ctx: &IrContext,
    array_new_data: wasm_dialect::ArrayNewData,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArrayNewData {
        array_type_index: array_new_data.type_idx(ctx),
        array_data_index: array_new_data.data_idx(ctx),
    });
    Ok(())
}

/// Handle array.new_fixed operation
pub(crate) fn handle_array_new_fixed(
    ctx: &IrContext,
    array_new_fixed: wasm_dialect::ArrayNewFixed,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(array_new_fixed.op_ref());
    let array_size =
        u32::try_from(operands.len()).expect("array.new_fixed has at most u32::MAX elements");
    function.instruction(&Instruction::ArrayNewFixed {
        array_type_index: array_new_fixed.type_idx(ctx),
        array_size,
    });
    Ok(())
}

/// Handle array.get operation
pub(crate) fn handle_array_get(
    ctx: &IrContext,
    array_get_op: wasm_dialect::ArrayGet,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArrayGet(array_get_op.type_idx(ctx)));
    Ok(())
}

/// Handle array.get_s operation (sign-extending load)
pub(crate) fn handle_array_get_s(
    ctx: &IrContext,
    array_get_s_op: wasm_dialect::ArrayGetS,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArrayGetS(array_get_s_op.type_idx(ctx)));
    Ok(())
}

/// Handle array.get_u operation (zero-extending load)
pub(crate) fn handle_array_get_u(
    ctx: &IrContext,
    array_get_u_op: wasm_dialect::ArrayGetU,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArrayGetU(array_get_u_op.type_idx(ctx)));
    Ok(())
}

/// Handle array.set operation
pub(crate) fn handle_array_set(
    ctx: &IrContext,
    array_set_op: wasm_dialect::ArraySet,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArraySet(array_set_op.type_idx(ctx)));
    Ok(())
}

/// Handle array.copy operation
pub(crate) fn handle_array_copy(
    ctx: &IrContext,
    array_copy_op: wasm_dialect::ArrayCopy,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::ArrayCopy {
        array_type_index_dst: array_copy_op.dst_type_idx(ctx),
        array_type_index_src: array_copy_op.src_type_idx(ctx),
    });
    Ok(())
}
