//! Control flow operation handlers for wasm backend.
//!
//! This module handles WebAssembly control flow operations:
//! - wasm.if (conditional branching with regions)
//! - wasm.block (structured block)
//! - wasm.loop (loop construct)
//! - wasm.br (unconditional branch)
//! - wasm.br_if (conditional branch)
//! - wasm.br_table (indexed branch)
//!
//! A region's result and a branch's value reach their label on the stack,
//! and loop-carried values are written to their locals before the loop and
//! its back edges, so the handlers write only the structure.

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::refs::OpRef;
use wasm_encoder::{BlockType, Function, Instruction};

use crate::stack_effect::structured_result_type;
use crate::{CompilationError, CompilationResult};

use super::super::{ModuleInfo, emit_region_ops, helpers};

/// The Wasm block type of a structured operation.
fn block_type(
    ctx: &IrContext,
    op: OpRef,
    module_info: &ModuleInfo,
) -> CompilationResult<BlockType> {
    match structured_result_type(ctx, op) {
        Some(ty) => Ok(BlockType::Result(helpers::type_to_valtype(
            ctx,
            ty,
            &module_info.type_idx_by_type,
        )?)),
        None => Ok(BlockType::Empty),
    }
}

/// Handle wasm.if operation
pub(crate) fn handle_if(
    ctx: &IrContext,
    op: OpRef,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let block_type = block_type(ctx, op, module_info)?;
    let then_region = ctx
        .op_region(op, 0)
        .ok_or_else(|| CompilationError::invalid_module("wasm.if missing then region"))?;
    let else_region = ctx.op_region(op, 1);
    if matches!(block_type, BlockType::Result(_)) && else_region.is_none() {
        return Err(CompilationError::invalid_module(
            "wasm.if with result requires else region",
        ));
    }

    function.instruction(&Instruction::If(block_type));
    emit_region_ops(ctx, then_region, module_info, function)?;
    if let Some(else_region) = else_region {
        function.instruction(&Instruction::Else);
        emit_region_ops(ctx, else_region, module_info, function)?;
    }
    function.instruction(&Instruction::End);
    Ok(())
}

/// Handle wasm.block operation
pub(crate) fn handle_block(
    ctx: &IrContext,
    op: OpRef,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let block_type = block_type(ctx, op, module_info)?;
    let region = ctx
        .op_region(op, 0)
        .ok_or_else(|| CompilationError::invalid_module("wasm.block missing body region"))?;

    function.instruction(&Instruction::Block(block_type));
    emit_region_ops(ctx, region, module_info, function)?;
    function.instruction(&Instruction::End);
    Ok(())
}

/// Handle wasm.loop operation
pub(crate) fn handle_loop(
    ctx: &IrContext,
    op: OpRef,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let block_type = block_type(ctx, op, module_info)?;
    let region = ctx
        .op_region(op, 0)
        .ok_or_else(|| CompilationError::invalid_module("wasm.loop missing body region"))?;

    function.instruction(&Instruction::Loop(block_type));
    emit_region_ops(ctx, region, module_info, function)?;
    function.instruction(&Instruction::End);
    Ok(())
}

/// Handle wasm.br operation
pub(crate) fn handle_br(
    ctx: &IrContext,
    br_op: wasm_dialect::Br,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::Br(br_op.target(ctx)));
    Ok(())
}

/// Handle wasm.br_if operation
pub(crate) fn handle_br_if(
    ctx: &IrContext,
    br_if_op: wasm_dialect::BrIf,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    function.instruction(&Instruction::BrIf(br_if_op.target(ctx)));
    Ok(())
}

/// Handle wasm.br_table operation
pub(crate) fn handle_br_table(
    ctx: &IrContext,
    br_table_op: wasm_dialect::BrTable,
    function: &mut Function,
) -> CompilationResult<()> {
    let targets: Vec<u32> = br_table_op.targets(ctx).collect();
    function.instruction(&Instruction::BrTable(
        targets.into(),
        br_table_op.default(ctx),
    ));
    Ok(())
}
