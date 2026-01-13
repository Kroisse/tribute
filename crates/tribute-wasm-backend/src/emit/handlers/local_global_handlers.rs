//! Local and global variable handlers for wasm backend.
//!
//! This module handles WebAssembly local and global variable operations:
//! - wasm.local_get, wasm.local_set, wasm.local_tee
//! - wasm.global_get, wasm.global_set

use trunk_ir::dialect::wasm;
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;

use super::super::{FunctionEmitContext, ModuleInfo, emit_operands, set_result_local};

/// Handle wasm.local_get operation
pub(crate) fn handle_local_get<'db>(
    db: &'db dyn salsa::Database,
    local_op: wasm::LocalGet<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = local_op.index(db);
    function.instruction(&Instruction::LocalGet(index));
    set_result_local(db, &local_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle wasm.local_set operation
pub(crate) fn handle_local_set<'db>(
    db: &'db dyn salsa::Database,
    local_op: wasm::LocalSet<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = local_op.operation();
    let operands = op.operands(db);
    let index = local_op.index(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    function.instruction(&Instruction::LocalSet(index));
    Ok(())
}

/// Handle wasm.local_tee operation
pub(crate) fn handle_local_tee<'db>(
    db: &'db dyn salsa::Database,
    local_op: wasm::LocalTee<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = local_op.operation();
    let operands = op.operands(db);
    let index = local_op.index(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    function.instruction(&Instruction::LocalTee(index));
    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle wasm.global_get operation
pub(crate) fn handle_global_get<'db>(
    db: &'db dyn salsa::Database,
    global_op: wasm::GlobalGet<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = global_op.index(db);
    function.instruction(&Instruction::GlobalGet(index));
    set_result_local(db, &global_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle wasm.global_set operation
pub(crate) fn handle_global_set<'db>(
    db: &'db dyn salsa::Database,
    global_op: wasm::GlobalSet<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = global_op.operation();
    let operands = op.operands(db);
    let index = global_op.index(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    function.instruction(&Instruction::GlobalSet(index));
    Ok(())
}
