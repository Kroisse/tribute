//! Constant value handlers for wasm backend.
//!
//! This module handles the emission of constant values (i32, i64, f32, f64).

use trunk_ir::dialect::wasm;
use trunk_ir::{DialectOp, Operation};
use wasm_encoder::{Function, Instruction};

use crate::CompilationResult;

use super::super::set_result_local;

/// Handle i32.const operation
pub(crate) fn handle_i32_const<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let const_op = wasm::I32Const::from_operation(db, *op).map_err(|_| {
        crate::errors::CompilationErrorKind::InvalidModule("expected i32.const".to_string())
    })?;
    let value = const_op.value(db);
    function.instruction(&Instruction::I32Const(value));
    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle i64.const operation
pub(crate) fn handle_i64_const<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let const_op = wasm::I64Const::from_operation(db, *op).map_err(|_| {
        crate::errors::CompilationErrorKind::InvalidModule("expected i64.const".to_string())
    })?;
    let value = const_op.value(db);
    function.instruction(&Instruction::I64Const(value));
    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle f32.const operation
pub(crate) fn handle_f32_const<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let const_op = wasm::F32Const::from_operation(db, *op).map_err(|_| {
        crate::errors::CompilationErrorKind::InvalidModule("expected f32.const".to_string())
    })?;
    let value = const_op.value(db);
    function.instruction(&Instruction::F32Const(value.into()));
    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle f64.const operation
pub(crate) fn handle_f64_const<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let const_op = wasm::F64Const::from_operation(db, *op).map_err(|_| {
        crate::errors::CompilationErrorKind::InvalidModule("expected f64.const".to_string())
    })?;
    let value = const_op.value(db);
    function.instruction(&Instruction::F64Const(value.into()));
    set_result_local(db, op, ctx, function)?;
    Ok(())
}
