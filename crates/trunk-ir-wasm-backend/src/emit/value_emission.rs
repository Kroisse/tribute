//! Value emission helpers for wasm backend.
//!
//! This module provides functions for emitting values and operands
//! in WebAssembly code generation.

use tracing::debug;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::refs::{ValueDef, ValueRef};
use wasm_encoder::{AbstractHeapType, HeapType, Instruction};

use crate::CompilationResult;

use super::helpers::{is_nil_type, value_type};

/// Emit operands for an operation.
///
/// For each operand value:
/// - Emit local.get if the value has a local mapping
/// - Emit ref.null none for nil type values
/// - Handle block arguments by using their index directly
/// - Report error for stale value references
pub(super) fn emit_operands(
    ctx: &IrContext,
    operands: &[ValueRef],
    emit_ctx: &super::FunctionEmitContext,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    for &value in operands {
        // Try direct lookup first
        if let Some(index) = emit_ctx.value_locals.get(&value) {
            function.instruction(&Instruction::LocalGet(*index));
            continue;
        }

        // Nil type values need ref.null none on the stack (e.g., empty closure environments)
        // Check this AFTER local lookup since nil values may be stored in locals
        let ty = value_type(ctx, value);
        if is_nil_type(ctx, ty) {
            debug!(
                "emit_operands: emitting ref.null none for nil type value {:?}",
                ctx.value_def(value)
            );
            function.instruction(&Instruction::RefNull(HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            }));
            continue;
        }

        // Handle stale block argument references (issue #43)
        if let ValueDef::BlockArg(_, idx) = ctx.value_def(value) {
            function.instruction(&Instruction::LocalGet(idx));
            continue;
        }

        // If operand not found and not a block arg, this is an ERROR - stale value reference!
        if let ValueDef::OpResult(stale_op, _) = ctx.value_def(value) {
            // Debug: print all operands for context
            tracing::error!("emit_operands: stale value detected in operands list:");
            for (i, &v) in operands.iter().enumerate() {
                let def_info = match ctx.value_def(v) {
                    ValueDef::OpResult(op, _) => {
                        let op_data = ctx.op(op);
                        format!("{}.{}", op_data.dialect, op_data.name)
                    }
                    ValueDef::BlockArg(bid, _) => format!("block_arg({:?})", bid),
                };
                tracing::error!("  operand[{}]: {:?} -> {}", i, v, def_info);
            }
            let op_data = ctx.op(stale_op);
            panic!(
                "emit_operands: stale SSA value: {}.{} (value references should have been resolved before emit)",
                op_data.dialect, op_data.name,
            );
        }
    }
    Ok(())
}

/// Emit a single value (local.get or block arg fallback).
pub(super) fn emit_value(
    ctx: &IrContext,
    value: ValueRef,
    emit_ctx: &super::FunctionEmitContext,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    // Try direct lookup first
    if let Some(index) = emit_ctx.value_locals.get(&value) {
        function.instruction(&Instruction::LocalGet(*index));
        return Ok(());
    }

    // Handle stale block argument references
    if let ValueDef::BlockArg(_, idx) = ctx.value_def(value) {
        function.instruction(&Instruction::LocalGet(idx));
        return Ok(());
    }

    // If operand not found and not a block arg, this is an error.
    let ValueDef::OpResult(stale_op, _) = ctx.value_def(value) else {
        unreachable!("ValueDef only has BlockArg and OpResult variants");
    };
    let op_data = ctx.op(stale_op);
    panic!(
        "emit_value: stale SSA value: {}.{}",
        op_data.dialect, op_data.name,
    )
}
