//! Value emission helpers for wasm backend.
//!
//! This module provides functions for emitting values and operands
//! in WebAssembly code generation.

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::dialect::tribute;
use trunk_ir::{Attribute, BlockId, IdVec, Symbol, Type, Value, ValueDef};
use wasm_encoder::{AbstractHeapType, HeapType, Instruction};

use crate::{CompilationError, CompilationResult};

use super::helpers::{is_nil_type, value_type};

/// Emit operands for an operation.
///
/// For each operand value:
/// - Emit local.get if the value has a local mapping
/// - Emit ref.null none for nil type values
/// - Handle block arguments by using their index directly
/// - Report error for stale value references
pub(super) fn emit_operands<'db>(
    db: &'db dyn salsa::Database,
    operands: &IdVec<Value<'db>>,
    ctx: &super::FunctionEmitContext<'db>,
    block_arg_types: &HashMap<(BlockId, usize), Type<'db>>,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    for value in operands.iter() {
        // Try direct lookup first
        if let Some(index) = ctx.value_locals.get(value) {
            function.instruction(&Instruction::LocalGet(*index));
            continue;
        }

        // Nil type values need ref.null none on the stack (e.g., empty closure environments)
        // Check this AFTER local lookup since nil values may be stored in locals
        if let Some(ty) = value_type(db, *value, block_arg_types)
            && is_nil_type(db, ty)
        {
            debug!(
                "emit_operands: emitting ref.null none for nil type value {:?}",
                value.def(db)
            );
            function.instruction(&Instruction::RefNull(HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            }));
            continue;
        }

        // Handle stale block argument references (issue #43)
        // The resolver creates operands that reference OLD block arguments, but value_locals
        // only contains NEW block arguments. For block args, we can use the index directly
        // since parameters are always locals 0, 1, 2, etc.
        if let ValueDef::BlockArg(_block_id) = value.def(db) {
            let index = value.index(db) as u32;
            function.instruction(&Instruction::LocalGet(index));
            continue;
        }

        // If operand not found and not a block arg, this is an ERROR - stale value reference!
        if let ValueDef::OpResult(stale_op) = value.def(db) {
            // For tribute.var, try to find what it references by looking at its name attribute
            if stale_op.dialect(db) == tribute::DIALECT_NAME()
                && stale_op.name(db) == tribute::VAR()
            {
                if let Some(Attribute::Symbol(var_name)) =
                    stale_op.attributes(db).get(&Symbol::new("name"))
                {
                    tracing::error!(
                        "emit_operands: stale SSA value: tribute.var '{}' index={} (var references should have been resolved)",
                        var_name,
                        value.index(db)
                    );
                } else {
                    tracing::error!(
                        "emit_operands: stale SSA value: tribute.var (no name) index={}",
                        value.index(db)
                    );
                }
            } else {
                tracing::error!(
                    "emit_operands: stale SSA value: {}.{} index={}",
                    stale_op.dialect(db),
                    stale_op.name(db),
                    value.index(db)
                );
            }
            return Err(CompilationError::invalid_module(
                "stale SSA value in wasm backend (missing local mapping)",
            ));
        }
    }
    Ok(())
}

/// Emit a single value (local.get or block arg fallback).
pub(super) fn emit_value<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    ctx: &super::FunctionEmitContext<'db>,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    // Try direct lookup first
    if let Some(index) = ctx.value_locals.get(&value) {
        function.instruction(&Instruction::LocalGet(*index));
        return Ok(());
    }

    // Handle stale block argument references
    if let ValueDef::BlockArg(_block_id) = value.def(db) {
        let index = value.index(db) as u32;
        function.instruction(&Instruction::LocalGet(index));
        return Ok(());
    }

    // If operand not found and not a block arg, this is an error
    if let ValueDef::OpResult(stale_op) = value.def(db) {
        tracing::error!(
            "stale SSA value: {}.{} index={}",
            stale_op.dialect(db),
            stale_op.name(db),
            value.index(db)
        );
    }
    Err(CompilationError::invalid_module(
        "stale SSA value in wasm backend (missing local mapping)",
    ))
}
