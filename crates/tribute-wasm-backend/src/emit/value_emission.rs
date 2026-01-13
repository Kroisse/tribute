//! Value emission helpers for wasm backend.
//!
//! This module provides functions for emitting values, operands, and type conversions
//! (boxing/unboxing) in WebAssembly code generation.

use std::collections::HashMap;

use tracing::debug;
use tribute_ir::dialect::{tribute, tribute_rt};
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{
    Attribute, BlockId, DialectOp, DialectType, IdVec, Operation, Symbol, Type, Value, ValueDef,
};
use wasm_encoder::{AbstractHeapType, HeapType, Instruction};

use crate::gc_types::BOXED_F64_IDX;
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

/// Emit operands with boxing when calling generic functions.
///
/// If a parameter expects anyref (type.var) but the operand is a concrete type (Int, Float),
/// we need to box the value.
pub(super) fn emit_operands_with_boxing<'db>(
    db: &'db dyn salsa::Database,
    operands: &IdVec<Value<'db>>,
    param_types: &IdVec<Type<'db>>,
    ctx: &super::FunctionEmitContext<'db>,
    module_info: &super::ModuleInfo<'db>,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    let mut param_iter = param_types.iter();

    for value in operands.iter() {
        // Get the corresponding parameter type (must stay synchronized with operands)
        let Some(param_ty) = param_iter.next().copied() else {
            return Err(CompilationError::invalid_module(
                "wasm.call operand count exceeds callee param count",
            ));
        };

        // Nil type values need ref.null none on the stack (e.g., empty closure environments)
        if let Some(ty) = value_type(db, *value, &module_info.block_arg_types)
            && is_nil_type(db, ty)
        {
            debug!(
                "emit_operands_with_boxing: emitting ref.null none for nil type value {:?}",
                value.def(db)
            );
            function.instruction(&Instruction::RefNull(HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            }));
            continue;
        }

        // Emit the value (local.get)
        emit_value(db, *value, ctx, function)?;

        // Check if boxing is needed
        // If parameter expects anyref (type.var) AND doesn't have a concrete type index, box the operand
        // Types with a type index (like struct types) are already reference types and don't need boxing
        // Use effective_types to get the actual computed type, falling back to IR type
        if tribute::is_type_var(db, param_ty)
            && !module_info.type_idx_by_type.contains_key(&param_ty)
        {
            // Use effective type if available (computed during local allocation),
            // otherwise fall back to IR result type
            let operand_ty = ctx
                .effective_types
                .get(value)
                .copied()
                .or_else(|| value_type(db, *value, &module_info.block_arg_types));
            if let Some(operand_ty) = operand_ty {
                debug!(
                    "emit_operands_with_boxing: param expects anyref, operand effective_ty={}.{}",
                    operand_ty.dialect(db),
                    operand_ty.name(db)
                );
                emit_boxing(db, operand_ty, function)?;
            }
        }
    }

    if param_iter.len() != 0 {
        return Err(CompilationError::invalid_module(
            "wasm.call operand count is less than callee param count",
        ));
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

/// Emit boxing instructions to convert a concrete type to anyref.
///
/// - Int (i32) → i31ref: use ref.i31 directly
/// - Float (f64) → BoxedF64 struct: wrap in a struct with single f64 field
pub(super) fn emit_boxing<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    debug!("emit_boxing: type={}.{}", ty.dialect(db), ty.name(db));
    if tribute_rt::is_int(db, ty) || tribute_rt::is_nat(db, ty) {
        debug!("  -> boxing Int/Nat to i31ref");
        // Int/Nat (i32) → i31ref (direct, 31-bit values)
        function.instruction(&Instruction::RefI31);
        Ok(())
    } else if tribute_rt::is_float(db, ty) || core::F64::from_type(db, ty).is_some() {
        // Float (f64) → BoxedF64 struct
        // Create a struct with the f64 value
        function.instruction(&Instruction::StructNew(BOXED_F64_IDX));
        Ok(())
    } else {
        // For reference types (structs, etc.), no boxing needed - they're already subtypes of anyref
        // Just leave the value as-is on the stack
        Ok(())
    }
}

/// Emit unboxing instructions to convert anyref to a concrete type.
///
/// - i31ref → Int (i32): extract i32 directly
/// - BoxedF64 → Float (f64): cast and extract f64 field
pub(super) fn emit_unboxing<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    function: &mut wasm_encoder::Function,
) -> CompilationResult<()> {
    if tribute_rt::is_int(db, ty) {
        // anyref → i31ref → Int (i32)
        // Cast anyref to i31ref, extract i32 (signed)
        function.instruction(&Instruction::RefCastNullable(HeapType::I31));
        function.instruction(&Instruction::I31GetS);
        Ok(())
    } else if tribute_rt::is_nat(db, ty) {
        // anyref → i31ref → Nat (i32)
        // Cast anyref to i31ref, extract u32 (unsigned)
        function.instruction(&Instruction::RefCastNullable(HeapType::I31));
        function.instruction(&Instruction::I31GetU);
        Ok(())
    } else if tribute_rt::is_float(db, ty) || core::F64::from_type(db, ty).is_some() {
        // anyref → BoxedF64 → Float (f64)
        // Cast to BoxedF64 struct, then extract f64 field
        function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(
            BOXED_F64_IDX,
        )));
        function.instruction(&Instruction::StructGet {
            struct_type_index: BOXED_F64_IDX,
            field_index: 0,
        });
        Ok(())
    } else {
        // For reference types, assume no unboxing needed
        Ok(())
    }
}

/// Infer the actual result type for a call operation.
///
/// For generic function calls where the IR result type is `type.var`,
/// we infer the concrete type from the operand types.
pub(super) fn infer_call_result_type<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    result_ty: Type<'db>,
    func_types: &HashMap<Symbol, core::Func<'db>>,
    block_arg_types: &HashMap<(BlockId, usize), Type<'db>>,
    func_return_type: Option<Type<'db>>,
) -> Type<'db> {
    // Handle wasm.call_indirect - if result is polymorphic but function returns funcref, use funcref
    if wasm::CallIndirect::matches(db, *op) {
        let is_polymorphic_result =
            tribute::is_type_var(db, result_ty) || wasm::Anyref::from_type(db, result_ty).is_some();
        if let Some(func_ret_ty) = func_return_type {
            let func_returns_funcref = wasm::Funcref::from_type(db, func_ret_ty).is_some()
                || core::Func::from_type(db, func_ret_ty).is_some();
            if is_polymorphic_result && func_returns_funcref {
                return wasm::Funcref::new(db).as_type();
            }
        }
        return result_ty;
    }

    // Only handle wasm.call operations
    let call_op = match wasm::Call::from_operation(db, *op) {
        Ok(c) => c,
        Err(_) => return result_ty,
    };

    // Get the callee
    let callee = call_op.callee(db);

    // Look up the callee's function type
    let callee_ty = match func_types.get(&callee) {
        Some(ty) => ty,
        None => return result_ty,
    };

    // Check if the callee returns type.var (generic)
    let return_ty = callee_ty.result(db);
    if !tribute::is_type_var(db, return_ty) {
        // Callee returns a concrete type, use it
        return return_ty;
    }

    // Infer concrete type from first operand (works for identity-like functions)
    if let Some(operand_ty) = op
        .operands(db)
        .first()
        .and_then(|v| value_type(db, *v, block_arg_types))
        .filter(|ty| !tribute::is_type_var(db, *ty))
    {
        return operand_ty;
    }

    result_ty
}
