//! Control flow operation handlers for wasm backend.
//!
//! This module handles WebAssembly control flow operations:
//! - wasm.if (conditional branching with regions)
//! - wasm.block (structured block)
//! - wasm.loop (loop construct)
//! - wasm.br (unconditional branch)
//! - wasm.br_if (conditional branch)

use tracing::debug;
use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{DialectType, Operation, Type, Value};
use wasm_encoder::{BlockType, Function, HeapType, Instruction, RefType, ValType};

use crate::{CompilationError, CompilationResult};

use super::super::{
    FunctionEmitContext, ModuleInfo, emit_operands, emit_region_ops, is_nil_type,
    region_result_value, set_result_local, type_to_valtype,
};

// ============================================================================
// Helper functions for control flow
// ============================================================================

/// Emit local.get for a value
fn emit_value_get<'db>(
    value: Value<'db>,
    ctx: &FunctionEmitContext<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = ctx
        .value_locals
        .get(&value)
        .ok_or_else(|| CompilationError::invalid_module("value missing local mapping"))?;
    function.instruction(&Instruction::LocalGet(*index));
    Ok(())
}

// ============================================================================
// Control flow handlers
// ============================================================================

/// Handle wasm.if operation
pub(crate) fn handle_if<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = op.operands(db);
    let result_ty = op.results(db).first().copied();

    // Check if we can actually get a result value from the then region
    let then_region_result = op
        .regions(db)
        .first()
        .and_then(|r| region_result_value(db, r));
    let then_has_result_value = then_region_result.is_some();

    debug!(
        "wasm.if: then_has_result_value={}, result_ty={:?}",
        then_has_result_value,
        result_ty.map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db)))
    );

    // Determine if we should use a result type
    // Use IR result type directly - wasm_type_concrete should have resolved placeholders
    let has_result = then_has_result_value && matches!(result_ty, Some(ty) if !is_nil_type(db, ty));

    let block_type = compute_block_type(db, has_result, result_ty, module_info)?;

    if operands.len() != 1 {
        return Err(CompilationError::invalid_module(
            "wasm.if expects a single condition operand",
        ));
    }
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    function.instruction(&Instruction::If(block_type));

    let regions = op.regions(db);
    let then_region = regions
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.if missing then region"))?;

    // Emit then branch
    let then_result = if has_result {
        Some(region_result_value(db, then_region).ok_or_else(|| {
            CompilationError::invalid_module("wasm.if then region missing result value")
        })?)
    } else {
        None
    };
    emit_region_ops(db, then_region, ctx, module_info, function)?;
    if let Some(value) = then_result {
        emit_value_get(value, ctx, function)?;
    }

    // Emit else branch if present
    if let Some(else_region) = regions.get(1) {
        let else_result = if has_result {
            Some(region_result_value(db, else_region).ok_or_else(|| {
                CompilationError::invalid_module("wasm.if else region missing result value")
            })?)
        } else {
            None
        };
        function.instruction(&Instruction::Else);
        emit_region_ops(db, else_region, ctx, module_info, function)?;
        if let Some(value) = else_result {
            emit_value_get(value, ctx, function)?;
        }
    } else if has_result {
        return Err(CompilationError::invalid_module(
            "wasm.if with result requires else region",
        ));
    }

    function.instruction(&Instruction::End);
    if has_result {
        set_result_local(db, op, ctx, function)?;
    }
    Ok(())
}

/// Compute the WASM block type from result type
fn compute_block_type<'db>(
    db: &'db dyn salsa::Database,
    has_result: bool,
    result_ty: Option<Type<'db>>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<BlockType> {
    if !has_result {
        return Ok(BlockType::Empty);
    }

    let ty = result_ty.expect("result_ty should be Some when has_result is true");

    // IMPORTANT: Check primitive types BEFORE type_idx_by_type lookup.
    // Primitive types should use their native WASM types, not GC struct references.

    // core.func types should use funcref block type
    if core::Func::from_type(db, ty).is_some() {
        debug!(
            "block_type: using funcref for core.func type {}.{}",
            ty.dialect(db),
            ty.name(db)
        );
        return Ok(BlockType::Result(ValType::Ref(RefType::FUNCREF)));
    }

    // Check for core primitive types (i32, i64, f32, f64)
    if core::I32::from_type(db, ty).is_some() {
        debug!("block_type: using i32 for core.i32");
        return Ok(BlockType::Result(ValType::I32));
    }
    if core::I64::from_type(db, ty).is_some() {
        debug!("block_type: using i64 for core.i64");
        return Ok(BlockType::Result(ValType::I64));
    }
    if core::F32::from_type(db, ty).is_some() {
        debug!("block_type: using f32 for core.f32");
        return Ok(BlockType::Result(ValType::F32));
    }
    if core::F64::from_type(db, ty).is_some() {
        debug!("block_type: using f64 for core.f64");
        return Ok(BlockType::Result(ValType::F64));
    }

    // Check for tribute_rt primitive types (int, nat, bool -> i32, float -> f64)
    // TODO: This layer shouldn't need to know about tribute_rt types.
    // Consider adding a type normalization pass that converts tribute_rt.int -> core.i32, etc.
    // before reaching the WASM backend.
    if tribute_rt::Int::from_type(db, ty).is_some()
        || tribute_rt::Nat::from_type(db, ty).is_some()
        || tribute_rt::Bool::from_type(db, ty).is_some()
    {
        debug!(
            "block_type: using i32 for primitive type {}.{}",
            ty.dialect(db),
            ty.name(db)
        );
        return Ok(BlockType::Result(ValType::I32));
    }
    if tribute_rt::Float::from_type(db, ty).is_some() {
        debug!("block_type: using f64 for tribute_rt.float");
        return Ok(BlockType::Result(ValType::F64));
    }

    if let Some(&type_idx) = module_info.type_idx_by_type.get(&ty) {
        // ADT types - use concrete GC type reference
        debug!(
            "block_type: using concrete type_idx={} for {}.{}",
            type_idx,
            ty.dialect(db),
            ty.name(db)
        );
        return Ok(BlockType::Result(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        })));
    }

    debug!(
        "block_type: no type_idx for {}.{}, using type_to_valtype",
        ty.dialect(db),
        ty.name(db)
    );
    Ok(BlockType::Result(type_to_valtype(
        db,
        ty,
        &module_info.type_idx_by_type,
    )?))
}

/// Handle wasm.block operation
pub(crate) fn handle_block<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    // Use IR result type directly - wasm_type_concrete should have resolved placeholders
    let result_ty = op.results(db).first().copied();
    let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));

    let block_type = compute_block_type(db, has_result, result_ty, module_info)?;

    function.instruction(&Instruction::Block(block_type));

    let region = op
        .regions(db)
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.block missing body region"))?;
    emit_region_ops(db, region, ctx, module_info, function)?;

    if has_result {
        let value = region_result_value(db, region).ok_or_else(|| {
            CompilationError::invalid_module("wasm.block body missing result value")
        })?;
        emit_value_get(value, ctx, function)?;
    }

    function.instruction(&Instruction::End);
    if has_result {
        set_result_local(db, op, ctx, function)?;
    }
    Ok(())
}

/// Handle wasm.loop operation
pub(crate) fn handle_loop<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    // Use IR result type directly - wasm_type_concrete should have resolved placeholders
    let result_ty = op.results(db).first().copied();
    let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));

    let block_type = compute_block_type(db, has_result, result_ty, module_info)?;

    function.instruction(&Instruction::Loop(block_type));

    let region = op
        .regions(db)
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.loop missing body region"))?;
    emit_region_ops(db, region, ctx, module_info, function)?;

    if has_result {
        let value = region_result_value(db, region).ok_or_else(|| {
            CompilationError::invalid_module("wasm.loop body missing result value")
        })?;
        emit_value_get(value, ctx, function)?;
    }

    function.instruction(&Instruction::End);
    if has_result {
        set_result_local(db, op, ctx, function)?;
    }
    Ok(())
}

/// Handle wasm.br operation
pub(crate) fn handle_br<'db>(
    db: &'db dyn salsa::Database,
    br_op: wasm::Br<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let depth = br_op.target(db);
    function.instruction(&Instruction::Br(depth));
    Ok(())
}

/// Handle wasm.br_if operation
pub(crate) fn handle_br_if<'db>(
    db: &'db dyn salsa::Database,
    br_if_op: wasm::BrIf<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = br_if_op.operation();
    let operands = op.operands(db);

    if operands.len() != 1 {
        return Err(CompilationError::invalid_module(
            "wasm.br_if expects a single condition operand",
        ));
    }

    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let depth = br_if_op.target(db);
    function.instruction(&Instruction::BrIf(depth));
    Ok(())
}
