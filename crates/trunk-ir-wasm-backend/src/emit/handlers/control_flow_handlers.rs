//! Control flow operation handlers for wasm backend.
//!
//! This module handles WebAssembly control flow operations:
//! - wasm.if (conditional branching with regions)
//! - wasm.block (structured block)
//! - wasm.loop (loop construct)
//! - wasm.br (unconditional branch)
//! - wasm.br_if (conditional branch)

use tracing::debug;
use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use wasm_encoder::{BlockType, Function, HeapType, Instruction, RefType, ValType};

use crate::{CompilationError, CompilationResult};

use super::super::helpers;
use super::super::value_emission::emit_operands;
use super::super::{
    FunctionEmitContext, ModuleInfo, NestingKind, emit_region_ops_nested, region_result_value,
    set_result_local,
};

// ============================================================================
// Helper functions for control flow
// ============================================================================

/// Emit local.get for a value
fn emit_value_get(
    ctx: &IrContext,
    value: ValueRef,
    emit_ctx: &FunctionEmitContext,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = emit_ctx.value_locals.get(&value).ok_or_else(|| {
        let def_info = match ctx.value_def(value) {
            trunk_ir::refs::ValueDef::OpResult(op, _) => {
                let op_data = ctx.op(op);
                format!(
                    "OpResult from {}.{} at {:?}",
                    op_data.dialect, op_data.name, op_data.location
                )
            }
            trunk_ir::refs::ValueDef::BlockArg(block_id, idx) => {
                format!("BlockArg({:?}) index={}", block_id, idx)
            }
        };
        tracing::error!(
            "value missing local mapping: value={:?}, def={}",
            value,
            def_info
        );
        CompilationError::invalid_module("value missing local mapping")
    })?;
    function.instruction(&Instruction::LocalGet(*index));
    Ok(())
}

// ============================================================================
// Control flow handlers
// ============================================================================

/// Handle wasm.if operation
pub(crate) fn handle_if(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
    nesting: &[NestingKind],
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    let result_ty = ctx.op_result_types(op).first().copied();

    // Check if we can actually get a result value from the then region
    let regions = &ctx.op(op).regions;
    let then_region_result = regions.first().and_then(|r| region_result_value(ctx, *r));
    let then_has_result_value = then_region_result.is_some();

    debug!(
        "wasm.if: then_has_result_value={}, result_ty={:?}",
        then_has_result_value,
        result_ty.map(|ty| {
            let data = ctx.types.get(ty);
            format!("{}.{}", data.dialect, data.name)
        })
    );

    // Determine if we should use a result type
    // Use IR result type directly - type variables are resolved at AST level
    let has_result =
        then_has_result_value && matches!(result_ty, Some(ty) if !helpers::is_nil_type(ctx, ty));

    let block_type = compute_block_type(ctx, has_result, result_ty, module_info)?;

    if operands.len() != 1 {
        return Err(CompilationError::invalid_module(
            "wasm.if expects a single condition operand",
        ));
    }
    emit_operands(ctx, operands, emit_ctx, function)?;
    function.instruction(&Instruction::If(block_type));

    let then_region = regions
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.if missing then region"))?;

    // Push If nesting for child regions
    let mut child_nesting = nesting.to_vec();
    child_nesting.push(NestingKind::If);

    // Emit then branch
    emit_region_ops_nested(
        ctx,
        *then_region,
        emit_ctx,
        module_info,
        function,
        &child_nesting,
    )?;
    if has_result && let Some(value) = region_result_value(ctx, *then_region) {
        emit_value_get(ctx, value, emit_ctx, function)?;
    }

    // Emit else branch if present
    if let Some(else_region) = regions.get(1) {
        function.instruction(&Instruction::Else);
        emit_region_ops_nested(
            ctx,
            *else_region,
            emit_ctx,
            module_info,
            function,
            &child_nesting,
        )?;
        if has_result && let Some(value) = region_result_value(ctx, *else_region) {
            emit_value_get(ctx, value, emit_ctx, function)?;
        }
    } else if has_result {
        return Err(CompilationError::invalid_module(
            "wasm.if with result requires else region",
        ));
    }

    function.instruction(&Instruction::End);
    if has_result {
        set_result_local(ctx, op, emit_ctx, function)?;
    }
    Ok(())
}

/// Compute the WASM block type from result type
fn compute_block_type(
    ctx: &IrContext,
    has_result: bool,
    result_ty: Option<TypeRef>,
    module_info: &ModuleInfo,
) -> CompilationResult<BlockType> {
    if !has_result {
        return Ok(BlockType::Empty);
    }

    let ty = result_ty.expect("result_ty should be Some when has_result is true");

    // IMPORTANT: Check primitive types BEFORE type_idx_by_type lookup.
    // Primitive types should use their native WASM types, not GC struct references.

    // core.func types should use funcref block type
    if helpers::is_type(ctx, ty, "core", "func") {
        debug!(
            "block_type: using funcref for core.func type {}.{}",
            ctx.types.get(ty).dialect,
            ctx.types.get(ty).name
        );
        return Ok(BlockType::Result(ValType::Ref(RefType::FUNCREF)));
    }

    // Check for core primitive types (i32, i64, f32, f64)
    if helpers::is_type(ctx, ty, "core", "i32") {
        debug!("block_type: using i32 for core.i32");
        return Ok(BlockType::Result(ValType::I32));
    }
    if helpers::is_type(ctx, ty, "core", "i64") {
        debug!("block_type: using i64 for core.i64");
        return Ok(BlockType::Result(ValType::I64));
    }
    if helpers::is_type(ctx, ty, "core", "f32") {
        debug!("block_type: using f32 for core.f32");
        return Ok(BlockType::Result(ValType::F32));
    }
    if helpers::is_type(ctx, ty, "core", "f64") {
        debug!("block_type: using f64 for core.f64");
        return Ok(BlockType::Result(ValType::F64));
    }

    if let Some(&type_idx) = module_info.type_idx_by_type.get(&ty) {
        // ADT types - use concrete GC type reference
        debug!(
            "block_type: using concrete type_idx={} for {}.{}",
            type_idx,
            ctx.types.get(ty).dialect,
            ctx.types.get(ty).name
        );
        return Ok(BlockType::Result(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        })));
    }

    debug!(
        "block_type: no type_idx for {}.{}, using type_to_valtype",
        ctx.types.get(ty).dialect,
        ctx.types.get(ty).name
    );
    Ok(BlockType::Result(helpers::type_to_valtype(
        ctx,
        ty,
        &module_info.type_idx_by_type,
    )?))
}

/// Handle wasm.block operation
pub(crate) fn handle_block(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
    nesting: &[NestingKind],
) -> CompilationResult<()> {
    // Use IR result type directly - type variables are resolved at AST level
    let result_ty = ctx.op_result_types(op).first().copied();
    let has_result = matches!(result_ty, Some(ty) if !helpers::is_nil_type(ctx, ty));

    let block_type = compute_block_type(ctx, has_result, result_ty, module_info)?;

    function.instruction(&Instruction::Block(block_type));

    let regions = &ctx.op(op).regions;
    let region = regions
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.block missing body region"))?;

    let mut child_nesting = nesting.to_vec();
    child_nesting.push(NestingKind::Block);
    emit_region_ops_nested(
        ctx,
        *region,
        emit_ctx,
        module_info,
        function,
        &child_nesting,
    )?;

    if has_result && let Some(value) = region_result_value(ctx, *region) {
        emit_value_get(ctx, value, emit_ctx, function)?;
    }

    function.instruction(&Instruction::End);
    if has_result {
        set_result_local(ctx, op, emit_ctx, function)?;
    }
    Ok(())
}

/// Handle wasm.loop operation
pub(crate) fn handle_loop(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
    nesting: &[NestingKind],
) -> CompilationResult<()> {
    // Use IR result type directly - type variables are resolved at AST level
    let result_ty = ctx.op_result_types(op).first().copied();
    let has_result = matches!(result_ty, Some(ty) if !helpers::is_nil_type(ctx, ty));

    let block_type = compute_block_type(ctx, has_result, result_ty, module_info)?;

    let regions = &ctx.op(op).regions;
    let region = regions
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.loop missing body region"))?;

    // Collect loop arg locals from the body's block arguments
    let body_block = ctx
        .region(*region)
        .blocks
        .first()
        .ok_or_else(|| CompilationError::invalid_module("wasm.loop body has no block"))?;
    let block_args = ctx.block_args(*body_block);
    let arg_locals: Vec<u32> = (0..block_args.len())
        .map(|i| {
            let block_arg = ctx.block_arg(*body_block, i as u32);
            emit_ctx
                .value_locals
                .get(&block_arg)
                .copied()
                .ok_or_else(|| {
                    CompilationError::invalid_module("wasm.loop block arg missing local")
                })
        })
        .collect::<Result<Vec<u32>, _>>()?;

    // Initialize loop-carried variables from init operands
    let init_operands = ctx.op_operands(op);
    assert_eq!(
        init_operands.len(),
        arg_locals.len(),
        "wasm.loop init operands and block arg locals must have the same length"
    );
    for (init_val, &arg_local) in init_operands.iter().zip(&arg_locals) {
        emit_value_get(ctx, *init_val, emit_ctx, function)?;
        function.instruction(&Instruction::LocalSet(arg_local));
    }

    function.instruction(&Instruction::Loop(block_type));

    let mut child_nesting = nesting.to_vec();
    child_nesting.push(NestingKind::Loop { arg_locals });
    emit_region_ops_nested(
        ctx,
        *region,
        emit_ctx,
        module_info,
        function,
        &child_nesting,
    )?;

    if has_result && let Some(value) = region_result_value(ctx, *region) {
        emit_value_get(ctx, value, emit_ctx, function)?;
    }

    function.instruction(&Instruction::End);
    if has_result {
        set_result_local(ctx, op, emit_ctx, function)?;
    }
    Ok(())
}

/// Handle wasm.br operation
pub(crate) fn handle_br(
    ctx: &IrContext,
    br_op: wasm_dialect::Br,
    function: &mut Function,
) -> CompilationResult<()> {
    let depth = br_op.target(ctx);
    function.instruction(&Instruction::Br(depth));
    Ok(())
}

/// Handle wasm.br_if operation
pub(crate) fn handle_br_if(
    ctx: &IrContext,
    br_if_op: wasm_dialect::BrIf,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(br_if_op.op_ref());

    if operands.len() != 1 {
        return Err(CompilationError::invalid_module(
            "wasm.br_if expects a single condition operand",
        ));
    }

    emit_operands(ctx, operands, emit_ctx, function)?;
    let depth = br_if_op.target(ctx);
    function.instruction(&Instruction::BrIf(depth));
    Ok(())
}
