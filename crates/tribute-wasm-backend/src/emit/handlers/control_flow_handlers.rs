//! Control flow operation handlers for wasm backend.
//!
//! This module handles WebAssembly control flow operations:
//! - wasm.if (conditional branching with regions)
//! - wasm.block (structured block)
//! - wasm.loop (loop construct)
//! - wasm.br (unconditional branch)
//! - wasm.br_if (conditional branch)

use tracing::debug;
use tribute_ir::dialect::tribute;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{DialectType, Operation, Type, Value, ValueDef};
use wasm_encoder::{
    AbstractHeapType, BlockType, Function, HeapType, Instruction, RefType, ValType,
};

use crate::{CompilationError, CompilationResult};

use super::super::{
    FunctionEmitContext, ModuleInfo, emit_operands, emit_region_ops, is_nil_type, is_step_type,
    region_result_value, set_result_local, type_to_valtype,
};

// ============================================================================
// Helper functions for control flow
// ============================================================================

/// Check if a type is polymorphic (type_var or anyref).
/// These types need special handling for control flow result types.
fn is_polymorphic_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    tribute::is_type_var(db, ty) || wasm::Anyref::from_type(db, ty).is_some()
}

/// Try to infer a concrete effective type from a control flow operation's first region.
/// Returns the inferred type if it's more concrete than type_var/anyref, None otherwise.
fn infer_region_effective_type<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
) -> Option<Type<'db>> {
    let region = op.regions(db).first()?;
    let result_value = region_result_value(db, region)?;

    // First try effective_types (populated during function setup)
    #[allow(clippy::collapsible_if)]
    if let Some(&ty) = ctx.effective_types.get(&result_value) {
        if !is_polymorphic_type(db, ty) {
            return Some(ty);
        }
    }

    // If not in effective_types, try to get the type from the value's definition
    // This handles remapped operations in resume functions
    #[allow(clippy::collapsible_if)]
    if let ValueDef::OpResult(def_op) = result_value.def(db) {
        if let Some(result_ty) = def_op.results(db).get(result_value.index(db)).copied() {
            if !is_polymorphic_type(db, result_ty) {
                return Some(result_ty);
            }
        }
    }

    None
}

/// Upgrade a polymorphic type to Step if the function returns Step.
/// Used for wasm.block/wasm.loop result types.
fn upgrade_polymorphic_to_step<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    func_return_type: Option<Type<'db>>,
) -> Type<'db> {
    if !is_polymorphic_type(db, ty) {
        return ty;
    }

    if let Some(ret_ty) = func_return_type
        && is_step_type(db, ret_ty)
    {
        return crate::gc_types::step_marker_type(db);
    }
    ty
}

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

    // First try to infer effective type from branches
    let branch_eff_ty = infer_region_effective_type(db, op, ctx);

    // Check if we can actually get a result value from the then region
    let then_region_result = op
        .regions(db)
        .first()
        .and_then(|r| region_result_value(db, r));
    let then_has_result_value = then_region_result.is_some();

    // Check if function returns Step - if so, if blocks should also produce Step
    let func_returns_step = ctx
        .func_return_type
        .map(|ty| is_step_type(db, ty))
        .unwrap_or(false);

    debug!(
        "wasm.if: then_has_result_value={}, branch_eff_ty={:?}, func_returns_step={}, result_ty={:?}",
        then_has_result_value,
        branch_eff_ty.map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db))),
        func_returns_step,
        result_ty.map(|ty| format!("{}.{}", ty.dialect(db), ty.name(db)))
    );

    // Determine if we should use a result type
    // Only set has_result if we can actually find a result value
    let has_result = if !then_has_result_value {
        false
    } else if let Some(eff_ty) = branch_eff_ty {
        !is_nil_type(db, eff_ty)
    } else if func_returns_step && result_ty.is_some() {
        // Function returns Step, so if with result should produce Step
        true
    } else {
        matches!(result_ty, Some(ty) if !is_nil_type(db, ty))
    };

    // For wasm.if with results, we need to determine the actual block type.
    // If the IR result type is polymorphic or nil but the effective result type
    // from the then/else branches is concrete, we must use the effective type.
    let effective_ty = determine_if_effective_type(
        db,
        has_result,
        branch_eff_ty,
        func_returns_step,
        result_ty,
        ctx,
    );

    let block_type = compute_block_type(db, has_result, effective_ty, module_info)?;

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
        emit_branch_result_cast(db, effective_ty, &value, ctx, module_info, function, "then")?;
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
            emit_branch_result_cast(db, effective_ty, &value, ctx, module_info, function, "else")?;
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

/// Determine the effective type for wasm.if block
fn determine_if_effective_type<'db>(
    db: &'db dyn salsa::Database,
    has_result: bool,
    branch_eff_ty: Option<Type<'db>>,
    func_returns_step: bool,
    result_ty: Option<Type<'db>>,
    ctx: &FunctionEmitContext<'db>,
) -> Option<Type<'db>> {
    if !has_result {
        return None;
    }

    // Try branch effective type first (handles Step in resume functions)
    if let Some(eff_ty) = branch_eff_ty {
        if !is_nil_type(db, eff_ty) {
            debug!(
                "wasm.if: using then branch effective type {}.{}",
                eff_ty.dialect(db),
                eff_ty.name(db)
            );
            return Some(eff_ty);
        }
        return result_ty;
    }

    if func_returns_step {
        // Function returns Step, use Step as the block type
        debug!("wasm.if: using Step type because function returns Step");
        return Some(crate::gc_types::step_marker_type(db));
    }

    if let Some(ir_ty) = result_ty {
        if tribute::is_type_var(db, ir_ty)
            && let Some(ret_ty) = ctx.func_return_type
            && !is_polymorphic_type(db, ret_ty)
        {
            // Fallback to function return type for polymorphic IR types
            debug!(
                "wasm.if: using function return type {}.{} instead of type_var",
                ret_ty.dialect(db),
                ret_ty.name(db)
            );
            return Some(ret_ty);
        }
        return Some(ir_ty);
    }

    None
}

/// Compute the WASM block type from effective type
fn compute_block_type<'db>(
    db: &'db dyn salsa::Database,
    has_result: bool,
    effective_ty: Option<Type<'db>>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<BlockType> {
    if !has_result {
        return Ok(BlockType::Empty);
    }

    let eff_ty = effective_ty.expect("effective_ty should be Some when has_result is true");

    // IMPORTANT: Check core.func BEFORE type_idx_by_type lookup.
    // core.func types should always use funcref block type, not concrete struct types.
    if core::Func::from_type(db, eff_ty).is_some() {
        debug!(
            "wasm.if block_type: using funcref for core.func type {}.{}",
            eff_ty.dialect(db),
            eff_ty.name(db)
        );
        return Ok(BlockType::Result(ValType::Ref(RefType::FUNCREF)));
    }

    if let Some(&type_idx) = module_info.type_idx_by_type.get(&eff_ty) {
        // ADT types - use concrete GC type reference
        debug!(
            "wasm.if block_type: using concrete type_idx={} for {}.{}",
            type_idx,
            eff_ty.dialect(db),
            eff_ty.name(db)
        );
        return Ok(BlockType::Result(ValType::Ref(RefType {
            nullable: true,
            heap_type: HeapType::Concrete(type_idx),
        })));
    }

    debug!(
        "wasm.if block_type: no type_idx for {}.{}, using type_to_valtype",
        eff_ty.dialect(db),
        eff_ty.name(db)
    );
    Ok(BlockType::Result(type_to_valtype(
        db,
        eff_ty,
        &module_info.type_idx_by_type,
    )?))
}

/// Emit cast for branch result if needed
fn emit_branch_result_cast<'db>(
    db: &'db dyn salsa::Database,
    effective_ty: Option<Type<'db>>,
    value: &Value<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
    branch_name: &str,
) -> CompilationResult<()> {
    let (Some(eff_ty), Some(value_ty)) = (effective_ty, ctx.effective_types.get(value)) else {
        return Ok(());
    };

    if !tribute::is_type_var(db, *value_ty) && wasm::Anyref::from_type(db, *value_ty).is_none() {
        return Ok(());
    }

    if core::Func::from_type(db, eff_ty).is_some() {
        // core.func types need cast to funcref (abstract type)
        debug!(
            "wasm.if {}: casting anyref branch result to funcref",
            branch_name
        );
        function.instruction(&Instruction::RefCastNullable(HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Func,
        }));
    } else if let Some(&type_idx) = module_info.type_idx_by_type.get(&eff_ty) {
        // ADT types need cast to concrete struct type
        debug!(
            "wasm.if {}: casting anyref branch result to (ref null {})",
            branch_name, type_idx
        );
        function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(type_idx)));
    }

    Ok(())
}

/// Handle wasm.block operation
pub(crate) fn handle_block<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    // Upgrade polymorphic block result type to Step if function returns Step
    let result_ty = op
        .results(db)
        .first()
        .map(|ty| upgrade_polymorphic_to_step(db, *ty, ctx.func_return_type));
    let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));

    let block_type = if has_result {
        BlockType::Result(type_to_valtype(
            db,
            result_ty.expect("block result type"),
            &module_info.type_idx_by_type,
        )?)
    } else {
        BlockType::Empty
    };

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
    // Upgrade polymorphic loop result type to Step if function returns Step
    let result_ty = op
        .results(db)
        .first()
        .map(|ty| upgrade_polymorphic_to_step(db, *ty, ctx.func_return_type));
    let has_result = matches!(result_ty, Some(ty) if !is_nil_type(db, ty));

    let block_type = if has_result {
        BlockType::Result(type_to_valtype(
            db,
            result_ty.expect("loop result type"),
            &module_info.type_idx_by_type,
        )?)
    } else {
        BlockType::Empty
    };

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
