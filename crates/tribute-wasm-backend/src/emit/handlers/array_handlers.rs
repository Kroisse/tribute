//! Array operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC array operations.

use trunk_ir::dialect::wasm;
use trunk_ir::{Operation, Type};
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::{
    ModuleInfo, emit_operands, is_closure_struct_type, set_result_local, value_type,
};

/// Get type_idx from operation attributes or inferred type.
///
/// Priority: type_idx attr > type attr > inferred_type (from result/operand)
fn get_type_idx<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    inferred_type: Option<Type<'db>>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<u32> {
    use crate::gc_types::{ATTR_TYPE, ATTR_TYPE_IDX, CLOSURE_STRUCT_IDX};
    use trunk_ir::Attribute;

    let attrs = op.attributes(db);

    // First try type_idx attribute
    if let Some(trunk_ir::Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
        return Ok(*idx as u32);
    }

    // Fall back to type attribute (legacy, will be removed)
    if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
        if is_closure_struct_type(db, *ty) {
            return Ok(CLOSURE_STRUCT_IDX);
        }
        if let Some(&type_idx) = module_info.type_idx_by_type.get(ty) {
            return Ok(type_idx);
        }
    }

    // Fall back to inferred type
    if let Some(ty) = inferred_type {
        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
        if is_closure_struct_type(db, ty) {
            return Ok(CLOSURE_STRUCT_IDX);
        }
        if let Some(&type_idx) = module_info.type_idx_by_type.get(&ty) {
            return Ok(type_idx);
        }
    }

    Err(CompilationError::missing_attribute("type or type_idx"))
}

/// Handle array.new operation
pub(crate) fn handle_array_new<'db>(
    db: &'db dyn salsa::Database,
    array_new_op: wasm::ArrayNew<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_new_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from result type (type_idx attr may not be set during IR generation)
    let inferred_type = op.results(db).first().copied();
    let type_idx = get_type_idx(db, &op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayNew(type_idx));
    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle array.new_default operation
pub(crate) fn handle_array_new_default<'db>(
    db: &'db dyn salsa::Database,
    array_new_default_op: wasm::ArrayNewDefault<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_new_default_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from result type
    let inferred_type = op.results(db).first().copied();
    let type_idx = get_type_idx(db, &op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayNewDefault(type_idx));
    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle array.get operation
pub(crate) fn handle_array_get<'db>(
    db: &'db dyn salsa::Database,
    array_get_op: wasm::ArrayGet<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands
        .first()
        .and_then(|v| value_type(db, *v, &module_info.block_arg_types));
    let type_idx = get_type_idx(db, &op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayGet(type_idx));
    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle array.get_s operation (sign-extending load)
pub(crate) fn handle_array_get_s<'db>(
    db: &'db dyn salsa::Database,
    array_get_s_op: wasm::ArrayGetS<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_s_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands
        .first()
        .and_then(|v| value_type(db, *v, &module_info.block_arg_types));
    let type_idx = get_type_idx(db, &op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayGetS(type_idx));
    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle array.get_u operation (zero-extending load)
pub(crate) fn handle_array_get_u<'db>(
    db: &'db dyn salsa::Database,
    array_get_u_op: wasm::ArrayGetU<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_get_u_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands
        .first()
        .and_then(|v| value_type(db, *v, &module_info.block_arg_types));
    let type_idx = get_type_idx(db, &op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArrayGetU(type_idx));
    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle array.set operation
pub(crate) fn handle_array_set<'db>(
    db: &'db dyn salsa::Database,
    array_set_op: wasm::ArraySet<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_set_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from operand[0] (the array ref)
    let inferred_type = operands
        .first()
        .and_then(|v| value_type(db, *v, &module_info.block_arg_types));
    let type_idx = get_type_idx(db, &op, inferred_type, module_info)?;

    function.instruction(&Instruction::ArraySet(type_idx));
    Ok(())
}

/// Handle array.copy operation
pub(crate) fn handle_array_copy<'db>(
    db: &'db dyn salsa::Database,
    array_copy_op: wasm::ArrayCopy<'db>,
    ctx: &super::super::FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = array_copy_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    function.instruction(&Instruction::ArrayCopy {
        array_type_index_dst: array_copy_op.dst_type_idx(db),
        array_type_index_src: array_copy_op.src_type_idx(db),
    });
    Ok(())
}
