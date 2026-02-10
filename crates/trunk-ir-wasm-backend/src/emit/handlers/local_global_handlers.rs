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
///
/// When reading from an anyref global, the result may need to be cast to the
/// specific type expected by the IR result type. This happens for globals like
/// $yield_cont which store different continuation types as anyref.
pub(crate) fn handle_global_get<'db>(
    db: &'db dyn salsa::Database,
    global_op: wasm::GlobalGet<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let index = global_op.index(db);
    function.instruction(&Instruction::GlobalGet(index));

    // Check if the global's actual type is anyref but the IR result type is more specific.
    // If so, insert a ref.cast to narrow the type.
    if let Some(global_def) = module_info.globals.get(index as usize)
        && is_anyref_valtype(&global_def.valtype)
    {
        let op = global_op.operation();
        if let Some(result_ty) = op.results(db).first().copied()
            && let Ok(result_valtype) =
                super::super::type_to_valtype(db, result_ty, &module_info.type_idx_by_type)
            && let Some(heap_type) = concrete_heap_type(&result_valtype)
        {
            function.instruction(&Instruction::RefCastNullable(heap_type));
        }
    }

    set_result_local(db, &global_op.operation(), ctx, function)?;
    Ok(())
}

/// Check if a ValType is anyref.
fn is_anyref_valtype(vt: &wasm_encoder::ValType) -> bool {
    matches!(
        vt,
        wasm_encoder::ValType::Ref(wasm_encoder::RefType {
            nullable: true,
            heap_type: wasm_encoder::HeapType::Abstract {
                shared: false,
                ty: wasm_encoder::AbstractHeapType::Any
            }
        })
    )
}

/// Extract a concrete HeapType from a ValType, if present.
fn concrete_heap_type(vt: &wasm_encoder::ValType) -> Option<wasm_encoder::HeapType> {
    match vt {
        wasm_encoder::ValType::Ref(rt) => match &rt.heap_type {
            ht @ wasm_encoder::HeapType::Concrete(_) => Some(*ht),
            _ => None,
        },
        _ => None,
    }
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
