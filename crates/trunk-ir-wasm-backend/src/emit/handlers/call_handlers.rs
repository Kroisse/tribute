//! Call operation handlers for wasm backend.
//!
//! This module handles WebAssembly function call operations:
//! - wasm.call (direct function call)
//! - wasm.call_indirect (indirect function call via i32 table index)
//! - wasm.return_call (tail call)

use tracing::debug;
use trunk_ir::Symbol;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueDef};
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::helpers;
use super::super::value_emission::{emit_operands, emit_value};
use super::super::{FunctionEmitContext, ModuleInfo, resolve_callee, set_result_local};

/// Handle wasm.call operation
pub(crate) fn handle_call(
    ctx: &IrContext,
    call_op: arena_wasm::Call,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = call_op.op_ref();
    let operands = ctx.op_operands(op);
    let callee = call_op.callee(ctx);
    let target = resolve_callee(callee, module_info)?;

    // Boxing/unboxing for generic calls is now handled by the boxing pass
    // (tribute-passes/src/boxing.rs) which inserts explicit tribute_rt.box_*/unbox_* ops.
    // These are lowered to wasm instructions by tribute_rt_to_wasm.rs.
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::Call(target));

    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle wasm.call_indirect operation
pub(crate) fn handle_call_indirect(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);

    // wasm.call_indirect: indirect function call via i32 table index
    // All indirect calls use table-based call_indirect (no call_ref).
    // Operands: [table_idx, arg1, arg2, ..., argN]
    // WebAssembly expects: [arg1, arg2, ..., argN, table_idx]

    if operands.is_empty() {
        return Err(CompilationError::invalid_module(
            "wasm.call_indirect requires at least a table index operand",
        ));
    }

    // The callee (i32 table index) is the FIRST operand, followed by args.
    let first_operand = operands[0];
    let first_operand_ty = helpers::value_type(ctx, first_operand);

    // All call_indirect operations must use i32 table index
    if !helpers::is_type(ctx, first_operand_ty, "core", "i32") {
        let data = ctx.types.get(first_operand_ty);
        return Err(CompilationError::invalid_module(format!(
            "call_indirect first operand must be i32 table index, got {}.{}",
            data.dialect, data.name
        )));
    }

    debug!(
        "call_indirect: first_operand_ty={}.{}",
        ctx.types.get(first_operand_ty).dialect,
        ctx.types.get(first_operand_ty).name
    );

    // Debug: trace the value definition
    match ctx.value_def(first_operand) {
        ValueDef::OpResult(def_op, _) => {
            let op_data = ctx.op(def_op);
            let result_types = ctx.op_result_types(def_op);
            debug!(
                "call_indirect: first_operand defined by {}.{}, results={:?}",
                op_data.dialect,
                op_data.name,
                result_types
                    .iter()
                    .map(|t| {
                        let td = ctx.types.get(*t);
                        format!("{}.{}", td.dialect, td.name)
                    })
                    .collect::<Vec<_>>()
            );
        }
        ValueDef::BlockArg(block_id, idx) => {
            debug!(
                "call_indirect: first_operand is block arg from block {:?} idx {}",
                block_id, idx
            );
        }
    }

    // Build parameter types (all operands except first which is funcref/table_idx)
    // After normalize_primitive_types pass, anyref types are already wasm.anyref.
    // Note: core::Nil is NOT normalized - it uses (ref null none) which is
    // a subtype of anyref, so it can be passed without boxing.
    let anyref_ty = module_info
        .common_types
        .anyref
        .expect("anyref type not pre-interned");
    let normalize_param_type = |ty: TypeRef| -> TypeRef {
        // After normalize_primitive_types pass:
        // - tribute_rt.any → wasm.anyref
        // So we only need to check for wasm.anyref
        if helpers::is_type(ctx, ty, "wasm", "anyref") {
            anyref_ty
        } else {
            ty
        }
    };
    let param_types: Vec<TypeRef> = operands
        .iter()
        .skip(1)
        .map(|v| {
            let ty = helpers::value_type(ctx, *v);
            normalize_param_type(ty)
        })
        .collect();

    // Get result type - use enclosing function's return type if it's funcref
    // and the call_indirect has anyref result. This is needed because
    // WebAssembly GC has separate type hierarchies for anyref and funcref,
    // so we can't cast between them.
    let result_types = ctx.op_result_types(op);
    let mut result_ty = result_types.first().copied().ok_or_else(|| {
        CompilationError::invalid_module("wasm.call_indirect must have a result type")
    })?;

    // If result type is anyref but enclosing function returns funcref or Step,
    // upgrade the result type accordingly. This is needed because WebAssembly GC has separate
    // type hierarchies, and effectful functions return Step for yield bubbling.
    // Note: type variables are resolved at AST level before IR generation.
    let funcref_ty = module_info
        .common_types
        .funcref
        .expect("funcref type not pre-interned");
    if let Some(func_ret_ty) = emit_ctx.func_return_type {
        let is_anyref_result = helpers::is_type(ctx, result_ty, "wasm", "anyref");
        let func_returns_funcref = helpers::is_type(ctx, func_ret_ty, "wasm", "funcref")
            || helpers::is_type(ctx, func_ret_ty, "core", "func");
        // Check for Step type (trampoline-based effect system)
        let func_returns_step = helpers::is_step_type(ctx, func_ret_ty);
        if is_anyref_result && func_returns_funcref {
            debug!("call_indirect emit: upgrading anyref result to funcref for enclosing function");
            result_ty = funcref_ty;
        } else if is_anyref_result && func_returns_step {
            debug!("call_indirect emit: upgrading anyref result to Step for enclosing function");
            result_ty = module_info
                .common_types
                .step
                .expect("step type not pre-interned");
        }
    }

    // Normalize result type: anyref stays as anyref for polymorphic dispatch
    // This must match the normalization done in collect_call_indirect_types
    if helpers::should_normalize_to_anyref(ctx, result_ty) {
        debug!(
            "call_indirect emit: normalizing result {}.{} to anyref",
            ctx.types.get(result_ty).dialect,
            ctx.types.get(result_ty).name
        );
        result_ty = anyref_ty;
    }

    // Look up type index for the function type.
    // The type must have been pre-registered by collect_call_indirect_types.
    // We construct a lookup key by building param+result TypeRef list.
    let func_type = find_func_type_in_registry(ctx, &param_types, result_ty, module_info)?;

    debug!(
        "call_indirect emit: looking up func_type with result={}.{}",
        ctx.types.get(result_ty).dialect,
        ctx.types.get(result_ty).name
    );

    // Get or compute type_idx
    let attrs = &ctx.op(op).attributes;
    let type_idx = match attr_u32(attrs, Symbol::new("type_idx")) {
        Ok(idx) => {
            debug!("call_indirect emit: using type_idx from attribute: {}", idx);
            idx
        }
        Err(_) => {
            // Look up type index
            let idx = module_info
                .type_idx_by_type
                .get(&func_type)
                .copied()
                .ok_or_else(|| {
                    debug!(
                        "call_indirect emit: func_type not found in type_idx_by_type! func_type={:?}",
                        func_type
                    );
                    CompilationError::invalid_module(
                        "wasm.call_indirect function type not registered in type section",
                    )
                })?;
            debug!("call_indirect emit: looked up type_idx: {}", idx);
            idx
        }
    };

    // call_indirect with i32 table index
    // IR operand order: [table_idx, arg1, arg2, ...]
    // WebAssembly stack order: [arg1, arg2, ..., table_idx]
    let table = attr_u32(attrs, Symbol::new("table")).unwrap_or(0);

    // Emit arguments first (operands[1..])
    for &operand in operands.iter().skip(1) {
        emit_value(ctx, operand, emit_ctx, function)?;
    }

    // Emit the table index (operands[0])
    emit_value(ctx, operands[0], emit_ctx, function)?;

    function.instruction(&Instruction::CallIndirect {
        type_index: type_idx,
        table_index: table,
    });

    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle wasm.return_call operation (tail call)
pub(crate) fn handle_return_call(
    ctx: &IrContext,
    return_call_op: arena_wasm::ReturnCall,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(return_call_op.op_ref());
    let callee = return_call_op.callee(ctx);
    let target = resolve_callee(callee, module_info)?;

    // Boxing for generic calls is now handled by the boxing pass
    emit_operands(ctx, operands, emit_ctx, function)?;

    function.instruction(&Instruction::ReturnCall(target));
    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

use super::super::helpers::attr_u32;

/// Find a core.func type in the type_idx_by_type registry by matching params and result.
/// Returns the TypeRef if found. This avoids needing &mut IrContext for interning.
fn find_func_type_in_registry(
    ctx: &IrContext,
    params: &[TypeRef],
    result: TypeRef,
    module_info: &ModuleInfo,
) -> CompilationResult<TypeRef> {
    let core_sym = Symbol::new("core");
    let func_sym = Symbol::new("func");
    let expected_len = params.len() + 1;

    // Search through registered func types (from imports, funcs, and call_indirect collection)
    for &ty_ref in module_info.type_idx_by_type.keys() {
        let data = ctx.types.get(ty_ref);
        if data.dialect != core_sym || data.name != func_sym {
            continue;
        }
        if data.params.len() != expected_len {
            continue;
        }
        // Check params match (all but last)
        let (ty_params, ty_result) = data.params.split_at(data.params.len() - 1);
        if ty_params == params && ty_result[0] == result {
            return Ok(ty_ref);
        }
    }
    // Also check func_types map
    for &ty_ref in module_info.func_types.values() {
        let data = ctx.types.get(ty_ref);
        if data.dialect != core_sym || data.name != func_sym {
            continue;
        }
        if data.params.len() != expected_len {
            continue;
        }
        let (ty_params, ty_result) = data.params.split_at(data.params.len() - 1);
        if ty_params == params && ty_result[0] == result {
            return Ok(ty_ref);
        }
    }
    Err(CompilationError::invalid_module(
        "wasm.call_indirect function type not registered in type section",
    ))
}
