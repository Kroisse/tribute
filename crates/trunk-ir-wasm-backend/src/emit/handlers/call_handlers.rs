//! Call operation handlers for wasm backend.
//!
//! This module handles WebAssembly function call operations:
//! - wasm.call (direct function call)
//! - wasm.call_indirect (indirect function call via i32 table index)
//! - wasm.return_call (tail call)
//! - wasm.return_call_indirect (indirect tail call)

use tracing::debug;
use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::refs::{OpRef, TypeRef, ValueDef};
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::helpers;
use super::super::value_emission::{emit_operands, emit_value};
use super::super::{FunctionEmitContext, ModuleInfo, resolve_callee, set_result_local};

/// Handle wasm.call operation
pub(crate) fn handle_call(
    ctx: &IrContext,
    call_op: wasm_dialect::Call,
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
        .ok_or_else(|| CompilationError::invalid_module("anyref type not pre-interned"))?;
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
        .ok_or_else(|| CompilationError::invalid_module("funcref type not pre-interned"))?;
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
                .ok_or_else(|| CompilationError::invalid_module("step type not pre-interned"))?;
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

    // The lowering pattern leaves a placeholder `type_idx = 0` on this op.
    // Resolve the exact collected signature index at emission time.
    let attrs = &ctx.op(op).attributes;
    let type_idx = module_info
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

    // call_indirect with i32 table index
    // IR operand order: [table_idx, arg1, arg2, ...]
    // WebAssembly stack order: [arg1, arg2, ..., table_idx]
    let table = match attrs.get("table") {
        Some(_) => attr_u32(attrs, Symbol::new("table"))?,
        None => 0,
    };

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
    return_call_op: wasm_dialect::ReturnCall,
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

/// Handle wasm.return_call_indirect operation (indirect tail call).
pub(crate) fn handle_return_call_indirect(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    if operands.is_empty() {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect requires at least a table index operand",
        ));
    }
    if !ctx.op_result_types(op).is_empty() {
        return Err(CompilationError::invalid_module(
            "wasm.return_call_indirect must not produce a result",
        ));
    }

    let table_index = operands[0];
    let table_index_ty = helpers::value_type(ctx, table_index);
    if !helpers::is_type(ctx, table_index_ty, "core", "i32") {
        let data = ctx.types.get(table_index_ty);
        return Err(CompilationError::invalid_module(format!(
            "return_call_indirect first operand must be i32 table index, got {}.{}",
            data.dialect, data.name
        )));
    }

    let param_types: Vec<TypeRef> = operands
        .iter()
        .skip(1)
        .map(|value| helpers::value_type(ctx, *value))
        .collect();
    let func_type = find_void_func_type_in_registry(ctx, &param_types, module_info)?;
    // The lowering pattern leaves a placeholder `type_idx = 0` on this op.
    // Tail calls must use the index assigned to the exact collected signature;
    // unlike ordinary calls, an attribute here is never authoritative.
    let type_index = module_info
        .type_idx_by_type
        .get(&func_type)
        .copied()
        .ok_or_else(|| {
            CompilationError::invalid_module(
                "wasm.return_call_indirect function type not registered in type section",
            )
        })?;
    let attrs = &ctx.op(op).attributes;
    let table = match attrs.get("table") {
        Some(_) => attr_u32(attrs, Symbol::new("table"))?,
        None => 0,
    };

    for &operand in operands.iter().skip(1) {
        emit_value(ctx, operand, emit_ctx, function)?;
    }
    emit_value(ctx, table_index, emit_ctx, function)?;
    function.instruction(&Instruction::ReturnCallIndirect {
        type_index,
        table_index: table,
    });
    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

use super::super::helpers::attr_u32;

/// Find a core.func type in the `type_idx_by_type` / `func_types` registries by
/// matching params and result.
///
/// core.func types encode a single result followed by parameters in `TypeData.params`:
/// `[result, param1, .., paramN]`.
///
/// This performs a linear O(n) scan over the registries to avoid requiring
/// `&mut IrContext` for interning a new type. The trade-off is O(n) per
/// `call_indirect` emission, which is acceptable for current module sizes.
/// If this becomes a bottleneck, a dedicated index keyed by (params, result)
/// could be built during `collect_module_info`.
fn find_func_type_in_registry(
    ctx: &IrContext,
    params: &[TypeRef],
    result: TypeRef,
    module_info: &ModuleInfo,
) -> CompilationResult<TypeRef> {
    // Search through registered func types (from imports, funcs, and call_indirect collection)
    for &ty_ref in module_info.type_idx_by_type.keys() {
        if helpers::func_type_parts(ctx, ty_ref)
            .is_some_and(|(ty_params, ty_result)| ty_result == result && ty_params == params)
        {
            return Ok(ty_ref);
        }
    }
    // Also check func_types map
    for &ty_ref in module_info.func_types.values() {
        if helpers::func_type_parts(ctx, ty_ref)
            .is_some_and(|(ty_params, ty_result)| ty_result == result && ty_params == params)
        {
            return Ok(ty_ref);
        }
    }
    Err(CompilationError::invalid_module(
        "wasm.call_indirect function type not registered in type section",
    ))
}

fn find_void_func_type_in_registry(
    ctx: &IrContext,
    params: &[TypeRef],
    module_info: &ModuleInfo,
) -> CompilationResult<TypeRef> {
    for &ty_ref in module_info.type_idx_by_type.keys() {
        if helpers::func_type_parts(ctx, ty_ref).is_some_and(|(ty_params, ty_result)| {
            helpers::is_nil_type(ctx, ty_result) && ty_params == params
        }) {
            return Ok(ty_ref);
        }
    }
    for &ty_ref in module_info.func_types.values() {
        if helpers::func_type_parts(ctx, ty_ref).is_some_and(|(ty_params, ty_result)| {
            helpers::is_nil_type(ctx, ty_result) && ty_params == params
        }) {
            return Ok(ty_ref);
        }
    }
    Err(CompilationError::invalid_module(
        "wasm.return_call_indirect void function type not registered in type section",
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use trunk_ir::Span;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::{Location, TypeDataBuilder};
    use wasm_encoder::ValType;

    use crate::emit::CommonTypes;

    use super::*;

    #[test]
    fn return_call_indirect_emits_registered_nonzero_type_index() {
        let mut ctx = IrContext::new();
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let nil_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());
        let func_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(nil_ty)
                .build(),
        );
        let table = wasm_dialect::i32_const(&mut ctx, location, i32_ty, 0);
        let table_result = table.result(&ctx);
        let tail = wasm_dialect::return_call_indirect(&mut ctx, location, vec![table_result], 0, 0);
        let emit_ctx = FunctionEmitContext {
            value_locals: HashMap::from([(table_result, 0)]),
            effective_types: HashMap::new(),
            func_return_type: None,
        };
        let module_info = ModuleInfo {
            type_idx_by_type: HashMap::from([(func_ty, 7)]),
            ..ModuleInfo::default()
        };
        let mut function = Function::new([(1, ValType::I32)]);

        handle_return_call_indirect(&ctx, tail.op_ref(), &emit_ctx, &module_info, &mut function)
            .expect("tail indirect call should emit");

        let body = function.into_raw_body();
        assert!(
            body.ends_with(&[0x13, 0x07, 0x00]),
            "registered type index must be emitted in return_call_indirect"
        );
    }

    #[test]
    fn call_indirect_emits_registered_nonzero_type_index() {
        let mut ctx = IrContext::new();
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let anyref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref")).build());
        let funcref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("funcref")).build());
        let func_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(i32_ty)
                .build(),
        );
        let table = wasm_dialect::i32_const(&mut ctx, location, i32_ty, 0);
        let table_result = table.result(&ctx);
        let call =
            wasm_dialect::call_indirect(&mut ctx, location, vec![table_result], vec![i32_ty], 0, 0);
        let call_result = call.results(&ctx)[0];
        let emit_ctx = FunctionEmitContext {
            value_locals: HashMap::from([(table_result, 0), (call_result, 1)]),
            effective_types: HashMap::new(),
            func_return_type: None,
        };
        let module_info = ModuleInfo {
            type_idx_by_type: HashMap::from([(func_ty, 7)]),
            common_types: CommonTypes {
                anyref: Some(anyref_ty),
                funcref: Some(funcref_ty),
                ..CommonTypes::default()
            },
            ..ModuleInfo::default()
        };
        let mut function = Function::new([(2, ValType::I32)]);

        handle_call_indirect(&ctx, call.op_ref(), &emit_ctx, &module_info, &mut function)
            .expect("ordinary indirect call should emit");

        let body = function.into_raw_body();
        assert!(body.windows(3).any(|window| window == [0x11, 0x07, 0x00]));
        assert!(!body.windows(3).any(|window| window == [0x13, 0x07, 0x00]));
    }
}
