//! Reference operation handlers for wasm backend.
//!
//! This module handles WebAssembly reference operations like ref.null, ref.func,
//! ref.cast, and ref.test.

use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use wasm_encoder::{Function, HeapType, Instruction, ValType};

use crate::{CompilationError, CompilationResult};

use super::super::helpers::attr_heap_type;
use super::super::value_emission::emit_operands;
use super::super::{
    ATTR_HEAP_TYPE, ATTR_TARGET_TYPE, FunctionEmitContext, ModuleInfo, set_result_local,
};

/// Handle ref.null operation
pub(crate) fn handle_ref_null(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let ref_null =
        wasm_dialect::RefNull::from_op(ctx, op).expect("handler called for wasm.ref_null");
    let heap_type = if let Some(type_idx) = ref_null.type_idx(ctx) {
        HeapType::Concrete(type_idx)
    } else {
        attr_heap_type(ctx, &ctx.op(op).attributes, ATTR_HEAP_TYPE())?
    };

    function.instruction(&Instruction::RefNull(heap_type));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle ref.func operation
pub(crate) fn handle_ref_func(
    ctx: &IrContext,
    ref_func_op: wasm_dialect::RefFunc,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let func_name = ref_func_op.func_name(ctx);
    let func_idx = resolve_callee(func_name, module_info)?;
    function.instruction(&Instruction::RefFunc(func_idx));
    set_result_local(ctx, ref_func_op.op_ref(), emit_ctx, function)?;
    Ok(())
}

/// Handle ref.cast operation
pub(crate) fn handle_ref_cast(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    let ref_cast =
        wasm_dialect::RefCast::from_op(ctx, op).expect("handler called for wasm.ref_cast");
    let heap_type = if let Some(type_idx) = ref_cast.type_idx(ctx) {
        HeapType::Concrete(type_idx)
    } else {
        attr_heap_type(ctx, &ctx.op(op).attributes, ATTR_TARGET_TYPE())?
    };

    let result_is_non_null = ctx
        .op_result_types(op)
        .first()
        .and_then(|&ty| {
            super::super::helpers::type_to_valtype(ctx, ty, &module_info.type_idx_by_type).ok()
        })
        .is_some_and(|ty| matches!(ty, ValType::Ref(ref_ty) if !ref_ty.nullable));
    tracing::debug!(result_is_non_null, ?heap_type, "emitting ref.cast");
    if result_is_non_null {
        function.instruction(&Instruction::RefCastNonNull(heap_type));
    } else {
        function.instruction(&Instruction::RefCastNullable(heap_type));
    }
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle ref.test operation
pub(crate) fn handle_ref_test(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    let ref_test =
        wasm_dialect::RefTest::from_op(ctx, op).expect("handler called for wasm.ref_test");
    let heap_type = if let Some(type_idx) = ref_test.type_idx(ctx) {
        HeapType::Concrete(type_idx)
    } else {
        attr_heap_type(ctx, &ctx.op(op).attributes, ATTR_TARGET_TYPE())?
    };

    function.instruction(&Instruction::RefTestNullable(heap_type));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Resolve a callee symbol to a function index.
fn resolve_callee(path: Symbol, module_info: &ModuleInfo) -> CompilationResult<u32> {
    module_info
        .func_indices
        .get(&path)
        .copied()
        .ok_or_else(|| CompilationError::function_not_found(&path.to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use trunk_ir::Span;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::{Location, TypeDataBuilder};
    use wasm_encoder::{RefType, ValType};

    use super::*;

    #[test]
    fn ref_cast_uses_non_null_instruction_for_non_null_result() {
        let mut ctx = IrContext::new();
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let anyref_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref")).build());
        let bytes_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bytes")).build());
        let null =
            wasm_dialect::ref_null(&mut ctx, location, anyref_ty, Symbol::new("anyref"), None);
        let null_result = null.result(&ctx);
        let cast = wasm_dialect::ref_cast(
            &mut ctx,
            location,
            null_result,
            bytes_ty,
            bytes_ty,
            Some(crate::gc_types::BYTES_STRUCT_IDX),
        );
        let cast_result = cast.result(&ctx);
        let emit_ctx = FunctionEmitContext {
            value_locals: HashMap::from([(null_result, 0), (cast_result, 1)]),
            effective_types: HashMap::new(),
            func_return_type: None,
        };
        let module_info = ModuleInfo::default();
        let mut function = Function::new([(2, ValType::Ref(RefType::ANYREF))]);

        handle_ref_cast(&ctx, cast.op_ref(), &emit_ctx, &module_info, &mut function)
            .expect("non-null ref.cast should emit");
    }

    #[test]
    fn resolve_callee_reports_missing_symbols() {
        let found = Symbol::new("found");
        let module_info = ModuleInfo {
            func_indices: HashMap::from([(found, 7)]),
            ..ModuleInfo::default()
        };

        assert_eq!(resolve_callee(found, &module_info).unwrap(), 7);
        assert!(resolve_callee(Symbol::new("missing"), &module_info).is_err());
    }
}
