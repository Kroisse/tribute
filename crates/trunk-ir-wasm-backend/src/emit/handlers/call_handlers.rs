//! Call operation handlers for wasm backend.
//!
//! This module handles WebAssembly function call operations:
//! - wasm.call (direct function call)
//! - wasm.call_indirect (indirect function call via i32 table index)
//! - wasm.return_call (tail call)

use tracing::debug;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{DialectType, IdVec, Symbol, ValueDef};
use wasm_encoder::{Function, Instruction};

use crate::{CompilationError, CompilationResult};

use super::super::{
    FunctionEmitContext, ModuleInfo, attr_u32, emit_operands, emit_value, is_step_type,
    resolve_callee, set_result_local, value_type,
};

/// Handle wasm.call operation
pub(crate) fn handle_call<'db>(
    db: &'db dyn salsa::Database,
    call_op: wasm::Call<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = call_op.operation();
    let operands = op.operands(db);
    let callee = call_op.callee(db);
    let target = resolve_callee(callee, module_info)?;

    // Boxing/unboxing for generic calls is now handled by the boxing pass
    // (tribute-passes/src/boxing.rs) which inserts explicit tribute_rt.box_*/unbox_* ops.
    // These are lowered to wasm instructions by tribute_rt_to_wasm.rs.
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    function.instruction(&Instruction::Call(target));

    set_result_local(db, &op, ctx, function)?;
    Ok(())
}

/// Handle wasm.call_indirect operation
pub(crate) fn handle_call_indirect<'db>(
    db: &'db dyn salsa::Database,
    op: &trunk_ir::Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = op.operands(db);

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
    let first_operand = operands.first().copied().unwrap();
    let first_operand_ty = value_type(db, first_operand, &module_info.block_arg_types);

    // All call_indirect operations must use i32 table index
    debug_assert!(
        first_operand_ty.is_some_and(|ty| core::I32::from_type(db, ty).is_some()),
        "call_indirect first operand must be i32 table index, got {:?}",
        first_operand_ty.map(|ty| {
            ty.dialect(db)
                .with_str(|d| ty.name(db).with_str(|n| format!("{}.{}", d, n)))
        })
    );

    debug!(
        "call_indirect: first_operand_ty={:?}",
        first_operand_ty.map(|ty| {
            ty.dialect(db)
                .with_str(|d| ty.name(db).with_str(|n| format!("{}.{}", d, n)))
        })
    );

    // Debug: trace the value definition
    match first_operand.def(db) {
        ValueDef::OpResult(def_op) => {
            debug!(
                "call_indirect: first_operand defined by {}.{}, results={:?}",
                def_op.dialect(db),
                def_op.name(db),
                def_op
                    .results(db)
                    .iter()
                    .map(|t| {
                        t.dialect(db)
                            .with_str(|d| t.name(db).with_str(|n| format!("{}.{}", d, n)))
                    })
                    .collect::<Vec<_>>()
            );
        }
        ValueDef::BlockArg(block_id) => {
            debug!(
                "call_indirect: first_operand is block arg from block {:?} idx {}",
                block_id,
                first_operand.index(db)
            );
        }
    }

    // Build parameter types (all operands except first which is funcref/table_idx)
    // After normalize_primitive_types pass, anyref types are already wasm.anyref.
    // Note: core::Nil is NOT normalized - it uses (ref null none) which is
    // a subtype of anyref, so it can be passed without boxing.
    let anyref_ty = wasm::Anyref::new(db).as_type();
    let normalize_param_type = |ty: trunk_ir::Type<'db>| -> trunk_ir::Type<'db> {
        // After normalize_primitive_types pass:
        // - tribute_rt.any → wasm.anyref
        // So we only need to check for wasm.anyref
        if wasm::Anyref::from_type(db, ty).is_some() {
            anyref_ty
        } else {
            ty
        }
    };
    let param_types: IdVec<trunk_ir::Type<'db>> = operands
        .iter()
        .skip(1)
        .filter_map(|v| value_type(db, *v, &module_info.block_arg_types))
        .map(normalize_param_type)
        .collect();

    // Get result type - use enclosing function's return type if it's funcref
    // and the call_indirect has anyref result. This is needed because
    // WebAssembly GC has separate type hierarchies for anyref and funcref,
    // so we can't cast between them.
    let mut result_ty = op.results(db).first().copied().ok_or_else(|| {
        CompilationError::invalid_module("wasm.call_indirect must have a result type")
    })?;

    // If result type is anyref but enclosing function returns funcref or Step,
    // upgrade the result type accordingly. This is needed because WebAssembly GC has separate
    // type hierarchies, and effectful functions return Step for yield bubbling.
    // Note: type variables are resolved at AST level before IR generation.
    let funcref_ty = wasm::Funcref::new(db).as_type();
    if let Some(func_ret_ty) = ctx.func_return_type {
        let is_anyref_result = wasm::Anyref::from_type(db, result_ty).is_some();
        let func_returns_funcref = wasm::Funcref::from_type(db, func_ret_ty).is_some()
            || core::Func::from_type(db, func_ret_ty).is_some();
        // Check for Step type (trampoline-based effect system)
        let func_returns_step = is_step_type(db, func_ret_ty);
        if is_anyref_result && func_returns_funcref {
            debug!("call_indirect emit: upgrading anyref result to funcref for enclosing function");
            result_ty = funcref_ty;
        } else if is_anyref_result && func_returns_step {
            debug!("call_indirect emit: upgrading anyref result to Step for enclosing function");
            result_ty = crate::gc_types::step_marker_type(db);
        }
    }

    // Normalize result type: anyref stays as anyref for polymorphic dispatch
    // This must match the normalization done in collect_call_indirect_types
    if crate::emit::helpers::should_normalize_to_anyref(db, result_ty) {
        debug!(
            "call_indirect emit: normalizing result {} to anyref",
            result_ty
                .dialect(db)
                .with_str(|d| result_ty.name(db).with_str(|n| format!("{}.{}", d, n)))
        );
        result_ty = anyref_ty;
    }

    // Construct function type
    let func_type = core::Func::new(db, param_types, result_ty).as_type();

    debug!(
        "call_indirect emit: looking up func_type with result={}.{}",
        result_ty.dialect(db),
        result_ty.name(db)
    );

    // Get or compute type_idx
    let type_idx = match attr_u32(op.attributes(db), Symbol::new("type_idx")) {
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
    let table = attr_u32(op.attributes(db), Symbol::new("table")).unwrap_or(0);

    // Emit arguments first (operands[1..])
    for operand in operands.iter().skip(1) {
        emit_value(db, *operand, ctx, function)?;
    }

    // Emit the table index (operands[0])
    emit_value(db, operands[0], ctx, function)?;

    function.instruction(&Instruction::CallIndirect {
        type_index: type_idx,
        table_index: table,
    });

    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle wasm.return_call operation (tail call)
pub(crate) fn handle_return_call<'db>(
    db: &'db dyn salsa::Database,
    return_call_op: wasm::ReturnCall<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = return_call_op.operation();
    let operands = op.operands(db);
    let callee = return_call_op.callee(db);
    let target = resolve_callee(callee, module_info)?;

    // Boxing for generic calls is now handled by the boxing pass
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    function.instruction(&Instruction::ReturnCall(target));
    Ok(())
}
