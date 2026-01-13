//! Call operation handlers for wasm backend.
//!
//! This module handles WebAssembly function call operations:
//! - wasm.call (direct function call)
//! - wasm.call_indirect (indirect function call via funcref or table)
//! - wasm.return_call (tail call)

use tracing::debug;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::{DialectType, IdVec, Symbol, ValueDef};
use wasm_encoder::{Function, HeapType, Instruction};

use tribute_ir::dialect::{tribute, tribute_rt};

use crate::{CompilationError, CompilationResult};

use super::super::{
    FunctionEmitContext, ModuleInfo, attr_u32, emit_operands, emit_operands_with_boxing,
    emit_unboxing, emit_value, is_closure_struct_type, is_step_type, resolve_callee,
    set_result_local, value_type,
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

    // Check if we need boxing for generic function calls
    if let Some(callee_ty) = module_info.func_types.get(&callee) {
        let param_types = callee_ty.params(db);
        emit_operands_with_boxing(db, operands, &param_types, ctx, module_info, function)?;
    } else {
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    }

    function.instruction(&Instruction::Call(target));

    // Check if we need unboxing for the return value
    if let Some(callee_ty) = module_info.func_types.get(&callee) {
        let return_ty = callee_ty.result(db);
        // If callee returns anyref (type.var), we need to unbox to the expected concrete type.
        // Since type inference doesn't propagate instantiated types to the IR,
        // we infer the result type from the first operand's type (works for identity-like functions).
        if tribute::is_type_var(db, return_ty)
            && !module_info.type_idx_by_type.contains_key(&return_ty)
        {
            // Try to infer concrete type from first operand
            if let Some(operand_ty) = operands
                .first()
                .and_then(|v| value_type(db, *v, &module_info.block_arg_types))
                .filter(|ty| !tribute::is_type_var(db, *ty))
            {
                emit_unboxing(db, operand_ty, function)?;
            }
        }
    }

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

    // wasm.call_indirect: indirect function call
    // Operands: [arg1, arg2, ..., argN, funcref]
    // The funcref is the last operand (on top of stack in WebAssembly)
    //
    // If the last operand is a reference type (funcref/anyref), we use call_ref.
    // If it's an i32 (table index), we use the traditional call_indirect.

    if operands.is_empty() {
        return Err(CompilationError::invalid_module(
            "wasm.call_indirect requires at least a funcref operand",
        ));
    }

    // In func.call_indirect IR, the callee (funcref) is the FIRST operand, followed by args.
    // But WebAssembly expects: [args..., funcref/table_idx] with the call target last on stack.
    // Check the first operand to determine if we have a funcref (use call_ref) or i32 (use call_indirect).
    let first_operand = operands.first().copied().unwrap();
    let first_operand_ty = value_type(db, first_operand, &module_info.block_arg_types);
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

    let is_ref_type = first_operand_ty.is_some_and(|ty| {
        let is_funcref = wasm::Funcref::from_type(db, ty).is_some();
        let is_anyref = wasm::Anyref::from_type(db, ty).is_some();
        let is_core_func = core::Func::from_type(db, ty).is_some();
        // Check if this is a closure struct (adt.struct with name "_closure")
        // Closure structs contain (funcref, anyref) and are used for call_indirect
        let is_closure_struct = is_closure_struct_type(db, ty);
        debug!(
            "call_indirect: is_funcref={}, is_anyref={}, is_core_func={}, is_closure_struct={}",
            is_funcref, is_anyref, is_core_func, is_closure_struct
        );
        is_funcref || is_anyref || is_core_func || is_closure_struct
    });
    debug!("call_indirect: is_ref_type={}", is_ref_type);

    // Build parameter types (all operands except first which is funcref)
    // Normalize IR types to wasm types - primitive IR types that might be boxed
    // (in polymorphic handlers) should use anyref.
    let anyref_ty = wasm::Anyref::new(db).as_type();
    let normalize_param_type = |ty: trunk_ir::Type<'db>| -> trunk_ir::Type<'db> {
        if tribute_rt::is_int(db, ty)
            || tribute_rt::is_nat(db, ty)
            || tribute_rt::is_bool(db, ty)
            || tribute_rt::is_float(db, ty)
            || tribute::is_type_var(db, ty)
            || core::Nil::from_type(db, ty).is_some()
        {
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

    // If result type is anyref/type_var but enclosing function returns funcref or Step,
    // upgrade the result type accordingly. This is needed because WebAssembly GC has separate
    // type hierarchies, and effectful functions return Step for yield bubbling.
    let funcref_ty = wasm::Funcref::new(db).as_type();
    if let Some(func_ret_ty) = ctx.func_return_type {
        let is_anyref_result = wasm::Anyref::from_type(db, result_ty).is_some();
        let is_type_var_result = result_ty.dialect(db) == Symbol::new("tribute")
            && result_ty.name(db) == Symbol::new("type_var");
        let is_polymorphic_result = is_anyref_result || is_type_var_result;
        let func_returns_funcref = wasm::Funcref::from_type(db, func_ret_ty).is_some()
            || core::Func::from_type(db, func_ret_ty).is_some();
        // Check for Step type (trampoline-based effect system)
        let func_returns_step = is_step_type(db, func_ret_ty);
        if is_polymorphic_result && func_returns_funcref {
            debug!(
                "call_indirect emit: upgrading polymorphic result to funcref for enclosing function"
            );
            result_ty = funcref_ty;
        } else if is_polymorphic_result && func_returns_step {
            debug!(
                "call_indirect emit: upgrading polymorphic result to Step for enclosing function"
            );
            result_ty = crate::gc_types::step_marker_type(db);
        }
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

    if is_ref_type {
        // Use call_ref for typed function references
        // IR operand order: [funcref, arg1, arg2, ...]
        // WebAssembly stack order: [arg1, arg2, ..., funcref]
        // So we emit args first (operands[1..]), then funcref (operands[0])

        // Emit arguments (all operands except first funcref)
        for (i, operand) in operands.iter().skip(1).enumerate() {
            debug!(
                "call_indirect: emitting arg {} of type {:?}",
                i,
                value_type(db, *operand, &module_info.block_arg_types).map(|t| t
                    .dialect(db)
                    .with_str(|d| t.name(db).with_str(|n| format!("{}.{}", d, n))))
            );
            match operand.def(db) {
                ValueDef::OpResult(def_op) => {
                    debug!(
                        "  arg {} defined by {}.{}",
                        i,
                        def_op.dialect(db),
                        def_op.name(db)
                    );
                }
                ValueDef::BlockArg(block_id) => {
                    debug!("  arg {} is block arg from {:?}", i, block_id);
                }
            }
            emit_value(db, *operand, ctx, function)?;
        }

        // Emit the funcref (first operand)
        emit_value(db, first_operand, ctx, function)?;

        // Cast anyref/closure struct to typed function reference if needed
        // Closure struct (adt.struct with name "_closure") contains funcref in field 0.
        // When we extract the funcref via struct_get, the IR type may still be adt.struct,
        // but the actual wasm value is funcref. Cast to the concrete function type.
        if let Some(ty) = first_operand_ty
            && (wasm::Anyref::from_type(db, ty).is_some()
                || core::Func::from_type(db, ty).is_some()
                || is_closure_struct_type(db, ty))
        {
            // Cast to (ref null func_type)
            function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(type_idx)));
        }

        // Emit call_ref with the function type index
        function.instruction(&Instruction::CallRef(type_idx));

        // The call_ref returns the declared result type of the called function.
        // But the local where we store the result may have a different (concrete) type.
        // We need to cast the result to match the local's type.
        //
        // Note: This is a workaround for unresolved type variables (tribute.type_var)
        // in the IR. The proper fix would be to resolve types earlier in the pipeline.
    } else {
        // Traditional call_indirect with i32 table index
        let table = attr_u32(op.attributes(db), Symbol::new("table")).unwrap_or(0);

        // Emit all operands (arguments first, then table index)
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

        function.instruction(&Instruction::CallIndirect {
            type_index: type_idx,
            table_index: table,
        });
    }

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

    // Check if we need boxing for generic function calls
    if let Some(callee_ty) = module_info.func_types.get(&callee) {
        let param_types = callee_ty.params(db);
        emit_operands_with_boxing(db, operands, &param_types, ctx, module_info, function)?;
    } else {
        emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    }

    // Note: Return unboxing is not needed for tail calls since
    // the caller's return type should match the callee's.
    function.instruction(&Instruction::ReturnCall(target));
    Ok(())
}
