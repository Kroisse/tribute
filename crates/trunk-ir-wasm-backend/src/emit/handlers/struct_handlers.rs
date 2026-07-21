//! Struct operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC struct operations (struct.new, struct.get, struct.set).

use tracing::debug;
use trunk_ir::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use wasm_encoder::{Function, HeapType, Instruction, StorageType, ValType};

use crate::CompilationResult;
use crate::gc_types::{GcTypeDef, STEP_IDX};

use super::super::helpers::{self, value_type};
use super::super::value_emission::emit_operands;
use super::super::{FunctionEmitContext, ModuleInfo, set_result_local};

/// Handle struct.new operation
pub(crate) fn handle_struct_new(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    // struct_new needs all field values on the stack, including nil types.
    // emit_operands handles nil types by emitting ref.null none.
    emit_operands(ctx, operands, emit_ctx, function)?;

    let type_idx = wasm_dialect::StructNew::from_op(ctx, op)
        .expect("handler called for wasm.struct_new")
        .type_idx(ctx);

    function.instruction(&Instruction::StructNew(type_idx));
    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Handle struct.get operation
pub(crate) fn handle_struct_get(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Check if operand is abstract type (anyref/structref) and needs casting to concrete struct type
    // This happens when:
    // - A closure is captured (stored as anyref) and later used
    // - A continuation field is typed as structref but needs access to concrete struct fields
    let operand_abstract_type = operands.first().and_then(|op_val| {
        let ty = value_type(ctx, *op_val);
        if helpers::is_type(ctx, ty, "wasm", "anyref") {
            Some("anyref")
        } else if helpers::is_type(ctx, ty, "wasm", "structref") {
            Some("structref")
        } else {
            None
        }
    });
    let needs_cast_from_abstract = operand_abstract_type.is_some();

    let struct_get =
        wasm_dialect::StructGet::from_op(ctx, op).expect("handler called for wasm.struct_get");
    let type_idx = struct_get.type_idx(ctx);
    let field_idx = struct_get.field_idx(ctx);
    debug!(
        "struct_get: emitting StructGet with type_idx={}, field_idx={}, operand_abstract_type={:?}",
        type_idx, field_idx, operand_abstract_type
    );

    // If operand was abstract type (anyref/structref), cast it to the concrete struct type first
    if needs_cast_from_abstract {
        debug!(
            "struct_get: casting {:?} to struct type_idx={}",
            operand_abstract_type.unwrap_or("unknown"),
            type_idx
        );
        function.instruction(&Instruction::RefCastNullable(HeapType::Concrete(type_idx)));
    }

    function.instruction(&Instruction::StructGet {
        struct_type_index: type_idx,
        field_index: field_idx,
    });

    // Check if boxing is needed: result local expects anyref but struct field is i64
    // This happens when extracting values from structs where the IR uses generic/wrapper
    // types but the actual struct field contains a primitive (i64).
    let needs_boxing =
        check_struct_get_needs_boxing(ctx, op, emit_ctx, module_info, type_idx, field_idx);

    if needs_boxing {
        debug!("struct_get: boxing i64 to i31ref for anyref local");
        function.instruction(&Instruction::I32WrapI64);
        function.instruction(&Instruction::RefI31);
    } else {
        // Check if struct field type is anyref but IR result type is more specific.
        // This happens when reading from Step.value (anyref field) where the IR
        // expects a concrete type. Insert ref.cast to narrow the type.
        let field_is_anyref = module_info
            .gc_types
            .get(type_idx as usize)
            .and_then(|gc_type| match gc_type {
                GcTypeDef::Struct(fields) => fields.get(field_idx as usize),
                _ => None,
            })
            .map(|field| {
                matches!(
                    field.element_type,
                    StorageType::Val(ValType::Ref(wasm_encoder::RefType {
                        nullable: true,
                        heap_type: HeapType::Abstract {
                            shared: false,
                            ty: wasm_encoder::AbstractHeapType::Any,
                        },
                    }))
                )
            })
            .unwrap_or(false);

        // Skip concrete ref.cast for Step.value (field_idx=1): it stores
        // heterogeneous anyref values (i31ref, struct ref, null) by design.
        // The correct narrowing casts are inserted at the IR level.
        let is_step_value_field = type_idx == STEP_IDX && field_idx == 1;

        if field_is_anyref
            && !is_step_value_field
            && let Some(&result_ty) = ctx.op_result_types(op).first()
            && let Ok(result_valtype) =
                helpers::type_to_valtype(ctx, result_ty, &module_info.type_idx_by_type)
            && let ValType::Ref(rt) = &result_valtype
            && let ht @ HeapType::Concrete(_) = &rt.heap_type
        {
            debug!("struct_get: casting anyref field to concrete type {:?}", ht);
            function.instruction(&Instruction::RefCastNullable(*ht));
        }
    }

    set_result_local(ctx, op, emit_ctx, function)?;
    Ok(())
}

/// Check if struct.get result needs boxing (i64 to i31ref)
fn check_struct_get_needs_boxing(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    type_idx: u32,
    field_idx: u32,
) -> bool {
    let result_types = ctx.op_result_types(op);
    if result_types.is_empty() {
        return false;
    }

    let result_value = ctx.op_result(op, 0);

    // Check if the result local would be anyref by examining the effective type
    let local_type = emit_ctx
        .effective_types
        .get(&result_value)
        .copied()
        .or_else(|| result_types.first().copied());

    // Check if the result expects anyref
    // Note: type variables are resolved at AST level before IR generation
    let expects_anyref = local_type
        .map(|ty| helpers::is_type(ctx, ty, "wasm", "anyref"))
        .unwrap_or(false);

    // Check if the struct field is i64
    let field_is_i64 = module_info
        .gc_types
        .get(type_idx as usize)
        .map(|gc_type| {
            if let GcTypeDef::Struct(fields) = gc_type {
                fields
                    .get(field_idx as usize)
                    .map(|field| matches!(field.element_type, StorageType::Val(ValType::I64)))
                    .unwrap_or(false)
            } else {
                false
            }
        })
        .unwrap_or(false);

    debug!(
        "struct_get boxing check: expects_anyref={}, field_is_i64={}, type_idx={}, field_idx={}",
        expects_anyref, field_is_i64, type_idx, field_idx
    );

    expects_anyref && field_is_i64
}

/// Handle struct.set operation
pub(crate) fn handle_struct_set(
    ctx: &IrContext,
    struct_set_op: wasm_dialect::StructSet,
    emit_ctx: &FunctionEmitContext,
    _module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = struct_set_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    let type_idx = struct_set_op.type_idx(ctx);
    let field_idx = struct_set_op.field_idx(ctx);

    function.instruction(&Instruction::StructSet {
        struct_type_index: type_idx,
        field_index: field_idx,
    });
    Ok(())
}
