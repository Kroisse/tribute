//! Struct operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC struct operations (struct.new, struct.get, struct.set).

use std::collections::BTreeMap;

use tracing::debug;
use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueDef, ValueRef};
use trunk_ir::types::Attribute;
use wasm_encoder::{Function, HeapType, Instruction, StorageType, ValType};

use crate::gc_types::{
    ATTR_FIELD_COUNT, ATTR_TYPE, ATTR_TYPE_IDX, CLOSURE_STRUCT_IDX, GcTypeDef, STEP_IDX,
};
use crate::{CompilationError, CompilationResult};

use super::super::helpers::{self, get_type_idx_from_attrs, is_closure_struct_type, value_type};
use super::super::value_emission::emit_operands;
use super::super::{ATTR_TARGET_TYPE, FunctionEmitContext, ModuleInfo, set_result_local};

/// Handle struct.new operation
pub(crate) fn handle_struct_new(
    ctx: &IrContext,
    op: OpRef,
    emit_ctx: &FunctionEmitContext,
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = ctx.op_operands(op);
    // struct_new needs all field values on the stack, including nil types.
    // emit_operands handles nil types by emitting ref.null none.
    emit_operands(ctx, operands, emit_ctx, function)?;

    let attrs = &ctx.op(op).attributes;
    let field_count = operands.len();
    let result_type = ctx.op_result_types(op).first().copied();

    // Priority: explicit type_idx attr > type attr > placeholder result type > inferred result type
    // type_idx attribute takes highest precedence (set by wasm_gc_type_assign pass)
    let type_idx = if let Some(Attribute::Int(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
        Some(u32::try_from(*idx).map_err(|_| {
            CompilationError::invalid_attribute(format!(
                "struct_new: type_idx value {} out of u32 range",
                idx
            ))
        })?)
    } else if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
        if helpers::is_type(ctx, *ty, "wasm", "structref") {
            // Use placeholder map for wasm.structref
            // All (type, field_count) pairs are registered by collect_gc_types upfront
            module_info
                .placeholder_struct_type_idx
                .get(&(*ty, field_count))
                .copied()
        } else if is_closure_struct_type(ctx, *ty) {
            // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
            Some(CLOSURE_STRUCT_IDX)
        } else {
            // Regular type
            module_info.type_idx_by_type.get(ty).copied()
        }
    } else if let Some(ty) = result_type
        && helpers::is_type(ctx, ty, "wasm", "structref")
    {
        // Result type is a placeholder (wasm.structref) - use placeholder map
        module_info
            .placeholder_struct_type_idx
            .get(&(ty, field_count))
            .copied()
    } else if let Some(ty) = result_type {
        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
        if is_closure_struct_type(ctx, ty) {
            Some(CLOSURE_STRUCT_IDX)
        } else {
            // Infer type from result type (non-placeholder)
            module_info.type_idx_by_type.get(&ty).copied()
        }
    } else {
        None
    }
    .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;

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
    let attrs = &ctx.op(op).attributes;

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

    // CRITICAL: For struct_get, the type_idx MUST match the operand's actual type.
    // We need to trace through ref.cast operations to find the actual type,
    // because the IR type might be different from the wasm type after casting.
    let operand = operands.first().copied();
    let type_idx = resolve_struct_get_type_idx(ctx, op, operand, attrs, module_info)?;

    let field_idx = attr_field_idx(attrs)?;
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

/// Resolve the type_idx for struct.get operation by tracing through ref.cast operations
fn resolve_struct_get_type_idx(
    ctx: &IrContext,
    op: OpRef,
    operand: Option<ValueRef>,
    attrs: &BTreeMap<Symbol, Attribute>,
    module_info: &ModuleInfo,
) -> CompilationResult<u32> {
    let Some(op_val) = operand else {
        debug!("struct_get: no operand, using fallback");
        return get_type_idx_from_attrs(ctx, attrs, None, &module_info.type_idx_by_type)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"));
    };

    // Check if the operand was defined by a ref.cast - if so, use its target type
    if let ValueDef::OpResult(def_op, _) = ctx.value_def(op_val) {
        let op_data = ctx.op(def_op);
        debug!(
            "struct_get: operand defined by {}.{}",
            op_data.dialect, op_data.name
        );
        if wasm_dialect::RefCast::matches(ctx, def_op) {
            return resolve_from_ref_cast(ctx, op, def_op, attrs, module_info);
        }
        // Not a ref_cast, use normal lookup
        let inferred_type = Some(value_type(ctx, op_val));
        return resolve_type_idx_from_inferred(ctx, inferred_type, attrs, module_info, "_closure");
    }

    // Block arg - use normal lookup
    let inferred_type = Some(value_type(ctx, op_val));
    resolve_type_idx_from_inferred(ctx, inferred_type, attrs, module_info, "block arg _closure")
}

/// Helper to resolve type_idx from inferred type
fn resolve_type_idx_from_inferred(
    ctx: &IrContext,
    inferred_type: Option<TypeRef>,
    attrs: &BTreeMap<Symbol, Attribute>,
    module_info: &ModuleInfo,
    closure_debug_label: &str,
) -> CompilationResult<u32> {
    let Some(inferred) = inferred_type else {
        return get_type_idx_from_attrs(ctx, attrs, None, &module_info.type_idx_by_type)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"));
    };

    // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
    if is_closure_struct_type(ctx, inferred) {
        debug!(
            "struct_get: using CLOSURE_STRUCT_IDX for {} type",
            closure_debug_label
        );
        return Ok(CLOSURE_STRUCT_IDX);
    }

    if let Some(&idx) = module_info.type_idx_by_type.get(&inferred) {
        let data = ctx.types.get(inferred);
        debug!(
            "struct_get: using type_idx={} for {}.{}",
            idx, data.dialect, data.name
        );
        return Ok(idx);
    }

    get_type_idx_from_attrs(ctx, attrs, Some(inferred), &module_info.type_idx_by_type)
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
}

/// Resolve type_idx from a ref.cast operation
fn resolve_from_ref_cast(
    ctx: &IrContext,
    struct_get_op: OpRef,
    ref_cast_op: OpRef,
    struct_get_attrs: &BTreeMap<Symbol, Attribute>,
    module_info: &ModuleInfo,
) -> CompilationResult<u32> {
    let def_attrs = &ctx.op(ref_cast_op).attributes;
    if let Some(Attribute::Type(target_ty)) = def_attrs.get(&ATTR_TARGET_TYPE()) {
        // For placeholder types like wasm.structref, we MUST use field_count
        // to distinguish between different concrete types with same abstract type.
        let is_placeholder = helpers::is_type(ctx, *target_ty, "wasm", "structref");

        if is_placeholder {
            // For placeholder types like wasm.structref, try to find the concrete type
            // from struct_get's type attribute first (this has the actual adt.struct/typeref type)
            if let Some(Attribute::Type(struct_ty)) = struct_get_attrs.get(&ATTR_TYPE()) {
                // Try direct lookup with the concrete struct type
                if let Some(&idx) = module_info.type_idx_by_type.get(struct_ty) {
                    let data = ctx.types.get(*struct_ty);
                    debug!(
                        "struct_get: using struct_get type attr for type_idx={} ({}.{})",
                        idx, data.dialect, data.name
                    );
                    return Ok(idx);
                }
            }

            // Use placeholder lookup with field_count as fallback
            let field_count = if let Some(Attribute::Int(fc)) = def_attrs.get(&ATTR_FIELD_COUNT()) {
                debug!("struct_get: ref_cast (placeholder) has field_count={}", *fc);
                usize::try_from(*fc).map_err(|_| {
                    CompilationError::invalid_attribute(format!(
                        "struct_get: field_count value {} out of usize range",
                        fc
                    ))
                })?
            } else if let Some(Attribute::Type(ty)) = struct_get_attrs.get(&ATTR_TYPE()) {
                debug!(
                    "struct_get: ref_cast (placeholder) has NO field_count, inferring from type attr"
                );
                get_struct_field_count(ctx, *ty).ok_or_else(|| {
                    CompilationError::invalid_attribute(
                        "struct_get: placeholder structref requires field_count but type attr has no field info",
                    )
                })?
            } else {
                return Err(CompilationError::missing_attribute(
                    "field_count or type on placeholder structref ref_cast",
                ));
            };
            debug!(
                "struct_get: looking up placeholder ({}.{}, field_count={})",
                ctx.types.get(*target_ty).dialect,
                ctx.types.get(*target_ty).name,
                field_count
            );
            let result = module_info
                .placeholder_struct_type_idx
                .get(&(*target_ty, field_count))
                .copied();
            debug!("struct_get: placeholder lookup result = {:?}", result);
            result.ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
        } else if let Some(&idx) = module_info.type_idx_by_type.get(target_ty) {
            // Non-placeholder concrete type - use direct lookup
            let data = ctx.types.get(*target_ty);
            debug!(
                "struct_get: using ref_cast direct type_idx={} for {}.{}",
                idx, data.dialect, data.name
            );
            Ok(idx)
        } else {
            // Non-placeholder but not found - try placeholder lookup as fallback
            let field_count = if let Some(Attribute::Int(fc)) = def_attrs.get(&ATTR_FIELD_COUNT()) {
                usize::try_from(*fc).map_err(|_| {
                    CompilationError::invalid_attribute(format!(
                        "struct_get: field_count value {} out of usize range",
                        fc
                    ))
                })?
            } else if let Some(Attribute::Type(ty)) = struct_get_attrs.get(&ATTR_TYPE()) {
                get_struct_field_count(ctx, *ty).ok_or_else(|| {
                    CompilationError::invalid_attribute(
                        "struct_get: placeholder structref requires field_count but type attr has no field info",
                    )
                })?
            } else {
                return Err(CompilationError::missing_attribute(
                    "field_count or type on placeholder structref ref_cast",
                ));
            };
            module_info
                .placeholder_struct_type_idx
                .get(&(*target_ty, field_count))
                .copied()
                .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
        }
    } else {
        // No target_type attr on ref_cast, fall back
        debug!("struct_get: ref_cast has NO target_type attribute!");
        let inferred_type = ctx.op_result_types(struct_get_op).first().copied();
        get_type_idx_from_attrs(
            ctx,
            struct_get_attrs,
            inferred_type,
            &module_info.type_idx_by_type,
        )
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
    }
}

/// Get the number of fields in an adt.struct type (arena version).
/// Returns None if the type is not an adt.struct or has no field info.
fn get_struct_field_count(ctx: &IrContext, ty: TypeRef) -> Option<usize> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return None;
    }
    // In arena types, the "fields" attribute stores field information
    match data.attrs.get(&Symbol::new("fields")) {
        Some(Attribute::List(fields)) => Some(fields.len()),
        _ => {
            // Field types are stored as type params in arena adt.struct types
            if !data.params.is_empty() {
                Some(data.params.len())
            } else {
                None
            }
        }
    }
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
    module_info: &ModuleInfo,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = struct_set_op.op_ref();
    let operands = ctx.op_operands(op);
    emit_operands(ctx, operands, emit_ctx, function)?;

    // Infer type from operand[0] (the struct ref)
    let inferred_type = operands.first().map(|v| value_type(ctx, *v));
    let type_idx = get_type_idx_from_attrs(
        ctx,
        &ctx.op(op).attributes,
        inferred_type,
        &module_info.type_idx_by_type,
    )
    .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
    let field_idx = struct_set_op.field_idx(ctx);

    function.instruction(&Instruction::StructSet {
        struct_type_index: type_idx,
        field_index: field_idx,
    });
    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

use super::super::helpers::attr_field_idx;
