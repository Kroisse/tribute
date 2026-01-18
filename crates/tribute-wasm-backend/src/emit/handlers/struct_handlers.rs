//! Struct operation handlers for wasm backend.
//!
//! This module handles WebAssembly GC struct operations (struct.new, struct.get, struct.set).

use tracing::debug;
use tribute_ir::dialect::{adt, tribute};
use trunk_ir::dialect::{cont, wasm};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, ValueDef};
use wasm_encoder::{Function, HeapType, Instruction, StorageType, ValType};

use crate::gc_types::{ATTR_TYPE, ATTR_TYPE_IDX, CLOSURE_STRUCT_IDX, GcTypeDef};
use crate::{CompilationError, CompilationResult};

use super::super::{
    ATTR_TARGET_TYPE, FunctionEmitContext, ModuleInfo, attr_field_idx, emit_operands,
    get_type_idx_from_attrs, is_closure_struct_type, set_result_local, value_type,
};

/// Handle struct.new operation
pub(crate) fn handle_struct_new<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = op.operands(db);
    // struct_new needs all field values on the stack, including nil types.
    // emit_operands handles nil types by emitting ref.null none.
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    let attrs = op.attributes(db);
    let field_count = operands.len();
    let result_type = op.results(db).first().copied();

    // Priority: explicit type_idx attr > type attr > placeholder result type > inferred result type
    // type_idx attribute takes highest precedence (set by wasm_gc_type_assign pass)
    let type_idx = if let Some(Attribute::IntBits(idx)) = attrs.get(&ATTR_TYPE_IDX()) {
        Some(*idx as u32)
    } else if let Some(Attribute::Type(ty)) = attrs.get(&ATTR_TYPE()) {
        if wasm::Structref::from_type(db, *ty).is_some() {
            // Use placeholder map for wasm.structref
            // All (type, field_count) pairs are registered by collect_gc_types upfront
            module_info
                .placeholder_struct_type_idx
                .get(&(*ty, field_count))
                .copied()
        } else if is_closure_struct_type(db, *ty) {
            // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
            Some(CLOSURE_STRUCT_IDX)
        } else {
            // Regular type
            module_info.type_idx_by_type.get(ty).copied()
        }
    } else if let Some(ty) = result_type
        && wasm::Structref::from_type(db, ty).is_some()
    {
        // Result type is a placeholder (wasm.structref) - use placeholder map
        module_info
            .placeholder_struct_type_idx
            .get(&(ty, field_count))
            .copied()
    } else if let Some(ty) = result_type {
        // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
        if is_closure_struct_type(db, ty) {
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
    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle struct.get operation
pub(crate) fn handle_struct_get<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;
    let attrs = op.attributes(db);

    // Check if operand is abstract type (anyref/structref/continuation) and needs casting to concrete struct type
    // This happens when:
    // - A closure is captured (stored as anyref) and later used
    // - A continuation field is typed as structref but needs access to concrete struct fields
    // - cont::Continuation type is stored as structref in wasm but accessed with concrete struct ops
    let operand_abstract_type = operands.first().and_then(|op_val| {
        let ty = value_type(db, *op_val, &module_info.block_arg_types)?;
        if wasm::Anyref::from_type(db, ty).is_some() {
            Some("anyref")
        } else if wasm::Structref::from_type(db, ty).is_some() {
            Some("structref")
        } else if cont::Continuation::from_type(db, ty).is_some() {
            // cont::Continuation is stored as structref in wasm
            Some("cont.continuation")
        } else {
            None
        }
    });
    let needs_cast_from_abstract = operand_abstract_type.is_some();

    // CRITICAL: For struct_get, the type_idx MUST match the operand's actual type.
    // We need to trace through ref.cast operations to find the actual type,
    // because the IR type might be different from the wasm type after casting.
    let operand = operands.first().copied();
    let type_idx = resolve_struct_get_type_idx(db, op, operand, attrs, module_info)?;

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
    let needs_boxing = check_struct_get_needs_boxing(db, op, ctx, module_info, type_idx, field_idx);

    if needs_boxing {
        debug!("struct_get: boxing i64 to i31ref for anyref local");
        function.instruction(&Instruction::I32WrapI64);
        function.instruction(&Instruction::RefI31);
    }

    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Resolve the type_idx for struct.get operation by tracing through ref.cast operations
fn resolve_struct_get_type_idx<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    operand: Option<trunk_ir::Value<'db>>,
    attrs: &trunk_ir::Attrs<'db>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<u32> {
    let Some(op_val) = operand else {
        debug!("struct_get: no operand, using fallback");
        return get_type_idx_from_attrs(db, attrs, None, &module_info.type_idx_by_type)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"));
    };

    // Check if the operand was defined by a ref.cast - if so, use its target type
    if let ValueDef::OpResult(def_op) = op_val.def(db) {
        debug!(
            "struct_get: operand defined by {}.{} at index {}",
            def_op.dialect(db),
            def_op.name(db),
            op_val.index(db)
        );
        if wasm::RefCast::matches(db, def_op) {
            return resolve_from_ref_cast(db, op, def_op, attrs, module_info);
        }
        // Not a ref_cast, use normal lookup
        let inferred_type = value_type(db, op_val, &module_info.block_arg_types);
        return resolve_type_idx_from_inferred(db, inferred_type, attrs, module_info, "_closure");
    }

    // Block arg - use normal lookup
    let inferred_type = value_type(db, op_val, &module_info.block_arg_types);
    resolve_type_idx_from_inferred(db, inferred_type, attrs, module_info, "block arg _closure")
}

/// Helper to resolve type_idx from inferred type
fn resolve_type_idx_from_inferred<'db>(
    db: &'db dyn salsa::Database,
    inferred_type: Option<trunk_ir::Type<'db>>,
    attrs: &trunk_ir::Attrs<'db>,
    module_info: &ModuleInfo<'db>,
    closure_debug_label: &str,
) -> CompilationResult<u32> {
    let Some(inferred) = inferred_type else {
        return get_type_idx_from_attrs(db, attrs, None, &module_info.type_idx_by_type)
            .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"));
    };

    // Special case: _closure struct type uses builtin CLOSURE_STRUCT_IDX
    if is_closure_struct_type(db, inferred) {
        debug!(
            "struct_get: using CLOSURE_STRUCT_IDX for {} type",
            closure_debug_label
        );
        return Ok(CLOSURE_STRUCT_IDX);
    }

    if let Some(&idx) = module_info.type_idx_by_type.get(&inferred) {
        debug!(
            "struct_get: using type_idx={} for {}.{}",
            idx,
            inferred.dialect(db),
            inferred.name(db)
        );
        return Ok(idx);
    }

    get_type_idx_from_attrs(db, attrs, Some(inferred), &module_info.type_idx_by_type)
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
}

/// Resolve type_idx from a ref.cast operation
fn resolve_from_ref_cast<'db>(
    db: &'db dyn salsa::Database,
    struct_get_op: &Operation<'db>,
    ref_cast_op: Operation<'db>,
    struct_get_attrs: &trunk_ir::Attrs<'db>,
    module_info: &ModuleInfo<'db>,
) -> CompilationResult<u32> {
    let def_attrs = ref_cast_op.attributes(db);
    if let Some(Attribute::Type(target_ty)) = def_attrs.get(&ATTR_TARGET_TYPE()) {
        // For placeholder types like wasm.structref, we MUST use field_count
        // to distinguish between different concrete types with same abstract type.
        let is_placeholder = wasm::Structref::from_type(db, *target_ty).is_some();

        if is_placeholder {
            // For placeholder types like wasm.structref, try to find the concrete type
            // from struct_get's type attribute first (this has the actual adt.struct/typeref type)
            if let Some(Attribute::Type(struct_ty)) = struct_get_attrs.get(&ATTR_TYPE()) {
                // Try direct lookup with the concrete struct type
                if let Some(&idx) = module_info.type_idx_by_type.get(struct_ty) {
                    debug!(
                        "struct_get: using struct_get type attr for type_idx={} ({}.{})",
                        idx,
                        struct_ty.dialect(db),
                        struct_ty.name(db)
                    );
                    return Ok(idx);
                }
            }

            // Use placeholder lookup with field_count as fallback
            let field_count =
                if let Some(Attribute::IntBits(fc)) = def_attrs.get(&Symbol::new("field_count")) {
                    debug!("struct_get: ref_cast (placeholder) has field_count={}", *fc);
                    *fc as usize
                } else {
                    debug!("struct_get: ref_cast (placeholder) has NO field_count!");
                    // Last resort - use struct_get's type attr
                    if let Some(Attribute::Type(ty)) = struct_get_attrs.get(&ATTR_TYPE()) {
                        adt::get_struct_fields(db, *ty)
                            .map(|f| f.len())
                            .unwrap_or(0)
                    } else {
                        0
                    }
                };
            debug!(
                "struct_get: looking up placeholder ({}.{}, field_count={})",
                target_ty.dialect(db),
                target_ty.name(db),
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
            debug!(
                "struct_get: using ref_cast direct type_idx={} for {}.{}",
                idx,
                target_ty.dialect(db),
                target_ty.name(db)
            );
            Ok(idx)
        } else {
            // Non-placeholder but not found - try placeholder lookup as fallback
            let field_count =
                if let Some(Attribute::IntBits(fc)) = def_attrs.get(&Symbol::new("field_count")) {
                    *fc as usize
                } else if let Some(Attribute::Type(ty)) = struct_get_attrs.get(&ATTR_TYPE()) {
                    adt::get_struct_fields(db, *ty)
                        .map(|f| f.len())
                        .unwrap_or(0)
                } else {
                    0
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
        let inferred_type = struct_get_op.results(db).first().copied();
        get_type_idx_from_attrs(
            db,
            struct_get_attrs,
            inferred_type,
            &module_info.type_idx_by_type,
        )
        .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))
    }
}

/// Check if struct.get result needs boxing (i64 to i31ref)
fn check_struct_get_needs_boxing<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    type_idx: u32,
    field_idx: u32,
) -> bool {
    if op.results(db).is_empty() {
        return false;
    }

    let result_value = op.result(db, 0);

    // Check if the result local would be anyref by examining the effective type
    let local_type = ctx
        .effective_types
        .get(&result_value)
        .copied()
        .or_else(|| op.results(db).first().copied());

    let expects_anyref = local_type
        .map(|ty| wasm::Anyref::from_type(db, ty).is_some() || tribute::is_type_var(db, ty))
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
pub(crate) fn handle_struct_set<'db>(
    db: &'db dyn salsa::Database,
    struct_set_op: wasm::StructSet<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let op = struct_set_op.operation();
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    // Infer type from operand[0] (the struct ref)
    let inferred_type = operands
        .first()
        .and_then(|v| value_type(db, *v, &module_info.block_arg_types));
    let type_idx = get_type_idx_from_attrs(
        db,
        op.attributes(db),
        inferred_type,
        &module_info.type_idx_by_type,
    )
    .ok_or_else(|| CompilationError::missing_attribute("type or type_idx"))?;
    let field_idx = struct_set_op.field_idx(db);

    function.instruction(&Instruction::StructSet {
        struct_type_index: type_idx,
        field_index: field_idx,
    });
    Ok(())
}
