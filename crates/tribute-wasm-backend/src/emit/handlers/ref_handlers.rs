//! Reference operation handlers for wasm backend.
//!
//! This module handles WebAssembly reference operations like ref.null, ref.func,
//! ref.cast, and ref.test.

use crate::{CompilationError, CompilationResult};
use trunk_ir::dialect::wasm;
use trunk_ir::{Attribute, DialectType, Operation, Symbol};
use wasm_encoder::{AbstractHeapType, Function, HeapType, Instruction};

use super::super::{
    ATTR_HEAP_TYPE, ATTR_TARGET_TYPE, FunctionEmitContext, ModuleInfo, attr_heap_type,
    emit_operands, get_type_idx_from_attrs, set_result_local,
};

/// Handle ref.null operation
pub(crate) fn handle_ref_null<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let attrs = op.attributes(db);
    // Infer type from result type
    let inferred_type = op.results(db).first().copied();

    // Try heap_type attribute first. If present but invalid, propagate error.
    let heap_type = if attrs.get(&ATTR_HEAP_TYPE()).is_some() {
        attr_heap_type(db, attrs, ATTR_HEAP_TYPE())?
    } else {
        // Attribute not present - fall back to type inference
        get_type_idx_from_attrs(db, attrs, inferred_type, &module_info.type_idx_by_type)
            .map(HeapType::Concrete)
            .ok_or_else(|| CompilationError::missing_attribute("heap_type or type"))?
    };

    function.instruction(&Instruction::RefNull(heap_type));
    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle ref.func operation
pub(crate) fn handle_ref_func<'db>(
    db: &'db dyn salsa::Database,
    ref_func_op: wasm::RefFunc<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let func_name = ref_func_op.func_name(db);
    let func_idx = resolve_callee(func_name, module_info)?;
    function.instruction(&Instruction::RefFunc(func_idx));
    set_result_local(db, &ref_func_op.operation(), ctx, function)?;
    Ok(())
}

/// Handle ref.cast operation
pub(crate) fn handle_ref_cast<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    let attrs = op.attributes(db);
    // Infer type from result type (the target type it casts to)
    let inferred_type = op.results(db).first().copied();

    // Check if this is a placeholder struct type (wasm.structref with field_count)
    // If so, use the concrete type index from the placeholder map
    let heap_type = if let Some(Attribute::Type(target_ty)) = attrs.get(&ATTR_TARGET_TYPE()) {
        if wasm::Structref::from_type(db, *target_ty).is_some() {
            // structref placeholder - try to find concrete type via type attribute first
            // The `type` attribute contains the actual adt.struct/typeref type
            if let Some(Attribute::Type(concrete_ty)) = attrs.get(&Symbol::new("type")) {
                if let Some(&type_idx) = module_info.type_idx_by_type.get(concrete_ty) {
                    tracing::debug!(
                        "ref_cast: using type attr for concrete type_idx={} ({}.{})",
                        type_idx,
                        concrete_ty.dialect(db),
                        concrete_ty.name(db)
                    );
                    HeapType::Concrete(type_idx)
                } else if let Some(Attribute::IntBits(fc)) = attrs.get(&Symbol::new("field_count"))
                {
                    // Fall back to placeholder lookup
                    if let Some(&type_idx) = module_info
                        .placeholder_struct_type_idx
                        .get(&(*target_ty, *fc as usize))
                    {
                        tracing::debug!(
                            "ref_cast: found placeholder type_idx={} for field_count={}",
                            type_idx,
                            fc
                        );
                        HeapType::Concrete(type_idx)
                    } else {
                        tracing::debug!(
                            "ref_cast: placeholder lookup FAILED for field_count={}, falling back to abstract structref",
                            fc
                        );
                        HeapType::Abstract {
                            shared: false,
                            ty: AbstractHeapType::Struct,
                        }
                    }
                } else {
                    HeapType::Abstract {
                        shared: false,
                        ty: AbstractHeapType::Struct,
                    }
                }
            } else if let Some(Attribute::IntBits(fc)) = attrs.get(&Symbol::new("field_count")) {
                if let Some(&type_idx) = module_info
                    .placeholder_struct_type_idx
                    .get(&(*target_ty, *fc as usize))
                {
                    tracing::debug!(
                        "ref_cast: found placeholder type_idx={} for field_count={}",
                        type_idx,
                        fc
                    );
                    HeapType::Concrete(type_idx)
                } else {
                    tracing::debug!(
                        "ref_cast: placeholder lookup FAILED for field_count={}, falling back to abstract structref",
                        fc
                    );
                    // Fallback to abstract structref if not found
                    HeapType::Abstract {
                        shared: false,
                        ty: AbstractHeapType::Struct,
                    }
                }
            } else {
                // No field_count - fall back to abstract structref
                HeapType::Abstract {
                    shared: false,
                    ty: AbstractHeapType::Struct,
                }
            }
        } else {
            // Non-placeholder type - try attr_heap_type first, then type lookup
            attr_heap_type(db, attrs, ATTR_TARGET_TYPE())
                .ok()
                .or_else(|| {
                    // Try to look up non-wasm types (adt.struct, tribute_rt.any, etc.) in type_idx_by_type
                    module_info
                        .type_idx_by_type
                        .get(target_ty)
                        .map(|&idx| HeapType::Concrete(idx))
                })
                .or_else(|| {
                    // Fall back to using inferred type
                    get_type_idx_from_attrs(db, attrs, inferred_type, &module_info.type_idx_by_type)
                        .map(HeapType::Concrete)
                })
                .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?
        }
    } else {
        attr_heap_type(db, attrs, ATTR_TARGET_TYPE())
            .ok()
            .or_else(|| {
                get_type_idx_from_attrs(db, attrs, inferred_type, &module_info.type_idx_by_type)
                    .map(HeapType::Concrete)
            })
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?
    };

    tracing::debug!(
        "ref_cast: emitting RefCastNullable with heap_type={:?}",
        heap_type
    );
    function.instruction(&Instruction::RefCastNullable(heap_type));
    set_result_local(db, op, ctx, function)?;
    Ok(())
}

/// Handle ref.test operation
pub(crate) fn handle_ref_test<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    ctx: &FunctionEmitContext<'db>,
    module_info: &ModuleInfo<'db>,
    function: &mut Function,
) -> CompilationResult<()> {
    let operands = op.operands(db);
    emit_operands(db, operands, ctx, &module_info.block_arg_types, function)?;

    let attrs = op.attributes(db);
    // ref_test result is i32, target type must be in attribute (can't infer)
    // Try target_type attribute first. If present but invalid, propagate error.
    let heap_type = if attrs.get(&ATTR_TARGET_TYPE()).is_some() {
        attr_heap_type(db, attrs, ATTR_TARGET_TYPE())?
    } else {
        // Attribute not present - fall back to type inference
        get_type_idx_from_attrs(db, attrs, None, &module_info.type_idx_by_type)
            .map(HeapType::Concrete)
            .ok_or_else(|| CompilationError::missing_attribute("target_type or type"))?
    };

    function.instruction(&Instruction::RefTestNullable(heap_type));
    set_result_local(db, op, ctx, function)?;
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
