//! Call indirect type collection for wasm backend emission.
//!
//! This module handles the collection of function types used in call_indirect
//! operations, ref_func declarations, and related type inference.

use std::collections::{HashMap, HashSet};

use tracing::debug;
use trunk_ir::dialect::{core, func, wasm};
use trunk_ir::{Attribute, BlockId, DialectOp, DialectType, IdVec, Region, Symbol, Type};

use crate::errors::CompilationResult;

use super::helpers::{is_closure_struct_type, is_step_type, value_type};

trunk_ir::symbols! {
    ATTR_FUNC_NAME => "func_name",
}

/// Collect function types used in call_indirect operations.
///
/// This function walks the IR to find all call_indirect operations and registers
/// their function types in the type section. It handles:
/// - Polymorphic function types (anyref params/results)
/// - Result type upgrade (anyref → funcref/Step based on enclosing function)
///
/// Returns a vector of (type_idx, func_type) pairs sorted by type index.
pub(crate) fn collect_call_indirect_types<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    type_idx_by_type: &mut HashMap<Type<'db>, u32>,
    block_arg_types: &HashMap<(BlockId, usize), Type<'db>>,
    gc_type_count: usize,
    func_type_count: usize,
) -> CompilationResult<Vec<(u32, core::Func<'db>)>> {
    fn collect_from_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        type_idx_by_type: &mut HashMap<Type<'db>, u32>,
        next_type_idx: &mut u32,
        new_types: &mut Vec<(u32, core::Func<'db>)>,
        block_arg_types: &HashMap<(BlockId, usize), Type<'db>>,
        enclosing_func_return_ty: Option<Type<'db>>,
    ) -> CompilationResult<()> {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                debug!(
                    "collect_call_indirect_types: visiting op {}.{}, enclosing_func_return_ty={:?}",
                    op.dialect(db),
                    op.name(db),
                    enclosing_func_return_ty.map(|t| {
                        t.dialect(db)
                            .with_str(|d| t.name(db).with_str(|n| format!("{}.{}", d, n)))
                    })
                );
                // Check if this is a function definition to track return type
                // NOTE: In lowered wasm IR, functions are wasm.func, not func.func
                let func_return_ty = if let Ok(wasm_fn) = wasm::Func::from_operation(db, *op) {
                    // Get the function's return type from wasm.func
                    let func_type = wasm_fn.r#type(db);
                    debug!(
                        "collect_call_indirect_types: found wasm.func, type={}.{}",
                        func_type.dialect(db),
                        func_type.name(db)
                    );
                    if let Some(func_ty) = core::Func::from_type(db, func_type) {
                        let ret_ty = func_ty.result(db);
                        debug!(
                            "collect_call_indirect_types: wasm.func return type={}.{}",
                            ret_ty.dialect(db),
                            ret_ty.name(db)
                        );
                        Some(ret_ty)
                    } else {
                        debug!("collect_call_indirect_types: wasm.func r#type is not core.func");
                        None
                    }
                } else if let Ok(func) = func::Func::from_operation(db, *op) {
                    // Also check for func.func (in case IR isn't fully lowered)
                    let func_type = func.r#type(db);
                    core::Func::from_type(db, func_type).map(|func_ty| func_ty.result(db))
                } else {
                    None
                };

                // Use the new function return type if we're entering a function,
                // otherwise keep the enclosing one
                let current_func_return_ty = func_return_ty.or(enclosing_func_return_ty);

                // Recursively process nested regions
                for nested in op.regions(db).iter() {
                    collect_from_region(
                        db,
                        nested,
                        type_idx_by_type,
                        next_type_idx,
                        new_types,
                        block_arg_types,
                        current_func_return_ty,
                    )?;
                }

                // Check if this is a call_indirect
                if wasm::CallIndirect::matches(db, *op) {
                    // Build function type from operands and results
                    let operands = op.operands(db);

                    if operands.is_empty() {
                        continue; // Skip invalid call_indirect
                    }

                    // The callee (i32 table index) is the FIRST operand, followed by args.
                    // All indirect calls use table-based call_indirect.
                    let first_operand = operands.first().copied().unwrap();
                    let first_operand_ty = value_type(db, first_operand, block_arg_types);
                    let callee_is_first = first_operand_ty.is_some_and(|ty| {
                        // First operand should be i32 table index or closure struct
                        // (closure struct's first field is the i32 table index)
                        core::I32::from_type(db, ty).is_some() || is_closure_struct_type(db, ty)
                    });

                    // Helper to normalize IR types to wasm types for call_indirect.
                    // Types that are already anyref (after normalize_primitive_types pass)
                    // should remain anyref in the signature.
                    let anyref_ty = wasm::Anyref::new(db).as_type();
                    let normalize_param_type = |ty: Type<'db>| -> Type<'db> {
                        // After normalize_primitive_types pass:
                        // - tribute_rt.any → wasm.anyref
                        // So we only need to check for wasm.anyref
                        if wasm::Anyref::from_type(db, ty).is_some() {
                            anyref_ty
                        } else {
                            ty
                        }
                    };

                    // Callee (i32 table index) is FIRST operand, params are operands[1..]
                    assert!(
                        callee_is_first,
                        "call_indirect first operand must be i32 table index or closure struct, got {:?}",
                        first_operand_ty.map(|ty| {
                            ty.dialect(db)
                                .with_str(|d| ty.name(db).with_str(|n| format!("{}.{}", d, n)))
                        })
                    );
                    let param_types: IdVec<Type<'db>> = operands
                        .iter()
                        .skip(1)
                        .filter_map(|v| value_type(db, *v, block_arg_types))
                        .map(normalize_param_type)
                        .collect();

                    // Result type - use enclosing function's return type if it's funcref
                    // and the call_indirect has anyref result. This is needed because
                    // WebAssembly GC has separate type hierarchies for anyref and funcref,
                    // so we can't cast between them.
                    let mut result_ty = match op.results(db).first().copied() {
                        Some(ty) => ty,
                        None => continue, // Skip if no result
                    };

                    // If result type is anyref but enclosing function returns funcref,
                    // use funcref as the result type. This is needed because WebAssembly GC has
                    // separate type hierarchies for anyref and funcref - you can't cast between them.
                    let funcref_ty = wasm::Funcref::new(db).as_type();
                    debug!(
                        "collect_call_indirect_types: result_ty={}.{}, enclosing_func_return_ty={:?}",
                        result_ty.dialect(db),
                        result_ty.name(db),
                        enclosing_func_return_ty.map(|t| {
                            t.dialect(db)
                                .with_str(|d| t.name(db).with_str(|n| format!("{}.{}", d, n)))
                        })
                    );
                    if let Some(func_ret_ty) = enclosing_func_return_ty {
                        // Check if result is anyref (polymorphic type)
                        // Note: type variables are resolved at AST level before IR generation
                        let is_anyref_result = wasm::Anyref::from_type(db, result_ty).is_some();
                        let func_returns_funcref = wasm::Funcref::from_type(db, func_ret_ty)
                            .is_some()
                            || core::Func::from_type(db, func_ret_ty).is_some();
                        // Check for Step type (trampoline-based effect system)
                        let func_returns_step = is_step_type(db, func_ret_ty);
                        debug!(
                            "collect_call_indirect_types: is_anyref={}, func_returns_funcref={}, func_returns_step={}",
                            is_anyref_result, func_returns_funcref, func_returns_step
                        );
                        if is_anyref_result && func_returns_funcref {
                            debug!(
                                "collect_call_indirect_types: upgrading polymorphic result to funcref \
                                 for enclosing function that returns funcref"
                            );
                            result_ty = funcref_ty;
                        } else if is_anyref_result && func_returns_step {
                            // When enclosing function returns Step (for trampoline effect system),
                            // upgrade polymorphic call_indirect results to Step too.
                            // This ensures closure/continuation calls return the right type.
                            debug!(
                                "collect_call_indirect_types: upgrading polymorphic result to Step \
                                 for enclosing function that returns Step"
                            );
                            result_ty = crate::gc_types::step_marker_type(db);
                        }
                    }

                    // Normalize result type: anyref stays as anyref for polymorphic dispatch
                    // This must match the normalization done in call_handlers for emit
                    if crate::emit::helpers::should_normalize_to_anyref(db, result_ty) {
                        debug!(
                            "collect_call_indirect_types: normalizing result {} to anyref",
                            result_ty.dialect(db).with_str(|d| result_ty
                                .name(db)
                                .with_str(|n| format!("{}.{}", d, n)))
                        );
                        result_ty = anyref_ty;
                    }

                    // Create function type
                    let func_ty = core::Func::new(db, param_types, result_ty);
                    let func_type = func_ty.as_type();

                    // Register if not already registered, and collect new types
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        type_idx_by_type.entry(func_type)
                    {
                        let idx = *next_type_idx;
                        *next_type_idx += 1;
                        e.insert(idx);
                        new_types.push((idx, func_ty));
                        debug!(
                            "collect_call_indirect_types: registered new func type idx={}, params={:?}, result={}.{}",
                            idx,
                            func_ty
                                .params(db)
                                .iter()
                                .map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
                                .collect::<Vec<_>>(),
                            result_ty.dialect(db),
                            result_ty.name(db)
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // Start with the next available type index (after GC types AND function definition types)
    // GC types are indices 0..gc_type_count
    // Function definition types are indices gc_type_count..gc_type_count+func_type_count
    // call_indirect types should start after that
    let mut next_type_idx = (gc_type_count + func_type_count) as u32;
    let mut new_types = Vec::new();

    collect_from_region(
        db,
        &module.body(db),
        type_idx_by_type,
        &mut next_type_idx,
        &mut new_types,
        block_arg_types,
        None, // No enclosing function at module level
    )?;

    // Sort by type index to ensure they are emitted in order
    new_types.sort_by_key(|(idx, _)| *idx);

    Ok(new_types)
}

/// Collect function names referenced via wasm.ref_func.
///
/// These functions need to be declared in a declarative elem segment.
pub(crate) fn collect_ref_funcs<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> HashSet<Symbol> {
    fn collect_from_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        ref_funcs: &mut HashSet<Symbol>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                // Recursively process nested regions
                for nested in op.regions(db).iter() {
                    collect_from_region(db, nested, ref_funcs);
                }

                // Check if this is a ref_func
                if wasm::RefFunc::matches(db, *op)
                    && let Some(Attribute::Symbol(func_name)) =
                        op.attributes(db).get(&ATTR_FUNC_NAME())
                {
                    ref_funcs.insert(*func_name);
                }
            }
        }
    }

    let mut ref_funcs = HashSet::new();
    collect_from_region(db, &module.body(db), &mut ref_funcs);
    ref_funcs
}

/// Check if the module contains any call_indirect operations.
pub(crate) fn has_call_indirect<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> bool {
    fn check_region<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> bool {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                // Check nested regions first
                for nested in op.regions(db).iter() {
                    if check_region(db, nested) {
                        return true;
                    }
                }

                // Check if this is a call_indirect
                if wasm::CallIndirect::matches(db, *op) {
                    return true;
                }
            }
        }
        false
    }

    check_region(db, &module.body(db))
}
