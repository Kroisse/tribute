//! Call indirect type collection for wasm backend emission.
//!
//! This module handles the collection of function types used in call_indirect
//! operations, ref_func declarations, and related type inference.

use std::collections::{BTreeMap, HashMap, HashSet};

use tracing::debug;
use trunk_ir::Symbol;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::Module;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::refs::{RegionRef, TypeRef};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeData};
use trunk_ir::smallvec::SmallVec;

use crate::errors::CompilationResult;

use super::helpers;

/// Intern a core.func type from params and result type.
fn intern_func_type(ctx: &mut IrContext, params: &[TypeRef], result_ty: TypeRef) -> TypeRef {
    let mut all_params: SmallVec<[TypeRef; 4]> = params.into();
    all_params.push(result_ty); // last param is the return type
    ctx.types.intern(TypeData {
        dialect: Symbol::new("core"),
        name: Symbol::new("func"),
        params: all_params,
        attrs: Default::default(),
    })
}

/// Intern a simple wasm type with no params or attrs.
fn intern_simple_wasm_type(ctx: &mut IrContext, name: &str) -> TypeRef {
    ctx.types.intern(TypeData {
        dialect: Symbol::new("wasm"),
        name: Symbol::from_dynamic(name),
        params: Default::default(),
        attrs: Default::default(),
    })
}

/// Intern an adt.struct type with the given name attribute.
fn intern_named_adt_struct(ctx: &mut IrContext, name: &'static str) -> TypeRef {
    let mut attrs = BTreeMap::new();
    attrs.insert(
        Symbol::new("name"),
        ArenaAttribute::Symbol(Symbol::new(name)),
    );
    ctx.types.intern(TypeData {
        dialect: Symbol::new("adt"),
        name: Symbol::new("struct"),
        params: Default::default(),
        attrs,
    })
}

/// Format a TypeRef as "dialect.name" for debug logging.
fn fmt_type(ctx: &IrContext, ty: TypeRef) -> String {
    let data = ctx.types.get(ty);
    format!("{}.{}", data.dialect, data.name)
}

/// Collect function types used in call_indirect operations.
///
/// This function walks the IR to find all call_indirect operations and registers
/// their function types in the type section. It handles:
/// - Polymorphic function types (anyref params/results)
/// - Result type upgrade (anyref -> funcref/Step based on enclosing function)
///
/// Returns a vector of (type_idx, func_type) pairs sorted by type index.
pub(crate) fn collect_call_indirect_types(
    ctx: &mut IrContext,
    module: Module,
    type_idx_by_type: &mut HashMap<TypeRef, u32>,
    gc_type_count: usize,
    func_type_count: usize,
) -> CompilationResult<Vec<(u32, TypeRef)>> {
    fn collect_from_region(
        ctx: &mut IrContext,
        region_ref: RegionRef,
        type_idx_by_type: &mut HashMap<TypeRef, u32>,
        next_type_idx: &mut u32,
        new_types: &mut Vec<(u32, TypeRef)>,
        enclosing_func_return_ty: Option<TypeRef>,
    ) -> CompilationResult<()> {
        let blocks: SmallVec<[_; 4]> = ctx.region(region_ref).blocks.clone();
        for &block_ref in &blocks {
            let ops: SmallVec<[_; 4]> = ctx.block(block_ref).ops.clone();
            for &op in &ops {
                {
                    let op_data = ctx.op(op);
                    debug!(
                        "collect_call_indirect_types: visiting op {}.{}, enclosing_func_return_ty={:?}",
                        op_data.dialect,
                        op_data.name,
                        enclosing_func_return_ty.map(|t| fmt_type(ctx, t))
                    );
                }
                // Check if this is a function definition to track return type
                // NOTE: In lowered wasm IR, functions are wasm.func, not func.func
                let func_return_ty = if let Ok(wasm_fn) = arena_wasm::Func::from_op(ctx, op) {
                    // Get the function's return type from wasm.func
                    let func_type = wasm_fn.r#type(ctx);
                    debug!(
                        "collect_call_indirect_types: found wasm.func, type={}",
                        fmt_type(ctx, func_type)
                    );
                    if let Some((_, ret_ty)) = helpers::func_type_parts(ctx, func_type) {
                        debug!(
                            "collect_call_indirect_types: wasm.func return type={}",
                            fmt_type(ctx, ret_ty)
                        );
                        Some(ret_ty)
                    } else {
                        debug!("collect_call_indirect_types: wasm.func r#type is not core.func");
                        None
                    }
                } else if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                    // Also check for func.func (in case IR isn't fully lowered)
                    let func_type = func.r#type(ctx);
                    helpers::func_type_parts(ctx, func_type).map(|(_, ret_ty)| ret_ty)
                } else {
                    None
                };

                // Use the new function return type if we're entering a function,
                // otherwise keep the enclosing one
                let current_func_return_ty = func_return_ty.or(enclosing_func_return_ty);

                // Recursively process nested regions
                let regions: SmallVec<[_; 4]> = ctx.op(op).regions.clone();
                for &nested in &regions {
                    collect_from_region(
                        ctx,
                        nested,
                        type_idx_by_type,
                        next_type_idx,
                        new_types,
                        current_func_return_ty,
                    )?;
                }

                // Check if this is a call_indirect
                if arena_wasm::CallIndirect::matches(ctx, op) {
                    // Build function type from operands and results
                    let operands: Vec<_> = ctx.op_operands(op).to_vec();

                    if operands.is_empty() {
                        continue; // Skip invalid call_indirect
                    }

                    // The callee (i32 table index) is the FIRST operand, followed by args.
                    // All indirect calls use table-based call_indirect.
                    let first_operand = operands[0];
                    let first_operand_ty = helpers::value_type(ctx, first_operand);
                    let callee_is_first = {
                        // First operand should be i32 table index or closure struct
                        // (closure struct's first field is the i32 table index)
                        helpers::is_type(ctx, first_operand_ty, "core", "i32")
                            || helpers::is_closure_struct_type(ctx, first_operand_ty)
                    };

                    // Normalize IR types to wasm types for call_indirect.
                    // Types that are already anyref (after normalize_primitive_types pass)
                    // should remain anyref in the signature.
                    let anyref_ty = intern_simple_wasm_type(ctx, "anyref");

                    // Callee (i32 table index) is FIRST operand, params are operands[1..]
                    assert!(
                        callee_is_first,
                        "call_indirect first operand must be i32 table index or closure struct, got {:?}",
                        fmt_type(ctx, first_operand_ty)
                    );
                    let param_types: Vec<TypeRef> = operands
                        .iter()
                        .skip(1)
                        .map(|v| {
                            let ty = helpers::value_type(ctx, *v);
                            // After normalize_primitive_types pass:
                            // - tribute_rt.any -> wasm.anyref
                            // So we only need to check for wasm.anyref
                            if helpers::is_type(ctx, ty, "wasm", "anyref") {
                                anyref_ty
                            } else {
                                ty
                            }
                        })
                        .collect();

                    // Result type - use enclosing function's return type if it's funcref
                    // and the call_indirect has anyref result. This is needed because
                    // WebAssembly GC has separate type hierarchies for anyref and funcref,
                    // so we can't cast between them.
                    let result_types: Vec<_> = ctx.op_result_types(op).to_vec();
                    let mut result_ty = match result_types.first().copied() {
                        Some(ty) => ty,
                        None => continue, // Skip if no result
                    };

                    // If result type is anyref but enclosing function returns funcref,
                    // use funcref as the result type. This is needed because WebAssembly GC has
                    // separate type hierarchies for anyref and funcref - you can't cast between them.
                    let funcref_ty = intern_simple_wasm_type(ctx, "funcref");
                    debug!(
                        "collect_call_indirect_types: result_ty={}, enclosing_func_return_ty={:?}",
                        fmt_type(ctx, result_ty),
                        enclosing_func_return_ty.map(|t| fmt_type(ctx, t))
                    );
                    if let Some(func_ret_ty) = enclosing_func_return_ty {
                        // Check if result is anyref (polymorphic type)
                        // Note: type variables are resolved at AST level before IR generation
                        let is_anyref_result = helpers::is_type(ctx, result_ty, "wasm", "anyref");
                        let func_returns_funcref =
                            helpers::is_type(ctx, func_ret_ty, "wasm", "funcref")
                                || helpers::is_type(ctx, func_ret_ty, "core", "func");
                        // Check for Step type (trampoline-based effect system)
                        let func_returns_step = helpers::is_step_type(ctx, func_ret_ty);
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
                            result_ty = intern_named_adt_struct(ctx, "_Step");
                        }
                    }

                    // Normalize result type: anyref stays as anyref for polymorphic dispatch
                    // This must match the normalization done in call_handlers for emit
                    if helpers::should_normalize_to_anyref(ctx, result_ty) {
                        debug!(
                            "collect_call_indirect_types: normalizing result {} to anyref",
                            fmt_type(ctx, result_ty)
                        );
                        result_ty = anyref_ty;
                    }

                    // Create function type
                    let func_type = intern_func_type(ctx, &param_types, result_ty);

                    // Register if not already registered, and collect new types
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        type_idx_by_type.entry(func_type)
                    {
                        let idx = *next_type_idx;
                        *next_type_idx += 1;
                        e.insert(idx);
                        new_types.push((idx, func_type));
                        debug!(
                            "collect_call_indirect_types: registered new func type idx={}, params={:?}, result={}",
                            idx,
                            param_types
                                .iter()
                                .map(|t| fmt_type(ctx, *t))
                                .collect::<Vec<_>>(),
                            fmt_type(ctx, result_ty)
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

    let body = module.body(ctx).unwrap();
    collect_from_region(
        ctx,
        body,
        type_idx_by_type,
        &mut next_type_idx,
        &mut new_types,
        None, // No enclosing function at module level
    )?;

    // Sort by type index to ensure they are emitted in order
    new_types.sort_by_key(|(idx, _)| *idx);

    Ok(new_types)
}

/// Collect function names referenced via wasm.ref_func.
///
/// These functions need to be declared in a declarative elem segment.
pub(crate) fn collect_ref_funcs(ctx: &IrContext, module: Module) -> HashSet<Symbol> {
    fn collect_from_region(
        ctx: &IrContext,
        region_ref: RegionRef,
        ref_funcs: &mut HashSet<Symbol>,
    ) {
        for &block_ref in &ctx.region(region_ref).blocks {
            for &op in &ctx.block(block_ref).ops {
                // Recursively process nested regions
                for &nested in &ctx.op(op).regions {
                    collect_from_region(ctx, nested, ref_funcs);
                }

                // Check if this is a ref_func
                if let Ok(ref_func_op) = arena_wasm::RefFunc::from_op(ctx, op) {
                    ref_funcs.insert(ref_func_op.func_name(ctx));
                }
            }
        }
    }

    let mut ref_funcs = HashSet::new();
    let body = module.body(ctx).unwrap();
    collect_from_region(ctx, body, &mut ref_funcs);
    ref_funcs
}

/// Check if the module contains any call_indirect operations.
pub(crate) fn has_call_indirect(ctx: &IrContext, module: Module) -> bool {
    fn check_region(ctx: &IrContext, region_ref: RegionRef) -> bool {
        for &block_ref in &ctx.region(region_ref).blocks {
            for &op in &ctx.block(block_ref).ops {
                // Check nested regions first
                for &nested in &ctx.op(op).regions {
                    if check_region(ctx, nested) {
                        return true;
                    }
                }

                // Check if this is a call_indirect
                if arena_wasm::CallIndirect::matches(ctx, op) {
                    return true;
                }
            }
        }
        false
    }

    let body = module.body(ctx).unwrap();
    check_region(ctx, body)
}
