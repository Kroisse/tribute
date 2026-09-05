//! Call indirect type collection for wasm backend emission.
//!
//! This module handles the collection of function types used in call_indirect
//! operations, ref_func declarations, and related type inference.

use std::collections::{HashMap, HashSet};

use tracing::debug;
use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::func;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::op_interface::IndirectCallLikeOps;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{RegionRef, TypeRef};
use trunk_ir::smallvec::SmallVec;
use trunk_ir::types::TypeData;

use crate::errors::{CompilationError, CompilationResult};

use super::helpers::{self, intern_named_adt_struct};

/// Intern a func.func_sig type from params and result type.
fn intern_func_type(ctx: &mut IrContext, params: &[TypeRef], result_ty: TypeRef) -> TypeRef {
    func::func_sig(ctx, params.iter().copied(), [result_ty]).as_type_ref()
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
                let func_return_ty = if let Ok(wasm_fn) = wasm_dialect::Func::from_op(ctx, op) {
                    // Get the function's return type from wasm.func
                    let func_type = wasm_fn.r#type(ctx);
                    debug!(
                        "collect_call_indirect_types: found wasm.func, type={}",
                        fmt_type(ctx, func_type)
                    );
                    let (_, ret_ty) =
                        helpers::func_type_parts(ctx, func_type).ok_or_else(|| {
                            CompilationError::type_error(
                                "Wasm indirect-call collection requires valid one-result func.func_sig",
                            )
                        })?;
                    Some(ret_ty)
                } else if let Ok(func) = func::Func::from_op(ctx, op) {
                    let (_, ret_ty) =
                        helpers::func_type_parts(ctx, func.r#type(ctx)).ok_or_else(|| {
                            CompilationError::type_error(
                                "Wasm indirect-call collection requires valid one-result func.func_sig",
                            )
                        })?;
                    Some(ret_ty)
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

                // `return_call_indirect` carries the exact physical callable
                // contract from the shared ABI pass. Validate and register it
                // before handling ordinary calls, whose historical path still
                // derives a signature from their result-producing operation.
                if wasm_dialect::ReturnCallIndirect::matches(ctx, op) {
                    let func_type = helpers::exact_return_call_indirect_signature(ctx, op)?;
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        type_idx_by_type.entry(func_type)
                    {
                        let index = *next_type_idx;
                        *next_type_idx += 1;
                        entry.insert(index);
                        new_types.push((index, func_type));
                    }
                    continue;
                }

                // Check if this is a result-producing call_indirect.
                if wasm_dialect::CallIndirect::matches(ctx, op) {
                    // When source lowering retained an exact contract, use it
                    // as the sole type-section key. It must not be recreated
                    // from the erased table index and operands.
                    if IndirectCallLikeOps::exact_signature(ctx, op).is_some() {
                        let func_type = helpers::exact_call_indirect_signature(ctx, op)?;
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            type_idx_by_type.entry(func_type)
                        {
                            let index = *next_type_idx;
                            *next_type_idx += 1;
                            entry.insert(index);
                            new_types.push((index, func_type));
                        }
                        continue;
                    }

                    // Build function type from the interface's callee and
                    // argument accessors when no exact contract was retained.
                    let Some(first_operand) = IndirectCallLikeOps::callee(ctx, op) else {
                        continue;
                    };
                    // The callee (i32 table index) is the FIRST operand, followed by args.
                    // All indirect calls use table-based call_indirect.
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
                    let Some(args) = IndirectCallLikeOps::arguments(ctx, op) else {
                        continue; // Skip invalid call_indirect
                    };

                    // Callee (i32 table index) is FIRST operand, params are operands[1..]
                    assert!(
                        callee_is_first,
                        "call_indirect first operand must be i32 table index or closure struct, got {:?}",
                        fmt_type(ctx, first_operand_ty)
                    );
                    let param_types: Vec<TypeRef> = args
                        .iter()
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
                                || helpers::is_type(ctx, func_ret_ty, "func", "func_sig");
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
    let mut staged_indices = type_idx_by_type.clone();

    let body = module.body(ctx).unwrap();
    collect_from_region(
        ctx,
        body,
        &mut staged_indices,
        &mut next_type_idx,
        &mut new_types,
        None, // No enclosing function at module level
    )?;

    // Sort by type index to ensure they are emitted in order
    new_types.sort_by_key(|(idx, _)| *idx);

    *type_idx_by_type = staged_indices;
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
                if let Ok(ref_func_op) = wasm_dialect::RefFunc::from_op(ctx, op) {
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

/// Check if the module contains any table-based indirect transfer.
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

                if wasm_dialect::CallIndirect::matches(ctx, op)
                    || wasm_dialect::ReturnCallIndirect::matches(ctx, op)
                {
                    return true;
                }
            }
        }
        false
    }

    let body = module.body(ctx).unwrap();
    check_region(ctx, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_function_type_uses_input_and_result_accessors() {
        let mut ctx = IrContext::new();
        let result = intern_simple_wasm_type(&mut ctx, "i32");
        let first = intern_simple_wasm_type(&mut ctx, "i64");
        let second = intern_simple_wasm_type(&mut ctx, "f32");

        let func_ty = intern_func_type(&mut ctx, &[first, second], result);

        assert_eq!(
            helpers::func_type_parts(&ctx, func_ty),
            Some((&[first, second][..], result))
        );
    }
}

#[cfg(test)]
mod result_list_tests {
    use super::*;
    #[test]
    fn review_late_signature_error_preserves_seeded_indices() {
        for dialect in ["wasm", "func"] {
            let mut ctx = IrContext::new();
            let text = format!(
                "core.module @m {{
                wasm.func {{sym_name = @good, type = func.func_sig<() -> core.i32>}} {{
                    %callee = wasm.i32_const {{value = 0}} : core.i32
                    %value = wasm.call_indirect %callee : core.i32
                    wasm.return %value
                }}
                {dialect}.func {{sym_name = @bad, type = func.func_sig<() -> ()>}} {{}}
            }}"
            );
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &text);
            let seed = func::func_sig(&mut ctx, [], []).as_type_ref();
            let mut indices = HashMap::from([(seed, 7)]);
            let before = indices.clone();
            let error =
                collect_call_indirect_types(&mut ctx, module, &mut indices, 8, 0).unwrap_err();
            assert!(error.to_string().contains("one-result func.func_sig"));
            assert_eq!(indices, before);
            let bad = module.ops(&ctx)[1];
            trunk_ir::rewrite::erase_op(&mut ctx, bad);
            let added = collect_call_indirect_types(&mut ctx, module, &mut indices, 8, 0).unwrap();
            assert_eq!(
                added.len(),
                1,
                "the earlier call must register a new signature"
            );
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[&seed], 7);
        }
    }

    #[test]
    fn unsupported_resultless_function_is_rejected_before_collection() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            "core.module @m { wasm.func {sym_name = @f, type = func.func_sig<() -> ()>} { wasm.return } }",
        );
        let before = trunk_ir::printer::print_module(&ctx, module.op());
        let mut indices = HashMap::new();
        let error = collect_call_indirect_types(&mut ctx, module, &mut indices, 0, 0).unwrap_err();
        assert!(error.to_string().contains("one-result func.func_sig"));
        assert!(indices.is_empty());
        assert_eq!(trunk_ir::printer::print_module(&ctx, module.op()), before);
    }
}
