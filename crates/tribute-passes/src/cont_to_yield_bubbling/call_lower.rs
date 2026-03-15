//! Expand effectful calls into Done/Shift branches with continuation chaining.
//!
//! For each effectful call in an effectful function's body:
//! - Done path: unwrap value, continue with remaining code
//! - Shift path: chain continuation with remaining code
//!
//! This resolves #336 (sequential effectful calls).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use trunk_ir::IrMapping;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location};

use super::ResumeCounter;
use super::shift_lower::fresh_resume_name;
use super::types::{YieldBubblingTypes, adt_struct_type, is_yield_result_type};

/// Specification for generating a chain function.
pub(crate) struct ChainFuncSpec {
    /// Chain function name (also used as self-reference for re-chaining)
    pub(crate) name: String,
    /// State type: [inner_cont(anyref), live_var_0(anyref), ...]
    pub(crate) state_type: TypeRef,
    /// State field descriptors
    #[allow(dead_code)]
    pub(crate) state_fields: Vec<(Symbol, TypeRef)>,
    /// Original types of live variables (for casting from anyref)
    pub(crate) original_live_types: Vec<TypeRef>,
    /// Original values of live variables (for remapping in chain function)
    pub(crate) original_live_values: Vec<ValueRef>,
    /// Block args from the original function (evidence param, etc.) for remapping
    pub(crate) original_block_args: Vec<ValueRef>,
    /// The original call result value (to remap)
    pub(crate) call_result_value: ValueRef,
    /// The original call result type (before YieldResult conversion)
    pub(crate) call_result_type: TypeRef,
    /// Snapshot of remaining op data for the chain function body
    pub(crate) remaining_op_snapshots: Vec<OpSnapshot>,
    /// Location
    pub(crate) location: Location,
    /// Effectful function names (for nested call detection)
    #[allow(dead_code)]
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    /// Module name
    #[allow(dead_code)]
    pub(crate) module_name: Symbol,
}

/// Snapshot of an operation's data (captured before modification).
pub(crate) struct OpSnapshot {
    pub(crate) location: Location,
    pub(crate) dialect: Symbol,
    pub(crate) name: Symbol,
    pub(crate) operands: Vec<ValueRef>,
    /// Original result ValueRefs — used to map old→new results when cloning
    pub(crate) result_values: Vec<ValueRef>,
    pub(crate) result_types: Vec<TypeRef>,
    pub(crate) attributes: Vec<(Symbol, Attribute)>,
    pub(crate) regions: Vec<RegionRef>,
}

pub(crate) type ChainSpecs = Rc<RefCell<Vec<ChainFuncSpec>>>;

/// Shared context for call lowering, grouping arguments that are passed
/// through the entire call chain unchanged.
pub(crate) struct CallLowerCtx<'a> {
    pub(crate) effectful_funcs: &'a Rc<HashSet<Symbol>>,
    pub(crate) types: &'a YieldBubblingTypes,
    pub(crate) chain_specs: &'a ChainSpecs,
    pub(crate) chain_counter: &'a ResumeCounter,
    pub(crate) module_name: Symbol,
    pub(crate) module_body: RegionRef,
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Expand effectful calls in all effectful functions.
///
/// This runs AFTER pattern matching (so effectful calls return YieldResult)
/// and BEFORE truncation (so the full function body is available).
pub(crate) fn lower_effectful_calls(
    ctx: &mut IrContext,
    module: Module,
    effectful_funcs: &Rc<HashSet<Symbol>>,
    types: &YieldBubblingTypes,
    chain_specs: &ChainSpecs,
    chain_counter: &ResumeCounter,
    module_name: Symbol,
) {
    let module_body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };
    let lc = CallLowerCtx {
        effectful_funcs,
        types,
        chain_specs,
        chain_counter,
        module_name,
        module_body,
    };
    lower_effectful_calls_impl(ctx, module, &lc, None);
}

/// Expand effectful calls only in the specified target functions.
///
/// Used for post-processing newly generated resume/chain functions.
pub(crate) fn lower_effectful_calls_for_funcs(
    ctx: &mut IrContext,
    module: Module,
    lc: &CallLowerCtx<'_>,
    target_funcs: &[Symbol],
) {
    lower_effectful_calls_impl(ctx, module, lc, Some(target_funcs));
}

/// Shared implementation: expand effectful calls in effectful functions.
///
/// When `target_funcs` is `Some`, only processes the specified functions.
/// When `None`, processes all effectful functions in the module.
fn lower_effectful_calls_impl(
    ctx: &mut IrContext,
    _module: Module,
    lower_ctx: &CallLowerCtx<'_>,
    target_funcs: Option<&[Symbol]>,
) {
    let blocks: Vec<BlockRef> = ctx.region(lower_ctx.module_body).blocks.to_vec();
    for block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            let Ok(func) = arena_func::Func::from_op(ctx, op) else {
                continue;
            };
            let func_name = func.sym_name(ctx);
            if !lower_ctx.effectful_funcs.contains(&func_name) {
                continue;
            }
            if target_funcs.is_some_and(|targets| !targets.contains(&func_name)) {
                continue;
            }
            let body = func.body(ctx);
            let entry_block = ctx.region(body).blocks[0];
            let func_entry_args = ctx.block_args(entry_block).to_vec();
            expand_calls_in_region(ctx, body, lower_ctx, &func_entry_args);
        }
    }
}

fn expand_calls_in_region(
    ctx: &mut IrContext,
    region: RegionRef,
    lower_ctx: &CallLowerCtx<'_>,
    func_entry_block_args: &[ValueRef],
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        expand_first_effectful_call(ctx, block, lower_ctx, func_entry_block_args);
    }
}

// ============================================================================
// Call Expansion
// ============================================================================

/// Expand the first effectful call in a block into Done/Shift branches.
fn expand_first_effectful_call(
    ctx: &mut IrContext,
    block: BlockRef,
    lower_ctx: &CallLowerCtx<'_>,
    func_entry_block_args: &[ValueRef],
) -> bool {
    let lc = lower_ctx;
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    // Find first effectful call
    let mut call_index = None;
    for (i, &op) in ops.iter().enumerate() {
        if is_effectful_call(ctx, op, lc.effectful_funcs) {
            call_index = Some(i);
            break;
        }
    }

    let call_index = match call_index {
        Some(i) => i,
        None => return false,
    };

    let call_op = ops[call_index];
    let call_results = ctx.op_results(call_op);
    if call_results.is_empty() {
        return false;
    }
    let call_result = call_results[0];
    let location = ctx.op(call_op).location;

    // Skip if call result is consumed by an scf.loop (handler dispatch).
    // The handler dispatch loop needs the raw YieldResult to do Done/Shift branching itself.
    if call_index + 1 < ops.len() {
        let next_op = ops[call_index + 1];
        if arena_scf::Loop::from_op(ctx, next_op).is_ok() {
            let loop_operands = ctx.op_operands(next_op);
            if !loop_operands.is_empty() && loop_operands[0] == call_result {
                return false;
            }
        }
    }

    // Get original result type from callee function signature
    let original_result_ty =
        get_callee_original_result_type(ctx, call_op, lc.effectful_funcs, lc.module_body)
            .unwrap_or(lc.types.anyref);

    let remaining_ops: Vec<OpRef> = ops[call_index + 1..].to_vec();

    // If the only remaining op is func.return, the function already returns
    // YieldResult directly — no Done/Shift expansion needed.
    if remaining_ops.len() == 1 && arena_func::Return::from_op(ctx, remaining_ops[0]).is_ok() {
        return false;
    }

    // Compute live variables at call point
    let used_in_remaining = collect_used_values(ctx, &remaining_ops);
    let mut live_vars: Vec<(ValueRef, TypeRef)> = Vec::new();
    let mut seen_values: HashSet<ValueRef> = HashSet::new();

    // Function entry block args (evidence param, etc.) — may be referenced
    // from nested Done branches even though they're not in the current block.
    for &arg in func_entry_block_args {
        if used_in_remaining.contains(&arg) && seen_values.insert(arg) {
            live_vars.push((arg, ctx.value_ty(arg)));
        }
    }
    // Current block args (may overlap with func entry args at top level)
    for &arg in ctx.block_args(block) {
        if used_in_remaining.contains(&arg) && seen_values.insert(arg) {
            live_vars.push((arg, ctx.value_ty(arg)));
        }
    }
    // Op results before the call
    for &op in &ops[..call_index] {
        for &v in ctx.op_results(op) {
            if used_in_remaining.contains(&v) {
                live_vars.push((v, ctx.value_ty(v)));
            }
        }
    }

    // Snapshot remaining ops BEFORE any modification
    let snapshots: Vec<OpSnapshot> = remaining_ops
        .iter()
        .map(|&op| snapshot_op(ctx, op))
        .collect();

    // Generate chain function
    let chain_name = fresh_resume_name(lc.chain_counter);
    let chain_name_sym = Symbol::from_dynamic(&chain_name);

    // Build chain state type: [inner_cont, live_var_0, live_var_1, ...]
    let mut state_fields: Vec<(Symbol, TypeRef)> = Vec::new();
    state_fields.push((Symbol::new("inner_cont"), lc.types.anyref));
    for (i, _) in live_vars.iter().enumerate() {
        state_fields.push((
            Symbol::from_dynamic(&format!("field_{}", i)),
            lc.types.anyref,
        ));
    }
    let state_name_str = format!(
        "__ChainState_{}",
        chain_name.trim_start_matches("__yb_resume_")
    );
    let state_name = Symbol::from_dynamic(&state_name_str);
    let state_type = adt_struct_type(ctx, state_name, &state_fields);

    // Record chain spec
    lc.chain_specs.borrow_mut().push(ChainFuncSpec {
        name: chain_name.clone(),
        state_type,
        state_fields: state_fields.clone(),
        original_live_types: live_vars.iter().map(|(_, ty)| *ty).collect(),
        original_live_values: live_vars.iter().map(|(v, _)| *v).collect(),
        original_block_args: func_entry_block_args.to_vec(),
        call_result_value: call_result,
        call_result_type: original_result_ty,
        remaining_op_snapshots: snapshots,
        location,
        effectful_funcs: Rc::clone(lc.effectful_funcs),
        module_name: lc.module_name,
    });

    // Remove remaining ops from block
    for &op in &remaining_ops {
        ctx.remove_op_from_block(block, op);
    }

    // Build Done branch: unwrap value, move remaining ops in
    let done_branch = build_inline_done_branch(
        ctx,
        location,
        call_result,
        original_result_ty,
        &remaining_ops,
        lc,
        func_entry_block_args,
    );

    // Build Shift branch: chain continuation
    let shift_branch = build_inline_shift_branch(
        ctx,
        location,
        call_result,
        &live_vars,
        chain_name_sym,
        state_type,
        lc.types,
    );

    // Add is_done check
    let is_done = arena_adt::variant_is(
        ctx,
        location,
        call_result,
        lc.types.i1,
        lc.types.yield_result,
        Symbol::new("Done"),
    );
    ctx.push_op(block, is_done.op_ref());

    // Add scf.if
    let if_op = arena_scf::r#if(
        ctx,
        location,
        is_done.result(ctx),
        lc.types.yield_result,
        done_branch,
        shift_branch,
    );
    ctx.push_op(block, if_op.op_ref());

    // Add func.return
    let ret_op = arena_func::r#return(ctx, location, [if_op.result(ctx)]);
    ctx.push_op(block, ret_op.op_ref());

    true
}

// ============================================================================
// Done Branch
// ============================================================================

/// Build the Done branch of an effectful call expansion.
///
/// Unwraps the YieldResult::Done value, remaps the call result,
/// and executes remaining operations.
fn build_inline_done_branch(
    ctx: &mut IrContext,
    location: Location,
    call_result: ValueRef,
    original_result_ty: TypeRef,
    remaining_ops: &[OpRef],
    lower_ctx: &CallLowerCtx<'_>,
    func_entry_block_args: &[ValueRef],
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    let t = lower_ctx.types;

    // Extract value from YieldResult::Done
    let get_val = arena_adt::variant_get(
        ctx,
        location,
        call_result,
        t.anyref,
        t.yield_result,
        Symbol::new("Done"),
        0,
    );
    ctx.push_op(block, get_val.op_ref());

    // Cast to original type
    let unwrapped = if original_result_ty != t.anyref {
        let cast = arena_core::unrealized_conversion_cast(
            ctx,
            location,
            get_val.result(ctx),
            original_result_ty,
        );
        ctx.push_op(block, cast.op_ref());
        cast.result(ctx)
    } else {
        get_val.result(ctx)
    };

    // Move remaining ops into this block with value remapping
    let mut remap: HashMap<ValueRef, ValueRef> = HashMap::new();
    remap.insert(call_result, unwrapped);

    let mut last_value: Option<ValueRef> = None;

    for &op in remaining_ops {
        // Handle func.return → scf.yield(Done(value))
        if arena_func::Return::from_op(ctx, op).is_ok() {
            if let Some(&ret_val) = ctx.op_operands(op).first() {
                let remapped = resolve_remap(&remap, ret_val);
                yield_value_or_done(ctx, block, location, remapped, t);
            }
            continue;
        }

        // Remap operands
        let old_operands = ctx.op_operands(op).to_vec();
        let remapped_operands: Vec<ValueRef> = old_operands
            .iter()
            .map(|&v| resolve_remap(&remap, v))
            .collect();

        let needs_rebuild = remapped_operands != old_operands;

        if needs_rebuild {
            let op_data = ctx.op(op);
            let op_location = op_data.location;
            let op_dialect = op_data.dialect;
            let op_name = op_data.name;
            let op_attrs: Vec<(Symbol, Attribute)> = op_data
                .attributes
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            let op_regions: Vec<RegionRef> = op_data.regions.to_vec();
            let result_types = ctx.op_result_types(op).to_vec();

            let cloned_regions: Vec<RegionRef> = op_regions
                .iter()
                .map(|&r| {
                    let mut mapping = IrMapping::new();
                    for (&old_v, &new_v) in &remap {
                        mapping.map_value(old_v, new_v);
                    }
                    ctx.clone_region(r, &mut mapping)
                })
                .collect();

            let mut builder = OperationDataBuilder::new(op_location, op_dialect, op_name)
                .operands(remapped_operands)
                .results(result_types);
            for (k, v) in op_attrs {
                builder = builder.attr(k, v);
            }
            for r in cloned_regions {
                builder = builder.region(r);
            }
            let new_data = builder.build(ctx);
            let new_op = ctx.create_op(new_data);

            let old_results = ctx.op_results(op).to_vec();
            let new_results = ctx.op_results(new_op).to_vec();
            for (old_r, new_r) in old_results.iter().zip(new_results.iter()) {
                if old_r != new_r {
                    remap.insert(*old_r, *new_r);
                }
            }

            ctx.push_op(block, new_op);
            if !new_results.is_empty() {
                last_value = Some(new_results[0]);
            }
        } else {
            ctx.push_op(block, op);
            let results = ctx.op_results(op);
            if !results.is_empty() {
                last_value = Some(results[0]);
            }
        }
    }

    // If no func.return was found, wrap the last value as Done
    let has_yield = {
        let ops = ctx.block(block).ops.to_vec();
        ops.last()
            .map(|&op| arena_scf::Yield::from_op(ctx, op).is_ok())
            .unwrap_or(false)
    };

    if !has_yield && let Some(val) = last_value {
        yield_value_or_done(ctx, block, location, val, t);
    }

    let region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    });

    // Recursively expand effectful calls in this Done branch
    expand_calls_in_region(ctx, region, lower_ctx, func_entry_block_args);

    region
}

// ============================================================================
// Shift Branch
// ============================================================================

/// Build the Shift branch of an effectful call expansion.
///
/// Extracts ShiftInfo, captures live variables into chain state,
/// creates a chained continuation, and returns YieldResult::Shift.
fn build_inline_shift_branch(
    ctx: &mut IrContext,
    location: Location,
    call_result: ValueRef,
    live_vars: &[(ValueRef, TypeRef)],
    chain_name: Symbol,
    state_type: TypeRef,
    types: &YieldBubblingTypes,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract ShiftInfo from YieldResult::Shift
    let get_info = arena_adt::variant_get(
        ctx,
        location,
        call_result,
        types.shift_info,
        types.yield_result,
        Symbol::new("Shift"),
        0,
    );
    ctx.push_op(block, get_info.op_ref());
    let shift_info = get_info.result(ctx);

    // Extract inner continuation (field 3 of ShiftInfo)
    let get_inner_cont =
        arena_adt::struct_get(ctx, location, shift_info, types.anyref, types.shift_info, 3);
    ctx.push_op(block, get_inner_cont.op_ref());
    let inner_cont = get_inner_cont.result(ctx);

    // Build chain state: [inner_cont, live_var_0, live_var_1, ...]
    let mut state_values: Vec<ValueRef> = vec![inner_cont];
    for &(val, _ty) in live_vars {
        let anyref_val = arena_core::unrealized_conversion_cast(ctx, location, val, types.anyref);
        ctx.push_op(block, anyref_val.op_ref());
        state_values.push(anyref_val.result(ctx));
    }

    let state_op = arena_adt::struct_new(ctx, location, state_values, types.anyref, state_type);
    ctx.push_op(block, state_op.op_ref());

    // Cast state to anyref
    let state_anyref =
        arena_core::unrealized_conversion_cast(ctx, location, state_op.result(ctx), types.anyref);
    ctx.push_op(block, state_anyref.op_ref());

    // Create chained Continuation { resume_fn: chain_fn, state: state_anyref }
    let chain_fn_const = arena_func::constant(ctx, location, types.ptr, chain_name);
    ctx.push_op(block, chain_fn_const.op_ref());

    let chained_cont = arena_adt::struct_new(
        ctx,
        location,
        vec![chain_fn_const.result(ctx), state_anyref.result(ctx)],
        types.anyref,
        types.continuation,
    );
    ctx.push_op(block, chained_cont.op_ref());

    // Cast chained continuation to anyref
    let chained_anyref = arena_core::unrealized_conversion_cast(
        ctx,
        location,
        chained_cont.result(ctx),
        types.anyref,
    );
    ctx.push_op(block, chained_anyref.op_ref());

    // Build new ShiftInfo: same value/prompt/op_idx but with chained continuation
    let get_sv =
        arena_adt::struct_get(ctx, location, shift_info, types.anyref, types.shift_info, 0);
    ctx.push_op(block, get_sv.op_ref());
    let get_prompt =
        arena_adt::struct_get(ctx, location, shift_info, types.i32, types.shift_info, 1);
    ctx.push_op(block, get_prompt.op_ref());
    let get_op_idx =
        arena_adt::struct_get(ctx, location, shift_info, types.i32, types.shift_info, 2);
    ctx.push_op(block, get_op_idx.op_ref());

    let new_info = arena_adt::struct_new(
        ctx,
        location,
        vec![
            get_sv.result(ctx),
            get_prompt.result(ctx),
            get_op_idx.result(ctx),
            chained_anyref.result(ctx),
        ],
        types.shift_info,
        types.shift_info,
    );
    ctx.push_op(block, new_info.op_ref());

    // Build YieldResult::Shift with new info
    let yr = arena_adt::variant_new(
        ctx,
        location,
        [new_info.result(ctx)],
        types.yield_result,
        types.yield_result,
        Symbol::new("Shift"),
    );
    ctx.push_op(block, yr.op_ref());

    let yield_op = arena_scf::r#yield(ctx, location, [yr.result(ctx)]);
    ctx.push_op(block, yield_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

// ============================================================================
// Chain Function Generation
// ============================================================================

/// Generate a chain function from a spec.
///
/// Chain function signature: `(Evidence, anyref) -> YieldResult`
///
/// The chain function:
/// 1. Extracts inner continuation and live vars from state
/// 2. Resumes inner continuation with resume value
/// 3. If Done: extract value, run remaining ops, return Done
/// 4. If Shift: re-chain (self-reference), return Shift
pub(crate) fn create_chain_function(
    ctx: &mut IrContext,
    spec: &ChainFuncSpec,
    types: &YieldBubblingTypes,
) -> OpRef {
    use trunk_ir::types::TypeDataBuilder;

    let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);
    let location = spec.location;
    let name = Symbol::from_dynamic(&spec.name);

    // Function type: (evidence, anyref) -> YieldResult
    // Use Layout A: params[0] = return type, params[1..] = parameter types
    let func_ty_ref = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(types.yield_result)
            .param(evidence_ty)
            .param(types.anyref)
            .build(),
    );

    let body_block = ctx.create_block(BlockData {
        location,
        args: vec![
            trunk_ir::context::BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            },
            trunk_ir::context::BlockArgData {
                ty: types.anyref,
                attrs: Default::default(),
            },
        ],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    let ev_arg = ctx.block_args(body_block)[0];
    let wrapper_arg = ctx.block_args(body_block)[1];

    // === Prologue: extract state ===

    // Cast wrapper to ResumeWrapper
    let wrapper_cast =
        arena_core::unrealized_conversion_cast(ctx, location, wrapper_arg, types.resume_wrapper);
    ctx.push_op(body_block, wrapper_cast.op_ref());

    // Extract resume_value (field 1)
    let get_rv = arena_adt::struct_get(
        ctx,
        location,
        wrapper_cast.result(ctx),
        types.anyref,
        types.resume_wrapper,
        1,
    );
    ctx.push_op(body_block, get_rv.op_ref());
    let resume_value = get_rv.result(ctx);

    // Extract state (field 0)
    let get_state = arena_adt::struct_get(
        ctx,
        location,
        wrapper_cast.result(ctx),
        types.anyref,
        types.resume_wrapper,
        0,
    );
    ctx.push_op(body_block, get_state.op_ref());

    // Cast state to chain state type
    let state_cast = arena_core::unrealized_conversion_cast(
        ctx,
        location,
        get_state.result(ctx),
        spec.state_type,
    );
    ctx.push_op(body_block, state_cast.op_ref());
    let state_val = state_cast.result(ctx);

    // Extract inner_cont (field 0 of chain state)
    let get_inner_cont =
        arena_adt::struct_get(ctx, location, state_val, types.anyref, spec.state_type, 0);
    ctx.push_op(body_block, get_inner_cont.op_ref());
    let inner_cont_anyref = get_inner_cont.result(ctx);

    // Extract live variables (fields 1..N)
    let mut restored_live_vars: Vec<ValueRef> = Vec::new();
    let mut value_map: HashMap<ValueRef, ValueRef> = HashMap::new();

    for (i, (original_val, original_ty)) in spec
        .original_live_values
        .iter()
        .zip(spec.original_live_types.iter())
        .enumerate()
    {
        let get_field = arena_adt::struct_get(
            ctx,
            location,
            state_val,
            types.anyref,
            spec.state_type,
            (i + 1) as u32, // +1 because field 0 is inner_cont
        );
        ctx.push_op(body_block, get_field.op_ref());

        let cast = arena_core::unrealized_conversion_cast(
            ctx,
            location,
            get_field.result(ctx),
            *original_ty,
        );
        ctx.push_op(body_block, cast.op_ref());

        value_map.insert(*original_val, cast.result(ctx));
        restored_live_vars.push(cast.result(ctx));
    }

    // === Resume inner continuation ===

    // Cast inner_cont to Continuation struct
    let inner_cont_cast = arena_core::unrealized_conversion_cast(
        ctx,
        location,
        inner_cont_anyref,
        types.continuation,
    );
    ctx.push_op(body_block, inner_cont_cast.op_ref());

    // Extract resume_fn (field 0) — typed as core.ptr (not RC-managed)
    let get_fn = arena_adt::struct_get(
        ctx,
        location,
        inner_cont_cast.result(ctx),
        types.ptr,
        types.continuation,
        0,
    );
    ctx.push_op(body_block, get_fn.op_ref());

    // Extract inner state (field 1)
    let get_inner_state = arena_adt::struct_get(
        ctx,
        location,
        inner_cont_cast.result(ctx),
        types.anyref,
        types.continuation,
        1,
    );
    ctx.push_op(body_block, get_inner_state.op_ref());

    // Build inner ResumeWrapper { state: inner_state, resume_value: rv }
    let inner_wrapper = arena_adt::struct_new(
        ctx,
        location,
        vec![get_inner_state.result(ctx), resume_value],
        types.anyref,
        types.resume_wrapper,
    );
    ctx.push_op(body_block, inner_wrapper.op_ref());

    // Cast to anyref for call_indirect
    let inner_wrapper_anyref = arena_core::unrealized_conversion_cast(
        ctx,
        location,
        inner_wrapper.result(ctx),
        types.anyref,
    );
    ctx.push_op(body_block, inner_wrapper_anyref.op_ref());

    // Call indirect: resume inner continuation
    let inner_yr = arena_func::call_indirect(
        ctx,
        location,
        get_fn.result(ctx),
        vec![ev_arg, inner_wrapper_anyref.result(ctx)],
        types.yield_result,
    );
    ctx.push_op(body_block, inner_yr.op_ref());
    let inner_yr_val = inner_yr.result(ctx);

    // === Check Done/Shift ===

    let is_done = arena_adt::variant_is(
        ctx,
        location,
        inner_yr_val,
        types.i1,
        types.yield_result,
        Symbol::new("Done"),
    );
    ctx.push_op(body_block, is_done.op_ref());

    // === Done branch: extract value, run remaining ops ===
    let done_branch =
        build_chain_done_branch(ctx, location, inner_yr_val, spec, types, &value_map, ev_arg);

    // === Shift branch: re-chain ===
    let shift_branch = build_chain_shift_branch(
        ctx,
        location,
        inner_yr_val,
        &restored_live_vars,
        spec,
        types,
    );

    let if_op = arena_scf::r#if(
        ctx,
        location,
        is_done.result(ctx),
        types.yield_result,
        done_branch,
        shift_branch,
    );
    ctx.push_op(body_block, if_op.op_ref());

    // Return
    let ret_op = arena_func::r#return(ctx, location, [if_op.result(ctx)]);
    ctx.push_op(body_block, ret_op.op_ref());

    let body_region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![body_block],
        parent_op: None,
    });

    let func_data = OperationDataBuilder::new(location, Symbol::new("func"), Symbol::new("func"))
        .attr("sym_name", Attribute::Symbol(name))
        .attr("type", Attribute::Type(func_ty_ref))
        .region(body_region)
        .build(ctx);

    ctx.create_op(func_data)
}

/// Build the Done branch of a chain function.
///
/// Extracts the inner result value and executes remaining ops with remapping.
fn build_chain_done_branch(
    ctx: &mut IrContext,
    location: Location,
    inner_yr: ValueRef,
    spec: &ChainFuncSpec,
    types: &YieldBubblingTypes,
    base_remap: &HashMap<ValueRef, ValueRef>,
    chain_ev_arg: ValueRef,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract value from inner YieldResult::Done
    let get_val = arena_adt::variant_get(
        ctx,
        location,
        inner_yr,
        types.anyref,
        types.yield_result,
        Symbol::new("Done"),
        0,
    );
    ctx.push_op(block, get_val.op_ref());

    // Cast to original call result type
    let unwrapped = if spec.call_result_type != types.anyref {
        let cast = arena_core::unrealized_conversion_cast(
            ctx,
            location,
            get_val.result(ctx),
            spec.call_result_type,
        );
        ctx.push_op(block, cast.op_ref());
        cast.result(ctx)
    } else {
        get_val.result(ctx)
    };

    // Build value map: base_remap + call_result → unwrapped + block_arg mappings
    let mut remap = base_remap.clone();
    remap.insert(spec.call_result_value, unwrapped);

    // Map original block args → chain function's evidence arg
    // The first block arg is typically the evidence parameter.
    // Block args that are live vars are already in base_remap, but
    // they might also be used directly in snapshot operands.
    for &block_arg in &spec.original_block_args {
        // Default: map to evidence arg (first block arg is typically evidence)
        remap.entry(block_arg).or_insert(chain_ev_arg);
    }

    // Clone remaining ops from snapshots
    let mut last_value: Option<ValueRef> = None;

    for snap in &spec.remaining_op_snapshots {
        // Handle func.return → scf.yield(Done(value))
        if snap.dialect == Symbol::new("func") && snap.name == Symbol::new("return") {
            if let Some(&ret_val) = snap.operands.first() {
                let remapped = resolve_remap(&remap, ret_val);
                yield_value_or_done(ctx, block, location, remapped, types);
            }
            continue;
        }

        // Clone the op with remapping
        let remapped_operands: Vec<ValueRef> = snap
            .operands
            .iter()
            .map(|&v| resolve_remap(&remap, v))
            .collect();

        // Deep-clone nested regions to avoid "region already belongs" panic
        let cloned_regions: Vec<RegionRef> = snap
            .regions
            .iter()
            .map(|&r| {
                let mut mapping = IrMapping::new();
                for (&old_v, &new_v) in &remap {
                    mapping.map_value(old_v, new_v);
                }
                ctx.clone_region(r, &mut mapping)
            })
            .collect();

        let mut builder = OperationDataBuilder::new(snap.location, snap.dialect, snap.name)
            .operands(remapped_operands)
            .results(snap.result_types.clone());
        for (k, v) in &snap.attributes {
            builder = builder.attr(*k, v.clone());
        }
        for r in cloned_regions {
            builder = builder.region(r);
        }
        let new_data = builder.build(ctx);
        let new_op = ctx.create_op(new_data);

        ctx.push_op(block, new_op);
        let new_results = ctx.op_results(new_op).to_vec();

        // Map old result ValueRefs → new result ValueRefs for subsequent ops
        for (old_r, new_r) in snap.result_values.iter().zip(new_results.iter()) {
            if old_r != new_r {
                remap.insert(*old_r, *new_r);
            }
        }

        if !new_results.is_empty() {
            last_value = Some(new_results[0]);
        }
    }

    // Ensure we have a yield
    let has_yield = {
        let ops = ctx.block(block).ops.to_vec();
        ops.last()
            .map(|&op| arena_scf::Yield::from_op(ctx, op).is_ok())
            .unwrap_or(false)
    };

    if !has_yield && let Some(val) = last_value {
        yield_value_or_done(ctx, block, location, val, types);
    }

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build the Shift branch of a chain function (re-chain with self).
fn build_chain_shift_branch(
    ctx: &mut IrContext,
    location: Location,
    inner_yr: ValueRef,
    restored_live_vars: &[ValueRef],
    spec: &ChainFuncSpec,
    types: &YieldBubblingTypes,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract new ShiftInfo from inner YieldResult::Shift
    let get_info = arena_adt::variant_get(
        ctx,
        location,
        inner_yr,
        types.shift_info,
        types.yield_result,
        Symbol::new("Shift"),
        0,
    );
    ctx.push_op(block, get_info.op_ref());
    let new_shift_info = get_info.result(ctx);

    // Extract new inner continuation (field 3)
    let get_new_inner = arena_adt::struct_get(
        ctx,
        location,
        new_shift_info,
        types.anyref,
        types.shift_info,
        3,
    );
    ctx.push_op(block, get_new_inner.op_ref());
    let new_inner_cont = get_new_inner.result(ctx);

    // Build new chain state: [new_inner_cont, live_var_0, live_var_1, ...]
    let mut new_state_values: Vec<ValueRef> = vec![new_inner_cont];
    for &lv in restored_live_vars {
        let anyref_val = arena_core::unrealized_conversion_cast(ctx, location, lv, types.anyref);
        ctx.push_op(block, anyref_val.op_ref());
        new_state_values.push(anyref_val.result(ctx));
    }

    let new_state = arena_adt::struct_new(
        ctx,
        location,
        new_state_values,
        types.anyref,
        spec.state_type,
    );
    ctx.push_op(block, new_state.op_ref());

    let new_state_anyref =
        arena_core::unrealized_conversion_cast(ctx, location, new_state.result(ctx), types.anyref);
    ctx.push_op(block, new_state_anyref.op_ref());

    // Self-reference: func.constant @__chain_N
    let self_name = Symbol::from_dynamic(&spec.name);
    let chain_fn = arena_func::constant(ctx, location, types.ptr, self_name);
    ctx.push_op(block, chain_fn.op_ref());

    // Build new Continuation
    let new_cont = arena_adt::struct_new(
        ctx,
        location,
        vec![chain_fn.result(ctx), new_state_anyref.result(ctx)],
        types.anyref,
        types.continuation,
    );
    ctx.push_op(block, new_cont.op_ref());

    let new_cont_anyref =
        arena_core::unrealized_conversion_cast(ctx, location, new_cont.result(ctx), types.anyref);
    ctx.push_op(block, new_cont_anyref.op_ref());

    // Build new ShiftInfo: copy value/prompt/op_idx, replace continuation
    let get_sv = arena_adt::struct_get(
        ctx,
        location,
        new_shift_info,
        types.anyref,
        types.shift_info,
        0,
    );
    ctx.push_op(block, get_sv.op_ref());
    let get_prompt = arena_adt::struct_get(
        ctx,
        location,
        new_shift_info,
        types.i32,
        types.shift_info,
        1,
    );
    ctx.push_op(block, get_prompt.op_ref());
    let get_op_idx = arena_adt::struct_get(
        ctx,
        location,
        new_shift_info,
        types.i32,
        types.shift_info,
        2,
    );
    ctx.push_op(block, get_op_idx.op_ref());

    let rebuilt_info = arena_adt::struct_new(
        ctx,
        location,
        vec![
            get_sv.result(ctx),
            get_prompt.result(ctx),
            get_op_idx.result(ctx),
            new_cont_anyref.result(ctx),
        ],
        types.shift_info,
        types.shift_info,
    );
    ctx.push_op(block, rebuilt_info.op_ref());

    // YieldResult::Shift
    let yr = arena_adt::variant_new(
        ctx,
        location,
        [rebuilt_info.result(ctx)],
        types.yield_result,
        types.yield_result,
        Symbol::new("Shift"),
    );
    ctx.push_op(block, yr.op_ref());

    let yield_op = arena_scf::r#yield(ctx, location, [yr.result(ctx)]);
    ctx.push_op(block, yield_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Check if an operation is a call to an effectful function.
fn is_effectful_call(ctx: &IrContext, op: OpRef, effectful_funcs: &HashSet<Symbol>) -> bool {
    if let Ok(call) = arena_func::Call::from_op(ctx, op) {
        effectful_funcs.contains(&call.callee(ctx)) && !ctx.op_results(op).is_empty()
    } else {
        false
    }
}

/// Get the original result type of a callee function (before YieldResult conversion).
pub(crate) fn get_callee_original_result_type(
    ctx: &IrContext,
    call_op: OpRef,
    effectful_funcs: &HashSet<Symbol>,
    module_body: RegionRef,
) -> Option<TypeRef> {
    let Ok(call) = arena_func::Call::from_op(ctx, call_op) else {
        return None;
    };
    let callee = call.callee(ctx);
    if !effectful_funcs.contains(&callee) {
        return None;
    }

    super::truncate::find_original_result_type(ctx, module_body, callee)
}

/// Snapshot an operation's data.
fn snapshot_op(ctx: &IrContext, op: OpRef) -> OpSnapshot {
    let data = ctx.op(op);
    OpSnapshot {
        location: data.location,
        dialect: data.dialect,
        name: data.name,
        operands: ctx.op_operands(op).to_vec(),
        result_values: ctx.op_results(op).to_vec(),
        result_types: ctx.op_result_types(op).to_vec(),
        attributes: data
            .attributes
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect(),
        regions: data.regions.to_vec(),
    }
}

/// Collect all values used as operands in a set of operations (including nested regions).
fn collect_used_values(ctx: &IrContext, ops: &[OpRef]) -> HashSet<ValueRef> {
    use std::ops::ControlFlow;
    use trunk_ir::walk;

    let mut used = HashSet::new();
    for &op in ops {
        for &operand in ctx.op_operands(op) {
            used.insert(operand);
        }
        for &region in &ctx.op(op).regions {
            let _ = walk::walk_region::<()>(ctx, region, &mut |nested_op| {
                for &operand in ctx.op_operands(nested_op) {
                    used.insert(operand);
                }
                ControlFlow::Continue(walk::WalkAction::Advance)
            });
        }
    }
    used
}

/// Resolve a value through the remap chain.
fn resolve_remap(remap: &HashMap<ValueRef, ValueRef>, val: ValueRef) -> ValueRef {
    let mut current = val;
    for _ in 0..10 {
        // limit to prevent infinite loops
        match remap.get(&current) {
            Some(&next) if next != current => current = next,
            _ => break,
        }
    }
    current
}

/// Emit `scf.yield` for a value, wrapping in `YieldResult::Done` if needed.
///
/// If the value is already a `YieldResult`, yields it directly.
/// Otherwise, casts to anyref, wraps in `Done`, then yields.
fn yield_value_or_done(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    value: ValueRef,
    types: &YieldBubblingTypes,
) {
    let val_ty = ctx.value_ty(value);
    if is_yield_result_type(ctx, val_ty) {
        let yield_op = arena_scf::r#yield(ctx, location, [value]);
        ctx.push_op(block, yield_op.op_ref());
    } else {
        let anyref_val = arena_core::unrealized_conversion_cast(ctx, location, value, types.anyref);
        ctx.push_op(block, anyref_val.op_ref());
        let done_op = arena_adt::variant_new(
            ctx,
            location,
            [anyref_val.result(ctx)],
            types.yield_result,
            types.yield_result,
            Symbol::new("Done"),
        );
        ctx.push_op(block, done_op.op_ref());
        let yield_op = arena_scf::r#yield(ctx, location, [done_op.result(ctx)]);
        ctx.push_op(block, yield_op.op_ref());
    }
}
