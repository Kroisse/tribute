//! Tail-Resumptive Dispatch pass.
//!
//! This pass runs after `resolve_evidence` and before `cont_to_libmprompt`.
//! It optimizes tail-resumptive handlers by generating TR dispatch functions
//! and inserting fast-path branches at shift sites.
//!
//! ## Overview
//!
//! For handlers where **all** suspend arms are tail-resumptive (`k(value)`),
//! continuation capture via `mp_yield` is unnecessary. Instead, a dispatch
//! function is generated that directly computes the resume value.
//!
//! ## Transformations
//!
//! 1. **Marker patching**: For eligible handlers, the Marker's `tr_dispatch_fn`
//!    field (initially null) is replaced with a `func.constant` pointing to
//!    the generated dispatch function.
//!
//! 2. **Shift site branching**: At each `cont.shift`, after evidence lookup,
//!    the `tr_dispatch_fn` is checked. If non-null, the dispatch function is
//!    called directly (fast path). Otherwise, the normal `cont.shift` path
//!    is taken.
//!
//! ## Generated dispatch function
//!
//! ```text
//! func.func @__tr_dispatch_N(%op_idx: i32, %shift_value: ptr) -> ptr {
//!     %hash_0 = arith.const <hash(Ability, op0)> : i32
//!     %is_0 = arith.cmp_eq %op_idx, %hash_0
//!     %result = scf.if(%is_0) {
//!         // ... compute resume value from %shift_value ...
//!         scf.yield %value0
//!     } else {
//!         %hash_1 = arith.const <hash(Ability, op1)> : i32
//!         %is_1 = arith.cmp_eq %op_idx, %hash_1
//!         %inner = scf.if(%is_1) {
//!             // ... compute resume value ...
//!             scf.yield %value1
//!         } else {
//!             func.unreachable
//!         }
//!         scf.yield %inner
//!     }
//!     func.return %result
//! }
//! ```

use std::collections::HashMap;

use tribute_ir::dialect::ability as arena_ability;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location};

use crate::cont_util::{SuspendArm, collect_suspend_arms, compute_op_idx};
use crate::tail_resumptive;

// ============================================================================
// Helper type constructors
// ============================================================================

fn i32_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types.intern(
        trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build(),
    )
}

fn i1_type_ref(ctx: &mut IrContext) -> TypeRef {
    ctx.types.intern(
        trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build(),
    )
}

// ============================================================================
// Analysis: find eligible handlers
// ============================================================================

/// Information about a handler that is fully tail-resumptive.
struct TrHandlerInfo {
    /// The suspend arms (all tail-resumptive).
    arms: Vec<SuspendArm>,
}

/// Information about a push_prompt + handler_dispatch pair.
struct PushPromptPair {
    /// The `cont.push_prompt` OpRef.
    push_prompt_op: OpRef,
    /// Associated TR handler info (if eligible).
    tr_handler: Option<TrHandlerInfo>,
}

/// Find push_prompt + handler_dispatch pairs in a block.
/// Returns info about whether the handler is fully TR-eligible.
fn find_push_prompt_pairs(ctx: &IrContext, block: BlockRef) -> Vec<PushPromptPair> {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut pairs = Vec::new();

    // First pass: collect handler_dispatch info by tag
    let mut dispatch_by_tag: HashMap<u32, (OpRef, Vec<SuspendArm>)> = HashMap::new();
    for &op in &ops {
        if let Ok(dispatch_op) = arena_cont::HandlerDispatch::from_op(ctx, op) {
            let tag = dispatch_op.tag(ctx);
            let body = dispatch_op.body(ctx);
            let arms = collect_suspend_arms(ctx, body);
            dispatch_by_tag.insert(tag, (op, arms));
        }
    }

    // Second pass: match push_prompt ops with their dispatchers
    for &op in &ops {
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            let push_prompt_op = arena_cont::PushPrompt::from_op(ctx, op).unwrap();
            let tag_attr = push_prompt_op.tag(ctx);
            let tag = match &tag_attr {
                Attribute::Int(v) => u32::try_from(*v).unwrap_or(0),
                _ => 0,
            };

            let tr_handler = dispatch_by_tag.get(&tag).and_then(|(dispatch_op, arms)| {
                // All arms must be tail-resumptive AND self-contained for eligibility
                if arms.is_empty() {
                    return None;
                }
                let all_eligible = arms
                    .iter()
                    .all(|arm| arm.tail_resumptive && is_arm_self_contained(ctx, arm.body));
                if all_eligible {
                    Some(TrHandlerInfo {
                        arms: collect_suspend_arms(ctx, {
                            let d =
                                arena_cont::HandlerDispatch::from_op(ctx, *dispatch_op).unwrap();
                            d.body(ctx)
                        }),
                    })
                } else {
                    None
                }
            });

            pairs.push(PushPromptPair {
                push_prompt_op: op,
                tr_handler,
            });
        }
    }

    pairs
}

// ============================================================================
// TR dispatch function generation
// ============================================================================

/// Check if a TR arm's value computation is self-contained (no live-ins).
///
/// For the dispatch function to work, all values used in the computation
/// must be either:
/// - Block args of the suspend body (remapped to dispatch function args)
/// - Results of ops within the body (cloned into dispatch function)
///
/// If any op uses a value from an outer scope (e.g., a captured variable),
/// the arm is NOT eligible for TR dispatch.
fn is_arm_self_contained(ctx: &IrContext, body: RegionRef) -> bool {
    let blocks = &ctx.region(body).blocks;
    let Some(&first_block) = blocks.first() else {
        return false;
    };

    let block_args: std::collections::HashSet<ValueRef> =
        ctx.block_args(first_block).iter().copied().collect();

    // Collect all values defined by ops in the body
    let mut defined_values: std::collections::HashSet<ValueRef> = block_args.clone();
    let block_ops = &ctx.block(first_block).ops;

    for &op in block_ops.iter() {
        // Add op results to defined values
        for &result in ctx.op_results(op).iter() {
            defined_values.insert(result);
        }
    }

    // Check that all operands of non-resume/yield ops reference defined values
    for &op in block_ops.iter() {
        // Skip cont.resume and scf.yield — they use %k which is a block arg
        if arena_cont::Resume::matches(ctx, op) || arena_scf::Yield::matches(ctx, op) {
            continue;
        }

        for &operand in ctx.op_operands(op).iter() {
            if !defined_values.contains(&operand) {
                return false;
            }
        }
    }

    // Also check the resume_value itself
    if let Some(info) = tail_resumptive::is_tail_resumptive(ctx, body)
        && !defined_values.contains(&info.resume_value)
    {
        return false;
    }

    true
}

/// Extract the value computation from a tail-resumptive suspend body.
///
/// A TR suspend body has the form:
/// ```text
/// ^bb0(%k: ptr, %sv: ptr):
///     ... value computation ops ...
///     %resume_value = ...
///     %result = cont.resume %k, %resume_value
///     scf.yield %result
/// ```
///
/// We extract all ops before cont.resume, and the resume_value.
/// These ops will be cloned into the dispatch function, with %sv remapped
/// to the dispatch function's %shift_value parameter.
struct TrArmExtraction {
    /// Ops to clone (excluding cont.resume and scf.yield)
    ops_to_clone: Vec<OpRef>,
    /// The value passed to cont.resume (resume_value)
    resume_value: ValueRef,
    /// The shift_value block arg (%sv, block_arg[1])
    shift_value_arg: ValueRef,
}

fn extract_tr_arm(ctx: &IrContext, body: RegionRef) -> Option<TrArmExtraction> {
    let blocks = &ctx.region(body).blocks;
    let &first_block = blocks.first()?;
    let block_args = ctx.block_args(first_block);
    if block_args.len() < 2 {
        return None;
    }
    let shift_value_arg = block_args[1];

    let info = tail_resumptive::is_tail_resumptive(ctx, body)?;

    // Collect ops before cont.resume (the value computation)
    let block_ops = &ctx.block(first_block).ops;
    let mut ops_to_clone = Vec::new();
    for &op in block_ops.iter() {
        if op == info.resume_op {
            break;
        }
        // Skip scf.yield (shouldn't appear before resume, but be safe)
        if arena_scf::Yield::matches(ctx, op) {
            break;
        }
        ops_to_clone.push(op);
    }

    Some(TrArmExtraction {
        ops_to_clone,
        resume_value: info.resume_value,
        shift_value_arg,
    })
}

/// Generate a TR dispatch function for a fully tail-resumptive handler.
///
/// The function signature is `(op_idx: i32, shift_value: ptr) -> ptr`.
/// It dispatches on op_idx to compute the appropriate resume value.
fn generate_tr_dispatch_func(
    ctx: &mut IrContext,
    loc: Location,
    func_name: Symbol,
    arms: &[SuspendArm],
) -> OpRef {
    let i32_ty = i32_type_ref(ctx);
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();

    // Create function type: (i32, ptr) -> ptr
    let func_ty = arena_core::func(ctx, ptr_ty, [i32_ty, ptr_ty], None).as_type_ref();

    // Create entry block with args: %op_idx: i32, %shift_value: ptr
    let entry_block = ctx.create_block(trunk_ir::context::BlockData {
        location: loc,
        args: vec![
            trunk_ir::context::BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            },
            trunk_ir::context::BlockArgData {
                ty: ptr_ty,
                attrs: Default::default(),
            },
        ],
        ops: Default::default(),
        parent_region: None,
    });

    let op_idx_arg = ctx.block_args(entry_block)[0];
    let shift_value_arg = ctx.block_args(entry_block)[1];

    // Build nested scf.if chain for dispatch
    let result_val =
        build_dispatch_chain(ctx, loc, entry_block, arms, 0, op_idx_arg, shift_value_arg);

    // Add func.return
    let ret = arena_func::r#return(ctx, loc, [result_val]);
    ctx.push_op(entry_block, ret.op_ref());

    // Create body region
    let body = ctx.create_region(trunk_ir::context::RegionData {
        location: loc,
        blocks: trunk_ir::smallvec::smallvec![entry_block],
        parent_op: None,
    });

    let func_op = arena_func::func(ctx, loc, func_name, func_ty, body);
    func_op.op_ref()
}

/// Build a nested chain of scf.if operations for dispatching on op_idx.
///
/// For arms [a0, a1, a2], generates:
/// ```text
/// %is_a0 = arith.cmp_eq %op_idx, <hash_a0>
/// scf.if(%is_a0) {
///     ... a0 value computation ...
///     scf.yield %value0
/// } else {
///     %is_a1 = arith.cmp_eq %op_idx, <hash_a1>
///     scf.if(%is_a1) {
///         ... a1 value computation ...
///         scf.yield %value1
///     } else {
///         ... a2 value computation (last arm, no further check needed) ...
///         scf.yield %value2
///     }
/// }
/// ```
fn build_dispatch_chain(
    ctx: &mut IrContext,
    loc: Location,
    parent_block: BlockRef,
    arms: &[SuspendArm],
    arm_idx: usize,
    op_idx_arg: ValueRef,
    shift_value_arg: ValueRef,
) -> ValueRef {
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();
    let i32_ty = i32_type_ref(ctx);
    let i1_ty = i1_type_ref(ctx);

    assert!(!arms.is_empty(), "TR dispatch must have at least one arm");

    // Last arm: no condition check needed, just emit the value computation
    if arm_idx == arms.len() - 1 {
        let arm = &arms[arm_idx];
        let result = emit_arm_value_computation(ctx, loc, parent_block, arm, shift_value_arg);
        return result;
    }

    let arm = &arms[arm_idx];

    // %hash = arith.const <expected_op_idx>
    let hash_const = arith::r#const(
        ctx,
        loc,
        i32_ty,
        Attribute::Int(arm.expected_op_idx as i128),
    );
    let hash_val = hash_const.result(ctx);
    ctx.push_op(parent_block, hash_const.op_ref());

    // %is_match = arith.cmp_eq %op_idx, %hash
    let cmp = arith::cmp_eq(ctx, loc, op_idx_arg, hash_val, i1_ty);
    let cmp_val = cmp.result(ctx);
    ctx.push_op(parent_block, cmp.op_ref());

    // Then branch: emit this arm's value computation
    let then_block = ctx.create_block(trunk_ir::context::BlockData {
        location: loc,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let then_result = emit_arm_value_computation(ctx, loc, then_block, arm, shift_value_arg);
    let then_yield = arena_scf::r#yield(ctx, loc, [then_result]);
    ctx.push_op(then_block, then_yield.op_ref());
    let then_region = ctx.create_region(trunk_ir::context::RegionData {
        location: loc,
        blocks: trunk_ir::smallvec::smallvec![then_block],
        parent_op: None,
    });

    // Else branch: recurse for remaining arms
    let else_block = ctx.create_block(trunk_ir::context::BlockData {
        location: loc,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let else_result = build_dispatch_chain(
        ctx,
        loc,
        else_block,
        arms,
        arm_idx + 1,
        op_idx_arg,
        shift_value_arg,
    );
    let else_yield = arena_scf::r#yield(ctx, loc, [else_result]);
    ctx.push_op(else_block, else_yield.op_ref());
    let else_region = ctx.create_region(trunk_ir::context::RegionData {
        location: loc,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });

    // scf.if(%is_match) { then } else { else }
    let if_op = arena_scf::r#if(ctx, loc, cmp_val, ptr_ty, then_region, else_region);
    let if_result = if_op.result(ctx);
    ctx.push_op(parent_block, if_op.op_ref());

    if_result
}

/// Emit the value computation for a single TR arm.
///
/// Clones the ops from the suspend body into the target block,
/// remapping the shift_value block arg to the dispatch function's parameter.
fn emit_arm_value_computation(
    ctx: &mut IrContext,
    loc: Location,
    target_block: BlockRef,
    arm: &SuspendArm,
    shift_value_arg: ValueRef,
) -> ValueRef {
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();

    let extraction = match extract_tr_arm(ctx, arm.body) {
        Some(e) => e,
        None => {
            // Fallback: return shift_value as-is (shouldn't happen for TR arms)
            return shift_value_arg;
        }
    };

    // Build value remapping: original shift_value -> dispatch function's shift_value
    let mut remap: HashMap<ValueRef, ValueRef> = HashMap::new();
    remap.insert(extraction.shift_value_arg, shift_value_arg);

    // Clone each op, remapping operands
    for &src_op in &extraction.ops_to_clone {
        let new_op = clone_op_with_remap(ctx, src_op, &mut remap);
        ctx.push_op(target_block, new_op);
    }

    // Get the remapped resume_value
    let result_val = if let Some(&remapped) = remap.get(&extraction.resume_value) {
        remapped
    } else {
        // resume_value is the shift_value itself (not produced by any cloned op)
        shift_value_arg
    };

    // Cast result to ptr if needed (dispatch function returns ptr)
    let result_ty = ctx.value_ty(result_val);
    if result_ty != ptr_ty {
        let cast_data = trunk_ir::context::OperationDataBuilder::new(
            loc,
            Symbol::new("core"),
            Symbol::new("unrealized_conversion_cast"),
        )
        .operand(result_val)
        .result(ptr_ty)
        .build(ctx);
        let cast_op = ctx.create_op(cast_data);
        let cast_result = ctx.op_result(cast_op, 0);
        ctx.push_op(target_block, cast_op);
        cast_result
    } else {
        result_val
    }
}

/// Clone an operation with operand remapping.
/// Updates remap with the new result values.
fn clone_op_with_remap(
    ctx: &mut IrContext,
    src_op: OpRef,
    remap: &mut HashMap<ValueRef, ValueRef>,
) -> OpRef {
    let op_data = ctx.op(src_op);
    let loc = op_data.location;
    let dialect = op_data.dialect;
    let name = op_data.name;

    // Remap operands
    let operands: Vec<ValueRef> = ctx
        .op_operands(src_op)
        .iter()
        .map(|&v| remap.get(&v).copied().unwrap_or(v))
        .collect();

    // Get result types
    let result_types: Vec<TypeRef> = ctx.op_result_types(src_op).to_vec();

    // Clone attributes
    let attributes = op_data.attributes.clone();

    // Build new op
    let mut builder = trunk_ir::context::OperationDataBuilder::new(loc, dialect, name);
    for operand in &operands {
        builder = builder.operand(*operand);
    }
    for result_ty in &result_types {
        builder = builder.result(*result_ty);
    }
    let mut op_data = builder.build(ctx);
    op_data.attributes = attributes;
    let new_op = ctx.create_op(op_data);

    // Update remap for results
    let old_results = ctx.op_results(src_op);
    let new_results = ctx.op_results(new_op);
    for (old_r, new_r) in old_results.iter().zip(new_results.iter()) {
        remap.insert(*old_r, *new_r);
    }

    new_op
}

// ============================================================================
// Marker patching
// ============================================================================

/// Find the `adt.struct_new` that creates the Marker for a given tag,
/// and replace its null tr_dispatch_fn field with a func.constant.
fn patch_marker_struct_new(
    ctx: &mut IrContext,
    block: BlockRef,
    push_prompt_op: OpRef,
    dispatch_func_name: Symbol,
) {
    let loc = ctx.op(push_prompt_op).location;
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();
    let marker_ty = arena_ability::marker_adt_type_ref(ctx);

    // Find the adt.struct_new ops that precede this push_prompt and create Markers.
    // The Marker struct_new has 3 fields and its type is the marker_ty.
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let push_prompt_idx = ops.iter().position(|&op| op == push_prompt_op);
    let Some(pp_idx) = push_prompt_idx else {
        return;
    };

    // Scan backwards from push_prompt to find adt.struct_new with marker_ty
    for &op in ops[..pp_idx].iter().rev() {
        if let Ok(struct_new_op) = arena_adt::StructNew::from_op(ctx, op) {
            let sn_ty = struct_new_op.r#type(ctx);
            if sn_ty != marker_ty {
                continue;
            }

            // This is a Marker struct_new. The 3rd operand (index 2) is tr_dispatch_fn.
            let operands = ctx.op_operands(op).to_vec();
            if operands.len() != 3 {
                continue;
            }

            let _old_tr_fn = operands[2];

            // Create func.constant for the dispatch function
            let func_const = arena_func::constant(ctx, loc, ptr_ty, dispatch_func_name);
            let new_tr_fn = func_const.result(ctx);
            ctx.insert_op_before(block, op, func_const.op_ref());

            // Create new struct_new with updated tr_dispatch_fn
            let new_struct = arena_adt::struct_new(
                ctx,
                loc,
                vec![operands[0], operands[1], new_tr_fn],
                marker_ty,
                marker_ty,
            );
            let new_result = ctx.op_result(new_struct.op_ref(), 0);
            let old_result = ctx.op_result(op, 0);
            ctx.replace_all_uses(old_result, new_result);

            ctx.insert_op_before(block, op, new_struct.op_ref());
            ctx.remove_op_from_block(block, op);

            // Don't break — there may be multiple markers for multi-ability handlers
        }
    }
}

// ============================================================================
// Shift site branching
// ============================================================================

/// Add TR fast-path branching at `cont.shift` sites.
///
/// For each `cont.shift`, after the existing evidence lookup + struct_get
/// sequence, adds:
/// ```text
/// %tr_fn = adt.struct_get(%marker, 2)   // tr_dispatch_fn
/// %null = arith.const 0 : i32
/// %null_ptr = unrealized_conversion_cast %null : ptr
/// %has_tr = arith.cmp_ne %tr_fn, %null_ptr
/// %result = scf.if(%has_tr) {
///     %op_idx = arith.const <hash> : i32
///     %tr_result = func.call_indirect %tr_fn(%op_idx, %shift_value)
///     scf.yield %tr_result
/// } else {
///     %shift_result = cont.shift(...)  // original shift
///     scf.yield %shift_result
/// }
/// ```
fn add_tr_branching_at_shift_sites(ctx: &mut IrContext, module: Module) {
    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = arena_func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };
        let body = func_op.body(ctx);
        add_tr_branching_in_region(ctx, body);
    }
}

fn add_tr_branching_in_region(ctx: &mut IrContext, region: RegionRef) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        add_tr_branching_in_block(ctx, block);
    }
}

fn add_tr_branching_in_block(ctx: &mut IrContext, block: BlockRef) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        // Recurse into nested regions first
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            add_tr_branching_in_region(ctx, region);
        }

        let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) else {
            continue;
        };

        let loc = ctx.op(op).location;
        let ability_ref = shift_op.ability_ref(ctx);
        let op_name = shift_op.op_name(ctx);

        // Get the ability name from ability_ref type for op_idx computation
        let ability_data = ctx.types.get(ability_ref);
        let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
            Some(Attribute::Symbol(s)) => Some(*s),
            _ => None,
        };

        let expected_op_idx = compute_op_idx(ability_name, Some(op_name));

        // Find the marker value. The shift's tag comes from `adt.struct_get(marker, 1)`.
        // We need to find the marker value to do `adt.struct_get(marker, 2)` for tr_dispatch_fn.
        let tag_val = shift_op.tag(ctx);
        let marker_val = find_marker_from_tag(ctx, tag_val);
        let Some(marker_val) = marker_val else {
            // Can't find marker — skip TR optimization for this shift
            continue;
        };

        let i32_ty = i32_type_ref(ctx);
        let i1_ty = i1_type_ref(ctx);
        let ptr_ty = arena_core::ptr(ctx).as_type_ref();
        let marker_ty = arena_ability::marker_adt_type_ref(ctx);

        let result_ty = ctx
            .op_result_types(op)
            .first()
            .copied()
            .unwrap_or_else(|| arena_core::nil(ctx).as_type_ref());

        // %tr_fn = adt.struct_get(%marker, field=2) -- tr_dispatch_fn
        let struct_get_tr = arena_adt::struct_get(ctx, loc, marker_val, ptr_ty, marker_ty, 2);
        let tr_fn_val = ctx.op_result(struct_get_tr.op_ref(), 0);
        ctx.insert_op_before(block, op, struct_get_tr.op_ref());

        // %null_ptr = arith.const 0 : ptr
        let null_ptr_const = arith::r#const(ctx, loc, ptr_ty, Attribute::Int(0));
        let null_ptr_val = null_ptr_const.result(ctx);
        ctx.insert_op_before(block, op, null_ptr_const.op_ref());

        // %has_tr = arith.cmp_ne %tr_fn, %null_ptr
        let has_tr = arith::cmp_ne(ctx, loc, tr_fn_val, null_ptr_val, i1_ty);
        let has_tr_val = has_tr.result(ctx);
        ctx.insert_op_before(block, op, has_tr.op_ref());

        // --- Then branch: TR fast path ---
        let then_block = ctx.create_block(trunk_ir::context::BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });

        // %op_idx = arith.const <hash>
        let op_idx_const =
            arith::r#const(ctx, loc, i32_ty, Attribute::Int(expected_op_idx as i128));
        let op_idx_val = op_idx_const.result(ctx);
        ctx.push_op(then_block, op_idx_const.op_ref());

        // Get shift value operands (operands after the tag)
        let shift_operands = ctx.op_operands(op).to_vec();
        // shift operands: [tag, ...values]
        // For TR, we pass the first value operand as shift_value
        let shift_value = if shift_operands.len() > 1 {
            shift_operands[1]
        } else {
            // No value — create a null pointer
            let sv_null = arith::r#const(ctx, loc, ptr_ty, Attribute::Int(0));
            let sv_val = sv_null.result(ctx);
            ctx.push_op(then_block, sv_null.op_ref());
            sv_val
        };

        // Cast shift_value to ptr if needed
        let shift_value_ty = ctx.value_ty(shift_value);
        let sv_as_ptr = if shift_value_ty != ptr_ty {
            let cast_data = trunk_ir::context::OperationDataBuilder::new(
                loc,
                Symbol::new("core"),
                Symbol::new("unrealized_conversion_cast"),
            )
            .operand(shift_value)
            .result(ptr_ty)
            .build(ctx);
            let cast_op = ctx.create_op(cast_data);
            let cast_val = ctx.op_result(cast_op, 0);
            ctx.push_op(then_block, cast_op);
            cast_val
        } else {
            shift_value
        };

        // %tr_result = func.call_indirect %tr_fn(%op_idx, %sv_as_ptr)
        let tr_call =
            arena_func::call_indirect(ctx, loc, tr_fn_val, [op_idx_val, sv_as_ptr], ptr_ty);
        let tr_result = tr_call.result(ctx);
        ctx.push_op(then_block, tr_call.op_ref());

        // Cast result back to expected type if needed
        let then_result = if result_ty != ptr_ty {
            let cast_data = trunk_ir::context::OperationDataBuilder::new(
                loc,
                Symbol::new("core"),
                Symbol::new("unrealized_conversion_cast"),
            )
            .operand(tr_result)
            .result(result_ty)
            .build(ctx);
            let cast_op = ctx.create_op(cast_data);
            let cast_val = ctx.op_result(cast_op, 0);
            ctx.push_op(then_block, cast_op);
            cast_val
        } else {
            tr_result
        };

        let then_yield = arena_scf::r#yield(ctx, loc, [then_result]);
        ctx.push_op(then_block, then_yield.op_ref());

        let then_region = ctx.create_region(trunk_ir::context::RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![then_block],
            parent_op: None,
        });

        // --- Else branch: original cont.shift path ---
        let else_block = ctx.create_block(trunk_ir::context::BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });

        // Find the op that follows the shift in the block — we'll insert scf.if before it.
        let current_ops = ctx.block(block).ops.to_vec();
        let shift_pos = current_ops.iter().position(|&o| o == op);
        let next_op_after_shift = shift_pos.and_then(|pos| current_ops.get(pos + 1)).copied();

        // Move the original shift op into the else branch
        ctx.remove_op_from_block(block, op);
        ctx.push_op(else_block, op);

        let shift_result = ctx.op_result(op, 0);

        let else_yield = arena_scf::r#yield(ctx, loc, [shift_result]);
        ctx.push_op(else_block, else_yield.op_ref());

        let else_region = ctx.create_region(trunk_ir::context::RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![else_block],
            parent_op: None,
        });

        // scf.if(%has_tr) { then } else { else }
        let if_op = arena_scf::r#if(ctx, loc, has_tr_val, result_ty, then_region, else_region);
        let if_result = if_op.result(ctx);

        // Insert scf.if at the position where cont.shift was
        if let Some(next) = next_op_after_shift {
            ctx.insert_op_before(block, next, if_op.op_ref());
        } else {
            ctx.push_op(block, if_op.op_ref());
        }

        // Replace all uses of the old shift result with the if result.
        // shift_result is now inside the else branch's scf.yield, so we need
        // to be careful not to replace that use. Do the replacement first,
        // then fix up the else yield.
        ctx.replace_all_uses(shift_result, if_result);
        // Fix: the else yield's operand was replaced — restore it
        ctx.set_op_operand(else_yield.op_ref(), 0, shift_result);
    }
}

/// Given the tag value from a cont.shift, trace back to find the marker value.
///
/// The tag comes from `adt.struct_get(marker, 1)`, so we look at the value def
/// of tag_val to find the struct_get, then get its operand (the marker).
fn find_marker_from_tag(ctx: &IrContext, tag_val: ValueRef) -> Option<ValueRef> {
    let def = ctx.value_def(tag_val);
    match def {
        trunk_ir::refs::ValueDef::OpResult(op, _) => {
            if arena_adt::StructGet::matches(ctx, op) {
                // The struct_get's operand is the marker
                let operands = ctx.op_operands(op);
                if !operands.is_empty() {
                    return Some(operands[0]);
                }
            }
            None
        }
        _ => None,
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Insert tail-resumptive dispatch optimization.
///
/// This pass:
/// 1. Finds fully tail-resumptive handlers
/// 2. Generates TR dispatch functions
/// 3. Patches Marker struct_new to include dispatch function pointer
/// 4. Adds fast-path branching at shift sites
pub fn insert_tr_dispatch(ctx: &mut IrContext, module: Module) {
    let Some(module_body) = module.body(ctx) else {
        return;
    };

    let module_block = match ctx.region(module_body).blocks.first().copied() {
        Some(b) => b,
        None => return,
    };

    let loc = ctx.op(module.op()).location;

    // Step 1: Find all func ops and analyze their handlers
    let func_ops: Vec<OpRef> = module.ops(ctx);
    let mut dispatch_counter = 0u32;

    for func_op_ref in func_ops {
        let Ok(func_op) = arena_func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };
        let body = func_op.body(ctx);
        let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();

        for block in blocks {
            let pairs = find_push_prompt_pairs(ctx, block);

            for pair in pairs {
                let Some(tr_handler) = pair.tr_handler else {
                    continue;
                };

                // Generate a unique dispatch function name
                let dispatch_name =
                    Symbol::from_dynamic(&format!("__tr_dispatch_{}", dispatch_counter));
                dispatch_counter += 1;

                // Generate the dispatch function
                let dispatch_func_op =
                    generate_tr_dispatch_func(ctx, loc, dispatch_name, &tr_handler.arms);

                // Add dispatch function to module
                let first_op = ctx.block(module_block).ops.first().copied();
                if let Some(first) = first_op {
                    ctx.insert_op_before(module_block, first, dispatch_func_op);
                } else {
                    ctx.push_op(module_block, dispatch_func_op);
                }

                // Patch the Marker struct_new ops for this push_prompt
                patch_marker_struct_new(ctx, block, pair.push_prompt_op, dispatch_name);
            }
        }
    }

    // Step 2: Add TR branching at all shift sites
    add_tr_branching_at_shift_sites(ctx, module);
}
