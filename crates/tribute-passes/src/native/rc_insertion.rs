//! Reference counting insertion pass.
//!
//! Automatically inserts `tribute_rt.retain` and `tribute_rt.release` operations
//! for pointer-typed (`core.ptr`) values in the native backend pipeline.
//!
//! ## Pipeline Position
//!
//! Runs after Phase 2.7 (`tribute_rt_to_clif` boxing lowering), where:
//! - All allocations are `clif.call @__tribute_alloc`
//! - Boxing ops are already lowered to clif
//! - Pointer types are all `core.ptr`
//! - `tribute_rt.retain`/`release` are preserved as legal ops
//!
//! ## RC Rules
//!
//! ### Retain (reference acquisition)
//!
//! | Situation | Action |
//! |-----------|--------|
//! | Function parameter (ptr) | `retain` at entry |
//! | `clif.call @__tribute_alloc` result | No retain (starts with refcount=1) |
//! | Other `clif.call` result (ptr) | No retain (ownership transfer) |
//! | `clif.store` with ptr value | `retain` before store |
//! | `clif.load` with ptr result | `retain` after load |
//!
//! ### Release (reference drop)
//!
//! | Situation | Action |
//! |-----------|--------|
//! | Last SSA use in block | `release` after last use |
//! | `clif.return` operand | No release (ownership transfer to caller) |
//! | Value dies in block (live-in but not live-out) | `release` at appropriate point |

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;
use trunk_ir::arena::TypeDataBuilder;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::clif as arena_clif;
use trunk_ir::arena::dialect::core as arena_core;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::arena::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};

use tribute_ir::arena::dialect::tribute_rt as arena_tribute_rt;

/// Check if a type is `core.ptr`.
fn is_ptr_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("ptr")
}

/// Check if a value is a pointer type.
fn is_ptr_value(ctx: &IrContext, value: ValueRef) -> bool {
    is_ptr_type(ctx, ctx.value_ty(value))
}

/// Check if an op is a block terminator.
fn is_terminator_op(ctx: &IrContext, op: OpRef) -> bool {
    arena_clif::Return::matches(ctx, op)
        || arena_clif::Jump::matches(ctx, op)
        || arena_clif::Brif::matches(ctx, op)
        || arena_clif::Trap::matches(ctx, op)
        || arena_clif::ReturnCall::matches(ctx, op)
        || arena_clif::BrTable::matches(ctx, op)
}

/// Check if a value is a static pointer (not RC-managed).
fn is_static_ptr(ctx: &IrContext, value: ValueRef) -> bool {
    let ValueDef::OpResult(def_op, _) = ctx.value_def(value) else {
        return false;
    };
    if arena_clif::SymbolAddr::matches(ctx, def_op) {
        return true;
    }
    if arena_clif::Iconst::matches(ctx, def_op) && is_ptr_type(ctx, ctx.value_ty(value)) {
        return true;
    }
    false
}

/// Check if a value is an intermediate allocation pointer.
fn is_alloc_intermediate(ctx: &IrContext, value: ValueRef) -> bool {
    let ValueDef::OpResult(def_op, _) = ctx.value_def(value) else {
        return false;
    };
    if let Ok(call_op) = arena_clif::Call::from_op(ctx, def_op) {
        return call_op.callee(ctx) == Symbol::new("__tribute_alloc");
    }
    if let Ok(_iadd_op) = arena_clif::Iadd::from_op(ctx, def_op) {
        let operands = ctx.op_operands(def_op).to_vec();
        if let Some(&lhs) = operands.first() {
            let ValueDef::OpResult(lhs_op, _) = ctx.value_def(lhs) else {
                return false;
            };
            if let Ok(call_op) = arena_clif::Call::from_op(ctx, lhs_op) {
                return call_op.callee(ctx) == Symbol::new("__tribute_alloc");
            }
        }
    }
    false
}

/// Infer allocation size by tracing the def chain.
fn infer_alloc_size(ctx: &IrContext, value: ValueRef) -> u64 {
    let ValueDef::OpResult(def_op, _) = ctx.value_def(value) else {
        return 0;
    };
    let Ok(call_op) = arena_clif::Call::from_op(ctx, def_op) else {
        if let Ok(_iadd_op) = arena_clif::Iadd::from_op(ctx, def_op) {
            let operands = ctx.op_operands(def_op).to_vec();
            if let Some(&lhs) = operands.first() {
                return infer_alloc_size(ctx, lhs);
            }
        }
        return 0;
    };
    if call_op.callee(ctx) != Symbol::new("__tribute_alloc") {
        return 0;
    }
    let args = ctx.op_operands(def_op).to_vec();
    let Some(&size_val) = args.first() else {
        return 0;
    };
    let ValueDef::OpResult(size_op, _) = ctx.value_def(size_val) else {
        return 0;
    };
    if let Ok(iconst_op) = arena_clif::Iconst::from_op(ctx, size_op) {
        return iconst_op.value(ctx) as u64;
    }
    0
}

// =============================================================================
// Liveness Analysis
// =============================================================================

/// Per-block liveness information.
struct LivenessInfo {
    def_set: HashMap<BlockRef, HashSet<ValueRef>>,
    live_in: HashMap<BlockRef, HashSet<ValueRef>>,
    live_out: HashMap<BlockRef, HashSet<ValueRef>>,
}

/// Collect all pointer-typed values in the function body.
fn collect_ptr_values(ctx: &IrContext, body: RegionRef) -> HashSet<ValueRef> {
    let mut ptr_values = HashSet::new();
    let blocks = ctx.region(body).blocks.to_vec();

    for &block in &blocks {
        // Block arguments
        for &arg_val in ctx.block_args(block) {
            if is_ptr_type(ctx, ctx.value_ty(arg_val)) {
                ptr_values.insert(arg_val);
            }
        }
        // Operation results
        for &op in &ctx.block(block).ops.to_vec() {
            for &result_val in ctx.op_results(op) {
                if is_ptr_type(ctx, ctx.value_ty(result_val)) && !is_static_ptr(ctx, result_val) {
                    ptr_values.insert(result_val);
                }
            }
        }
    }

    // Also check operands
    for &block in &blocks {
        for &op in &ctx.block(block).ops.to_vec() {
            for &operand in ctx.op_operands(op) {
                if is_ptr_value(ctx, operand) && !is_static_ptr(ctx, operand) {
                    ptr_values.insert(operand);
                }
            }
        }
    }

    ptr_values
}

/// Build alias map for unrealized_conversion_cast.
fn build_ptr_alias_map(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &mut HashSet<ValueRef>,
) -> HashMap<ValueRef, ValueRef> {
    let mut aliases = HashMap::new();
    let blocks = ctx.region(body).blocks.to_vec();

    for &block in &blocks {
        for &op in &ctx.block(block).ops.to_vec() {
            if arena_core::UnrealizedConversionCast::matches(ctx, op) {
                let operands = ctx.op_operands(op).to_vec();
                if let Some(&input) = operands.first() {
                    let root = if ptr_values.contains(&input) {
                        Some(input)
                    } else {
                        aliases.get(&input).copied()
                    };

                    if let Some(root) = root {
                        let output = ctx.op_result(op, 0);
                        let output_ty = ctx.value_ty(output);
                        let output_data = ctx.types.get(output_ty);
                        let is_integer_output = output_data.dialect == Symbol::new("core")
                            && (output_data.name == Symbol::new("i64")
                                || output_data.name == Symbol::new("i32"));
                        if !is_integer_output {
                            aliases.insert(output, root);
                            ptr_values.remove(&output);
                        }
                    }
                }
            }
        }
    }
    aliases
}

/// Build CFG successor map.
fn build_successor_map(ctx: &IrContext, body: RegionRef) -> HashMap<BlockRef, Vec<BlockRef>> {
    let mut successors = HashMap::new();
    let blocks = ctx.region(body).blocks.to_vec();

    for &block in &blocks {
        let ops = &ctx.block(block).ops;
        let mut succs = Vec::new();
        if let Some(&last_op) = ops.last() {
            succs.extend(ctx.op(last_op).successors.iter().copied());
        }
        successors.insert(block, succs);
    }

    successors
}

/// Compute use and def sets.
fn compute_use_def_sets(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
) -> (
    HashMap<BlockRef, HashSet<ValueRef>>,
    HashMap<BlockRef, HashSet<ValueRef>>,
) {
    let mut use_sets: HashMap<BlockRef, HashSet<ValueRef>> = HashMap::new();
    let mut def_sets: HashMap<BlockRef, HashSet<ValueRef>> = HashMap::new();
    let blocks = ctx.region(body).blocks.to_vec();

    for &block in &blocks {
        let mut uses = HashSet::new();
        let mut defs = HashSet::new();

        for &arg_val in ctx.block_args(block) {
            if is_ptr_type(ctx, ctx.value_ty(arg_val)) {
                defs.insert(arg_val);
            }
        }

        for &op in &ctx.block(block).ops.to_vec() {
            for &operand in ctx.op_operands(op) {
                if ptr_values.contains(&operand) && !defs.contains(&operand) {
                    uses.insert(operand);
                }
                if let Some(&aliased) = ptr_alias_map.get(&operand)
                    && !defs.contains(&aliased)
                {
                    uses.insert(aliased);
                }
            }
            for &result_val in ctx.op_results(op) {
                if is_ptr_type(ctx, ctx.value_ty(result_val))
                    && !is_static_ptr(ctx, result_val)
                    && ptr_values.contains(&result_val)
                {
                    defs.insert(result_val);
                }
            }
        }

        use_sets.insert(block, uses);
        def_sets.insert(block, defs);
    }

    (use_sets, def_sets)
}

/// Compute liveness via backward dataflow.
fn compute_liveness(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
) -> LivenessInfo {
    let (use_sets, def_sets) = compute_use_def_sets(ctx, body, ptr_values, ptr_alias_map);
    let successor_map = build_successor_map(ctx, body);
    let block_refs: Vec<BlockRef> = ctx.region(body).blocks.to_vec();

    let mut live_in: HashMap<BlockRef, HashSet<ValueRef>> = HashMap::new();
    let mut live_out: HashMap<BlockRef, HashSet<ValueRef>> = HashMap::new();

    for &b in &block_refs {
        live_in.insert(b, HashSet::new());
        live_out.insert(b, HashSet::new());
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &bid in block_refs.iter().rev() {
            let mut new_live_out = HashSet::new();
            if let Some(succs) = successor_map.get(&bid) {
                for succ in succs {
                    if let Some(succ_live_in) = live_in.get(succ) {
                        new_live_out.extend(succ_live_in.iter().copied());
                    }
                }
            }

            let use_b = use_sets.get(&bid).cloned().unwrap_or_default();
            let def_b = def_sets.get(&bid).cloned().unwrap_or_default();
            let mut new_live_in = use_b;
            for v in &new_live_out {
                if !def_b.contains(v) {
                    new_live_in.insert(*v);
                }
            }

            if new_live_in != *live_in.get(&bid).unwrap() {
                live_in.insert(bid, new_live_in);
                changed = true;
            }
            if new_live_out != *live_out.get(&bid).unwrap() {
                live_out.insert(bid, new_live_out);
                changed = true;
            }
        }
    }

    LivenessInfo {
        def_set: def_sets,
        live_in,
        live_out,
    }
}

// =============================================================================
// RC Insertion
// =============================================================================

/// Insertion plan for RC operations.
#[derive(Default)]
struct InsertionPlan {
    before: HashMap<usize, Vec<OpRef>>,
    after: HashMap<usize, Vec<OpRef>>,
    at_start: Vec<OpRef>,
}

/// Check if value has any use after given op index in block.
fn has_use_after(
    ctx: &IrContext,
    ops: &[OpRef],
    after_idx: usize,
    value: ValueRef,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
) -> bool {
    for &op in ops.iter().skip(after_idx + 1) {
        for &operand in ctx.op_operands(op) {
            if operand == value {
                return true;
            }
            if let Some(&aliased) = ptr_alias_map.get(&operand)
                && aliased == value
            {
                return true;
            }
        }
    }
    false
}

/// Check if value is defined before given op index.
fn is_defined_before(ctx: &IrContext, ops: &[OpRef], before_idx: usize, value: ValueRef) -> bool {
    match ctx.value_def(value) {
        ValueDef::BlockArg(_, _) => true,
        ValueDef::OpResult(def_op, _) => {
            for (i, &op) in ops.iter().enumerate() {
                if i >= before_idx {
                    return false;
                }
                if op == def_op {
                    return true;
                }
            }
            false
        }
    }
}

/// Check if an operation is `clif.call @__tribute_yield`.
fn is_yield_call(ctx: &IrContext, op: OpRef) -> bool {
    if let Ok(call_op) = arena_clif::Call::from_op(ctx, op) {
        call_op.callee(ctx) == Symbol::new("__tribute_yield")
    } else {
        false
    }
}

/// Insert reference counting operations for all pointer-typed values.
pub fn insert_rc(ctx: &mut IrContext, module: ArenaModule) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in module_ops {
        if let Ok(func_op) = arena_clif::Func::from_op(ctx, op) {
            let sym = func_op.sym_name(ctx);
            if sym.with_str(|s| s.starts_with(super::rtti::RELEASE_FN_PREFIX)) {
                continue;
            }
            let body = func_op.body(ctx);
            insert_rc_in_function(ctx, body);
        }
    }
}

/// Insert RC in a function body.
fn insert_rc_in_function(ctx: &mut IrContext, body: RegionRef) {
    let mut ptr_values = collect_ptr_values(ctx, body);

    if ptr_values.is_empty() {
        return;
    }

    let ptr_alias_map = build_ptr_alias_map(ctx, body, &mut ptr_values);
    let liveness = compute_liveness(ctx, body, &ptr_values, &ptr_alias_map);

    let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();
    for (block_idx, &block) in blocks.iter().enumerate() {
        insert_rc_in_block(
            ctx,
            block,
            block_idx == 0,
            &ptr_values,
            &liveness,
            &ptr_alias_map,
        );
    }
}

/// Insert RC ops in a single block.
fn insert_rc_in_block(
    ctx: &mut IrContext,
    block: BlockRef,
    is_entry: bool,
    ptr_values: &HashSet<ValueRef>,
    liveness: &LivenessInfo,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let loc = ctx.block(block).location;
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());

    let live_in = liveness.live_in.get(&block).cloned().unwrap_or_default();
    let live_out = liveness.live_out.get(&block).cloned().unwrap_or_default();

    // Compute per-value last use index
    let mut last_use_in_block: HashMap<ValueRef, usize> = HashMap::new();
    for (op_idx, &op) in ops.iter().enumerate() {
        for &operand in ctx.op_operands(op) {
            if ptr_values.contains(&operand) {
                last_use_in_block.insert(operand, op_idx);
            }
            if let Some(&aliased) = ptr_alias_map.get(&operand) {
                last_use_in_block.insert(aliased, op_idx);
            }
        }
    }

    // Returned values
    let mut returned_values: HashSet<ValueRef> = HashSet::new();
    if let Some(&last_op) = ops.last()
        && arena_clif::Return::matches(ctx, last_op)
    {
        for &operand in ctx.op_operands(last_op) {
            if ptr_values.contains(&operand) {
                returned_values.insert(operand);
            }
            if let Some(&aliased) = ptr_alias_map.get(&operand) {
                returned_values.insert(aliased);
            }
        }
    }

    let mut plan = InsertionPlan::default();

    // --- Retain insertions ---

    // 1. Entry block: retain each ptr parameter
    if is_entry {
        let args: Vec<ValueRef> = ctx.block_args(block).to_vec();
        for arg_val in args {
            if is_ptr_type(ctx, ctx.value_ty(arg_val)) {
                let retain_op = arena_tribute_rt::retain(ctx, loc, arg_val, ptr_ty);
                plan.at_start.push(retain_op.op_ref());
            }
        }
    }

    // 2. Retain before store of ptr, retain after load of ptr
    for (op_idx, &op) in ops.iter().enumerate() {
        if let Ok(_store_op) = arena_clif::Store::from_op(ctx, op) {
            let operands = ctx.op_operands(op).to_vec();
            if let Some(&stored_val) = operands.first()
                && is_ptr_value(ctx, stored_val)
                && !is_static_ptr(ctx, stored_val)
            {
                let op_loc = ctx.op(op).location;
                let retain_op = arena_tribute_rt::retain(ctx, op_loc, stored_val, ptr_ty);
                plan.before
                    .entry(op_idx)
                    .or_default()
                    .push(retain_op.op_ref());
            }
        }

        if arena_clif::Load::matches(ctx, op) {
            let result_ty = ctx.op_result_types(op).first().copied();
            if result_ty.is_some_and(|ty| is_ptr_type(ctx, ty)) {
                let load_result = ctx.op_result(op, 0);
                let op_loc = ctx.op(op).location;
                let retain_op = arena_tribute_rt::retain(ctx, op_loc, load_result, ptr_ty);
                plan.after
                    .entry(op_idx)
                    .or_default()
                    .push(retain_op.op_ref());
            }
        }
    }

    let defs_in_block = liveness.def_set.get(&block).cloned().unwrap_or_default();

    // 3. Yield handling
    for (op_idx, &op) in ops.iter().enumerate() {
        if !is_yield_call(ctx, op) {
            continue;
        }

        let live_across_yield: Vec<ValueRef> = {
            let mut live: Vec<ValueRef> = Vec::new();
            for v in &live_in {
                if live_out.contains(v) || has_use_after(ctx, &ops, op_idx, *v, ptr_alias_map) {
                    live.push(*v);
                }
            }
            for v in &defs_in_block {
                if (live_out.contains(v) || has_use_after(ctx, &ops, op_idx, *v, ptr_alias_map))
                    && is_defined_before(ctx, &ops, op_idx, *v)
                    && !live.contains(v)
                {
                    live.push(*v);
                }
            }
            live.sort_by_key(|v| match ctx.value_def(*v) {
                ValueDef::BlockArg(_, idx) => (0usize, idx),
                ValueDef::OpResult(def_op, idx) => {
                    let pos = ops.iter().position(|&o| o == def_op).unwrap_or(usize::MAX);
                    (pos.saturating_add(1), idx)
                }
            });
            live
        };

        if live_across_yield.is_empty() {
            continue;
        }

        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let i64_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
        let nil_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());
        let op_loc = ctx.op(op).location;

        let before_ops = plan.before.entry(op_idx).or_default();

        for v in &live_across_yield {
            let retain_op = arena_tribute_rt::retain(ctx, op_loc, *v, ptr_ty);
            before_ops.push(retain_op.op_ref());
        }

        let roots_count = live_across_yield.len();
        let array_size = (roots_count * 16) as i64;

        let alloc_size_const = arena_clif::iconst(ctx, op_loc, i64_ty, array_size);
        before_ops.push(alloc_size_const.op_ref());

        let alloc_call = arena_clif::call(
            ctx,
            op_loc,
            [alloc_size_const.result(ctx)],
            ptr_ty,
            Symbol::new("__tribute_alloc"),
        );
        before_ops.push(alloc_call.op_ref());
        let roots_ptr = alloc_call.result(ctx);

        for (i, v) in live_across_yield.iter().enumerate() {
            let offset = (i * 16) as i32;
            let store_ptr = arena_clif::store(ctx, op_loc, *v, roots_ptr, offset);
            before_ops.push(store_ptr.op_ref());
            let obj_alloc_size = infer_alloc_size(ctx, *v);
            let size_const = arena_clif::iconst(ctx, op_loc, i64_ty, obj_alloc_size as i64);
            before_ops.push(size_const.op_ref());
            let store_size =
                arena_clif::store(ctx, op_loc, size_const.result(ctx), roots_ptr, offset + 8);
            before_ops.push(store_size.op_ref());
        }

        let count_const = arena_clif::iconst(ctx, op_loc, i32_ty, roots_count as i64);
        before_ops.push(count_const.op_ref());

        let set_roots_call = arena_clif::call(
            ctx,
            op_loc,
            [roots_ptr, count_const.result(ctx)],
            nil_ty,
            Symbol::new("__tribute_yield_set_rc_roots"),
        );
        before_ops.push(set_roots_call.op_ref());

        let after_ops = plan.after.entry(op_idx).or_default();
        for v in &live_across_yield {
            let obj_alloc_size = infer_alloc_size(ctx, *v);
            let release_op = arena_tribute_rt::release(ctx, op_loc, *v, obj_alloc_size);
            after_ops.push(release_op.op_ref());
        }
    }

    // --- Release insertions ---
    let mut dying_values: HashSet<ValueRef> = HashSet::new();

    for v in &live_in {
        if !live_out.contains(v) && !returned_values.contains(v) {
            dying_values.insert(*v);
        }
    }
    for v in &defs_in_block {
        if !live_out.contains(v) && !returned_values.contains(v) && !is_alloc_intermediate(ctx, *v)
        {
            dying_values.insert(*v);
        }
    }

    let mut dying_sorted: Vec<ValueRef> = dying_values.into_iter().collect();
    dying_sorted.sort_by_key(|v| match ctx.value_def(*v) {
        ValueDef::BlockArg(_, idx) => (0usize, idx),
        ValueDef::OpResult(def_op, idx) => {
            let pos = ops.iter().position(|&o| o == def_op).unwrap_or(usize::MAX);
            (pos.saturating_add(1), idx)
        }
    });

    for v in &dying_sorted {
        if let Some(&last_use_idx) = last_use_in_block.get(v) {
            let last_op = ops[last_use_idx];
            if arena_clif::Return::matches(ctx, last_op) {
                continue;
            }
            let alloc_size = infer_alloc_size(ctx, *v);
            let op_loc = ctx.op(last_op).location;
            let release_op = arena_tribute_rt::release(ctx, op_loc, *v, alloc_size);
            if is_terminator_op(ctx, last_op) {
                if arena_clif::Jump::matches(ctx, last_op) {
                    continue;
                }
                plan.before
                    .entry(last_use_idx)
                    .or_default()
                    .push(release_op.op_ref());
            } else {
                plan.after
                    .entry(last_use_idx)
                    .or_default()
                    .push(release_op.op_ref());
            }
        } else if live_in.contains(v) {
            let alloc_size = infer_alloc_size(ctx, *v);
            let release_op = arena_tribute_rt::release(ctx, loc, *v, alloc_size);
            plan.at_start.push(release_op.op_ref());
        } else if let ValueDef::OpResult(def_op, _) = ctx.value_def(*v) {
            for (op_idx, &op) in ops.iter().enumerate() {
                if op == def_op {
                    let alloc_size = infer_alloc_size(ctx, *v);
                    let op_loc = ctx.op(op).location;
                    let release_op = arena_tribute_rt::release(ctx, op_loc, *v, alloc_size);
                    plan.after
                        .entry(op_idx)
                        .or_default()
                        .push(release_op.op_ref());
                    break;
                }
            }
        } else if let ValueDef::BlockArg(_, _) = ctx.value_def(*v) {
            let alloc_size = infer_alloc_size(ctx, *v);
            let release_op = arena_tribute_rt::release(ctx, loc, *v, alloc_size);
            plan.at_start.push(release_op.op_ref());
        }
    }

    // --- Apply insertion plan ---
    apply_insertion_plan(ctx, block, &ops, &plan);
}

/// Apply insertion plan by removing and re-inserting ops in order.
fn apply_insertion_plan(
    ctx: &mut IrContext,
    block: BlockRef,
    original_ops: &[OpRef],
    plan: &InsertionPlan,
) {
    let has_changes =
        !plan.at_start.is_empty() || !plan.before.is_empty() || !plan.after.is_empty();
    if !has_changes {
        return;
    }

    // Detach all original ops from the block
    for &op in original_ops {
        ctx.remove_op_from_block(block, op);
    }

    // Re-insert in order with plan insertions
    for &op in &plan.at_start {
        ctx.push_op(block, op);
    }

    for (idx, &op) in original_ops.iter().enumerate() {
        if let Some(before_ops) = plan.before.get(&idx) {
            for &bop in before_ops {
                ctx.push_op(block, bop);
            }
        }
        ctx.push_op(block, op);
        if let Some(after_ops) = plan.after.get(&idx) {
            for &aop in after_ops {
                ctx.push_op(block, aop);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::arena::context::IrContext;
    use trunk_ir::arena::parser::parse_test_module;
    use trunk_ir::arena::printer::print_module;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        insert_rc(&mut ctx, module);
        print_module(&ctx, module.op())
    }

    // =========================================================================
    // Snapshot tests
    // =========================================================================

    #[test]
    fn test_snapshot_simple_param() {
        // ptr parameter → load → return: retain at entry, release after last non-return use
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
}"#,
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_snapshot_alloc_store_return() {
        // alloc → store → return ptr: no RC for returned alloc (ownership transfer)
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.i32) -> core.ptr {
    %1 = clif.iconst {value = 16} : core.i64
    %2 = clif.call %1 {callee = @__tribute_alloc} : core.ptr
    %3 = clif.iconst {value = 8} : core.i64
    %4 = clif.iadd %2, %3 : core.ptr
    clif.store %0, %4 {offset = 0}
    clif.return %4
  }
}"#,
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_snapshot_multiple_uses() {
        // ptr param used in two loads — release after last use
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    %2 = clif.load %0 {offset = 4} : core.i32
    clif.return %1
  }
}"#,
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_snapshot_yield_with_live_ptr() {
        // ptr param + yield: RC root setup around yield
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr, %1: core.ptr) -> core.nil {
    %2 = clif.call %1 {callee = @__tribute_yield} : core.nil
    %3 = clif.load %0 {offset = 0} : core.i32
    clif.return %2
  }
}"#,
        );
        insta::assert_snapshot!(output);
    }

    // =========================================================================
    // Unit tests
    // =========================================================================

    #[test]
    fn test_symbol_addr_no_rc() {
        // clif.symbol_addr produces static pointer — no retain/release
        let output = run_pass(
            r#"core.module @test {
  clif.func @f() -> core.ptr {
    %0 = clif.symbol_addr {sym = @some_global} : core.ptr
    clif.return %0
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt.retain"),
            "symbol_addr should not be retained"
        );
        assert!(
            !output.contains("tribute_rt.release"),
            "symbol_addr should not be released"
        );
    }

    #[test]
    fn test_null_ptr_iconst_no_rc() {
        // clif.iconst 0 : ptr is a null pointer — no RC
        let output = run_pass(
            r#"core.module @test {
  clif.func @f() -> core.ptr {
    %0 = clif.iconst {value = 0} : core.ptr
    clif.return %0
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt.retain"),
            "null ptr iconst should not be retained"
        );
        assert!(
            !output.contains("tribute_rt.release"),
            "null ptr iconst should not be released"
        );
    }

    #[test]
    fn test_no_ptr_noop() {
        // i32-only function — no RC ops inserted
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.i32) -> core.i32 {
    %1 = clif.iconst {value = 42} : core.i32
    clif.return %1
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt"),
            "no ptr values means no RC ops"
        );
    }

    #[test]
    fn test_store_ptr_retains() {
        // store ptr into ptr: retain before store
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr, %1: core.ptr) -> core.nil {
    clif.store %0, %1 {offset = 0}
    %2 = clif.iconst {value = 0} : core.nil
    clif.return %2
  }
}"#,
        );
        assert!(
            output.contains("tribute_rt.retain"),
            "store of ptr should insert retain"
        );
    }

    #[test]
    fn test_load_ptr_retains() {
        // load ptr from ptr: retain after load
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.ptr {
    %1 = clif.load %0 {offset = 0} : core.ptr
    clif.return %1
  }
}"#,
        );
        // Should retain the loaded ptr and the parameter
        let retain_count = output.matches("tribute_rt.retain").count();
        assert!(
            retain_count >= 2,
            "should retain param and loaded ptr, got {retain_count}"
        );
    }

    #[test]
    fn test_unused_ptr_param_released() {
        // unused ptr param: retain + release both present
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.i32 {
    %1 = clif.iconst {value = 0} : core.i32
    clif.return %1
  }
}"#,
        );
        assert!(
            output.contains("tribute_rt.retain"),
            "unused ptr param should still be retained"
        );
        assert!(
            output.contains("tribute_rt.release"),
            "unused ptr param should be released"
        );
    }

    #[test]
    fn test_alloc_return_no_release() {
        // alloc and return: no release (ownership transfer)
        let output = run_pass(
            r#"core.module @test {
  clif.func @f() -> core.ptr {
    %0 = clif.iconst {value = 16} : core.i64
    %1 = clif.call %0 {callee = @__tribute_alloc} : core.ptr
    %2 = clif.iconst {value = 8} : core.i64
    %3 = clif.iadd %1, %2 : core.ptr
    clif.return %3
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt.release"),
            "returned alloc should not be released"
        );
    }
}
