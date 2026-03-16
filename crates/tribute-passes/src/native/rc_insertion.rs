//! Reference counting insertion pass.
//!
//! Automatically inserts `tribute_rt.retain` and `tribute_rt.release` operations
//! for `tribute_rt.anyref`-typed values in the native backend pipeline.
//!
//! Only `tribute_rt.anyref` values are RC-managed. Plain `core.ptr` values
//! (function pointers, continuations, null sentinels) are not affected.
//!
//! ## Pipeline Position
//!
//! Runs after Phase 2.7 (`tribute_rt_to_clif` boxing lowering), where:
//! - All allocations are `clif.call @__tribute_alloc`
//! - Boxing ops are already lowered to clif with `tribute_rt.anyref` result type
//! - `tribute_rt.retain`/`release` are preserved as legal ops
//!
//! ## RC Rules
//!
//! ### Retain (reference acquisition)
//!
//! | Situation | Action |
//! |-----------|--------|
//! | Function parameter (anyref) | `retain` at entry |
//! | `clif.call @__tribute_alloc` result | No retain (starts with refcount=1) |
//! | Other `clif.call` result (anyref) | No retain (ownership transfer) |
//! | `clif.store` with anyref value | `retain` before store |
//! | `clif.load` with anyref result | `retain` after load |
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
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif as arena_clif;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;
use trunk_ir::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};

use tribute_ir::dialect::tribute_rt as arena_tribute_rt;

/// Check if a type is `tribute_rt.anyref` (RC-managed reference type).
fn is_anyref_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("tribute_rt") && data.name == Symbol::new("anyref")
}

/// Check if a value is an anyref type (RC-managed).
fn is_anyref_value(ctx: &IrContext, value: ValueRef) -> bool {
    is_anyref_type(ctx, ctx.value_ty(value))
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
    if arena_clif::Iconst::matches(ctx, def_op) && is_anyref_type(ctx, ctx.value_ty(value)) {
        // Only treat null (zero) constants as unmanaged
        if let Ok(iconst) = arena_clif::Iconst::from_op(ctx, def_op)
            && iconst.value(ctx) == 0
        {
            return true;
        }
        return false;
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
            if is_anyref_type(ctx, ctx.value_ty(arg_val)) {
                ptr_values.insert(arg_val);
            }
        }
        // Operation results
        for &op in &ctx.block(block).ops.to_vec() {
            for &result_val in ctx.op_results(op) {
                if is_anyref_type(ctx, ctx.value_ty(result_val)) && !is_static_ptr(ctx, result_val)
                {
                    ptr_values.insert(result_val);
                }
            }
        }
    }

    // Also check operands
    for &block in &blocks {
        for &op in &ctx.block(block).ops.to_vec() {
            for &operand in ctx.op_operands(op) {
                if is_anyref_value(ctx, operand) && !is_static_ptr(ctx, operand) {
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
            if is_anyref_type(ctx, ctx.value_ty(arg_val)) {
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
                if is_anyref_type(ctx, ctx.value_ty(result_val))
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

/// Insert reference counting operations for all `tribute_rt.anyref`-typed values,
/// then lower all remaining `tribute_rt.anyref` types to `core.ptr`.
pub fn insert_rc(ctx: &mut IrContext, module: Module) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in &module_ops {
        if let Ok(func_op) = arena_clif::Func::from_op(ctx, *op) {
            let sym = func_op.sym_name(ctx);
            if sym.with_str(|s| s.starts_with(super::rtti::RELEASE_FN_PREFIX)) {
                continue;
            }
            let body = func_op.body(ctx);
            insert_rc_in_function(ctx, body);
        }
    }

    // After RC insertion, lower all remaining `tribute_rt.anyref` types to `core.ptr`.
    // This ensures anyref doesn't survive past RC insertion into the Cranelift emit phase.
    lower_anyref_to_ptr(ctx, module);
}

/// Rewrite all `tribute_rt.anyref` types to `core.ptr` in the module.
///
/// After RC insertion has used anyref to identify RC-managed values, the type
/// distinction is no longer needed. All anyref types are lowered to core.ptr
/// so that subsequent passes (resolve_casts, Cranelift emit) see only core types.
fn lower_anyref_to_ptr(ctx: &mut IrContext, module: Module) {
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();
    let anyref_ty = ctx.types.intern(
        trunk_ir::TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build(),
    );
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in module_ops {
        if let Ok(func_op) = arena_clif::Func::from_op(ctx, op) {
            // Rewrite function type attribute (anyref → ptr in params/return)
            let func_type = func_op.r#type(ctx);
            let new_func_type = rewrite_func_type(ctx, func_type, anyref_ty, ptr_ty);
            if new_func_type != func_type {
                ctx.op_mut(op).attributes.insert(
                    Symbol::new("type"),
                    trunk_ir::Attribute::Type(new_func_type),
                );
            }

            let body = func_op.body(ctx);
            lower_anyref_in_region(ctx, body, ptr_ty);
        }
    }
}

/// Rewrite anyref types to ptr in a core.func type.
fn rewrite_func_type(
    ctx: &mut IrContext,
    func_ty: TypeRef,
    anyref_ty: TypeRef,
    ptr_ty: TypeRef,
) -> TypeRef {
    rewrite_type_anyref(ctx, func_ty, anyref_ty, ptr_ty)
}

/// Rewrite anyref types to ptr in a region (function body).
fn lower_anyref_in_region(ctx: &mut IrContext, region: RegionRef, ptr_ty: TypeRef) {
    let anyref_ty = ctx.types.intern(
        trunk_ir::TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build(),
    );

    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        // Rewrite block argument types
        let arg_count = ctx.block_args(block).len();
        for idx in 0..arg_count {
            let arg_val = ctx.block_args(block)[idx];
            if is_anyref_type(ctx, ctx.value_ty(arg_val)) {
                ctx.set_block_arg_type(block, idx as u32, ptr_ty);
            }
        }

        // Rewrite operation result types and type attributes
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            let result_count = ctx.op_results(op).len();
            for idx in 0..result_count {
                let result_val = ctx.op_results(op)[idx];
                if is_anyref_type(ctx, ctx.value_ty(result_val)) {
                    ctx.set_op_result_type(op, idx as u32, ptr_ty);
                }
            }

            // Rewrite type attributes (e.g., clif.call_indirect's sig attribute)
            rewrite_op_type_attrs(ctx, op, anyref_ty, ptr_ty);

            // Recurse into nested regions
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for r in regions {
                lower_anyref_in_region(ctx, r, ptr_ty);
            }
        }
    }
}

/// Rewrite anyref types in operation type attributes.
fn rewrite_op_type_attrs(ctx: &mut IrContext, op: OpRef, anyref_ty: TypeRef, ptr_ty: TypeRef) {
    let attrs: Vec<(Symbol, trunk_ir::Attribute)> = ctx
        .op(op)
        .attributes
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    for (key, attr) in attrs {
        if let trunk_ir::Attribute::Type(ty) = attr {
            let new_ty = rewrite_type_anyref(ctx, ty, anyref_ty, ptr_ty);
            if new_ty != ty {
                ctx.op_mut(op)
                    .attributes
                    .insert(key, trunk_ir::Attribute::Type(new_ty));
            }
        }
    }
}

/// Recursively rewrite anyref in a type (handles core.func params).
fn rewrite_type_anyref(
    ctx: &mut IrContext,
    ty: TypeRef,
    anyref_ty: TypeRef,
    ptr_ty: TypeRef,
) -> TypeRef {
    if ty == anyref_ty {
        return ptr_ty;
    }
    let data = ctx.types.get(ty);
    // Only recurse into core.func types (which have params that may contain anyref)
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return ty;
    }
    // Collect params and attrs before mutating ctx
    let params: Vec<TypeRef> = data.params.to_vec();
    let dialect = data.dialect;
    let name = data.name;
    let attrs: Vec<(Symbol, trunk_ir::Attribute)> =
        data.attrs.iter().map(|(k, v)| (*k, v.clone())).collect();

    let mut changed = false;
    let new_params: Vec<TypeRef> = params
        .iter()
        .map(|&p| {
            let new_p = rewrite_type_anyref(ctx, p, anyref_ty, ptr_ty);
            if new_p != p {
                changed = true;
            }
            new_p
        })
        .collect();
    if !changed {
        return ty;
    }
    let mut builder = trunk_ir::TypeDataBuilder::new(dialect, name);
    for &p in &new_params {
        builder = builder.param(p);
    }
    for (key, attr) in attrs {
        builder = builder.attr(key, attr);
    }
    ctx.types.intern(builder.build())
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
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();

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
            if is_anyref_type(ctx, ctx.value_ty(arg_val)) {
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
                && is_anyref_value(ctx, stored_val)
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
            if result_ty.is_some_and(|ty| is_anyref_type(ctx, ty)) {
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
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

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
        // anyref parameter → load → return: retain at entry, release after last non-return use
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
}"#,
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_snapshot_alloc_store_return() {
        // alloc → store → return anyref: no RC for returned alloc (ownership transfer)
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.i32) -> tribute_rt.anyref {
    %1 = clif.iconst {value = 16} : core.i64
    %2 = clif.call %1 {callee = @__tribute_alloc} : core.ptr
    %3 = clif.iconst {value = 8} : core.i64
    %4 = clif.iadd %2, %3 : tribute_rt.anyref
    clif.store %0, %4 {offset = 0}
    clif.return %4
  }
}"#,
        );
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_snapshot_multiple_uses() {
        // anyref param used in two loads — release after last use
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    %2 = clif.load %0 {offset = 4} : core.i32
    clif.return %1
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
        // clif.symbol_addr produces core.ptr (static pointer) — no retain/release
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
    fn test_store_anyref_retains() {
        // store anyref into ptr: retain before store
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.ptr) -> core.nil {
    clif.store %0, %1 {offset = 0}
    %2 = clif.iconst {value = 0} : core.nil
    clif.return %2
  }
}"#,
        );
        assert!(
            output.contains("tribute_rt.retain"),
            "store of anyref should insert retain"
        );
    }

    #[test]
    fn test_load_anyref_retains() {
        // load anyref from ptr: retain after load
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> tribute_rt.anyref {
    %1 = clif.load %0 {offset = 0} : tribute_rt.anyref
    clif.return %1
  }
}"#,
        );
        // Should retain the loaded anyref and the parameter
        let retain_count = output.matches("tribute_rt.retain").count();
        assert!(
            retain_count >= 2,
            "should retain param and loaded anyref, got {retain_count}"
        );
    }

    #[test]
    fn test_unused_anyref_param_released() {
        // unused anyref param: retain + release both present
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.iconst {value = 0} : core.i32
    clif.return %1
  }
}"#,
        );
        assert!(
            output.contains("tribute_rt.retain"),
            "unused anyref param should still be retained"
        );
        assert!(
            output.contains("tribute_rt.release"),
            "unused anyref param should be released"
        );
    }

    #[test]
    fn test_alloc_return_no_release() {
        // alloc and return anyref: no release (ownership transfer)
        let output = run_pass(
            r#"core.module @test {
  clif.func @f() -> tribute_rt.anyref {
    %0 = clif.iconst {value = 16} : core.i64
    %1 = clif.call %0 {callee = @__tribute_alloc} : core.ptr
    %2 = clif.iconst {value = 8} : core.i64
    %3 = clif.iadd %1, %2 : tribute_rt.anyref
    clif.return %3
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt.release"),
            "returned alloc should not be released"
        );
    }

    #[test]
    fn test_core_ptr_no_rc() {
        // core.ptr parameters are NOT RC-managed — no retain/release
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
}"#,
        );
        assert!(
            !output.contains("tribute_rt.retain"),
            "core.ptr param should not be retained"
        );
        assert!(
            !output.contains("tribute_rt.release"),
            "core.ptr param should not be released"
        );
    }

    #[test]
    fn test_mixed_anyref_and_ptr() {
        // anyref + core.ptr params: only anyref gets RC
        let output = run_pass(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.ptr) -> core.i32 {
    %2 = clif.load %0 {offset = 0} : core.i32
    %3 = clif.load %1 {offset = 0} : core.i32
    clif.return %2
  }
}"#,
        );
        // Only 1 retain for the anyref param, none for core.ptr
        let retain_count = output.matches("tribute_rt.retain").count();
        assert_eq!(
            retain_count, 1,
            "should retain only anyref param, got {retain_count}"
        );
    }
}
