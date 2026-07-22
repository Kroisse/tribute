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
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core;
use trunk_ir::dominance::DominatorTree;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;
use trunk_ir::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};

use tribute_ir::dialect::tribute_rt;

use super::ownership_summary::{
    BorrowedUse, BorrowedUseKind, ParameterOwnership, TrustedOwnershipSummaries,
    classify_borrowed_use,
};

/// Policy for eliding RC ownership of proven borrowed function parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorrowedParameterPolicy {
    /// Preserve the current owned-parameter convention for every parameter.
    Preserve,
    /// Omit parameter RC only when all uses are proven non-escaping.
    ElideProvenBorrowed,
}

/// Policy for eliding ownership of proven field-derived temporaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporaryBorrowPolicy {
    Preserve,
    ElideProvenFieldBorrows,
}

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
    clif::Return::matches(ctx, op)
        || clif::Jump::matches(ctx, op)
        || clif::Brif::matches(ctx, op)
        || clif::Trap::matches(ctx, op)
        || clif::ReturnCall::matches(ctx, op)
        || clif::BrTable::matches(ctx, op)
}

/// Check if a value is a static pointer (not RC-managed).
fn is_static_ptr(ctx: &IrContext, value: ValueRef) -> bool {
    let ValueDef::OpResult(def_op, _) = ctx.value_def(value) else {
        return false;
    };
    if clif::SymbolAddr::matches(ctx, def_op) {
        return true;
    }
    if clif::Iconst::matches(ctx, def_op) && is_anyref_type(ctx, ctx.value_ty(value)) {
        // Only treat null (zero) constants as unmanaged
        if let Ok(iconst) = clif::Iconst::from_op(ctx, def_op)
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
    if let Ok(call_op) = clif::Call::from_op(ctx, def_op) {
        return call_op.callee(ctx) == Symbol::new("__tribute_alloc");
    }
    if let Ok(_iadd_op) = clif::Iadd::from_op(ctx, def_op) {
        let operands = ctx.op_operands(def_op).to_vec();
        if let Some(&lhs) = operands.first() {
            let ValueDef::OpResult(lhs_op, _) = ctx.value_def(lhs) else {
                return false;
            };
            if let Ok(call_op) = clif::Call::from_op(ctx, lhs_op) {
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
    let Ok(call_op) = clif::Call::from_op(ctx, def_op) else {
        if let Ok(_iadd_op) = clif::Iadd::from_op(ctx, def_op) {
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
    if let Ok(iconst_op) = clif::Iconst::from_op(ctx, size_op) {
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

#[derive(Default)]
struct TemporaryBorrowInfo {
    borrowed: HashSet<ValueRef>,
    lifetime_dependencies: HashMap<ValueRef, ValueRef>,
}

struct RcBorrowInfo<'a> {
    parameters: &'a HashSet<ValueRef>,
    temporaries: &'a HashSet<ValueRef>,
    lifetime_dependencies: &'a HashMap<ValueRef, ValueRef>,
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
            if core::UnrealizedConversionCast::matches(ctx, op) {
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
    lifetime_dependencies: &HashMap<ValueRef, ValueRef>,
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
                record_lifetime_dependencies(operand, lifetime_dependencies, |owner| {
                    if !defs.contains(&owner) {
                        uses.insert(owner);
                    }
                });
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
    lifetime_dependencies: &HashMap<ValueRef, ValueRef>,
) -> LivenessInfo {
    let (use_sets, def_sets) =
        compute_use_def_sets(ctx, body, ptr_values, ptr_alias_map, lifetime_dependencies);
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
    insert_rc_with_policies(
        ctx,
        module,
        BorrowedParameterPolicy::Preserve,
        TemporaryBorrowPolicy::Preserve,
    );
}

/// Insert reference counting operations with an explicit borrowed-parameter
/// policy, then lower all remaining `tribute_rt.anyref` types to `core.ptr`.
pub fn insert_rc_with_policy(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
) {
    insert_rc_with_policies(
        ctx,
        module,
        borrowed_parameters,
        TemporaryBorrowPolicy::Preserve,
    );
}

pub fn insert_rc_with_trusted_summaries(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    trusted_summaries: &TrustedOwnershipSummaries,
) {
    insert_rc_with_policies_and_trusted_summaries(
        ctx,
        module,
        borrowed_parameters,
        TemporaryBorrowPolicy::Preserve,
        trusted_summaries,
    );
}

/// Insert reference counting operations with independently selectable borrow
/// policies, then lower all remaining `tribute_rt.anyref` types to `core.ptr`.
///
/// Without trusted ownership summaries, `ElideProvenBorrowed` falls back to
/// preserving parameter ownership; this entrypoint does not perform local-only
/// borrowed-parameter elision.
pub fn insert_rc_with_policies(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    temporary_borrows: TemporaryBorrowPolicy,
) {
    insert_rc_impl(ctx, module, borrowed_parameters, temporary_borrows, None);
}

pub fn insert_rc_with_policies_and_trusted_summaries(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    temporary_borrows: TemporaryBorrowPolicy,
    trusted_summaries: &TrustedOwnershipSummaries,
) {
    insert_rc_impl(
        ctx,
        module,
        borrowed_parameters,
        temporary_borrows,
        Some(trusted_summaries),
    );
}

fn insert_rc_impl(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    temporary_borrows: TemporaryBorrowPolicy,
    trusted_summaries: Option<&TrustedOwnershipSummaries>,
) {
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();
    let trusted_summaries = trusted_summaries
        .map(|summaries| summaries.validated_for_clif(ctx, &module_ops))
        .unwrap_or_default();
    let borrow_safe_functions = borrow_safe_functions(ctx, &module_ops);

    for op in &module_ops {
        if let Ok(func_op) = clif::Func::from_op(ctx, *op) {
            let sym = func_op.sym_name(ctx);
            if sym.with_str(|s| s.starts_with(super::rtti::RELEASE_FN_PREFIX)) {
                continue;
            }
            let body = func_op.body(ctx);
            let function_policy = if borrow_safe_functions.contains(&sym) {
                borrowed_parameters
            } else {
                BorrowedParameterPolicy::Preserve
            };
            insert_rc_in_function(
                ctx,
                sym,
                body,
                function_policy,
                temporary_borrows,
                &trusted_summaries,
            );
        }
    }

    // After RC insertion, lower all remaining `tribute_rt.anyref` types to `core.ptr`.
    // This ensures anyref doesn't survive past RC insertion into the Cranelift emit phase.
    lower_anyref_to_ptr(ctx, module);
}

/// Functions whose callers are guaranteed to keep an owning frame alive for
/// the duration of an ordinary synchronous call.
///
/// A function is ineligible when it is externally callable, is the target of a
/// direct tail call, or its address is materialized for an indirect/escaping
/// call. These exclusions preserve the caller-lifetime premise of borrowed
/// parameters without requiring inter-procedural ownership summaries.
fn borrow_safe_functions(ctx: &IrContext, module_ops: &[OpRef]) -> HashSet<Symbol> {
    let mut candidates = HashSet::new();
    for &op in module_ops {
        if let Ok(func_op) = clif::Func::from_op(ctx, op)
            && !ctx.op(op).attributes.contains_key("abi")
        {
            candidates.insert(func_op.sym_name(ctx));
        }
    }

    let mut unsafe_callees = HashSet::new();
    for &op in module_ops {
        if let Ok(func_op) = clif::Func::from_op(ctx, op) {
            collect_borrow_unsafe_callees(ctx, func_op.body(ctx), &candidates, &mut unsafe_callees);
        }
    }
    candidates.retain(|symbol| !unsafe_callees.contains(symbol));
    candidates
}

fn collect_borrow_unsafe_callees(
    ctx: &IrContext,
    region: RegionRef,
    candidates: &HashSet<Symbol>,
    unsafe_callees: &mut HashSet<Symbol>,
) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if let Ok(return_call) = clif::ReturnCall::from_op(ctx, op) {
                let callee = return_call.callee(ctx);
                if candidates.contains(&callee) {
                    unsafe_callees.insert(callee);
                }
            }
            if let Ok(symbol_addr) = clif::SymbolAddr::from_op(ctx, op) {
                let symbol = symbol_addr.sym(ctx);
                if candidates.contains(&symbol) {
                    unsafe_callees.insert(symbol);
                }
            }
            for &nested in &ctx.op(op).regions {
                collect_borrow_unsafe_callees(ctx, nested, candidates, unsafe_callees);
            }
        }
    }
}

/// Rewrite all `tribute_rt.anyref` types to `core.ptr` in the module.
///
/// After RC insertion has used anyref to identify RC-managed values, the type
/// distinction is no longer needed. All anyref types are lowered to core.ptr
/// so that subsequent passes (resolve_casts, Cranelift emit) see only core types.
fn lower_anyref_to_ptr(ctx: &mut IrContext, module: Module) {
    let ptr_ty = core::ptr(ctx).as_type_ref();
    let anyref_ty = ctx.types.intern(
        trunk_ir::TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build(),
    );
    let Some(first_block) = module.first_block(ctx) else {
        return;
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();

    for op in module_ops {
        if let Ok(func_op) = clif::Func::from_op(ctx, op) {
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
fn insert_rc_in_function(
    ctx: &mut IrContext,
    function: Symbol,
    body: RegionRef,
    borrowed_parameter_policy: BorrowedParameterPolicy,
    temporary_borrow_policy: TemporaryBorrowPolicy,
    trusted_summaries: &HashMap<Symbol, Vec<ParameterOwnership>>,
) {
    let mut ptr_values = collect_ptr_values(ctx, body);

    if ptr_values.is_empty() {
        return;
    }

    let ptr_alias_map = build_ptr_alias_map(ctx, body, &mut ptr_values);
    let temporary_borrows = match temporary_borrow_policy {
        TemporaryBorrowPolicy::Preserve => TemporaryBorrowInfo::default(),
        TemporaryBorrowPolicy::ElideProvenFieldBorrows => {
            analyze_temporary_borrows(ctx, body, &ptr_values, &ptr_alias_map)
        }
    };
    let liveness = compute_liveness(
        ctx,
        body,
        &ptr_values,
        &ptr_alias_map,
        &temporary_borrows.lifetime_dependencies,
    );
    let borrowed_parameters = match borrowed_parameter_policy {
        BorrowedParameterPolicy::Preserve => HashSet::new(),
        BorrowedParameterPolicy::ElideProvenBorrowed => {
            analyze_borrowed_parameters(ctx, function, body, trusted_summaries)
        }
    };

    let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();
    let borrow_info = RcBorrowInfo {
        parameters: &borrowed_parameters,
        temporaries: &temporary_borrows.borrowed,
        lifetime_dependencies: &temporary_borrows.lifetime_dependencies,
    };
    for (block_idx, &block) in blocks.iter().enumerate() {
        insert_rc_in_block(
            ctx,
            block,
            block_idx == 0,
            &ptr_values,
            &liveness,
            &ptr_alias_map,
            &borrow_info,
        );
    }
}

fn analyze_temporary_borrows(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
) -> TemporaryBorrowInfo {
    let dominance = DominatorTree::compute(ctx, body);
    if !dominance.is_valid() {
        return TemporaryBorrowInfo::default();
    }

    let mut info = TemporaryBorrowInfo::default();
    for &block in &ctx.region(body).blocks {
        if !dominance.is_reachable(block) {
            continue;
        }
        for &op in &ctx.block(block).ops {
            if !clif::Load::matches(ctx, op) || ctx.op_results(op).len() != 1 {
                continue;
            }
            let temporary = ctx.op_result(op, 0);
            if !is_anyref_value(ctx, temporary) {
                continue;
            }
            let Some(&address) = ctx.op_operands(op).first() else {
                continue;
            };
            let Some(owner) = resolve_temporary_owner(
                ctx,
                address,
                ptr_values,
                ptr_alias_map,
                &mut HashSet::new(),
            ) else {
                continue;
            };
            if let Some(aliases) =
                temporary_is_proven_borrowed(ctx, body, op, temporary, owner, &dominance)
            {
                info.borrowed.insert(temporary);
                info.lifetime_dependencies.insert(temporary, owner);
                for alias in aliases {
                    info.lifetime_dependencies.insert(alias, owner);
                }
            }
        }
    }
    info
}

fn resolve_temporary_owner(
    ctx: &IrContext,
    value: ValueRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    visited: &mut HashSet<ValueRef>,
) -> Option<ValueRef> {
    if !visited.insert(value) {
        return None;
    }
    if ptr_values.contains(&value) {
        return Some(value);
    }
    if let Some(&owner) = ptr_alias_map.get(&value) {
        return resolve_temporary_owner(ctx, owner, ptr_values, ptr_alias_map, visited);
    }
    let ValueDef::OpResult(defining_op, _) = ctx.value_def(value) else {
        return None;
    };
    if core::UnrealizedConversionCast::matches(ctx, defining_op)
        || clif::Iadd::matches(ctx, defining_op)
    {
        let input = *ctx.op_operands(defining_op).first()?;
        return resolve_temporary_owner(ctx, input, ptr_values, ptr_alias_map, visited);
    }
    None
}

fn temporary_is_proven_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    load: OpRef,
    temporary: ValueRef,
    owner: ValueRef,
    dominance: &DominatorTree,
) -> Option<Vec<ValueRef>> {
    let load_block = ctx.op(load).parent_block?;
    if !value_dominates_op(ctx, body, owner, load, dominance) {
        return None;
    }

    let mut use_blocks = Vec::new();
    let mut aliases = Vec::new();
    let mut collector = TemporaryUseCollector {
        ctx,
        body,
        load,
        dominance,
        visited: HashSet::new(),
        use_blocks: &mut use_blocks,
        aliases: &mut aliases,
    };
    if !collector.collect(temporary) {
        return None;
    }

    for &user_block in &use_blocks {
        if user_block != load_block && dominance.predecessors(user_block).len() > 1 {
            return None;
        }
    }

    for (index, &left) in use_blocks.iter().enumerate() {
        for &right in &use_blocks[index + 1..] {
            if !dominance.dominates(left, right) && !dominance.dominates(right, left) {
                return None;
            }
        }
    }

    for &source in &ctx.region(body).blocks {
        for &successor in dominance.successors(source) {
            if dominance.dominates(successor, source)
                && dominance.dominates(load_block, source)
                && use_blocks
                    .iter()
                    .any(|&use_block| dominance.dominates(successor, use_block))
            {
                return None;
            }
        }
    }

    Some(aliases)
}

struct TemporaryUseCollector<'a> {
    ctx: &'a IrContext,
    body: RegionRef,
    load: OpRef,
    dominance: &'a DominatorTree,
    visited: HashSet<ValueRef>,
    use_blocks: &'a mut Vec<BlockRef>,
    aliases: &'a mut Vec<ValueRef>,
}

impl TemporaryUseCollector<'_> {
    fn collect(&mut self, value: ValueRef) -> bool {
        if !self.visited.insert(value) {
            return false;
        }
        for use_ in self.ctx.uses(value) {
            let user = use_.user;
            let Some(user_block) = self.ctx.op(user).parent_block else {
                return false;
            };
            if self.ctx.block(user_block).parent_region != Some(self.body)
                || !self.dominance.is_reachable(user_block)
                || !op_dominates_op(self.ctx, self.load, user, self.dominance)
            {
                return false;
            }
            let operand_index = use_.operand_index as usize;
            self.use_blocks.push(user_block);
            if core::UnrealizedConversionCast::matches(self.ctx, user)
                && operand_index == 0
                && self.ctx.op_operands(user).len() == 1
                && self.ctx.op_results(user).len() == 1
            {
                let alias = self.ctx.op_result(user, 0);
                self.aliases.push(alias);
                if !self.collect(alias) {
                    return false;
                }
            } else if !is_proven_temporary_use(self.ctx, user, operand_index) {
                return false;
            }
        }
        true
    }
}

fn value_dominates_op(
    ctx: &IrContext,
    body: RegionRef,
    value: ValueRef,
    op: OpRef,
    dominance: &DominatorTree,
) -> bool {
    match ctx.value_def(value) {
        ValueDef::BlockArg(block, _) => {
            ctx.block(block).parent_region == Some(body)
                && dominance.entry() == Some(block)
                && dominance.dominates(block, ctx.op(op).parent_block.unwrap_or(block))
        }
        ValueDef::OpResult(defining_op, _) => op_dominates_op(ctx, defining_op, op, dominance),
    }
}

fn op_dominates_op(
    ctx: &IrContext,
    defining_op: OpRef,
    user: OpRef,
    dominance: &DominatorTree,
) -> bool {
    let (Some(defining_block), Some(user_block)) =
        (ctx.op(defining_op).parent_block, ctx.op(user).parent_block)
    else {
        return false;
    };
    if defining_block != user_block {
        return dominance.dominates(defining_block, user_block);
    }
    let ops = &ctx.block(defining_block).ops;
    let defining_index = ops.iter().position(|&op| op == defining_op);
    let user_index = ops.iter().position(|&op| op == user);
    defining_index
        .zip(user_index)
        .is_some_and(|(defining, use_)| defining < use_)
}

fn is_proven_temporary_use(ctx: &IrContext, op: OpRef, operand_index: usize) -> bool {
    if clif::Load::matches(ctx, op) {
        return operand_index == 0;
    }
    if clif::Store::matches(ctx, op) {
        return operand_index == 1;
    }
    clif::Icmp::matches(ctx, op)
}

fn record_lifetime_dependencies(
    value: ValueRef,
    dependencies: &HashMap<ValueRef, ValueRef>,
    mut record: impl FnMut(ValueRef),
) {
    let mut current = value;
    let mut visited = HashSet::new();
    while let Some(&owner) = dependencies.get(&current) {
        if !visited.insert(owner) {
            break;
        }
        record(owner);
        current = owner;
    }
}

/// Return entry parameters whose complete use set is proven not to escape the
/// dynamic extent of the function call.
fn analyze_borrowed_parameters(
    ctx: &IrContext,
    function: Symbol,
    body: RegionRef,
    trusted_summaries: &HashMap<Symbol, Vec<ParameterOwnership>>,
) -> HashSet<ValueRef> {
    let Some(&entry) = ctx.region(body).blocks.first() else {
        return HashSet::new();
    };

    let Some(function_summary) = trusted_summaries.get(&function) else {
        return HashSet::new();
    };

    ctx.block_args(entry)
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| function_summary.get(*index) == Some(&ParameterOwnership::Borrowed))
        .map(|(_, parameter)| parameter)
        .filter(|&parameter| is_anyref_value(ctx, parameter))
        .filter(|&parameter| parameter_is_proven_borrowed(ctx, body, parameter, trusted_summaries))
        .collect()
}

fn parameter_is_proven_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    parameter: ValueRef,
    trusted_summaries: &HashMap<Symbol, Vec<ParameterOwnership>>,
) -> bool {
    value_is_proven_borrowed(ctx, body, parameter, trusted_summaries, &mut HashSet::new())
}

fn value_is_proven_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    value: ValueRef,
    trusted_summaries: &HashMap<Symbol, Vec<ParameterOwnership>>,
    visited: &mut HashSet<ValueRef>,
) -> bool {
    if !visited.insert(value) {
        return true;
    }

    ctx.uses(value).iter().all(|use_| {
        let op = use_.user;
        let operand_index = use_.operand_index as usize;
        match classify_borrowed_use(ctx, body, op, operand_index, borrowed_use_kind(ctx, op)) {
            BorrowedUse::Safe => true,
            BorrowedUse::TransparentAlias(alias) => {
                value_is_proven_borrowed(ctx, body, alias, trusted_summaries, visited)
            }
            BorrowedUse::DirectCall => {
                let call = clif::Call::from_op(ctx, op).expect("classified clif.call");
                trusted_summaries
                    .get(&call.callee(ctx))
                    .and_then(|summary| summary.get(operand_index))
                    == Some(&ParameterOwnership::Borrowed)
            }
            BorrowedUse::Escaping => false,
        }
    })
}

fn borrowed_use_kind(ctx: &IrContext, op: OpRef) -> BorrowedUseKind {
    if clif::Call::matches(ctx, op) {
        return BorrowedUseKind::DirectCall;
    }
    if clif::Load::matches(ctx, op) {
        return BorrowedUseKind::LoadAddress;
    }
    if clif::Store::matches(ctx, op) {
        return BorrowedUseKind::StoreAddress { address_operand: 1 };
    }
    if clif::Icmp::matches(ctx, op) {
        return BorrowedUseKind::Comparison;
    }
    if core::UnrealizedConversionCast::matches(ctx, op) {
        return BorrowedUseKind::TransparentAlias;
    }
    BorrowedUseKind::Escaping
}

/// Insert RC ops in a single block.
fn insert_rc_in_block(
    ctx: &mut IrContext,
    block: BlockRef,
    is_entry: bool,
    ptr_values: &HashSet<ValueRef>,
    liveness: &LivenessInfo,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    borrow_info: &RcBorrowInfo<'_>,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let loc = ctx.block(block).location;
    let ptr_ty = core::ptr(ctx).as_type_ref();

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
            record_lifetime_dependencies(operand, borrow_info.lifetime_dependencies, |owner| {
                last_use_in_block.insert(owner, op_idx);
            });
        }
    }

    // Returned values
    let mut returned_values: HashSet<ValueRef> = HashSet::new();
    if let Some(&last_op) = ops.last()
        && clif::Return::matches(ctx, last_op)
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
            if is_anyref_type(ctx, ctx.value_ty(arg_val))
                && !borrow_info.parameters.contains(&arg_val)
            {
                let retain_op = tribute_rt::retain(ctx, loc, arg_val, ptr_ty);
                plan.at_start.push(retain_op.op_ref());
            }
        }
    }

    // 2. Retain before store of ptr, retain after load of ptr
    for (op_idx, &op) in ops.iter().enumerate() {
        if let Ok(_store_op) = clif::Store::from_op(ctx, op) {
            let operands = ctx.op_operands(op).to_vec();
            if let Some(&stored_val) = operands.first()
                && is_anyref_value(ctx, stored_val)
                && !is_static_ptr(ctx, stored_val)
            {
                let op_loc = ctx.op(op).location;
                let retain_op = tribute_rt::retain(ctx, op_loc, stored_val, ptr_ty);
                plan.before
                    .entry(op_idx)
                    .or_default()
                    .push(retain_op.op_ref());
            }
        }

        if clif::Load::matches(ctx, op) {
            let result_ty = ctx.op_result_types(op).first().copied();
            if result_ty.is_some_and(|ty| is_anyref_type(ctx, ty)) {
                let load_result = ctx.op_result(op, 0);
                if borrow_info.temporaries.contains(&load_result) {
                    continue;
                }
                let op_loc = ctx.op(op).location;
                let retain_op = tribute_rt::retain(ctx, op_loc, load_result, ptr_ty);
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
        if !live_out.contains(v)
            && !returned_values.contains(v)
            && !borrow_info.parameters.contains(v)
            && !borrow_info.temporaries.contains(v)
        {
            dying_values.insert(*v);
        }
    }
    for v in &defs_in_block {
        if !live_out.contains(v)
            && !returned_values.contains(v)
            && !is_alloc_intermediate(ctx, *v)
            && !borrow_info.parameters.contains(v)
            && !borrow_info.temporaries.contains(v)
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
            if clif::Return::matches(ctx, last_op) || clif::Jump::matches(ctx, last_op) {
                continue;
            }
            let alloc_size = infer_alloc_size(ctx, *v);
            let op_loc = ctx.op(last_op).location;
            let release_op = tribute_rt::release(ctx, op_loc, *v, alloc_size);
            if is_terminator_op(ctx, last_op) {
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
            let release_op = tribute_rt::release(ctx, loc, *v, alloc_size);
            plan.at_start.push(release_op.op_ref());
        } else if let ValueDef::OpResult(def_op, _) = ctx.value_def(*v) {
            for (op_idx, &op) in ops.iter().enumerate() {
                if op == def_op {
                    let alloc_size = infer_alloc_size(ctx, *v);
                    let op_loc = ctx.op(op).location;
                    let release_op = tribute_rt::release(ctx, op_loc, *v, alloc_size);
                    plan.after
                        .entry(op_idx)
                        .or_default()
                        .push(release_op.op_ref());
                    break;
                }
            }
        } else if let ValueDef::BlockArg(_, _) = ctx.value_def(*v) {
            let alloc_size = infer_alloc_size(ctx, *v);
            let release_op = tribute_rt::release(ctx, loc, *v, alloc_size);
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
    use trunk_ir::validation::validate_use_chains;

    fn run_pass(ir: &str) -> String {
        run_pass_with_policy(ir, BorrowedParameterPolicy::Preserve)
    }

    fn run_pass_with_policy(ir: &str, policy: BorrowedParameterPolicy) -> String {
        run_pass_with_policies(ir, policy, TemporaryBorrowPolicy::Preserve)
    }

    fn run_pass_with_legacy_policy(ir: &str, policy: BorrowedParameterPolicy) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        insert_rc_with_policy(&mut ctx, module, policy);
        print_module(&ctx, module.op())
    }

    fn run_pass_with_policies(
        ir: &str,
        parameter_policy: BorrowedParameterPolicy,
        temporary_policy: TemporaryBorrowPolicy,
    ) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let validation = validate_use_chains(&ctx, module);
        assert!(
            validation.is_ok(),
            "input fixture must have valid SSA use chains: {validation}"
        );
        if parameter_policy == BorrowedParameterPolicy::ElideProvenBorrowed {
            let trusted =
                TrustedOwnershipSummaries::attach_locally_borrowed_for_tests(&mut ctx, module);
            insert_rc_with_policies_and_trusted_summaries(
                &mut ctx,
                module,
                parameter_policy,
                temporary_policy,
                &trusted,
            );
        } else {
            insert_rc_with_policies(&mut ctx, module, parameter_policy, temporary_policy);
        }
        let validation = validate_use_chains(&ctx, module);
        assert!(
            validation.is_ok(),
            "RC insertion must preserve SSA use chains: {validation}"
        );
        print_module(&ctx, module.op())
    }

    fn focused_rc_ops(output: &str) -> String {
        output
            .lines()
            .filter(|line| {
                line.contains("tribute_rt.retain") || line.contains("tribute_rt.release")
            })
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn assert_focused_rc(output: &str, expected: &str) {
        assert_eq!(focused_rc_ops(output), expected, "full IR:\n{output}");
    }

    fn run_temporary_borrow_comparison(ir: &str) -> (String, String) {
        let preserved = run_pass_with_policies(
            ir,
            BorrowedParameterPolicy::Preserve,
            TemporaryBorrowPolicy::Preserve,
        );
        let elided = run_pass_with_policies(
            ir,
            BorrowedParameterPolicy::Preserve,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        );
        (preserved, elided)
    }

    fn rc_counts(output: &str) -> (usize, usize) {
        (
            output.matches("tribute_rt.retain").count(),
            output.matches("tribute_rt.release").count(),
        )
    }

    #[test]
    fn parameter_and_temporary_borrow_policies_compose_independently() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    %2 = clif.load %1 {offset = 0} : core.i32
    clif.return %2
  }
}"#;

        let preserved = run_pass_with_policies(
            ir,
            BorrowedParameterPolicy::Preserve,
            TemporaryBorrowPolicy::Preserve,
        );
        let parameter_only = run_pass_with_policies(
            ir,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            TemporaryBorrowPolicy::Preserve,
        );
        let temporary_only = run_pass_with_policies(
            ir,
            BorrowedParameterPolicy::Preserve,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        );
        let composed = run_pass_with_policies(
            ir,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        );
        let legacy = run_pass_with_legacy_policy(ir, BorrowedParameterPolicy::ElideProvenBorrowed);

        assert_eq!(rc_counts(&preserved), (2, 2));
        assert_eq!(rc_counts(&parameter_only), (1, 1));
        assert_eq!(rc_counts(&temporary_only), (1, 1));
        assert_eq!(rc_counts(&composed), (0, 0));
        assert_eq!(rc_counts(&legacy), (2, 2));
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

    #[test]
    fn snapshot_temporary_field_borrow() {
        let (_, output) = run_temporary_borrow_comparison(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    %2 = clif.load %1 {offset = 0} : core.i32
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

    #[test]
    fn borrowed_read_only_parameter_omits_entry_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(&output, "");
    }

    #[test]
    fn borrowed_parameter_can_be_read_across_blocks() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
  ^bb0:
    clif.jump [^bb1]
  ^bb1:
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(&output, "");
    }

    #[test]
    fn borrowed_parameter_allows_store_address_and_comparison() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.i32) -> core.i8 {
    clif.store %1, %0 {offset = 0}
    %2 = clif.icmp %0, %0 {cond = @eq} : core.i8
    clif.return %2
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(&output, "");
    }

    #[test]
    fn returned_parameter_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> tribute_rt.anyref {
    clif.return %0
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(&output, "%1 = tribute_rt.retain %0 : core.ptr");
    }

    #[test]
    fn stored_parameter_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.ptr) -> core.nil {
    clif.store %0, %1 {offset = 0}
    clif.return
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%2 = tribute_rt.retain %0 : core.ptr\n",
                "%3 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn continuation_frame_capture_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.ptr) -> core.nil {
    clif.store %0, %1 {offset = 16}
    clif.return
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%2 = tribute_rt.retain %0 : core.ptr\n",
                "%3 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn opaque_call_parameter_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
    %1 = clif.call %0 {callee = @opaque} : core.nil
    clif.return
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn branch_forwarded_parameter_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
  ^bb0:
    clif.jump %0 [^bb1]
  ^bb1(%1: tribute_rt.anyref):
    %2 = clif.load %1 {offset = 0} : core.i32
    clif.return %2
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %2 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn loop_block_argument_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
  ^entry:
    clif.jump %0 [^loop]
  ^loop(%1: tribute_rt.anyref):
    %2 = clif.load %1 {offset = 0} : core.i32
    clif.jump %1 [^loop]
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(&output, "%1 = tribute_rt.retain %0 : core.ptr");
    }

    #[test]
    fn nested_region_capture_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
    func.func @capturing_closure() -> core.i32 {
      %1 = clif.load %0 {offset = 0} : core.i32
      func.return %1
    }
    clif.return
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn transparent_cast_allows_borrowed_read() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = core.unrealized_conversion_cast %0 : core.ptr
    %2 = clif.load %1 {offset = 0} : core.i32
    clif.return %2
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(&output, "");
    }

    #[test]
    fn escaping_cast_alias_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
    %1 = core.unrealized_conversion_cast %0 : core.ptr
    %2 = clif.call %1 {callee = @opaque} : core.nil
    clif.return
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn tail_call_target_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
  clif.func @caller(%0: tribute_rt.anyref) -> core.i32 {
    clif.return_call %0 {callee = @f}
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}\n",
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn address_taken_function_preserves_owned_rc() {
        let output = run_pass_with_policy(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 0} : core.i32
    clif.return %1
  }
  clif.func @address() -> core.ptr {
    %0 = clif.symbol_addr {sym = @f} : core.ptr
    clif.return %0
  }
}"#,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        );
        assert_focused_rc(
            &output,
            concat!(
                "%1 = tribute_rt.retain %0 : core.ptr\n",
                "tribute_rt.release %0 {alloc_size = 0}"
            ),
        );
    }

    #[test]
    fn temporary_borrow_in_dominated_subtree_elides_one_pair() {
        let (preserved, elided) = run_temporary_borrow_comparison(
            r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.i8) -> core.i32 {
  ^entry:
    clif.brif %1 [^borrow, ^exit]
  ^borrow:
    %2 = clif.load %0 {offset = 8} : tribute_rt.anyref
    clif.jump [^use]
  ^use:
    %3 = clif.load %2 {offset = 0} : core.i32
    clif.return %3
  ^exit:
    %4 = clif.iconst {value = 0} : core.i32
    clif.return %4
  }
}"#,
        );
        let (preserved_retains, preserved_releases) = rc_counts(&preserved);
        let (elided_retains, elided_releases) = rc_counts(&elided);
        assert_eq!(preserved_retains - elided_retains, 1, "{elided}");
        assert_eq!(preserved_releases - elided_releases, 1, "{elided}");
    }

    #[test]
    fn sibling_branch_uses_preserve_temporary_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.i8) -> core.i32 {
  ^entry:
    %2 = clif.load %0 {offset = 8} : tribute_rt.anyref
    clif.brif %1 [^left, ^right]
  ^left:
    %3 = clif.load %2 {offset = 0} : core.i32
    clif.return %3
  ^right:
    %4 = clif.load %2 {offset = 4} : core.i32
    clif.return %4
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn loop_carried_temporary_preserves_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
  ^entry:
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    clif.jump %1 [^loop]
  ^loop(%2: tribute_rt.anyref):
    %3 = clif.load %2 {offset = 0} : core.i32
    clif.jump %2 [^loop]
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn nested_region_capture_preserves_temporary_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    func.func @capture() -> core.i32 {
      %2 = clif.load %1 {offset = 0} : core.i32
      func.return %2
    }
    clif.return
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn cast_alias_preserves_temporary_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.i32 {
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    %2 = core.unrealized_conversion_cast %1 : core.ptr
    %3 = clif.call %2 {callee = @opaque} : core.i32
    clif.return %3
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn call_use_preserves_temporary_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    %2 = clif.call %1 {callee = @opaque} : core.nil
    clif.return
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn stored_temporary_preserves_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.ptr) -> core.nil {
    %2 = clif.load %0 {offset = 8} : tribute_rt.anyref
    clif.store %2, %1 {offset = 0}
    clif.return
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn join_use_preserves_temporary_ownership() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref, %1: core.i8) -> core.i32 {
  ^entry:
    %2 = clif.load %0 {offset = 8} : tribute_rt.anyref
    clif.brif %1 [^left, ^right]
  ^left:
    clif.jump [^join]
  ^right:
    clif.jump [^join]
  ^join:
    %3 = clif.load %2 {offset = 0} : core.i32
    clif.return %3
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
    }

    #[test]
    fn nested_field_borrows_have_independent_lifetimes() {
        let ir = r#"core.module @test {
  clif.func @f(%0: tribute_rt.anyref) -> core.nil {
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    %2 = clif.load %1 {offset = 8} : tribute_rt.anyref
    %3 = clif.call %2 {callee = @opaque} : core.nil
    clif.return
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        let (preserved_retains, preserved_releases) = rc_counts(&preserved);
        let (elided_retains, elided_releases) = rc_counts(&elided);
        assert_eq!(preserved_retains - elided_retains, 1, "{elided}");
        assert_eq!(preserved_releases - elided_releases, 1, "{elided}");
    }

    #[test]
    fn raw_allocation_address_preserves_temporary_ownership() {
        let ir = r#"core.module @test {
  clif.func @f() -> core.i32 {
    %0 = clif.iconst {value = 24} : core.i64
    %1 = clif.call %0 {callee = @__tribute_alloc} : core.ptr
    %2 = clif.iconst {value = 8} : core.i64
    %3 = clif.iadd %1, %2 : core.ptr
    %4 = clif.load %3 {offset = 8} : tribute_rt.anyref
    %5 = clif.load %4 {offset = 0} : core.i32
    clif.return %5
  }
}"#;
        let (preserved, elided) = run_temporary_borrow_comparison(ir);
        assert_eq!(focused_rc_ops(&preserved), focused_rc_ops(&elided));
        assert_eq!(rc_counts(&elided), (1, 1), "{elided}");
    }
}
