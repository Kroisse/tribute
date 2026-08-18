//! Reference counting insertion pass.
//!
//! Automatically inserts `tribute_rt.retain` and `tribute_rt.release` operations
//! for `tribute_rt.anyref`-typed values in the native backend pipeline.
//!
//! `tribute_rt.anyref` values and convention-proven physical closures are
//! RC-managed. Plain `core.ptr` values (function pointers, continuations,
//! null sentinels) are not affected.
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
    BorrowedUse, BorrowedUseKind, CALL_ARGUMENT_OWNERSHIP_ATTR, OWNERSHIP_CONTRACT_ID_ATTR,
    OwnershipContractError, PARAMETER_ENTRY_OWNERSHIP_ATTR, ParameterOwnership, RcOwnership,
    TrustedCallContract, TrustedOwnershipSummaries, ValidatedOwnershipContracts,
    classify_borrowed_use,
};

pub type RcInsertionError = OwnershipContractError;

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

fn is_core_ptr_value(ctx: &IrContext, value: ValueRef) -> bool {
    let data = ctx.types.get(ctx.value_ty(value));
    data.dialect == Symbol::new("core") && data.name == Symbol::new("ptr")
}

/// Check if an op is a block terminator.
fn is_terminator_op(ctx: &IrContext, op: OpRef) -> bool {
    clif::Return::matches(ctx, op)
        || clif::Jump::matches(ctx, op)
        || clif::Brif::matches(ctx, op)
        || clif::Trap::matches(ctx, op)
        || clif::ReturnCall::matches(ctx, op)
        || clif::ReturnCallIndirect::matches(ctx, op)
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

impl TemporaryBorrowInfo {
    fn extend(&mut self, other: Self) -> bool {
        let changed = other
            .borrowed
            .iter()
            .any(|value| !self.borrowed.contains(value))
            || other
                .lifetime_dependencies
                .iter()
                .any(|(value, owner)| self.lifetime_dependencies.get(value) != Some(owner));
        self.borrowed.extend(other.borrowed);
        self.lifetime_dependencies
            .extend(other.lifetime_dependencies);
        changed
    }
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
            if ptr_values.contains(&arg_val) {
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
                if ptr_values.contains(&result_val) {
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
    )
    .expect("legacy RC insertion without trusted CPS transfers");
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
    )
    .expect("legacy RC insertion without trusted CPS transfers");
}

pub fn insert_rc_with_trusted_summaries(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    trusted_summaries: &TrustedOwnershipSummaries,
) -> Result<(), RcInsertionError> {
    insert_rc_with_policies_and_trusted_summaries(
        ctx,
        module,
        borrowed_parameters,
        TemporaryBorrowPolicy::Preserve,
        trusted_summaries,
    )
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
) -> Result<(), RcInsertionError> {
    insert_rc_impl(ctx, module, borrowed_parameters, temporary_borrows, None)
}

pub fn insert_rc_with_policies_and_trusted_summaries(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    temporary_borrows: TemporaryBorrowPolicy,
    trusted_summaries: &TrustedOwnershipSummaries,
) -> Result<(), RcInsertionError> {
    insert_rc_impl(
        ctx,
        module,
        borrowed_parameters,
        temporary_borrows,
        Some(trusted_summaries),
    )
}

fn insert_rc_impl(
    ctx: &mut IrContext,
    module: Module,
    borrowed_parameters: BorrowedParameterPolicy,
    temporary_borrows: TemporaryBorrowPolicy,
    trusted_summaries: Option<&TrustedOwnershipSummaries>,
) -> Result<(), RcInsertionError> {
    let Some(first_block) = module.first_block(ctx) else {
        return Ok(());
    };
    let module_ops: Vec<OpRef> = ctx.block(first_block).ops.to_vec();
    let trusted = if let Some(summaries) = trusted_summaries {
        summaries.validated_for_clif(ctx, &module_ops)?
    } else {
        ValidatedOwnershipContracts {
            summaries: HashMap::new(),
            entry_contracts: HashMap::new(),
            call_contracts: HashMap::new(),
        }
    };
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
                &trusted.summaries,
                trusted.entry_contracts.get(&sym).map(Vec::as_slice),
                &trusted.call_contracts,
            );
        }
    }

    clear_contract_metadata(ctx, module.op());

    // After RC insertion, lower all remaining `tribute_rt.anyref` types to `core.ptr`.
    // This ensures anyref doesn't survive past RC insertion into the Cranelift emit phase.
    lower_anyref_to_ptr(ctx, module);
    Ok(())
}

fn clear_contract_metadata(ctx: &mut IrContext, op: OpRef) {
    let regions = ctx.op(op).regions.to_vec();
    let attributes = &mut ctx.op_mut(op).attributes;
    attributes.remove(PARAMETER_ENTRY_OWNERSHIP_ATTR);
    attributes.remove(CALL_ARGUMENT_OWNERSHIP_ATTR);
    attributes.remove(OWNERSHIP_CONTRACT_ID_ATTR);
    for region in regions {
        let blocks = ctx.region(region).blocks.clone();
        for block in blocks {
            let ops = ctx.block(block).ops.clone();
            for nested in ops {
                clear_contract_metadata(ctx, nested);
            }
        }
    }
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
#[allow(clippy::too_many_arguments)]
fn insert_rc_in_function(
    ctx: &mut IrContext,
    function: Symbol,
    body: RegionRef,
    borrowed_parameter_policy: BorrowedParameterPolicy,
    temporary_borrow_policy: TemporaryBorrowPolicy,
    trusted_summaries: &HashMap<Symbol, Vec<ParameterOwnership>>,
    entry_contract: Option<&[RcOwnership]>,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) {
    let consumed_parameters: HashSet<ValueRef> = entry_contract
        .into_iter()
        .flat_map(|entries| entries.iter().enumerate())
        .filter_map(|(index, entry)| {
            (*entry == RcOwnership::Consumed)
                .then(|| ctx.region(body).blocks.first())
                .flatten()
                .and_then(|&entry_block| ctx.block_args(entry_block).get(index).copied())
        })
        .collect();
    let mut ptr_values = collect_ptr_values(ctx, body);
    // `func_to_clif` erases convention-proven physical closures to `core.ptr`.
    // Their consumed entry contract is the exact provenance that keeps them in
    // RC accounting; arbitrary pointer block arguments remain unmanaged.
    ptr_values.extend(consumed_parameters.iter().copied());

    if ptr_values.is_empty() {
        return;
    }

    let initial_ptr_alias_map = build_ptr_alias_map(ctx, body, &mut ptr_values);
    let promoted_ptr_values = collect_adapted_tail_store_escapes(
        ctx,
        body,
        &ptr_values,
        &initial_ptr_alias_map,
        call_contracts,
    );
    ptr_values.extend(promoted_ptr_values.iter().copied());
    let ptr_alias_map = build_ptr_alias_map(ctx, body, &mut ptr_values);
    let mut temporary_borrows = match temporary_borrow_policy {
        TemporaryBorrowPolicy::Preserve => TemporaryBorrowInfo::default(),
        TemporaryBorrowPolicy::ElideProvenFieldBorrows => {
            analyze_temporary_borrows(ctx, body, &ptr_values, &ptr_alias_map, call_contracts)
        }
    };
    temporary_borrows.extend(analyze_abi_adapted_indirect_tail_transfer_borrows(
        ctx,
        body,
        &ptr_values,
        &ptr_alias_map,
        call_contracts,
    ));
    let liveness = compute_liveness(
        ctx,
        body,
        &ptr_values,
        &ptr_alias_map,
        &temporary_borrows.lifetime_dependencies,
    );
    let mut borrowed_parameters = match borrowed_parameter_policy {
        BorrowedParameterPolicy::Preserve => HashSet::new(),
        BorrowedParameterPolicy::ElideProvenBorrowed => {
            analyze_borrowed_parameters(ctx, function, body, trusted_summaries)
        }
    };
    borrowed_parameters.retain(|parameter| !consumed_parameters.contains(parameter));

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
            &promoted_ptr_values,
            &liveness,
            &ptr_alias_map,
            &borrow_info,
            &consumed_parameters,
            call_contracts,
        );
    }
}

/// Promote only physical closure loads that are proven to feed a trusted
/// indirect-tail transfer and whose otherwise-safe use chain stores that
/// closure as a value. The store receives its own RC unit; arbitrary pointers
/// and other escaping uses remain outside ownership accounting.
fn collect_adapted_tail_store_escapes(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) -> HashSet<ValueRef> {
    let dominance = DominatorTree::compute(ctx, body);
    if !dominance.is_valid() {
        return HashSet::new();
    }

    let known_owners = HashMap::new();
    let mut promoted = HashSet::new();
    for &block in &ctx.region(body).blocks {
        if !dominance.is_reachable(block) {
            continue;
        }
        for &op in &ctx.block(block).ops {
            if !clif::Load::matches(ctx, op) || ctx.op_results(op).len() != 1 {
                continue;
            }
            let loaded = ctx.op_result(op, 0);
            if !is_core_ptr_value(ctx, loaded)
                || !is_abi_adapted_indirect_tail_transfer(ctx, loaded, call_contracts)
                || !has_store_value_use(ctx, loaded, &mut HashSet::new())
            {
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
                &known_owners,
                &mut HashSet::new(),
            ) else {
                continue;
            };
            if temporary_is_proven_borrowed(
                ctx,
                body,
                op,
                loaded,
                owner,
                &dominance,
                call_contracts,
                true,
                true,
            )
            .is_some()
            {
                promoted.insert(loaded);
            }
        }
    }
    promoted
}

fn has_store_value_use(ctx: &IrContext, value: ValueRef, visited: &mut HashSet<ValueRef>) -> bool {
    if !visited.insert(value) {
        return false;
    }
    ctx.uses(value).iter().any(|use_| {
        let op = use_.user;
        let operand_index = use_.operand_index as usize;
        (clif::Store::matches(ctx, op) && operand_index == 0)
            || (core::UnrealizedConversionCast::matches(ctx, op)
                && operand_index == 0
                && ctx.op_operands(op).len() == 1
                && ctx.op_results(op).len() == 1
                && has_store_value_use(ctx, ctx.op_result(op, 0), visited))
    })
}

fn analyze_temporary_borrows(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) -> TemporaryBorrowInfo {
    let known_owners = HashMap::new();
    analyze_eligible_temporary_borrows(
        ctx,
        body,
        ptr_values,
        ptr_alias_map,
        call_contracts,
        &known_owners,
        is_anyref_value,
        false,
        false,
    )
}

/// Analyze only physical pointer loads whose trusted indirect-tail contract
/// requires a transferred RC unit. This correctness path is independent of
/// optional field-borrow elision.
fn analyze_abi_adapted_indirect_tail_transfer_borrows(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) -> TemporaryBorrowInfo {
    let mut info = TemporaryBorrowInfo::default();
    loop {
        let discovered = analyze_eligible_temporary_borrows(
            ctx,
            body,
            ptr_values,
            ptr_alias_map,
            call_contracts,
            &info.lifetime_dependencies,
            |ctx, temporary| {
                !is_anyref_value(ctx, temporary)
                    && is_abi_adapted_indirect_tail_transfer(ctx, temporary, call_contracts)
            },
            true,
            false,
        );
        if !info.extend(discovered) {
            return info;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_eligible_temporary_borrows(
    ctx: &IrContext,
    body: RegionRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
    known_owners: &HashMap<ValueRef, ValueRef>,
    is_eligible: impl Fn(&IrContext, ValueRef) -> bool,
    follow_descendant_loads: bool,
    allow_store_value: bool,
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
            if !is_eligible(ctx, temporary) {
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
                known_owners,
                &mut HashSet::new(),
            ) else {
                continue;
            };
            if let Some(aliases) = temporary_is_proven_borrowed(
                ctx,
                body,
                op,
                temporary,
                owner,
                &dominance,
                call_contracts,
                follow_descendant_loads,
                allow_store_value,
            ) {
                info.borrowed.insert(temporary);
                info.lifetime_dependencies.insert(temporary, owner);
                for alias in aliases {
                    info.borrowed.insert(alias);
                    info.lifetime_dependencies.insert(alias, owner);
                }
            }
        }
    }
    info
}

/// A physical `core.ptr` load may still carry an RC-managed value when the
/// exact trusted indirect-tail signature classifies that operand as a transfer.
/// Do not extend this exception to arbitrary pointer uses.
fn is_abi_adapted_indirect_tail_transfer(
    ctx: &IrContext,
    value: ValueRef,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) -> bool {
    fn reaches_transfer(
        ctx: &IrContext,
        value: ValueRef,
        call_contracts: &HashMap<OpRef, TrustedCallContract>,
        visited: &mut HashSet<ValueRef>,
    ) -> bool {
        if !visited.insert(value) {
            return false;
        }
        ctx.uses(value).iter().any(|use_| {
            let op = use_.user;
            let operand_index = use_.operand_index as usize;
            if core::UnrealizedConversionCast::matches(ctx, op)
                && operand_index == 0
                && ctx.op_operands(op).len() == 1
                && ctx.op_results(op).len() == 1
            {
                return reaches_transfer(ctx, ctx.op_result(op, 0), call_contracts, visited);
            }
            if clif::Load::matches(ctx, op) && operand_index == 0 && ctx.op_results(op).len() == 1 {
                let descendant = ctx.op_result(op, 0);
                return (is_anyref_value(ctx, descendant) || is_core_ptr_value(ctx, descendant))
                    && reaches_transfer(ctx, descendant, call_contracts, visited);
            }
            let Some(contract) = call_contracts.get(&op) else {
                return false;
            };
            contract.is_tail
                && contract.is_indirect
                && operand_index
                    .checked_sub(usize::from(contract.is_indirect))
                    .and_then(|index| contract.actions.get(index))
                    == Some(&RcOwnership::Transfer)
        })
    }

    reaches_transfer(ctx, value, call_contracts, &mut HashSet::new())
}

fn is_trusted_indirect_tail_callee(
    op: OpRef,
    operand_index: usize,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) -> bool {
    call_contracts
        .get(&op)
        .is_some_and(|contract| contract.is_tail && contract.is_indirect && operand_index == 0)
}

fn resolve_temporary_owner(
    ctx: &IrContext,
    value: ValueRef,
    ptr_values: &HashSet<ValueRef>,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    known_owners: &HashMap<ValueRef, ValueRef>,
    visited: &mut HashSet<ValueRef>,
) -> Option<ValueRef> {
    if !visited.insert(value) {
        return None;
    }
    if ptr_values.contains(&value) {
        return Some(value);
    }
    if let Some(&owner) = known_owners.get(&value) {
        return Some(owner);
    }
    if let Some(&owner) = ptr_alias_map.get(&value) {
        return resolve_temporary_owner(
            ctx,
            owner,
            ptr_values,
            ptr_alias_map,
            known_owners,
            visited,
        );
    }
    let ValueDef::OpResult(defining_op, _) = ctx.value_def(value) else {
        return None;
    };
    if core::UnrealizedConversionCast::matches(ctx, defining_op)
        || clif::Iadd::matches(ctx, defining_op)
    {
        let input = *ctx.op_operands(defining_op).first()?;
        return resolve_temporary_owner(
            ctx,
            input,
            ptr_values,
            ptr_alias_map,
            known_owners,
            visited,
        );
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn temporary_is_proven_borrowed(
    ctx: &IrContext,
    body: RegionRef,
    load: OpRef,
    temporary: ValueRef,
    owner: ValueRef,
    dominance: &DominatorTree,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
    follow_descendant_loads: bool,
    allow_store_value: bool,
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
        call_contracts,
        follow_descendant_loads,
        allow_store_value,
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
    call_contracts: &'a HashMap<OpRef, TrustedCallContract>,
    follow_descendant_loads: bool,
    allow_store_value: bool,
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
            } else if self.follow_descendant_loads
                && clif::Load::matches(self.ctx, user)
                && operand_index == 0
                && self.ctx.op_results(user).len() == 1
            {
                let descendant = self.ctx.op_result(user, 0);
                if is_anyref_value(self.ctx, descendant) || is_core_ptr_value(self.ctx, descendant)
                {
                    self.aliases.push(descendant);
                    if !self.collect(descendant) {
                        return false;
                    }
                }
            } else if !(self.allow_store_value
                && clif::Store::matches(self.ctx, user)
                && operand_index == 0)
                && !(self.follow_descendant_loads
                    && is_trusted_indirect_tail_callee(user, operand_index, self.call_contracts))
                && !is_proven_temporary_use(self.ctx, user, operand_index, self.call_contracts)
            {
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

fn is_proven_temporary_use(
    ctx: &IrContext,
    op: OpRef,
    operand_index: usize,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
) -> bool {
    if clif::Load::matches(ctx, op) {
        return operand_index == 0;
    }
    if clif::Store::matches(ctx, op) {
        return operand_index == 1;
    }
    if let Some(contract) = call_contracts.get(&op)
        && contract.is_tail
        && let Some(action_index) = operand_index.checked_sub(usize::from(contract.is_indirect))
    {
        return contract.actions.get(action_index) == Some(&RcOwnership::Transfer);
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
#[allow(clippy::too_many_arguments)]
fn insert_rc_in_block(
    ctx: &mut IrContext,
    block: BlockRef,
    is_entry: bool,
    ptr_values: &HashSet<ValueRef>,
    promoted_ptr_values: &HashSet<ValueRef>,
    liveness: &LivenessInfo,
    ptr_alias_map: &HashMap<ValueRef, ValueRef>,
    borrow_info: &RcBorrowInfo<'_>,
    consumed_parameters: &HashSet<ValueRef>,
    call_contracts: &HashMap<OpRef, TrustedCallContract>,
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
    let mut transferred_values = HashSet::new();

    for (op_idx, &op) in ops.iter().enumerate() {
        let Some(contract) = call_contracts.get(&op) else {
            continue;
        };
        let operands = ctx.op_operands(op).to_vec();
        let args = operands
            .get(usize::from(contract.is_indirect)..)
            .unwrap_or_default();
        if contract.is_tail {
            let mut transfers: HashMap<ValueRef, (ValueRef, usize)> = HashMap::new();
            for (&argument, action) in args.iter().zip(&contract.actions) {
                if *action != RcOwnership::Transfer {
                    continue;
                }
                let owned = ptr_alias_map.get(&argument).copied().unwrap_or(argument);
                transferred_values.insert(argument);
                transferred_values.insert(owned);
                transfers
                    .entry(owned)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((argument, 1));
            }
            for (owned, (argument, count)) in transfers {
                let borrowed = borrow_info.parameters.contains(&owned)
                    || borrow_info.parameters.contains(&argument)
                    || borrow_info.temporaries.contains(&owned)
                    || borrow_info.temporaries.contains(&argument);
                let retain_count = count.saturating_sub(usize::from(!borrowed));
                for _ in 0..retain_count {
                    let retain = tribute_rt::retain(ctx, ctx.op(op).location, argument, ptr_ty);
                    plan.before.entry(op_idx).or_default().push(retain.op_ref());
                }
            }
        } else {
            for (&argument, action) in args.iter().zip(&contract.actions) {
                if *action == RcOwnership::Acquire {
                    let retain = tribute_rt::retain(ctx, ctx.op(op).location, argument, ptr_ty);
                    plan.before.entry(op_idx).or_default().push(retain.op_ref());
                }
            }
        }
    }

    // --- Retain insertions ---

    // 1. Entry block: retain each ptr parameter
    if is_entry {
        let args: Vec<ValueRef> = ctx.block_args(block).to_vec();
        for arg_val in args {
            if is_anyref_type(ctx, ctx.value_ty(arg_val))
                && !borrow_info.parameters.contains(&arg_val)
                && !consumed_parameters.contains(&arg_val)
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
                && ((is_anyref_value(ctx, stored_val) && !is_static_ptr(ctx, stored_val))
                    || promoted_ptr_values.contains(&stored_val)
                    || consumed_parameters.contains(&stored_val)
                    || ptr_alias_map.get(&stored_val).is_some_and(|owner| {
                        promoted_ptr_values.contains(owner) || consumed_parameters.contains(owner)
                    }))
            {
                let op_loc = ctx.op(op).location;
                let retain_op = tribute_rt::retain(ctx, op_loc, stored_val, ptr_ty);
                plan.before
                    .entry(op_idx)
                    .or_default()
                    .push(retain_op.op_ref());
            }
        }

        if clif::Load::matches(ctx, op) && ctx.op_results(op).len() == 1 {
            let load_result = ctx.op_result(op, 0);
            if ((is_anyref_value(ctx, load_result) && !is_static_ptr(ctx, load_result))
                || promoted_ptr_values.contains(&load_result))
                && !borrow_info.temporaries.contains(&load_result)
            {
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
            && !transferred_values.contains(v)
            && !borrow_info.parameters.contains(v)
            && !borrow_info.temporaries.contains(v)
        {
            dying_values.insert(*v);
        }
    }
    for v in &defs_in_block {
        if !live_out.contains(v)
            && !returned_values.contains(v)
            && !transferred_values.contains(v)
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
    use crate::native::ownership_summary::compute_and_attach;
    use crate::native::type_converter::native_type_converter;
    use trunk_ir::Attribute;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::validation::validate_use_chains;
    use trunk_ir_cranelift_backend::passes::func_to_clif;

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
            )
            .expect("trusted RC insertion");
        } else {
            insert_rc_with_policies(&mut ctx, module, parameter_policy, temporary_policy)
                .expect("legacy RC insertion");
        }
        let validation = validate_use_chains(&ctx, module);
        assert!(
            validation.is_ok(),
            "RC insertion must preserve SSA use chains: {validation}"
        );
        print_module(&ctx, module.op())
    }

    fn run_trusted_cps_pass(
        ir: &str,
        temporary_policy: TemporaryBorrowPolicy,
    ) -> Result<String, String> {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let (type_converter, _) = native_type_converter(&mut ctx);
        let trusted = compute_and_attach(&mut ctx, module, &type_converter)
            .map_err(|error| error.to_string())?;
        func_to_clif::lower(&mut ctx, module, type_converter).map_err(|error| error.to_string())?;
        insert_rc_with_policies_and_trusted_summaries(
            &mut ctx,
            module,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            temporary_policy,
            &trusted,
        )
        .map_err(|error| error.to_string())?;
        Ok(print_module(&ctx, module.op()))
    }

    fn lowered_trusted_cps(ir: &str) -> (IrContext, Module, TrustedOwnershipSummaries) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let (type_converter, _) = native_type_converter(&mut ctx);
        let trusted = compute_and_attach(&mut ctx, module, &type_converter)
            .expect("ownership contract production");
        func_to_clif::lower(&mut ctx, module, type_converter).expect("func_to_clif");
        (ctx, module, trusted)
    }

    fn first_tail_call(ctx: &IrContext, module: Module, indirect: bool) -> OpRef {
        let block = module.first_block(ctx).expect("module body");
        ctx.block(block)
            .ops
            .iter()
            .filter_map(|&op| clif::Func::from_op(ctx, op).ok())
            .flat_map(|function| ctx.region(function.body(ctx)).blocks.iter().copied())
            .flat_map(|block| ctx.block(block).ops.iter().copied())
            .find(|&op| {
                if indirect {
                    clif::ReturnCallIndirect::matches(ctx, op)
                } else {
                    clif::ReturnCall::matches(ctx, op)
                }
            })
            .expect("tail call")
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
    fn direct_and_indirect_cps_tails_transfer_without_caller_cleanup() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @direct(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %0 {callee = @target, tribute.calling_convention = 2}
  }
  func.func @indirect(%0: core.ptr, %1: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref), tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("trusted CPS transfer");
        assert_eq!(output.matches("tribute_rt.retain").count(), 0, "{output}");
        assert_eq!(output.matches("tribute_rt.release").count(), 1, "{output}");
        assert!(
            output.contains("clif.return_call %0 {callee = @target"),
            "{output}"
        );
        assert!(
            output.contains("clif.return_call_indirect %0, %1"),
            "{output}"
        );
        assert!(
            output.contains("sig = core.func(core.nil, core.ptr)"),
            "{output}"
        );
        for body in output.split("clif.func").skip(1) {
            if body.contains("return_call") {
                assert_eq!(
                    body.lines()
                        .rfind(|line| !line.trim().is_empty())
                        .unwrap()
                        .trim(),
                    "}"
                );
                assert!(
                    !body
                        .split("return_call")
                        .nth(1)
                        .unwrap()
                        .contains("tribute_rt."),
                    "{body}"
                );
            }
        }
    }

    #[test]
    fn consumed_cps_entry_owns_even_when_borrow_summary_is_eligible() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    %1 = clif.load %0 {offset = 0} : core.i32
    func.unreachable
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("consumed CPS entry");
        assert!(!output.contains("tribute_rt.retain"), "{output}");
        assert_eq!(
            output.matches("tribute_rt.release %0").count(),
            1,
            "{output}"
        );
    }

    #[test]
    fn consumed_physical_entry_retain_per_owning_store() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  !handler = closure.closure(core.func(core.nil)) {tribute.calling_convention = 2, tribute.closure_environment_index = 0}
  func.func @producer(%0: core.i64, %1: core.i64, %2: core.i64, %3: core.i64, %4: !handler) -> core.nil attributes {tribute.calling_convention = 2} {
    %5 = clif.iconst {value = 32} : core.i64
    %6 = clif.call %5 {callee = @__tribute_alloc} : core.ptr
    clif.store %4, %6 {offset = 24}
    %7 = clif.call %5 {callee = @__tribute_alloc} : core.ptr
    clif.store %4, %7 {offset = 24}
    %8 = core.unrealized_conversion_cast %6 : !handler
    clif.store %8, %7 {offset = 40}
    func.unreachable
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("consumed physical entry stores");
        let producer = output
            .split("sym_name = @producer")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let stores: Vec<_> = producer
            .match_indices("clif.store %4,")
            .map(|(index, _)| index)
            .collect();
        assert_eq!(stores.len(), 2, "{producer}");
        assert_eq!(producer.matches("offset = 40").count(), 1, "{producer}");
        let retains: Vec<_> = producer
            .match_indices("tribute_rt.retain %4")
            .map(|(index, _)| index)
            .collect();
        assert_eq!(retains.len(), 2, "{producer}");
        assert!(
            retains[0] < stores[0] && stores[0] < retains[1] && retains[1] < stores[1],
            "{producer}"
        );
        let release = producer
            .find("tribute_rt.release %4")
            .unwrap_or_else(|| panic!("{producer}"));
        assert!(stores[1] < release, "{producer}");
    }

    #[test]
    fn plain_physical_entry_store_does_not_retain_core_ptr() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @producer(%0: core.i64, %1: core.i64, %2: core.i64, %3: core.i64, %4: core.ptr) -> core.nil attributes {tribute.calling_convention = 2} {
    %5 = clif.iconst {value = 32} : core.i64
    %6 = clif.call %5 {callee = @__tribute_alloc} : core.ptr
    clif.store %4, %6 {offset = 24}
    %7 = clif.call %5 {callee = @__tribute_alloc} : core.ptr
    clif.store %4, %7 {offset = 24}
    clif.store %6, %7 {offset = 40}
    func.unreachable
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("plain physical entry stores");
        let producer = output
            .split("sym_name = @producer")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        assert!(!producer.contains("tribute_rt.retain %4"), "{producer}");
        assert!(!producer.contains("tribute_rt.release %4"), "{producer}");
    }

    #[test]
    fn tail_transfer_multiplicity_acquires_only_additional_units() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %0, %0 {callee = @target, tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("trusted duplicate transfer");
        assert_eq!(output.matches("tribute_rt.retain").count(), 1, "{output}");
        assert_eq!(output.matches("tribute_rt.release").count(), 2, "{output}");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        assert!(
            caller.find("tribute_rt.retain").unwrap() < caller.find("clif.return_call").unwrap(),
            "{caller}"
        );
        assert!(!caller.contains("tribute_rt.release"), "{caller}");
    }

    #[test]
    fn borrowed_tail_operand_is_acquired_before_untransferred_owner_cleanup() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    clif.store %0, %0 {offset = 16}
    %1 = clif.load %0 {offset = 8} : tribute_rt.anyref
    func.tail_call %1 {callee = @target, tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        )
        .expect("borrowed tail acquisition");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let retain = caller
            .find("tribute_rt.retain %2")
            .unwrap_or_else(|| panic!("{caller}"));
        let release = caller
            .find("tribute_rt.release %0")
            .unwrap_or_else(|| panic!("{caller}"));
        let tail = caller
            .find("clif.return_call %2")
            .unwrap_or_else(|| panic!("{caller}"));
        assert!(retain < release && release < tail, "{caller}");
        assert!(!caller[tail..].contains("tribute_rt."), "{caller}");
    }

    #[test]
    fn borrowed_indirect_tail_alias_operand_is_acquired_before_untransferred_owner_cleanup() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    clif.store %1, %1 {offset = 16}
    %2 = core.unrealized_conversion_cast %1 : core.ptr
    %3 = clif.load %2 {offset = 8} : core.ptr
    func.tail_call_indirect %0, %3 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref), tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("borrowed indirect tail acquisition");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let load = caller
            .lines()
            .find(|line| line.contains("clif.load"))
            .unwrap_or_else(|| panic!("{caller}"));
        let loaded = load
            .split_once(" = ")
            .map(|(value, _)| value.trim())
            .unwrap_or_else(|| panic!("{caller}"));
        let retain_op = format!("tribute_rt.retain {loaded}");
        let retain = caller
            .find(&retain_op)
            .unwrap_or_else(|| panic!("{caller}"));
        assert_eq!(caller.matches(&retain_op).count(), 1, "{caller}");
        let release = caller
            .find("tribute_rt.release %1")
            .unwrap_or_else(|| panic!("{caller}"));
        let tail_op = format!("clif.return_call_indirect %0, {loaded}");
        let tail = caller.find(&tail_op).unwrap_or_else(|| panic!("{caller}"));
        let load = caller
            .find("clif.load")
            .unwrap_or_else(|| panic!("{caller}"));
        assert!(
            load < retain && retain < release && release < tail,
            "{caller}"
        );
        assert!(!caller[tail..].contains("tribute_rt."), "{caller}");
    }

    #[test]
    fn nested_borrowed_indirect_tail_operand_keeps_dispatcher_alive() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    clif.store %0, %0 {offset = 24}
    %1 = core.unrealized_conversion_cast %0 : core.ptr
    %2 = clif.load %1 {offset = 16} : core.ptr
    %3 = core.unrealized_conversion_cast %2 : core.ptr
    %4 = clif.load %3 {offset = 0} : core.ptr
    %5 = clif.load %3 {offset = 8} : core.ptr
    func.tail_call_indirect %4, %5 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref), tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("nested borrowed indirect tail acquisition");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let loaded_value = |offset| {
            caller
                .lines()
                .find(|line| line.contains("clif.load") && line.contains(offset))
                .and_then(|line| line.split_once(" = ").map(|(value, _)| value.trim()))
                .unwrap_or_else(|| panic!("{caller}"))
        };
        let dispatcher = loaded_value("offset = 16");
        let code = loaded_value("offset = 0");
        let environment = loaded_value("offset = 8");
        let retain_op = format!("tribute_rt.retain {environment}");
        let retain = caller
            .find(&retain_op)
            .unwrap_or_else(|| panic!("{caller}"));
        assert_eq!(caller.matches(&retain_op).count(), 1, "{caller}");
        assert!(
            !caller.contains(&format!("tribute_rt.retain {dispatcher}")),
            "{caller}"
        );
        let release = caller
            .find("tribute_rt.release %0")
            .unwrap_or_else(|| panic!("{caller}"));
        let tail_op = format!("clif.return_call_indirect {code}, {environment}");
        let tail = caller.find(&tail_op).unwrap_or_else(|| panic!("{caller}"));
        let environment_load = caller
            .find("offset = 8")
            .unwrap_or_else(|| panic!("{caller}"));
        assert!(
            environment_load < retain && retain < release && release < tail,
            "{caller}"
        );
        assert!(!caller[tail..].contains("tribute_rt."), "{caller}");
    }

    #[test]
    fn stored_borrowed_dispatcher_keeps_nested_tail_operands_alive() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    %1 = clif.load %0 {offset = 24} : core.ptr
    %2 = clif.iconst {value = 32} : core.i64
    %3 = clif.call %2 {callee = @__tribute_alloc} : core.ptr
    clif.store %1, %3 {offset = 24}
    %4 = clif.load %1 {offset = 0} : core.i64
    %5 = clif.load %1 {offset = 8} : core.ptr
    func.tail_call_indirect %4, %5 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref), tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("stored borrowed dispatcher acquisition");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let loaded_value = |offset| {
            caller
                .lines()
                .find(|line| line.contains("clif.load") && line.contains(offset))
                .and_then(|line| line.split_once(" = ").map(|(value, _)| value.trim()))
                .unwrap_or_else(|| panic!("{caller}"))
        };
        let dispatcher = loaded_value("offset = 24");
        let code = loaded_value("offset = 0");
        let environment = loaded_value("offset = 8");
        let dispatcher_retain = format!("tribute_rt.retain {dispatcher}");
        let dispatcher_load = caller
            .find("offset = 24")
            .unwrap_or_else(|| panic!("{caller}"));
        let root_release = caller
            .find("tribute_rt.release %0")
            .unwrap_or_else(|| panic!("{caller}"));
        let environment_retain = format!("tribute_rt.retain {environment}");
        let environment_retain = caller
            .find(&environment_retain)
            .unwrap_or_else(|| panic!("{caller}"));
        assert_eq!(caller.matches(&dispatcher_retain).count(), 2, "{caller}");
        let first_dispatcher_retain = caller
            .find(&dispatcher_retain)
            .unwrap_or_else(|| panic!("{caller}"));
        let dispatcher_release = caller
            .find(&format!("tribute_rt.release {dispatcher}"))
            .unwrap_or_else(|| panic!("{caller}"));
        let tail_op = format!("clif.return_call_indirect {code}, {environment}");
        let tail = caller.find(&tail_op).unwrap_or_else(|| panic!("{caller}"));
        assert!(
            dispatcher_load < first_dispatcher_retain
                && first_dispatcher_retain < root_release
                && environment_retain < dispatcher_release
                && dispatcher_release < tail,
            "{caller}"
        );
        assert!(!caller[tail..].contains("tribute_rt."), "{caller}");
    }

    #[test]
    fn plain_indirect_tail_store_does_not_promote_core_ptr() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    %1 = clif.load %0 {offset = 24} : core.ptr
    %2 = clif.iconst {value = 32} : core.i64
    %3 = clif.call %2 {callee = @__tribute_alloc} : core.ptr
    clif.store %1, %3 {offset = 24}
    %4 = clif.load %1 {offset = 0} : core.i64
    %5 = clif.load %1 {offset = 8} : core.ptr
    func.tail_call_indirect %4, %5 {func.indirect_call_signature = core.func(core.nil, core.ptr), tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("plain stored dispatcher stays unmanaged");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let dispatcher = caller
            .lines()
            .find(|line| line.contains("clif.load") && line.contains("offset = 24"))
            .and_then(|line| line.split_once(" = ").map(|(value, _)| value.trim()))
            .unwrap_or_else(|| panic!("{caller}"));
        assert!(
            !caller.contains(&format!("tribute_rt.retain {dispatcher}")),
            "{caller}"
        );
    }

    #[test]
    fn indirect_tail_plain_ptr_operand_preserves_loaded_anyref_ownership() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    %2 = clif.load %1 {offset = 8} : tribute_rt.anyref
    %3 = core.unrealized_conversion_cast %2 : core.ptr
    func.tail_call_indirect %0, %3 {func.indirect_call_signature = core.func(core.nil, core.ptr), tribute.calling_convention = 2}
  }
}"#,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        )
        .expect("plain indirect tail preserves loaded ownership");
        let caller = output
            .split("sym_name = @caller")
            .nth(1)
            .unwrap_or_else(|| panic!("{output}"));
        let load = caller
            .find("clif.load")
            .unwrap_or_else(|| panic!("{caller}"));
        let tail = caller
            .find("clif.return_call_indirect")
            .unwrap_or_else(|| panic!("{caller}"));
        assert!(caller[load..tail].contains("tribute_rt.retain"), "{caller}");
        assert!(!caller[tail..].contains("tribute_rt."), "{caller}");
    }

    #[test]
    fn ordinary_call_acquires_for_consumed_callee_and_return_still_transfers() {
        let output = run_trusted_cps_pass(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%0: tribute_rt.anyref) -> tribute_rt.anyref {
    %1 = func.call %0 {callee = @target} : core.nil
    func.return %0
  }
}"#,
            TemporaryBorrowPolicy::Preserve,
        )
        .expect("ordinary acquisition");
        let caller = output.split("sym_name = @caller").nth(1).unwrap();
        assert_eq!(
            caller.matches("tribute_rt.retain %0").count(),
            2,
            "{caller}"
        );
        assert!(
            caller.rfind("tribute_rt.retain %0").unwrap() < caller.find("clif.call %0").unwrap(),
            "{caller}"
        );
        assert!(!caller.contains("tribute_rt.release %0"), "{caller}");
        assert!(caller.contains("clif.return %0"), "{caller}");
    }

    #[test]
    fn malformed_trust_fails_before_rc_mutation() {
        let (mut ctx, module, trusted) = lowered_trusted_cps(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @other(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %0 {callee = @target, tribute.calling_convention = 2}
  }
}"#,
        );
        let tail = first_tail_call(&ctx, module, false);
        let original_id = ctx
            .op(tail)
            .attributes
            .get(OWNERSHIP_CONTRACT_ID_ATTR)
            .cloned()
            .expect("contract identity");
        ctx.op_mut(tail)
            .attributes
            .insert(Symbol::new(OWNERSHIP_CONTRACT_ID_ATTR), Attribute::Int(999));
        let before = print_module(&ctx, module.op());
        assert!(
            insert_rc_with_policies_and_trusted_summaries(
                &mut ctx,
                module,
                BorrowedParameterPolicy::ElideProvenBorrowed,
                TemporaryBorrowPolicy::Preserve,
                &trusted,
            )
            .is_err()
        );
        assert_eq!(print_module(&ctx, module.op()), before);

        ctx.op_mut(tail)
            .attributes
            .insert(Symbol::new(OWNERSHIP_CONTRACT_ID_ATTR), original_id);
        ctx.op_mut(tail).attributes.insert(
            Symbol::new("callee"),
            Attribute::Symbol(Symbol::new("other")),
        );
        let before = print_module(&ctx, module.op());
        assert!(
            insert_rc_with_policies_and_trusted_summaries(
                &mut ctx,
                module,
                BorrowedParameterPolicy::ElideProvenBorrowed,
                TemporaryBorrowPolicy::Preserve,
                &trusted,
            )
            .is_err()
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn indirect_signature_mismatch_fails_before_rc_mutation() {
        let (mut ctx, module, trusted) = lowered_trusted_cps(
            r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: tribute_rt.anyref, %2: core.i32) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1, %2 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref, core.i32), tribute.calling_convention = 2}
  }
}"#,
        );
        let tail = first_tail_call(&ctx, module, true);
        let nil = core::nil(&mut ctx).as_type_ref();
        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        let ptr = core::ptr(&mut ctx).as_type_ref();
        let changed_signature = core::func(&mut ctx, nil, [anyref, ptr]).as_type_ref();
        ctx.op_mut(tail)
            .attributes
            .insert(Symbol::new("sig"), Attribute::Type(changed_signature));
        let before = print_module(&ctx, module.op());
        assert!(
            insert_rc_with_policies_and_trusted_summaries(
                &mut ctx,
                module,
                BorrowedParameterPolicy::ElideProvenBorrowed,
                TemporaryBorrowPolicy::Preserve,
                &trusted,
            )
            .is_err()
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn zero_operand_indirect_tail_fails_before_rc_mutation() {
        for keep_metadata in [true, false] {
            let (mut ctx, module, trusted) = lowered_trusted_cps(
                r#"core.module @test {
  func.func @caller(%0: core.ptr, %1: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %0, %1 {func.indirect_call_signature = core.func(core.nil, tribute_rt.anyref), tribute.calling_convention = 2}
  }
}"#,
            );
            let tail = first_tail_call(&ctx, module, true);
            while !ctx.op_operands(tail).is_empty() {
                ctx.remove_op_operand(tail, 0);
            }
            if !keep_metadata {
                ctx.op_mut(tail)
                    .attributes
                    .remove(OWNERSHIP_CONTRACT_ID_ATTR);
                ctx.op_mut(tail)
                    .attributes
                    .remove(CALL_ARGUMENT_OWNERSHIP_ATTR);
            }
            let before = print_module(&ctx, module.op());
            let _: OwnershipContractError = insert_rc_with_policies_and_trusted_summaries(
                &mut ctx,
                module,
                BorrowedParameterPolicy::ElideProvenBorrowed,
                TemporaryBorrowPolicy::Preserve,
                &trusted,
            )
            .expect_err("zero-operand indirect tail must fail closed");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn nonfinal_tail_placement_fails_before_rc_mutation() {
        let (mut ctx, module, trusted) = lowered_trusted_cps(
            r#"core.module @test {
  func.func @target(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%0: tribute_rt.anyref) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %0 {callee = @target, tribute.calling_convention = 2}
    func.unreachable
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        assert!(
            insert_rc_with_policies_and_trusted_summaries(
                &mut ctx,
                module,
                BorrowedParameterPolicy::ElideProvenBorrowed,
                TemporaryBorrowPolicy::Preserve,
                &trusted,
            )
            .is_err()
        );
        assert_eq!(print_module(&ctx, module.op()), before);
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
