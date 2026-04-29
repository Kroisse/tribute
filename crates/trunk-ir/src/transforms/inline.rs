//! Function inlining pass for TrunkIR.
//!
//! Replaces `func.call` and `func.tail_call` ops whose callee has a single
//! top-level block by splicing the callee body directly into the caller:
//!
//! - Regular call: clone pre-return ops into the caller block before the
//!   call site, RAUW call results with the mapped return values, detach
//!   the call op.
//! - Tail call: same clone, then emit a `func.return` with the mapped
//!   return values in place of the tail call.
//!
//! This pass does **not** depend on the `cf` dialect. Structured control
//! flow inside a callee (via `scf.if`, `scf.match`, etc.) lives inside op
//! regions, so the callee's *top-level* block remains singular even for
//! complex bodies. Multi-block callees — which typically only exist after
//! `scf_to_cf` has already lowered them — fall out of scope for v1 and
//! return `InlineError::MultiBlockCallee`.

use crate::context::IrContext;
use crate::dialect::func;
use crate::ir_mapping::IrMapping;
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef, ValueRef};

// =========================================================================
// Errors
// =========================================================================

/// Reasons an inline attempt can fail at the primitive level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineError {
    /// The callee `func.func` has no body region (extern declaration).
    CalleeHasNoBody,
    /// The callee body region is empty (no entry block).
    CalleeHasEmptyBody,
    /// The callee body has more than one top-level block. The v1 inliner
    /// only handles single-block callees to stay independent of the `cf`
    /// dialect; multi-block inlining is a follow-up.
    MultiBlockCallee,
    /// The callee body's terminator is not `func.return`.
    CalleeHasNoReturn,
    /// The call-site operand count does not match the callee's parameter count.
    ArityMismatch {
        callee_params: usize,
        call_operands: usize,
    },
    /// The call op is not attached to a block (malformed state).
    CallOpDetached,
    /// The callee op is not a `func.func`.
    NotAFunc,
    /// The call op is neither `func.call` nor `func.tail_call`.
    NotACall,
}

// =========================================================================
// Public entry: inline_single_call
// =========================================================================

/// Inline a single call site. Handles both `func.call` and `func.tail_call`.
///
/// Requires the callee's body region to contain exactly one top-level block
/// terminated by a `func.return`. Structured control flow inside the callee
/// (`scf.if`, `scf.match`, etc.) is preserved as-is because it lives inside
/// nested op regions, not top-level blocks.
pub fn inline_single_call(
    ctx: &mut IrContext,
    call_op: OpRef,
    callee_func_op: OpRef,
) -> Result<(), InlineError> {
    let is_tail = func::TailCall::matches(ctx, call_op);
    let is_regular = func::Call::matches(ctx, call_op);
    if !is_tail && !is_regular {
        return Err(InlineError::NotACall);
    }

    let call_loc = ctx.op(call_op).location;
    let ret_values = splice_callee_body_before(ctx, call_op, callee_func_op)?;

    if is_tail {
        // Replace the tail-call terminator with an equivalent `func.return`.
        let caller_block = ctx
            .op(call_op)
            .parent_block
            .ok_or(InlineError::CallOpDetached)?;
        let new_ret = func::r#return(ctx, call_loc, ret_values);
        ctx.insert_op_before(caller_block, call_op, new_ret.op_ref());
        ctx.detach_op(call_op);
    } else {
        // Regular call: RAUW each call result with its mapped return value,
        // then detach the original call op.
        let call_results: Vec<_> = ctx.op_results(call_op).to_vec();
        for (call_result, ret_val) in call_results.iter().zip(ret_values.iter()) {
            ctx.replace_all_uses(*call_result, *ret_val);
        }
        ctx.detach_op(call_op);
    }

    Ok(())
}

/// Clone the callee's body (all ops before its `func.return`) into the caller
/// block immediately before `call_op`, threading `call_op`'s operands through
/// the callee's entry-block parameters. Returns the mapped return values that
/// `call_op`'s results should be replaced with; the caller decides how to
/// consume them (RAUW + detach for a regular call, new `func.return` for a
/// tail call).
///
/// This is the pure splicing primitive used by both [`inline_single_call`]
/// and the pattern-based inliner. It does *not* mutate `call_op` itself.
fn splice_callee_body_before(
    ctx: &mut IrContext,
    call_op: OpRef,
    callee_func_op: OpRef,
) -> Result<Vec<ValueRef>, InlineError> {
    if !func::Call::matches(ctx, call_op) && !func::TailCall::matches(ctx, call_op) {
        return Err(InlineError::NotACall);
    }
    if !func::Func::matches(ctx, callee_func_op) {
        return Err(InlineError::NotAFunc);
    }

    let caller_block = ctx
        .op(call_op)
        .parent_block
        .ok_or(InlineError::CallOpDetached)?;

    // Callee body & entry block (must be single-block).
    let callee_body = ctx
        .op(callee_func_op)
        .regions
        .first()
        .copied()
        .ok_or(InlineError::CalleeHasNoBody)?;
    let callee_blocks: Vec<_> = ctx.region(callee_body).blocks.iter().copied().collect();
    if callee_blocks.len() > 1 {
        return Err(InlineError::MultiBlockCallee);
    }
    let callee_entry = *callee_blocks
        .first()
        .ok_or(InlineError::CalleeHasEmptyBody)?;

    let callee_params: Vec<_> = ctx.block_args(callee_entry).to_vec();
    let call_operands: Vec<_> = ctx.op_operands(call_op).to_vec();
    if callee_params.len() != call_operands.len() {
        return Err(InlineError::ArityMismatch {
            callee_params: callee_params.len(),
            call_operands: call_operands.len(),
        });
    }

    // Split the callee block's ops into body (all ops except the terminator)
    // and the `func.return` terminator itself.
    let callee_ops: Vec<OpRef> = ctx.block(callee_entry).ops.iter().copied().collect();
    let (ret_op, body_ops) = callee_ops
        .split_last()
        .ok_or(InlineError::CalleeHasEmptyBody)?;
    if !func::Return::matches(ctx, *ret_op) {
        return Err(InlineError::CalleeHasNoReturn);
    }

    // Mapping: callee params → call-site operands. External references in the
    // cloned body pass through via `lookup_value_or_default`.
    let mut mapping = IrMapping::new();
    for (param, arg) in callee_params.iter().zip(call_operands.iter()) {
        mapping.map_value(*param, *arg);
    }

    // Clone each body op (in order) into the caller block before `call_op`.
    // `clone_op` registers old-result → new-result in `mapping`, so later
    // ops see the remapped SSA values.
    for &op in body_ops {
        let new_op = ctx.clone_op(op, &mut mapping);
        ctx.insert_op_before(caller_block, call_op, new_op);
    }

    // Map the callee's return operands through the value mapping. These are
    // what the call site's result values should effectively become.
    Ok(ctx
        .op_operands(*ret_op)
        .iter()
        .map(|v| mapping.lookup_value_or_default(*v))
        .collect())
}

// =========================================================================
// Policy
// =========================================================================

use super::call_graph::{CallGraph, recursive_functions};
use crate::rewrite::{Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter};
use crate::symbol::Symbol;
use crate::types::Attribute;
use std::collections::HashSet;
use std::sync::Arc;

/// Knobs for the inlining pass.
#[derive(Debug, Clone)]
pub struct InlineConfig {
    /// Inline callees whose body has at most this many ops (summed across
    /// all blocks and nested regions).
    pub size_threshold: usize,
    /// If true, always inline callees with exactly one static call site
    /// (provided they do not escape via `func.constant`).
    pub always_inline_single_call_site: bool,
    /// If true, inline even when the callee is referenced by `func.constant`
    /// somewhere. Defaults to false (conservative).
    pub inline_across_func_constant: bool,
}

impl Default for InlineConfig {
    fn default() -> Self {
        Self {
            size_threshold: 16,
            always_inline_single_call_site: true,
            inline_across_func_constant: false,
        }
    }
}

/// Count all ops in a `func.func`'s body (recursively through nested regions).
fn op_count(ctx: &IrContext, func_op: OpRef) -> usize {
    let regions: Vec<RegionRef> = ctx.op(func_op).regions.iter().copied().collect();
    regions.iter().map(|&r| region_op_count(ctx, r)).sum()
}

fn region_op_count(ctx: &IrContext, region: RegionRef) -> usize {
    let blocks: Vec<_> = ctx.region(region).blocks.iter().copied().collect();
    let mut total = 0usize;
    for block in blocks {
        let ops: Vec<_> = ctx.block(block).ops.iter().copied().collect();
        for op in ops {
            total += 1;
            let nested: Vec<_> = ctx.op(op).regions.iter().copied().collect();
            for r in nested {
                total += region_op_count(ctx, r);
            }
        }
    }
    total
}

/// Decide whether `callee` should be inlined based on the graph and config.
fn should_inline(
    graph: &CallGraph,
    config: &InlineConfig,
    recursive: &HashSet<Symbol>,
    ctx: &IrContext,
    callee: Symbol,
) -> bool {
    // Must have a body in this module.
    let Some(&callee_op) = graph.func_ops.get(&callee) else {
        return false;
    };
    if ctx.op(callee_op).regions.is_empty() {
        return false;
    }
    // Skip extern/ABI functions: they are externally callable and the body
    // may be empty or have calling-convention constraints.
    if ctx
        .op(callee_op)
        .attributes
        .contains_key(&Symbol::new("abi"))
    {
        return false;
    }
    // Skip recursive functions to avoid unbounded instantiation.
    if recursive.contains(&callee) {
        return false;
    }
    let escapes = graph.has_constant_ref.contains(&callee);

    // Single-call-site rule (only when the callee doesn't escape).
    if config.always_inline_single_call_site
        && !escapes
        && graph.call_site_count.get(&callee).copied().unwrap_or(0) == 1
    {
        return true;
    }

    // Size threshold.
    if !escapes || config.inline_across_func_constant {
        return op_count(ctx, callee_op) <= config.size_threshold;
    }

    false
}

// =========================================================================
// Pass entry + worklist
// =========================================================================

/// Summary of an inlining run.
#[derive(Debug, Default)]
pub struct InlineResult {
    pub inlined_count: usize,
}

/// Run one pass of function inlining over `module` using default
/// config. `am` is consumed for [`CallGraph`] access and invalidated
/// iff at least one call site was rewritten.
///
/// Construct `am` at the pipeline level so that other passes in the
/// same phase (e.g. a future canonicalize) can share the cached call
/// graph. See [`crate::analysis`] module docs for the orchestration
/// convention.
pub fn inline_functions(
    ctx: &mut IrContext,
    module: Module,
    am: &mut crate::analysis::AnalysisCache,
) -> InlineResult {
    inline_functions_with_config(ctx, module, InlineConfig::default(), am)
}

/// Run one pass of function inlining with custom config. See
/// [`inline_functions`] for the `am` injection convention.
///
/// Drives the [`InlineCallSite`] pattern through a
/// [`PatternApplicator`] one iteration at a time, **rebuilding the
/// [`CallGraph`] analysis at each iteration boundary**. This is needed
/// for the `always_inline_single_call_site` fast path to stay accurate
/// when earlier rewrites duplicate a previously-single-site callee
/// across multiple newly-exposed call sites; without the rebuild the
/// heuristic becomes traversal-order dependent and can aggressively
/// splice a large callee that no longer has a single site. Approach
/// matches option (b) in issue #680.
///
/// Loops until the applicator reports zero changes, with no artificial
/// cap. Termination is guaranteed by policy: [`should_inline`] never
/// targets recursive or escaping callees, so the subgraph being inlined
/// is a DAG and every iteration strictly consumes at least one call op
/// from a finite pool of inlineable call-chain depth. A deeply nested
/// wrapper chain (e.g. `a → b → c → … → leaf`) can therefore require
/// as many outer iterations as its depth, which the loop handles
/// rather than returning a silent partial result.
pub fn inline_functions_with_config(
    ctx: &mut IrContext,
    module: Module,
    config: InlineConfig,
    am: &mut crate::analysis::AnalysisCache,
) -> InlineResult {
    let mut inlined_count = 0usize;

    loop {
        let graph = am.get::<CallGraph>(ctx, module.op());
        let recursive = recursive_functions(&graph);

        let pattern = InlineCallSite::new(Arc::clone(&graph), recursive, config.clone());

        let applicator = PatternApplicator::new(TypeConverter::new())
            .add_pattern(pattern)
            .with_max_iterations(1);
        let result = applicator.apply_partial(ctx, module);
        if result.total_changes == 0 {
            break;
        }
        inlined_count += result.total_changes;
        // Rewrites happened this iteration — drop the cached graph so the
        // next `am.get::<CallGraph>` rebuilds against the current IR.
        am.invalidate::<CallGraph>(module.op());
    }

    InlineResult { inlined_count }
}

// =========================================================================
// InlineCallSite pattern
// =========================================================================

/// A per-call-site rewrite pattern that inlines a `func.call` /
/// `func.tail_call` whose callee satisfies [`InlineConfig`]-driven policy.
///
/// # Graph staleness
///
/// The captured [`CallGraph`] must be **accurate for the current IR**.
/// The `always_inline_single_call_site` rule in particular is sensitive
/// to stale `call_site_count`: once an earlier rewrite duplicates a
/// previously-single-site callee across several newly-exposed call
/// sites, the heuristic would otherwise take the single-site fast path
/// N times over, splicing a potentially large callee body into every
/// caller.
///
/// [`inline_functions_with_config`] handles this by rebuilding the
/// graph between applicator iterations. Other orchestrators that
/// compose [`InlineCallSite`] with additional patterns should either
/// do the same (rebuild + re-register) or accept the caveat that the
/// single-call-site rule may fire on stale counts.
pub struct InlineCallSite {
    graph: Arc<CallGraph>,
    recursive: HashSet<Symbol>,
    config: InlineConfig,
}

impl InlineCallSite {
    /// Build a pattern against `graph` + `recursive` using `config`.
    pub fn new(graph: Arc<CallGraph>, recursive: HashSet<Symbol>, config: InlineConfig) -> Self {
        Self {
            graph,
            recursive,
            config,
        }
    }
}

impl RewritePattern for InlineCallSite {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let is_tail = func::TailCall::matches(ctx, op);
        let is_regular = func::Call::matches(ctx, op);
        if !is_tail && !is_regular {
            return false;
        }

        let callee = match ctx.op(op).attributes.get(&Symbol::new("callee")) {
            Some(Attribute::Symbol(s)) => *s,
            _ => return false,
        };

        if !should_inline(&self.graph, &self.config, &self.recursive, ctx, callee) {
            return false;
        }

        let Some(&callee_op) = self.graph.func_ops.get(&callee) else {
            return false;
        };

        let call_loc = ctx.op(op).location;
        let ret_values = match splice_callee_body_before(ctx, op, callee_op) {
            Ok(v) => v,
            Err(_) => {
                // Policy said yes but primitive rejected (e.g. callee has
                // no body). Keep pass resilient.
                return false;
            }
        };

        if is_tail {
            let new_ret = func::r#return(ctx, call_loc, ret_values).op_ref();
            rewriter.replace_op(new_ret);
        } else {
            rewriter.erase_op(ret_values);
        }
        true
    }

    fn name(&self) -> &'static str {
        "InlineCallSite"
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod mechanics {
    use super::*;
    use crate::dialect::cf;
    use crate::location::Span;
    use crate::*;
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    /// Build `func.func @name(params) -> ret_ty { body_builder }`.
    fn build_func<F>(
        ctx: &mut IrContext,
        loc: Location,
        name: &str,
        param_tys: &[TypeRef],
        ret_ty: TypeRef,
        body_builder: F,
    ) -> OpRef
    where
        F: FnOnce(&mut IrContext, BlockRef, &[ValueRef]),
    {
        let fn_ty = {
            let mut params = vec![ret_ty];
            params.extend_from_slice(param_tys);
            ctx.types.intern(
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .params(params)
                    .build(),
            )
        };
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: param_tys
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: BTreeMap::new(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let args: Vec<_> = ctx.block_args(entry).to_vec();
        body_builder(ctx, entry, &args);
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, Symbol::from_dynamic(name), fn_ty, body).op_ref()
    }

    fn build_simple_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> OpRef {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        ctx.create_op(module_data)
    }

    /// Find the single func.call inside `func_op`'s body. Panics if there
    /// is none or more than one.
    fn find_call_in(ctx: &IrContext, func_op: OpRef) -> OpRef {
        use crate::walk::{WalkAction, walk_region};
        use std::ops::ControlFlow;
        let mut calls = Vec::new();
        let body = ctx.op(func_op).regions[0];
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
                calls.push(op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(
            calls.len(),
            1,
            "expected exactly one call, found {}",
            calls.len()
        );
        calls[0]
    }

    #[test]
    fn inline_single_block_callee_no_args() {
        // helper() -> i32 { return 42 }
        // caller() -> i32 { return helper() }
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let helper = build_func(&mut ctx, loc, "helper", &[], i32_ty, |ctx, entry, _args| {
            let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(42));
            ctx.push_op(entry, c.op_ref());
            let ret = func::r#return(ctx, loc, [c.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });
        let caller = build_func(&mut ctx, loc, "caller", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("helper"));
            let result = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [result]);
            ctx.push_op(entry, ret.op_ref());
        });

        let _module = build_simple_module(&mut ctx, loc, vec![helper, caller]);

        let call_op = find_call_in(&ctx, caller);
        inline_single_call(&mut ctx, call_op, helper).expect("inline should succeed");

        // Walk caller body, confirm no func.call remains, arith.const present.
        let body = ctx.op(caller).regions[0];
        let (has_call, has_const) = scan_body(&ctx, body);
        assert!(!has_call, "call should be gone after inlining");
        assert!(has_const, "inlined constant should be present");
    }

    #[test]
    fn inline_single_block_callee_with_args() {
        // helper(x) -> i32 { return x + 1 }
        // caller() -> i32 { let c = 10; return helper(c) }
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let helper = build_func(
            &mut ctx,
            loc,
            "helper",
            &[i32_ty],
            i32_ty,
            |ctx, entry, args| {
                let one = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(1));
                ctx.push_op(entry, one.op_ref());
                let add_data =
                    OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("add"))
                        .operand(args[0])
                        .operand(one.result(ctx))
                        .result(i32_ty)
                        .build(ctx);
                let add = ctx.create_op(add_data);
                ctx.push_op(entry, add);
                let add_result = ctx.op_results(add)[0];
                let ret = func::r#return(ctx, loc, [add_result]);
                ctx.push_op(entry, ret.op_ref());
            },
        );

        let caller = build_func(&mut ctx, loc, "caller", &[], i32_ty, |ctx, entry, _args| {
            let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(10));
            ctx.push_op(entry, c.op_ref());
            let call = func::call(ctx, loc, [c.result(ctx)], i32_ty, Symbol::new("helper"));
            let result = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [result]);
            ctx.push_op(entry, ret.op_ref());
        });

        let _module = build_simple_module(&mut ctx, loc, vec![helper, caller]);

        let call_op = find_call_in(&ctx, caller);
        inline_single_call(&mut ctx, call_op, helper).expect("inline should succeed");

        let body = ctx.op(caller).regions[0];
        let (has_call, _) = scan_body(&ctx, body);
        assert!(!has_call);
    }

    #[test]
    fn inline_multi_block_callee() {
        // helper(x) { if (x) { return x } else { return 0 } }
        // caller() { let c = 5; return helper(c) }
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Build helper with two blocks via cf.cond_br
        let helper = build_func(
            &mut ctx,
            loc,
            "helper",
            &[i32_ty],
            i32_ty,
            |ctx, entry, args| {
                // then/else blocks
                let then_b = ctx.create_block(BlockData {
                    location: loc,
                    args: vec![],
                    ops: smallvec![],
                    parent_region: None,
                });
                let else_b = ctx.create_block(BlockData {
                    location: loc,
                    args: vec![],
                    ops: smallvec![],
                    parent_region: None,
                });
                let cond_br = cf::cond_br(ctx, loc, args[0], then_b, else_b);
                ctx.push_op(entry, cond_br.op_ref());

                // then: return x
                let ret_then = func::r#return(ctx, loc, [args[0]]);
                ctx.push_op(then_b, ret_then.op_ref());

                // else: return 0
                let zero = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(0));
                ctx.push_op(else_b, zero.op_ref());
                let ret_else = func::r#return(ctx, loc, [zero.result(ctx)]);
                ctx.push_op(else_b, ret_else.op_ref());
            },
        );

        // After building the func we need to add the then/else blocks into the body region.
        // They were created but not attached. Let's add them now.
        let body = ctx.op(helper).regions[0];
        let entry_block = ctx.region(body).blocks[0];
        let then_b = ctx.op(ctx.block(entry_block).ops[0]).successors[0];
        let else_b = ctx.op(ctx.block(entry_block).ops[0]).successors[1];
        ctx.region_mut(body).blocks.push(then_b);
        ctx.region_mut(body).blocks.push(else_b);
        ctx.block_mut(then_b).parent_region = Some(body);
        ctx.block_mut(else_b).parent_region = Some(body);

        let caller = build_func(&mut ctx, loc, "caller", &[], i32_ty, |ctx, entry, _args| {
            let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(5));
            ctx.push_op(entry, c.op_ref());
            let call = func::call(ctx, loc, [c.result(ctx)], i32_ty, Symbol::new("helper"));
            let result = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [result]);
            ctx.push_op(entry, ret.op_ref());
        });

        let _module = build_simple_module(&mut ctx, loc, vec![helper, caller]);

        let call_op = find_call_in(&ctx, caller);
        // v1 rejects multi-block callees to stay independent of the cf dialect.
        let err = inline_single_call(&mut ctx, call_op, helper).unwrap_err();
        assert_eq!(err, InlineError::MultiBlockCallee);
    }

    #[test]
    fn inline_tail_call_single_block() {
        // helper() -> i32 { return 42 }
        // caller() -> i32 { tail_call helper() }
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let helper = build_func(&mut ctx, loc, "helper", &[], i32_ty, |ctx, entry, _args| {
            let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(42));
            ctx.push_op(entry, c.op_ref());
            let ret = func::r#return(ctx, loc, [c.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });
        let caller = build_func(&mut ctx, loc, "caller", &[], i32_ty, |ctx, entry, _args| {
            let tc = func::tail_call(ctx, loc, std::iter::empty(), Symbol::new("helper"));
            ctx.push_op(entry, tc.op_ref());
        });

        let _module = build_simple_module(&mut ctx, loc, vec![helper, caller]);

        let call_op = find_call_in(&ctx, caller);
        inline_single_call(&mut ctx, call_op, helper).expect("inline should succeed");

        let body = ctx.op(caller).regions[0];
        let (has_call, has_const) = scan_body(&ctx, body);
        assert!(!has_call);
        assert!(has_const, "inlined constant should be present");
        // func.return survives in cloned body (as caller's return).
        assert!(has_return(&ctx, body));
    }

    #[test]
    fn inline_arity_mismatch_returns_err() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // helper expects 1 arg, caller passes 0
        let helper = build_func(
            &mut ctx,
            loc,
            "helper",
            &[i32_ty],
            i32_ty,
            |ctx, entry, args| {
                let ret = func::r#return(ctx, loc, [args[0]]);
                ctx.push_op(entry, ret.op_ref());
            },
        );
        let caller = build_func(&mut ctx, loc, "caller", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(
                ctx,
                loc,
                std::iter::empty(), // no operands
                i32_ty,
                Symbol::new("helper"),
            );
            let result = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [result]);
            ctx.push_op(entry, ret.op_ref());
        });

        let _module = build_simple_module(&mut ctx, loc, vec![helper, caller]);

        let call_op = find_call_in(&ctx, caller);
        let err = inline_single_call(&mut ctx, call_op, helper).unwrap_err();
        assert!(
            matches!(err, InlineError::ArityMismatch { .. }),
            "expected ArityMismatch, got {err:?}"
        );
    }

    // -----------------------------------------------------------------
    // Helpers for assertions
    // -----------------------------------------------------------------

    fn scan_body(ctx: &IrContext, region: RegionRef) -> (bool, bool) {
        use crate::walk::{WalkAction, walk_region};
        use std::ops::ControlFlow;
        let mut has_call = false;
        let mut has_const = false;
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
                has_call = true;
            }
            if ctx.op(op).dialect == Symbol::new("arith") && ctx.op(op).name == Symbol::new("const")
            {
                has_const = true;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        (has_call, has_const)
    }

    fn has_return(ctx: &IrContext, region: RegionRef) -> bool {
        use crate::walk::{WalkAction, walk_region};
        use std::ops::ControlFlow;
        let mut found = false;
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            if func::Return::matches(ctx, op) {
                found = true;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found
    }
}

#[cfg(test)]
mod pass {
    use super::*;
    use crate::location::Span;
    use crate::*;
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn build_func<F>(
        ctx: &mut IrContext,
        loc: Location,
        name: &str,
        param_tys: &[TypeRef],
        ret_ty: TypeRef,
        body_builder: F,
    ) -> OpRef
    where
        F: FnOnce(&mut IrContext, BlockRef, &[ValueRef]),
    {
        let fn_ty = {
            let mut params = vec![ret_ty];
            params.extend_from_slice(param_tys);
            ctx.types.intern(
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                    .params(params)
                    .build(),
            )
        };
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: param_tys
                .iter()
                .map(|&ty| BlockArgData {
                    ty,
                    attrs: BTreeMap::new(),
                })
                .collect(),
            ops: smallvec![],
            parent_region: None,
        });
        let args: Vec<_> = ctx.block_args(entry).to_vec();
        body_builder(ctx, entry, &args);
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, Symbol::from_dynamic(name), fn_ty, body).op_ref()
    }

    fn build_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).unwrap()
    }

    fn count_calls_to(ctx: &IrContext, func_op: OpRef, callee: &str) -> usize {
        use crate::walk::{WalkAction, walk_region};
        use std::ops::ControlFlow;
        let target = Symbol::from_dynamic(callee);
        let mut count = 0;
        let body = ctx.op(func_op).regions[0];
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if func::Call::matches(ctx, op)
                && let Some(crate::types::Attribute::Symbol(s)) =
                    ctx.op(op).attributes.get(&Symbol::new("callee"))
                && *s == target
            {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn inlines_small_single_call_site_helper() {
        // helper() -> i32 { return 42 }   — 2 ops (const, return) → small
        // main() -> i32 { return helper() }
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let helper = build_func(&mut ctx, loc, "helper", &[], i32_ty, |ctx, entry, _args| {
            let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(42));
            ctx.push_op(entry, c.op_ref());
            let ret = func::r#return(ctx, loc, [c.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });
        let main = build_func(&mut ctx, loc, "main", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("helper"));
            let r = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [r]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![helper, main]);

        let mut am = crate::analysis::AnalysisCache::new();
        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 1);
        assert_eq!(count_calls_to(&ctx, main, "helper"), 0);
    }

    #[test]
    fn skips_recursive_function() {
        // f() { return f() }  — self-recursion → must not inline
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let f = build_func(&mut ctx, loc, "f", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("f"));
            let r = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [r]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![f]);

        let mut am = crate::analysis::AnalysisCache::new();
        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 0);
        assert_eq!(count_calls_to(&ctx, f, "f"), 1);
    }

    #[test]
    fn skips_when_escapes_via_func_constant() {
        // helper() { ... }
        // other() { %c = func.constant @helper; ... }  — helper escapes
        // main() { helper() }  — would be single call site, but escapes
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let helper = build_func(&mut ctx, loc, "helper", &[], i32_ty, |ctx, entry, _args| {
            let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(42));
            ctx.push_op(entry, c.op_ref());
            let ret = func::r#return(ctx, loc, [c.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });
        let fn_ty_helper = ctx.op(helper).attributes.get(&Symbol::new("type")).cloned();
        let helper_fn_ty = match fn_ty_helper {
            Some(Attribute::Type(t)) => t,
            _ => panic!("expected type attr"),
        };

        let other = build_func(&mut ctx, loc, "other", &[], i32_ty, |ctx, entry, _args| {
            let c = func::constant(ctx, loc, helper_fn_ty, Symbol::new("helper"));
            ctx.push_op(entry, c.op_ref());
            let z = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(0));
            ctx.push_op(entry, z.op_ref());
            let ret = func::r#return(ctx, loc, [z.result(ctx)]);
            ctx.push_op(entry, ret.op_ref());
        });

        let main = build_func(&mut ctx, loc, "main", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("helper"));
            let r = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [r]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![helper, other, main]);

        // Helper size is 2 ops (≤16) so size-threshold path would still accept
        // under default config. But inline_across_func_constant=false → skip.
        let mut am = crate::analysis::AnalysisCache::new();
        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 0);
        assert_eq!(count_calls_to(&ctx, main, "helper"), 1);
    }

    #[test]
    fn respects_size_threshold() {
        // helper is built with 20 ops; threshold=16 should skip it.
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let helper = build_func(&mut ctx, loc, "helper", &[], i32_ty, |ctx, entry, _args| {
            // Chain 18 arith.const ops + 1 dummy + 1 return = 20 ops
            let mut last = None;
            for i in 0..19 {
                let c = crate::dialect::arith::r#const(ctx, loc, i32_ty, Attribute::Int(i as i128));
                ctx.push_op(entry, c.op_ref());
                last = Some(c.result(ctx));
            }
            let ret = func::r#return(ctx, loc, [last.unwrap()]);
            ctx.push_op(entry, ret.op_ref());
        });
        // Two call sites so single-call-site rule doesn't trigger.
        let a = build_func(&mut ctx, loc, "a", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("helper"));
            let r = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [r]);
            ctx.push_op(entry, ret.op_ref());
        });
        let b = build_func(&mut ctx, loc, "b", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("helper"));
            let r = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [r]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![helper, a, b]);

        let mut am = crate::analysis::AnalysisCache::new();
        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 0);
    }

    #[test]
    fn skips_abi_function() {
        // helper has abi attr — externally callable, skip even if small.
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = crate::dialect::arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        let c_result = c.result(&ctx);
        ctx.push_op(entry, c.op_ref());
        let ret = func::r#return(&mut ctx, loc, [c_result]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });

        let fn_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .params(vec![i32_ty])
                .build(),
        );
        let helper_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("helper")))
            .attr("type", Attribute::Type(fn_ty))
            .attr("abi", Attribute::String("C".to_owned()))
            .region(body)
            .build(&mut ctx);
        let helper = ctx.create_op(helper_data);

        let main = build_func(&mut ctx, loc, "main", &[], i32_ty, |ctx, entry, _args| {
            let call = func::call(ctx, loc, std::iter::empty(), i32_ty, Symbol::new("helper"));
            let r = call.result(ctx);
            ctx.push_op(entry, call.op_ref());
            let ret = func::r#return(ctx, loc, [r]);
            ctx.push_op(entry, ret.op_ref());
        });
        let module = build_module(&mut ctx, loc, vec![helper, main]);

        let mut am = crate::analysis::AnalysisCache::new();
        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 0);
    }

    #[test]
    fn inline_with_am_invalidates_callgraph_after_mutation() {
        use crate::analysis::AnalysisCache;
        use crate::transforms::call_graph::CallGraph;

        // Small helper + one call site: inliner should fire and invalidate
        // the cached CallGraph.
        let input = r#"core.module @test {
  func.func @helper() -> core.i32 {
    %0 = arith.const {value = 42} : core.i32
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = func.call {callee = @helper} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let mut am = AnalysisCache::new();
        // Seed the cache by pulling the graph once.
        let pre = am.get::<CallGraph>(&ctx, module.op());
        assert_eq!(pre.call_site_count.get(&Symbol::new("helper")), Some(&1));

        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 1);

        // The pass rebuilds the call graph between applicator iterations
        // to keep the single-call-site heuristic accurate. After the run
        // the cache therefore holds a **fresh** graph that reflects the
        // rewritten IR, not the pre-pass one: no call sites remain.
        let post = am
            .get_cached::<CallGraph>(module.op())
            .expect("driver must leave a fresh graph cached after the pass");
        assert_eq!(post.call_site_count.get(&Symbol::new("helper")), None);
        // And a subsequent explicit `get` must coincide with the cached
        // Arc (no recomputation).
        let fetched = am.get::<CallGraph>(&ctx, module.op());
        assert!(std::sync::Arc::ptr_eq(&post, &fetched));
    }

    #[test]
    fn inline_with_am_leaves_cache_intact_when_no_rewrite() {
        use crate::analysis::AnalysisCache;
        use crate::transforms::call_graph::CallGraph;

        // Recursive helper — policy rejects it, so nothing is inlined and
        // the cached graph should still be valid.
        let input = r#"core.module @test {
  func.func @helper() -> core.i32 {
    %0 = func.call {callee = @helper} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let mut am = AnalysisCache::new();
        let before = am.get::<CallGraph>(&ctx, module.op());

        let result = inline_functions(&mut ctx, module, &mut am);
        assert_eq!(result.inlined_count, 0);

        // No rewrite happened → cache preserved, same Arc.
        let after = am
            .get_cached::<CallGraph>(module.op())
            .expect("cache should survive a no-op pass run");
        assert!(std::sync::Arc::ptr_eq(&before, &after));
    }

    #[test]
    fn single_call_site_rule_observes_rebuilt_counts_per_iteration() {
        use crate::analysis::AnalysisCache;

        // Declaration order puts the callers (`a`, `b`) before the chain
        // (`wrapper`, `large`). In iteration 1 the applicator first
        // inlines `wrapper` into `a` and `b` via the size threshold,
        // which duplicates the inner `call @large` into both callers.
        // Iteration 2 then sees two real call sites to `large`, so the
        // single-call-site fast path must *not* fire for `large` even
        // though the iter-1 graph captured `large` at one site.
        //
        // The expected end state preserves the pre-refactor policy:
        // `large` is inlined into `wrapper` (1 extra copy), but neither
        // `a` nor `b` is spliced; both keep `call @large`.
        let mut input = String::from(
            "core.module @test {\n\
  func.func @a() -> core.i32 {\n\
    %0 = func.call {callee = @wrapper} : core.i32\n\
    func.return %0\n\
  }\n\
  func.func @b() -> core.i32 {\n\
    %0 = func.call {callee = @wrapper} : core.i32\n\
    func.return %0\n\
  }\n\
  func.func @wrapper() -> core.i32 {\n\
    %0 = func.call {callee = @large} : core.i32\n\
    func.return %0\n\
  }\n\
  func.func @large() -> core.i32 {\n",
        );
        // Push `large` above the default size threshold (16) with 18
        // `arith.const` ops + the `func.return`.
        for i in 0..18 {
            input.push_str(&format!(
                "    %{i} = arith.const {{value = {i}}} : core.i32\n"
            ));
        }
        input.push_str("    func.return %17\n  }\n}");

        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, &input);

        let mut am = AnalysisCache::new();
        inline_functions(&mut ctx, module, &mut am);

        // After the pass, `a` and `b` must still contain exactly one
        // `func.call` (to `@large`) — not a copy of `large`'s body.
        let post = am.get::<CallGraph>(&ctx, module.op());
        for name in ["a", "b"] {
            let f = post
                .func_ops
                .get(&Symbol::new(name))
                .copied()
                .unwrap_or_else(|| panic!("{name} must be in the graph"));
            let body = ctx.op(f).regions[0];
            let mut calls = 0usize;
            let _ = crate::walk::walk_region::<()>(&ctx, body, &mut |op| {
                if func::Call::matches(&ctx, op) || func::TailCall::matches(&ctx, op) {
                    calls += 1;
                }
                std::ops::ControlFlow::Continue(crate::walk::WalkAction::Advance)
            });
            assert_eq!(
                calls, 1,
                "{name} must retain exactly one call after the pass; \
                 the single-call-site rule should not fire on `large` \
                 in iter 2 against a stale graph"
            );
        }
    }

    #[test]
    fn reaches_fixed_point_on_chain_of_inlineable_calls() {
        use crate::analysis::AnalysisCache;

        // inner() -> 1
        // helper() -> inner()
        // main() -> helper()
        //
        // All three are small, single-call-site, non-recursive. A single
        // invocation of `inline_functions` should drive the applicator to
        // a fixed point that eliminates both `func.call` sites in one run.
        let input = r#"core.module @test {
  func.func @inner() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    func.return %0
  }
  func.func @helper() -> core.i32 {
    %0 = func.call {callee = @inner} : core.i32
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = func.call {callee = @helper} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let mut am = AnalysisCache::new();
        let result = inline_functions(&mut ctx, module, &mut am);

        // Two call sites are eliminated in a single pass run.
        assert_eq!(result.inlined_count, 2);

        // Fresh graph must contain no call edges at all.
        let post = am.get::<CallGraph>(&ctx, module.op());
        assert!(post.call_site_count.is_empty());
    }

    #[test]
    fn composes_with_another_rewrite_pattern_in_same_applicator() {
        use crate::analysis::AnalysisCache;
        use crate::dialect::arith::fold_addi;
        use crate::rewrite::{PatternApplicator, TypeConverter};
        use crate::transforms::canonicalize::FoldDispatchPattern;

        // helper(%arg) -> %arg + 0
        // main()       -> helper(5)
        //
        // Running the inliner pattern alongside a single-entry
        // canonicalize fold dispatcher in the same applicator drives
        // both to a fixed point: inlining exposes `arith.addi %5, %0`
        // inside `main`, the fold dispatcher rewrites it via
        // `fold_addi`, and the final result contains a single constant
        // + return.
        let input = r#"core.module @test {
  func.func @helper(%arg: core.i32) -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    %1 = arith.addi %arg, %0 : core.i32
    func.return %1
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 5} : core.i32
    %1 = func.call %0 {callee = @helper} : core.i32
    func.return %1
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let mut am = AnalysisCache::new();
        let graph = am.get::<CallGraph>(&ctx, module.op());
        let recursive = recursive_functions(&graph);

        let inline_pattern =
            InlineCallSite::new(Arc::clone(&graph), recursive, InlineConfig::default());

        let total_changes = {
            let dispatcher = FoldDispatchPattern::single("arith", "addi", fold_addi);
            let applicator = PatternApplicator::new(TypeConverter::new())
                .add_pattern(inline_pattern)
                .add_pattern_box(Box::new(dispatcher));
            applicator.apply_partial(&mut ctx, module).total_changes
        };

        // The applicator must have fired both patterns: the inliner once
        // (call → spliced body) and the fold at least once on the exposed
        // `arith.addi`. With one call + one add this floor is two changes.
        assert!(
            total_changes >= 2,
            "expected at least one inline + one fold mutation, got {total_changes}"
        );

        // End state: `main` has no calls and no `arith.addi` — just the
        // const + return left. This is the composition property: the
        // inliner exposed the `arith.addi %5, %0` inside `main`, and the
        // fold pattern erased it in the same applicator sweep.
        let main_op = graph
            .func_ops
            .get(&Symbol::new("main"))
            .copied()
            .expect("main must be in the graph");
        let body = ctx.op(main_op).regions[0];
        let (has_call, has_add) = scan_main_body(&ctx, body);
        assert!(!has_call, "main should not contain any calls after run");
        assert!(
            !has_add,
            "main should not contain any arith.addi after composition"
        );
    }

    fn scan_main_body(ctx: &IrContext, region: crate::refs::RegionRef) -> (bool, bool) {
        use crate::walk::{WalkAction, walk_region};
        use std::ops::ControlFlow;
        let mut has_call = false;
        let mut has_add = false;
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
                has_call = true;
            }
            if ctx.op(op).dialect == Symbol::new("arith") && ctx.op(op).name == Symbol::new("addi")
            {
                has_add = true;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        (has_call, has_add)
    }
}
