//! Shared utilities for continuation lowering passes.
//!
//! This module contains functions and types used by `cont_to_yield_bubbling`
//! (and potentially other continuation backends).

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{RegionRef, ValueRef};

// ============================================================================
// Hash-Based Dispatch
// ============================================================================

/// Compute operation index using hash-based dispatch.
///
/// Delegates to `tribute_ir::dialect::ability::compute_op_idx`.
pub fn compute_op_idx(ability_ref: Option<Symbol>, op_name: Option<Symbol>) -> u32 {
    tribute_ir::dialect::ability::compute_op_idx(ability_ref, op_name)
}

// ============================================================================
// Handler Dispatch Utilities
// ============================================================================

// ============================================================================
// Region Utilities
// ============================================================================

/// Extract the result value from the last operation of an arena region.
///
/// Returns the first operand of a trailing `scf.yield`, or
/// the first result of the last operation otherwise.
pub fn get_region_result_value(ctx: &IrContext, region: RegionRef) -> Option<ValueRef> {
    let blocks = &ctx.region(region).blocks;
    let &last_block = blocks.last()?;
    let ops = &ctx.block(last_block).ops;
    let &last_op = ops.last()?;

    // If the last op is scf.yield, return its first operand (the yielded value)
    if let Ok(yield_op) = arena_scf::Yield::from_op(ctx, last_op) {
        return yield_op.values(ctx).first().copied();
    }

    // Otherwise, return the first result of the last op
    let results = ctx.op_results(last_op);
    if results.is_empty() {
        None
    } else {
        Some(results[0])
    }
}

/// Information about a suspend arm for dispatch (arena version).
pub struct SuspendArm {
    /// Expected op_idx for this arm
    pub expected_op_idx: u32,
    /// The body region containing the handler arm code
    pub body: RegionRef,
    /// Whether this arm is tail-resumptive (handler just does `k(value)`)
    pub tail_resumptive: bool,
}

/// Collect suspend arms from handler_dispatch's body region (arena version).
///
/// Uses hash-based dispatch: each arm's op_idx is computed from the ability
/// name and operation name via `compute_op_idx`.
pub fn collect_suspend_arms(ctx: &IrContext, body: RegionRef) -> Vec<SuspendArm> {
    let mut arms = Vec::new();
    let mut seen_op_indices: HashSet<u32> = HashSet::new();

    let blocks = &ctx.region(body).blocks;
    let Some(&first_block) = blocks.first() else {
        return arms;
    };

    for &op in &ctx.block(first_block).ops {
        // Try cont.yield first (tail-resumptive), then cont.suspend
        let (ability_ref_ty, op_name, body, is_tr) =
            if let Ok(yield_op) = arena_cont::Yield::from_op(ctx, op) {
                (
                    yield_op.ability_ref(ctx),
                    yield_op.op_name(ctx),
                    yield_op.body(ctx),
                    true,
                )
            } else if let Ok(suspend_op) = arena_cont::Suspend::from_op(ctx, op) {
                (
                    suspend_op.ability_ref(ctx),
                    suspend_op.op_name(ctx),
                    suspend_op.body(ctx),
                    false,
                )
            } else {
                continue;
            };

        let ability_data = ctx.types.get(ability_ref_ty);
        let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
            Some(trunk_ir::types::Attribute::Symbol(s)) => Some(*s),
            _ => panic!(
                "collect_suspend_arms: cont.suspend/yield has invalid ability_ref type: {:?}",
                ability_data,
            ),
        };

        let expected_op_idx = compute_op_idx(ability_name, Some(op_name));
        assert!(
            seen_op_indices.insert(expected_op_idx),
            "compute_op_idx collision in handler dispatch: op_idx {} appears twice \
             (ability={:?}, op={}). This indicates a hash collision that would \
             cause silent mis-dispatch at runtime.",
            expected_op_idx,
            ability_name,
            op_name,
        );
        arms.push(SuspendArm {
            expected_op_idx,
            body,
            tail_resumptive: is_tr,
        });
    }

    arms
}

/// Get the done region from handler_dispatch's body (arena version).
///
/// Finds the first `cont.done` child op and returns its body region.
pub fn get_done_region(ctx: &IrContext, body: RegionRef) -> Option<RegionRef> {
    let blocks = &ctx.region(body).blocks;
    let &first_block = blocks.first()?;

    for &op in &ctx.block(first_block).ops {
        if let Ok(done_op) = arena_cont::Done::from_op(ctx, op) {
            return Some(done_op.body(ctx));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::{BlockData, IrContext, RegionData};
    use trunk_ir::dialect::{arith, cont as arena_cont, scf as arena_scf};
    use trunk_ir::location::Span;
    use trunk_ir::refs::RegionRef;
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

    // ====================================================================
    // compute_op_idx
    // ====================================================================

    #[test]
    fn compute_op_idx_is_deterministic() {
        let idx1 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let idx2 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn compute_op_idx_differs_for_different_ops() {
        let ability = Some(Symbol::new("State"));
        let get_idx = compute_op_idx(ability, Some(Symbol::new("get")));
        let set_idx = compute_op_idx(ability, Some(Symbol::new("set")));
        assert_ne!(get_idx, set_idx);
    }

    #[test]
    fn compute_op_idx_differs_for_different_abilities() {
        let op = Some(Symbol::new("get"));
        let state_idx = compute_op_idx(Some(Symbol::new("State")), op);
        let console_idx = compute_op_idx(Some(Symbol::new("Console")), op);
        assert_ne!(state_idx, console_idx);
    }

    #[test]
    fn compute_op_idx_none_position_matters() {
        let sym = Symbol::new("test");
        let idx_none_some = compute_op_idx(None, Some(sym));
        let idx_some_none = compute_op_idx(Some(sym), None);
        assert_ne!(idx_none_some, idx_some_none);
    }

    #[test]
    fn compute_op_idx_within_range() {
        let idx = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        assert!(idx < 0x7FFFFFFF);
    }

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn ability_ref_type(ctx: &mut IrContext, name: Symbol) -> trunk_ir::refs::TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(name))
                .build(),
        )
    }

    /// Helper: create a simple body region with a const + yield.
    fn make_simple_body(ctx: &mut IrContext, loc: Location) -> RegionRef {
        let i32_ty = i32_type(ctx);
        let c = arith::r#const(ctx, loc, i32_ty, Attribute::Int(0));
        let c_result = c.result(ctx);
        let y = arena_scf::r#yield(ctx, loc, [c_result]);
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, c.op_ref());
        ctx.push_op(block, y.op_ref());
        ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        })
    }

    // get_region_result_value tests

    #[test]
    fn arena_get_region_result_value_returns_yield_operand() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let c_result = c.result(&ctx);
        let y = arena_scf::r#yield(&mut ctx, loc, [c_result]);
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, c.op_ref());
        ctx.push_op(block, y.op_ref());
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        let result = get_region_result_value(&ctx, region);
        assert_eq!(result, Some(c_result));
    }

    #[test]
    fn arena_get_region_result_value_returns_last_op_result() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, c.op_ref());
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        let result = get_region_result_value(&ctx, region);
        assert!(result.is_some());
    }

    #[test]
    fn arena_get_region_result_value_empty_region() {
        let (mut ctx, loc) = test_ctx();
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![],
            parent_op: None,
        });

        assert!(get_region_result_value(&ctx, region).is_none());
    }

    #[test]
    fn arena_get_region_result_value_empty_block() {
        let (mut ctx, loc) = test_ctx();
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        assert!(get_region_result_value(&ctx, region).is_none());
    }

    // collect_suspend_arms tests

    #[test]
    fn arena_collect_suspend_arms_extracts_all() {
        let (mut ctx, loc) = test_ctx();
        let state_ref_ty = ability_ref_type(&mut ctx, Symbol::new("State"));

        let done_body = make_simple_body(&mut ctx, loc);
        let get_body = make_simple_body(&mut ctx, loc);
        let set_body = make_simple_body(&mut ctx, loc);

        let done_op = arena_cont::done(&mut ctx, loc, done_body);
        let get_op = arena_cont::suspend(&mut ctx, loc, state_ref_ty, Symbol::new("get"), get_body);
        let set_op = arena_cont::suspend(&mut ctx, loc, state_ref_ty, Symbol::new("set"), set_body);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, done_op.op_ref());
        ctx.push_op(block, get_op.op_ref());
        ctx.push_op(block, set_op.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        let arms = collect_suspend_arms(&ctx, body);
        assert_eq!(arms.len(), 2);

        let expected_get = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let expected_set = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("set")));
        assert_eq!(arms[0].expected_op_idx, expected_get);
        assert_eq!(arms[1].expected_op_idx, expected_set);
    }

    #[test]
    fn arena_collect_suspend_arms_done_only() {
        let (mut ctx, loc) = test_ctx();
        let done_body = make_simple_body(&mut ctx, loc);
        let done_op = arena_cont::done(&mut ctx, loc, done_body);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, done_op.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        assert_eq!(collect_suspend_arms(&ctx, body).len(), 0);
    }

    // get_done_region tests

    #[test]
    fn arena_get_done_region_returns_region() {
        let (mut ctx, loc) = test_ctx();
        let done_body = make_simple_body(&mut ctx, loc);
        let done_op = arena_cont::done(&mut ctx, loc, done_body);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, done_op.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        assert!(get_done_region(&ctx, body).is_some());
    }

    #[test]
    fn arena_get_done_region_returns_none_when_no_done() {
        let (mut ctx, loc) = test_ctx();
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        assert!(get_done_region(&ctx, body).is_none());
    }
}
