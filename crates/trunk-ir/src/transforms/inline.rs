//! Function inlining pass for TrunkIR.
//!
//! Replaces `func.call` and `func.tail_call` ops with a cloned copy of the
//! callee's body, splicing it into the caller's CFG.
//!
//! # Strategy
//!
//! For regular `func.call` sites, split-and-splice is used uniformly even
//! for single-block callees — the continuation block's degeneracy can be
//! collapsed later by canonicalization. See
//! `transforms::scf_to_cf::replace_yield_with_br` for the same pattern
//! applied to SCF lowering.
//!
//! For `func.tail_call`, no split is required: the tail call is itself
//! a terminator and "callee return = caller return", so cloned `func.return`
//! ops remain valid in the caller's function body.

use std::collections::BTreeMap;

use crate::context::{BlockArgData, IrContext};
use crate::dialect::{cf, func};
use crate::ir_mapping::IrMapping;
use crate::ops::DialectOp;
use crate::refs::{BlockRef, OpRef, RegionRef, ValueRef};
use crate::rewrite::helpers::{erase_op, inline_region_blocks, split_block};
use crate::types::Location;

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
/// Steps common to both:
/// 1. Look up callee body, entry block, and parameters.
/// 2. Validate operand/return arity.
/// 3. Clone the callee body into a detached region, mapping callee params
///    to call-site operand values.
///
/// Regular call steps:
/// 4. Split caller block at the call op → continuation block. Add block
///    args for call results; RAUW results to those args.
/// 5. Splice cloned body before the continuation block.
/// 6. Append `cf.br cloned_entry` to the caller block as its new terminator.
/// 7. Rewrite each cloned `func.return %v` to `cf.br %v → continuation`.
/// 8. Erase the original call op (detached from continuation block).
///
/// Tail call steps:
/// 4'. Splice cloned body at the end of the caller's parent region
///     (no continuation block is needed).
/// 5'. Append `cf.br cloned_entry` to the caller block, replacing the
///     tail-call terminator.
/// 6'. Erase the original tail-call op. Cloned `func.return` ops remain
///     as-is — they now return from the *caller*.
pub fn inline_single_call(
    ctx: &mut IrContext,
    call_op: OpRef,
    callee_func_op: OpRef,
) -> Result<(), InlineError> {
    // --- Shared validation and cloning ---
    let is_tail = func::TailCall::matches(ctx, call_op);
    let is_regular = func::Call::matches(ctx, call_op);
    if !is_tail && !is_regular {
        return Err(InlineError::NotACall);
    }
    if !func::Func::matches(ctx, callee_func_op) {
        return Err(InlineError::NotAFunc);
    }

    let caller_block = ctx
        .op(call_op)
        .parent_block
        .ok_or(InlineError::CallOpDetached)?;
    let parent_region = ctx
        .block(caller_block)
        .parent_region
        .expect("caller block must belong to a region");

    let call_loc = ctx.op(call_op).location;

    // Callee body region & entry block
    let callee_body = ctx
        .op(callee_func_op)
        .regions
        .first()
        .copied()
        .ok_or(InlineError::CalleeHasNoBody)?;
    let callee_entry = ctx
        .region(callee_body)
        .blocks
        .first()
        .copied()
        .ok_or(InlineError::CalleeHasEmptyBody)?;

    // Callee params (entry block args) & call operands
    let callee_params: Vec<_> = ctx.block_args(callee_entry).to_vec();
    let call_operands: Vec<_> = ctx.op_operands(call_op).to_vec();
    if callee_params.len() != call_operands.len() {
        return Err(InlineError::ArityMismatch {
            callee_params: callee_params.len(),
            call_operands: call_operands.len(),
        });
    }

    // Deep-clone the callee body region. `clone_region` creates fresh entry
    // block args and maps old callee params → new cloned args. We pass the
    // call operands as `cf.br` block args, matching the new entry args.
    let mut mapping = IrMapping::new();
    let cloned_region = ctx.clone_region(callee_body, &mut mapping);
    let cloned_entry = mapping
        .lookup_block(callee_entry)
        .expect("clone_region must register entry block mapping");

    if is_tail {
        inline_tail_call(
            ctx,
            call_op,
            caller_block,
            parent_region,
            cloned_region,
            cloned_entry,
            &call_operands,
            call_loc,
        )
    } else {
        inline_regular_call(
            ctx,
            call_op,
            caller_block,
            parent_region,
            cloned_region,
            cloned_entry,
            &call_operands,
            call_loc,
        )
    }
}

// =========================================================================
// Regular call path
// =========================================================================

#[allow(clippy::too_many_arguments)]
fn inline_regular_call(
    ctx: &mut IrContext,
    call_op: OpRef,
    caller_block: BlockRef,
    parent_region: RegionRef,
    cloned_region: RegionRef,
    cloned_entry: BlockRef,
    call_operands: &[ValueRef],
    loc: Location,
) -> Result<(), InlineError> {
    // Snapshot result values & types before mutations.
    let call_results: Vec<_> = ctx.op_results(call_op).to_vec();
    let call_result_types: Vec<_> = ctx.op_result_types(call_op).to_vec();

    // 1. Split the caller block so call_op + everything after moves to
    //    `continuation_block`.
    let continuation_block = split_block(ctx, caller_block, call_op);

    // 2. Add block args on the continuation block matching the call's
    //    result types, then RAUW call results → continuation args.
    for (result, ty) in call_results.iter().zip(call_result_types.iter()) {
        let new_arg = ctx.add_block_arg(
            continuation_block,
            BlockArgData {
                ty: *ty,
                attrs: BTreeMap::new(),
            },
        );
        ctx.replace_all_uses(*result, new_arg);
    }

    // 3. Splice cloned body before the continuation block.
    inline_region_blocks(ctx, cloned_region, parent_region, Some(continuation_block));

    // 4. Append `cf.br cloned_entry(...call_operands)` to the caller block.
    //    The call operands become the cloned entry block's args.
    let br_op = cf::br(ctx, loc, call_operands.iter().copied(), cloned_entry);
    ctx.push_op(caller_block, br_op.op_ref());

    // 5. Rewrite each cloned `func.return %v` → `cf.br %v → continuation`.
    replace_returns_with_branches(ctx, cloned_entry, continuation_block, loc);

    // 6. Erase the original call op. After RAUW its results have no uses.
    erase_op(ctx, call_op);

    Ok(())
}

// =========================================================================
// Tail call path
// =========================================================================

#[allow(clippy::too_many_arguments)]
fn inline_tail_call(
    ctx: &mut IrContext,
    call_op: OpRef,
    caller_block: BlockRef,
    parent_region: RegionRef,
    cloned_region: RegionRef,
    cloned_entry: BlockRef,
    call_operands: &[ValueRef],
    loc: Location,
) -> Result<(), InlineError> {
    // 1. Splice cloned body at end of the caller's parent region. CFG
    //    correctness depends on `cf.br`, not on block order in the region.
    inline_region_blocks(ctx, cloned_region, parent_region, None);

    // 2. Replace the tail-call terminator with `cf.br cloned_entry(...ops)`.
    erase_op(ctx, call_op);
    let br_op = cf::br(ctx, loc, call_operands.iter().copied(), cloned_entry);
    ctx.push_op(caller_block, br_op.op_ref());

    // 3. Cloned `func.return` ops stay as-is: they now return from the caller.

    Ok(())
}

// =========================================================================
// Helpers
// =========================================================================

/// Walk the block graph starting from `entry_block` (all blocks reachable
/// within the same parent region) and replace each `func.return %v...` with
/// `cf.br %v... → target`.
fn replace_returns_with_branches(
    ctx: &mut IrContext,
    entry_block: BlockRef,
    target: BlockRef,
    loc: Location,
) {
    // Walk all blocks in the region that became the cloned body. Since
    // `inline_region_blocks` moved those blocks into `parent_region`, we'd
    // like to limit rewrites to the just-inlined blocks.
    //
    // Trick: collect all blocks reachable from `entry_block` via successor
    // edges. Since the cloned body is self-contained, this hits exactly the
    // inlined blocks without touching pre-existing ones.
    let mut visited: std::collections::HashSet<BlockRef> = std::collections::HashSet::new();
    let mut stack = vec![entry_block];
    while let Some(b) = stack.pop() {
        if !visited.insert(b) {
            continue;
        }
        let ops: Vec<OpRef> = ctx.block(b).ops.to_vec();
        for op in ops {
            if func::Return::matches(ctx, op) {
                let values: Vec<_> = ctx.op_operands(op).to_vec();
                let br = cf::br(ctx, loc, values, target);
                ctx.remove_op_from_block(b, op);
                ctx.push_op(b, br.op_ref());
                // Don't chase successors of a return op — there are none.
            } else {
                let successors = ctx.op(op).successors.clone();
                for succ in &successors {
                    stack.push(*succ);
                }
            }
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod mechanics {
    use super::*;
    use crate::location::Span;
    use crate::*;
    use smallvec::smallvec;

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
        inline_single_call(&mut ctx, call_op, helper).expect("inline should succeed");

        let body = ctx.op(caller).regions[0];
        // After inlining, there should be multiple blocks in caller's body.
        assert!(
            ctx.region(body).blocks.len() > 1,
            "multi-block callee should leave caller with multiple blocks"
        );
        let (has_call, _) = scan_body(&ctx, body);
        assert!(!has_call);
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
        matches!(err, InlineError::ArityMismatch { .. });
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
