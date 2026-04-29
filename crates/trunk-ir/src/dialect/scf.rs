//! Arena-based scf dialect.

#[crate::dialect(crate = crate)]
mod scf {
    fn r#if(cond: ()) -> result {
        #[region(then_region)]
        {}
        #[region(else_region)]
        {}
    }

    fn switch(discriminant: ()) {
        #[region(body)]
        {}
    }

    #[attr(value: any)]
    fn case() {
        #[region(body)]
        {}
    }

    fn default() {
        #[region(body)]
        {}
    }

    fn r#yield(#[rest] values: ()) {}

    fn r#loop(#[rest] init: ()) -> result {
        #[region(body)]
        {}
    }

    fn r#continue(#[rest] values: ()) {}

    fn r#break(value: ()) {}
}

// =========================================================================
// Canonicalization patterns
//
// `scf.if(arith.const Int(_) : core.i1)` is the first user of the
// non-fold escape hatch in `transforms::canonicalize`: the rewrite
// splices the chosen region's body into the parent block — a multi-op
// mutation that doesn't fit the single-op `FoldResult` shape.
// =========================================================================

use crate::context::IrContext;
use crate::dialect::arith::const_int_value;
use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueRef};
use crate::rewrite::{PatternRewriter, RewritePattern};

crate::register_canonicalize_pattern!(make_if_const_fold);

fn make_if_const_fold() -> Box<dyn RewritePattern> {
    Box::new(IfConstFold)
}

/// `scf.if(arith.const Int(c) : core.i1)` → splice the chosen region's
/// body into the parent block.
///
/// When the condition is a compile-time constant, exactly one branch
/// runs; the other is dead. The chosen region's `scf.yield <values>`
/// supplies the if op's results, so:
///
/// 1. Move every non-terminator op out of the active region's single
///    block, into the parent block, in original order, immediately
///    before the `scf.if` op. Captured operands stay valid because the
///    cloned ops keep referencing the same `ValueRef`s.
/// 2. RAUW the if op's results with the yield's operands.
/// 3. Erase the `scf.if`. Its other (dead) region is dropped along
///    with the orphaned yield from the active region.
///
/// Multi-block regions are left alone — they only appear post-`scf_to_cf`,
/// where this pattern doesn't run anyway. The pattern self-filters on
/// `scf.if` and bails out on any structural mismatch (yield arity,
/// non-`i1` const, malformed regions).
pub struct IfConstFold;

impl RewritePattern for IfConstFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !If::matches(ctx, op) {
            return false;
        }
        let if_op = match If::from_op(ctx, op) {
            Ok(o) => o,
            Err(_) => return false,
        };
        let cond = if_op.cond(ctx);

        let Some(cond_value) = const_int_value(ctx, cond) else {
            return false;
        };
        let active_region = if cond_value != 0 {
            if_op.then_region(ctx)
        } else {
            if_op.else_region(ctx)
        };

        // Active region must be a single block whose terminator is `scf.yield`.
        let blocks = ctx.region(active_region).blocks.to_vec();
        let [active_block] = blocks.as_slice() else {
            return false;
        };
        let active_block = *active_block;
        let region_ops: Vec<OpRef> = ctx.block(active_block).ops.to_vec();
        let Some((yield_op, body_ops)) = region_ops.split_last() else {
            return false;
        };
        if !Yield::matches(ctx, *yield_op) {
            return false;
        }

        // Yield arity must match the if op's result count.
        let yield_operands: Vec<ValueRef> = ctx.op_operands(*yield_op).to_vec();
        let if_result_count = ctx.op_results(op).len();
        if yield_operands.len() != if_result_count {
            return false;
        }

        // Splice body ops out into the parent block, before `op`.
        let parent_block = match ctx.op(op).parent_block {
            Some(b) => b,
            None => return false,
        };
        for body_op in body_ops {
            ctx.detach_op(*body_op);
            ctx.insert_op_before(parent_block, op, *body_op);
        }

        // Detach + destroy the yield — its only user was the implicit
        // edge to the if op's results, which we are about to erase.
        ctx.detach_op(*yield_op);
        ctx.remove_op(*yield_op);

        // Erase the if op, mapping its results to the yield's operands.
        rewriter.erase_op(yield_operands);
        true
    }

    fn name(&self) -> &'static str {
        "IfConstFold"
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::symbol::Symbol;
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only `IfConstFold` on `module`. Tests stay focused on this
    /// pattern even though the production `canonicalize` pass aggregates
    /// every registered fold and pattern.
    fn run_if_const_fold(ctx: &mut IrContext, module: Module) -> ApplyResult {
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(IfConstFold)
            .apply_partial(ctx, module)
    }

    fn count_ops(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> usize {
        let dialect_sym = Symbol::from_dynamic(dialect);
        let name_sym = Symbol::from_dynamic(name);
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect_sym && data.name == name_sym {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn if_const_true_splices_then_region() {
        // `scf.if(const true) { %a = arith.addi %x, %x; yield %a } { yield %x }`
        // → splice the addi into the parent block, replace the if's
        // result with %a (the yield's operand).
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.const {value = 1} : core.i1
    %r = scf.if %t : core.i32 {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_if_const_fold(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        // The then-region's `arith.addi` is now at the parent block.
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn if_const_false_splices_else_region() {
        // Mirror of the previous test but with const false: the else
        // region is the active one.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %f = arith.const {value = 0} : core.i1
    %r = scf.if %f : core.i32 {
      scf.yield %x
    } {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_if_const_fold(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn if_const_does_not_match_non_const_cond() {
        // The condition is a block argument — the pattern can't decide
        // which branch runs at compile time, so the if must stay.
        let input = r#"core.module @test {
  func.func @f(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      scf.yield %x
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_if_const_fold(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 1);
    }

    #[test]
    fn if_const_with_empty_active_region_just_forwards_yield() {
        // No body ops in the active region — the rewrite still fires:
        // the if is erased and its result becomes the yield's operand.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.const {value = 1} : core.i1
    %r = scf.if %t : core.i32 {
      scf.yield %x
    } {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_if_const_fold(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        // The else region's addi was dropped along with the if op (it
        // was unreachable). Only the parent's `func.return` remains for
        // `arith` ops, plus the const cond.
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
    }
}
