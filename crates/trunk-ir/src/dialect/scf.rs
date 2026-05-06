//! Arena-based scf dialect.

#[trunk_ir::dialect]
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
// Canonicalization folds
// =========================================================================

use crate::context::IrContext;
use crate::dialect::arith::const_int_value;
use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueRef};
use crate::transforms::canonicalize::FoldResult;

/// `scf.if(arith.const Int(c) : core.i1)` → splice the chosen region's
/// body into the parent block.
///
/// When the condition is a compile-time constant, exactly one branch
/// runs; the other is dead. The chosen region's `scf.yield <values>`
/// supplies the if op's results, so the fold names the body ops to
/// keep and the values that should take over the if op's result slots
/// — the canonicalize dispatcher does the splice + cleanup of the
/// dead branch and the chosen region's yield via [`FoldResult::Splice`].
///
/// Multi-block regions are left alone — they only appear
/// post-`scf_to_cf`, where this pass doesn't run anyway. Bails out on
/// any structural mismatch (yield arity, non-`i1` const, malformed
/// regions).
#[trunk_ir::canonicalize_fold(scf.r#if)]
pub(crate) fn fold_if(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let if_op = If::from_op(ctx, op).ok()?;
    let cond = if_op.cond(ctx);
    let cond_value = const_int_value(ctx, cond)?;
    let active_region = if cond_value != 0 {
        if_op.then_region(ctx)
    } else {
        if_op.else_region(ctx)
    };

    // Active region must be a single block whose terminator is `scf.yield`.
    let blocks = ctx.region(active_region).blocks.to_vec();
    let [active_block] = blocks.as_slice() else {
        return None;
    };
    let region_ops: Vec<OpRef> = ctx.block(*active_block).ops.to_vec();
    let (yield_op, body_ops) = region_ops.split_last()?;
    if !Yield::matches(ctx, *yield_op) {
        return None;
    }

    // Yield arity must match the if op's result count.
    let yield_operands: Vec<ValueRef> = ctx.op_operands(*yield_op).to_vec();
    if yield_operands.len() != ctx.op_results(op).len() {
        return None;
    }

    Some(FoldResult::Splice {
        body: body_ops.to_vec(),
        results: yield_operands,
    })
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
    use crate::transforms::canonicalize::{FoldDispatchPattern, folds_for_dialect};
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only this dialect's folds on `module` via a single
    /// [`FoldDispatchPattern`] — mirrors `arith.rs::run_arith_patterns`
    /// to keep per-dialect tests isolated from other dialects' folds.
    fn run_scf_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_folds(folds_for_dialect("scf"));
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
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

        let result = run_scf_patterns(&mut ctx, module);
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

        let result = run_scf_patterns(&mut ctx, module);
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

        let result = run_scf_patterns(&mut ctx, module);
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

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        // The else region's addi was dropped along with the if op (it
        // was unreachable). Only the parent's `func.return` remains for
        // `arith` ops, plus the const cond.
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
    }
}
