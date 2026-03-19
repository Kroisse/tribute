//! Tail-Resumptive Optimization (TRO) analysis and conversion.
//!
//! Most practical abilities (State, Reader, Writer, Console) are **tail-resumptive**:
//! the handler immediately resumes with `k(value)`, making continuation capture
//! unnecessary. This module detects such patterns and converts `ability.suspend` ops
//! to `ability.yield` so that downstream lowering passes can skip the resume/shift
//! overhead based on the operation type rather than an attribute.
//!
//! A `ability.suspend` body is tail-resumptive when:
//! 1. `%k` (block arg 0) is used exactly once
//! 2. That single use is a `ability.resume %k, %value` operation
//! 3. The result of `ability.resume` flows directly to `scf.yield` (tail position)

use trunk_ir::context::IrContext;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

use tribute_ir::dialect::ability;

/// Information about a tail-resumptive suspend body.
pub struct TailResumptiveInfo {
    /// The `ability.resume` operation in the body.
    pub resume_op: OpRef,
    /// The value passed to `ability.resume` (i.e., the value to yield directly).
    pub resume_value: ValueRef,
}

/// Analyze whether a `ability.suspend` body is tail-resumptive.
///
/// Returns `Some(TailResumptiveInfo)` if the body matches the pattern:
/// ```text
/// ^bb0(%k: continuation, %shift_args: any):
///     // ... ops that don't use %k ...
///     %result = ability.resume %k, %value
///     scf.yield %result
/// ```
pub fn is_tail_resumptive(ctx: &IrContext, suspend_body: RegionRef) -> Option<TailResumptiveInfo> {
    let blocks = &ctx.region(suspend_body).blocks;
    let &first_block = blocks.first()?;

    // Block arg 0 is the continuation `%k`
    let block_args = ctx.block_args(first_block);
    if block_args.is_empty() {
        return None;
    }
    let k = block_args[0];

    // Check: %k is used exactly once
    let uses = ctx.uses(k);
    if uses.len() != 1 {
        return None;
    }

    // Check: the single use is a `ability.resume` operation
    let use_op = uses[0].user;
    let resume = ability::Resume::from_op(ctx, use_op).ok()?;

    // Verify the continuation operand is indeed %k
    if resume.continuation(ctx) != k {
        return None;
    }

    let resume_value = resume.value(ctx);
    let resume_result = resume.result(ctx);

    // Check: ability.resume result flows directly to scf.yield (tail position)
    let resume_uses = ctx.uses(resume_result);
    if resume_uses.len() != 1 {
        return None;
    }

    let yield_op = resume_uses[0].user;
    if !arena_scf::Yield::matches(ctx, yield_op) {
        return None;
    }

    // Verify: scf.yield is the last op in the block (tail position)
    let block_ops = &ctx.block(first_block).ops;
    let &last_op = block_ops.last()?;
    if last_op != yield_op {
        return None;
    }

    // Verify: ability.resume is the second-to-last op
    if block_ops.len() < 2 {
        return None;
    }
    let &second_to_last = &block_ops[block_ops.len() - 2];
    if second_to_last != use_op {
        return None;
    }

    Some(TailResumptiveInfo {
        resume_op: use_op,
        resume_value,
    })
}

/// Pattern that converts `ability.suspend` to `ability.yield` when the body is tail-resumptive.
struct SuspendToYieldPattern;

impl RewritePattern for SuspendToYieldPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(suspend) = ability::Suspend::from_op(ctx, op) else {
            return false;
        };
        if is_tail_resumptive(ctx, suspend.body(ctx)).is_none() {
            return false;
        }

        let loc = ctx.op(op).location;
        let ability_ref = suspend.ability_ref(ctx);
        let op_name = suspend.op_name(ctx);
        let body = suspend.body(ctx);

        // Detach the body region from the old suspend op
        ctx.detach_region(body);

        // Create a new ability.yield op with the same attributes and body
        let new_op = ability::r#yield(ctx, loc, ability_ref, op_name, body);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Convert tail-resumptive `ability.suspend` ops to `ability.yield` in the module.
///
/// Uses `PatternApplicator` to walk all operations and replace eligible
/// `ability.suspend` ops with `ability.yield`.
pub fn convert_tail_resumptive(ctx: &mut IrContext, module: Module) {
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(SuspendToYieldPattern);
    applicator.apply_partial(ctx, module);
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::refs::OpRef;
    use trunk_ir::walk;

    /// Find all ability.suspend ops in the module.
    fn find_suspend_ops(ctx: &IrContext, module: trunk_ir::rewrite::Module) -> Vec<OpRef> {
        let body = module.body(ctx).unwrap();
        collect_ops_by_type::<ability::Suspend>(ctx, body)
    }

    /// Find all ability.yield ops in the module.
    fn find_yield_ops(ctx: &IrContext, module: trunk_ir::rewrite::Module) -> Vec<OpRef> {
        let body = module.body(ctx).unwrap();
        collect_ops_by_type::<ability::Yield>(ctx, body)
    }

    /// Collect ops matching a dialect op type from a region (recursive).
    fn collect_ops_by_type<T: DialectOp>(ctx: &IrContext, region: RegionRef) -> Vec<OpRef> {
        let mut result = Vec::new();
        let _ = walk::walk_region::<()>(ctx, region, &mut |op| {
            if T::matches(ctx, op) {
                result.push(op);
            }
            std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
        });
        result
    }

    // ====================================================================
    // is_tail_resumptive tests
    // ====================================================================

    #[test]
    fn simple_k_value_pattern_is_tr() {
        // Simple tail-resumptive: k(value) in tail position
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = ability.resume %k, %sv : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = ability::Suspend::from_op(&ctx, suspends[0]).unwrap();
        let info = is_tail_resumptive(&ctx, suspend.body(&ctx));
        assert!(info.is_some(), "simple k(value) pattern should be TR");
    }

    #[test]
    fn k_unused_is_not_tr() {
        // %k is not used at all
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %c = arith.const {value = 42} : core.i32
          %cast = core.unrealized_conversion_cast %c : core.ptr
          scf.yield %cast
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = ability::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "unused k should not be TR"
        );
    }

    #[test]
    fn k_used_twice_is_not_tr() {
        // %k used twice
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r1 = ability.resume %k, %sv : core.ptr
          %r2 = ability.resume %k, %r1 : core.ptr
          scf.yield %r2
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = ability::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "k used twice should not be TR"
        );
    }

    #[test]
    fn k_in_non_resume_is_not_tr() {
        // %k used in a func.call instead of ability.resume
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @some_func(%x: core.ptr) -> core.ptr {
    func.return %x
  }
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = func.call %k {callee = @some_func} : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = ability::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "k used in non-resume context should not be TR"
        );
    }

    #[test]
    fn resume_not_in_tail_position_is_not_tr() {
        // ability.resume is not in tail position (extra ops after)
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = ability.resume %k, %sv : core.ptr
          %c = arith.const {value = 1} : core.i32
          %cast = core.unrealized_conversion_cast %c : core.ptr
          scf.yield %cast
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = ability::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "resume not in tail position should not be TR"
        );
    }

    // ====================================================================
    // convert_tail_resumptive tests
    // ====================================================================

    #[test]
    fn convert_marks_tr_suspend() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = ability.resume %k, %sv : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
        );

        // Before conversion: 1 suspend, 0 yields
        assert_eq!(find_suspend_ops(&ctx, module).len(), 1);
        assert_eq!(find_yield_ops(&ctx, module).len(), 0);

        convert_tail_resumptive(&mut ctx, module);

        // After conversion: 0 suspends, 1 yield
        assert_eq!(
            find_suspend_ops(&ctx, module).len(),
            0,
            "TR suspend should be converted to yield"
        );
        let yields = find_yield_ops(&ctx, module);
        assert_eq!(yields.len(), 1, "Should have 1 ability.yield");
        assert!(
            ability::Yield::matches(&ctx, yields[0]),
            "Should be ability.yield"
        );
    }

    #[test]
    fn convert_does_not_mark_non_tr_suspend() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %c = arith.const {value = 42} : core.i32
          %cast = core.unrealized_conversion_cast %c : core.ptr
          scf.yield %cast
      }
    }
    func.return %1
  }
}"#,
        );

        convert_tail_resumptive(&mut ctx, module);

        // Non-TR suspend should remain as ability.suspend
        assert_eq!(
            find_suspend_ops(&ctx, module).len(),
            1,
            "non-TR suspend should NOT be converted"
        );
        assert_eq!(
            find_yield_ops(&ctx, module).len(),
            0,
            "Should have no ability.yield"
        );
    }

    #[test]
    fn convert_mixed_arms_converts_only_tr() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %yr = arith.const {value = 0} : core.ptr
    %hf = arith.const {value = 0} : core.ptr
    %1 = ability.handle_dispatch %yr, %hf {tag = 1, result_type = core.ptr} : core.ptr {
      ability.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = ability.resume %k, %sv : core.ptr
          scf.yield %r
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @set} {
        ^bb0(%k2: core.ptr, %sv2: core.ptr):
          %c = arith.const {value = 0} : core.i32
          %cast = core.unrealized_conversion_cast %c : core.ptr
          scf.yield %cast
      }
    }
    func.return %1
  }
}"#,
        );

        convert_tail_resumptive(&mut ctx, module);

        // First suspend (get) should be converted to yield
        let yields = find_yield_ops(&ctx, module);
        assert_eq!(yields.len(), 1, "get arm should be converted to yield");
        let yield_op = ability::Yield::from_op(&ctx, yields[0]).unwrap();
        assert_eq!(yield_op.op_name(&ctx), trunk_ir::Symbol::new("get"));

        // Second suspend (set) should remain as suspend
        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1, "set arm should remain as suspend");
        let suspend_op = ability::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert_eq!(suspend_op.op_name(&ctx), trunk_ir::Symbol::new("set"));
    }
}
