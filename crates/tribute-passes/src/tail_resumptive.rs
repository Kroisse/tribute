//! Tail-Resumptive Optimization (TRO) analysis and annotation.
//!
//! Most practical abilities (State, Reader, Writer, Console) are **tail-resumptive**:
//! the handler immediately resumes with `k(value)`, making continuation capture
//! unnecessary. This module detects such patterns and annotates `cont.suspend` ops
//! so that downstream lowering passes can skip the resume/shift overhead.
//!
//! A `cont.suspend` body is tail-resumptive when:
//! 1. `%k` (block arg 0) is used exactly once
//! 2. That single use is a `cont.resume %k, %value` operation
//! 3. The result of `cont.resume` flows directly to `scf.yield` (tail position)

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, ValueRef};
use trunk_ir::types::Attribute;

/// Information about a tail-resumptive suspend body.
pub struct TailResumptiveInfo {
    /// The `cont.resume` operation in the body.
    pub resume_op: OpRef,
    /// The value passed to `cont.resume` (i.e., the value to yield directly).
    pub resume_value: ValueRef,
}

/// Attribute key used to mark tail-resumptive `cont.suspend` ops.
pub const TAIL_RESUMPTIVE_ATTR: &str = "tail_resumptive";

/// Analyze whether a `cont.suspend` body is tail-resumptive.
///
/// Returns `Some(TailResumptiveInfo)` if the body matches the pattern:
/// ```text
/// ^bb0(%k: continuation, %shift_args: any):
///     // ... ops that don't use %k ...
///     %result = cont.resume %k, %value
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

    // Check: the single use is a `cont.resume` operation
    let use_op = uses[0].user;
    let resume = arena_cont::Resume::from_op(ctx, use_op).ok()?;

    // Verify the continuation operand is indeed %k
    if resume.continuation(ctx) != k {
        return None;
    }

    let resume_value = resume.value(ctx);
    let resume_result = resume.result(ctx);

    // Check: cont.resume result flows directly to scf.yield (tail position)
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

    // Verify: cont.resume is the second-to-last op
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

/// Annotate all tail-resumptive `cont.suspend` ops in the module.
///
/// Walks all operations recursively, finds `cont.suspend` ops, and sets a
/// `tail_resumptive` attribute on those that match the tail-resumptive pattern.
pub fn annotate_tail_resumptive(ctx: &mut IrContext, module: trunk_ir::Module) {
    let body = match module.body(ctx) {
        Some(body) => body,
        None => return,
    };

    // Collect all cont.suspend ops from the module
    let suspend_ops = collect_all_suspend_ops(ctx, body);

    for suspend_op in suspend_ops {
        let suspend = match arena_cont::Suspend::from_op(ctx, suspend_op) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let body = suspend.body(ctx);
        if is_tail_resumptive(ctx, body).is_some() {
            ctx.op_mut(suspend_op)
                .attributes
                .insert(Symbol::new(TAIL_RESUMPTIVE_ATTR), Attribute::Int(1));
        }
    }
}

/// Collect all `cont.suspend` ops in the module by walking the IR tree.
fn collect_all_suspend_ops(ctx: &IrContext, module_body: RegionRef) -> Vec<OpRef> {
    let mut result = Vec::new();
    collect_suspend_ops_from_region(ctx, module_body, &mut result);
    result
}

fn collect_suspend_ops_from_region(ctx: &IrContext, region: RegionRef, out: &mut Vec<OpRef>) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops.clone() {
            if arena_cont::Suspend::matches(ctx, op) {
                out.push(op);
            }
            // Recurse into nested regions
            for &nested_region in &ctx.op(op).regions.clone() {
                collect_suspend_ops_from_region(ctx, nested_region, out);
            }
        }
    }
}

/// Check if a `cont.suspend` op is marked as tail-resumptive.
pub fn is_marked_tail_resumptive(ctx: &IrContext, suspend_op: OpRef) -> bool {
    ctx.op(suspend_op)
        .attributes
        .get(&Symbol::new(TAIL_RESUMPTIVE_ATTR))
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;
    use trunk_ir::dialect::cont as arena_cont;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;

    /// Find all cont.suspend ops in the module.
    fn find_suspend_ops(ctx: &IrContext, module: trunk_ir::Module) -> Vec<OpRef> {
        let body = module.body(ctx).unwrap();
        collect_all_suspend_ops(ctx, body)
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
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = cont.resume %k, %sv : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = arena_cont::Suspend::from_op(&ctx, suspends[0]).unwrap();
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
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
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

        let suspend = arena_cont::Suspend::from_op(&ctx, suspends[0]).unwrap();
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
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r1 = cont.resume %k, %sv : core.ptr
          %r2 = cont.resume %k, %r1 : core.ptr
          scf.yield %r2
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);

        let suspend = arena_cont::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "k used twice should not be TR"
        );
    }

    #[test]
    fn k_in_non_resume_is_not_tr() {
        // %k used in a func.call instead of cont.resume
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @some_func(%x: core.ptr) -> core.ptr {
    func.return %x
  }
  func.func @test_fn() -> core.ptr {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
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

        let suspend = arena_cont::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "k used in non-resume context should not be TR"
        );
    }

    #[test]
    fn resume_not_in_tail_position_is_not_tr() {
        // cont.resume is not in tail position (extra ops after)
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = cont.resume %k, %sv : core.ptr
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

        let suspend = arena_cont::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert!(
            is_tail_resumptive(&ctx, suspend.body(&ctx)).is_none(),
            "resume not in tail position should not be TR"
        );
    }

    // ====================================================================
    // annotate_tail_resumptive tests
    // ====================================================================

    #[test]
    fn annotate_marks_tr_suspend() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = cont.resume %k, %sv : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
        );

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 1);
        assert!(!is_marked_tail_resumptive(&ctx, suspends[0]));

        annotate_tail_resumptive(&mut ctx, module);

        assert!(
            is_marked_tail_resumptive(&ctx, suspends[0]),
            "TR suspend should be marked"
        );
    }

    #[test]
    fn annotate_does_not_mark_non_tr_suspend() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
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

        annotate_tail_resumptive(&mut ctx, module);

        assert!(
            !is_marked_tail_resumptive(&ctx, suspends[0]),
            "non-TR suspend should NOT be marked"
        );
    }

    #[test]
    fn annotate_mixed_arms_marks_only_tr() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c0 = arith.const {value = 0} : core.i32
      scf.yield %c0
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.ptr} : core.ptr {
      cont.done {
        ^bb0(%v: core.ptr):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = cont.resume %k, %sv : core.ptr
          scf.yield %r
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @set} {
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

        let suspends = find_suspend_ops(&ctx, module);
        assert_eq!(suspends.len(), 2);

        annotate_tail_resumptive(&mut ctx, module);

        // First suspend (get) is TR
        let get_suspend = arena_cont::Suspend::from_op(&ctx, suspends[0]).unwrap();
        assert_eq!(get_suspend.op_name(&ctx), Symbol::new("get"));
        assert!(
            is_marked_tail_resumptive(&ctx, suspends[0]),
            "get arm should be TR"
        );

        // Second suspend (set) is NOT TR
        let set_suspend = arena_cont::Suspend::from_op(&ctx, suspends[1]).unwrap();
        assert_eq!(set_suspend.op_name(&ctx), Symbol::new("set"));
        assert!(
            !is_marked_tail_resumptive(&ctx, suspends[1]),
            "set arm should NOT be TR"
        );
    }
}
