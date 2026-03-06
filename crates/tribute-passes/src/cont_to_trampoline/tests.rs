//! Tests for cont_to_trampoline pass (arena-based).

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::parser::parse_test_module;
use trunk_ir::arena::printer::print_module;

fn run_pass(ir: &str) -> String {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    super::lower_cont_to_trampoline(&mut ctx, module)
        .expect("lower_cont_to_trampoline should succeed");
    print_module(&ctx, module.op())
}

// ============================================================================
// Test: Lower cont.shift in effectful function → trampoline ops
// ============================================================================

#[test]
fn test_lower_shift_basic() {
    let result = run_pass(
        r#"core.module @test {
  func.func @effectful_fn() -> core.ptr effects core.effect_row(core.ability_ref() {name = @State}) {
    %tag = arith.const {value = 42} : core.i32
    %val = arith.const {value = 1} : core.i32
    %val_ptr = core.unrealized_conversion_cast %val : core.ptr
    %result = cont.shift %tag, %val_ptr {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : core.ptr {
    }
    func.return %result
  }
}"#,
    );
    insta::assert_snapshot!(result);
}

// ============================================================================
// Test: Lower cont.resume → trampoline continuation extract + call
// ============================================================================

#[test]
fn test_lower_resume_basic() {
    let result = run_pass(
        r#"core.module @test {
  func.func @test_resume(%k: core.ptr, %val: core.ptr) -> core.ptr {
    %result = cont.resume %k, %val : core.ptr
    func.return %result
  }
}"#,
    );
    insta::assert_snapshot!(result);
}

// ============================================================================
// Test: Lower cont.push_prompt → yield check + dispatch
// TODO: LowerPushPromptPattern moves body ops without detaching from parent
//       block first, causing a panic. Needs fix in patterns.rs.
// ============================================================================

#[test]
#[ignore = "LowerPushPromptPattern has a bug: body ops not detached before insert"]
fn test_lower_push_prompt_basic() {
    let result = run_pass(
        r#"core.module @test {
  func.func @test_push_prompt() -> core.i32 {
    %0 = cont.push_prompt {tag = 1} : core.i32 {
      %1 = arith.const {value = 42} : core.i32
      scf.yield %1
    } {
    }
    func.return %0
  }
}"#,
    );
    insta::assert_snapshot!(result);
}

// ============================================================================
// Test: handler_dispatch with done only
// TODO: blocked by push_prompt lowering bug (same root cause)
// ============================================================================

#[test]
#[ignore = "blocked by LowerPushPromptPattern bug"]
fn test_handler_dispatch_done_only() {
    let result = run_pass(
        r#"core.module @test {
  func.func @test_done_only() -> core.i32 {
    %0 = cont.push_prompt {tag = 1} : core.i32 {
      %1 = arith.const {value = 42} : core.i32
      scf.yield %1
    } {
    }
    %2 = cont.handler_dispatch %0 {tag = 1, result_type = core.i32} : core.i32 {
      cont.done {
        ^bb0(%done_val: core.i32):
          scf.yield %done_val
      }
    }
    func.return %2
  }
}"#,
    );
    insta::assert_snapshot!(result);
}

// ============================================================================
// Test: handler_dispatch with single suspend arm
// TODO: blocked by push_prompt lowering bug (same root cause)
// ============================================================================

#[test]
#[ignore = "blocked by LowerPushPromptPattern bug"]
fn test_handler_dispatch_single_suspend_arm() {
    let result = run_pass(
        r#"core.module @test {
  func.func @__tribute_resume(%k: core.ptr, %sv: core.ptr) -> core.ptr {
    func.return %k
  }
  func.func @test_single_arm() -> core.i32 {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c = arith.const {value = 10} : core.i32
      scf.yield %c
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.i32} : core.i32 {
      cont.done {
        ^bb0(%v: core.i32):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = func.call %k, %sv {callee = @__tribute_resume} : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
    );
    insta::assert_snapshot!(result);
}

// ============================================================================
// Test: cont.drop passes through (not lowered by this pass)
// ============================================================================

#[test]
fn test_drop_passes_through() {
    let result = run_pass(
        r#"core.module @test {
  func.func @test_drop(%k: core.ptr) -> core.nil {
    cont.drop %k
    func.return
  }
}"#,
    );
    // cont.drop should remain (it's marked legal in the conversion target)
    assert!(
        result.contains("cont.drop"),
        "cont.drop should pass through, got:\n{}",
        result
    );
}
