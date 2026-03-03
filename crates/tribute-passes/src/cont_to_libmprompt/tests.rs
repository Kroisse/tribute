//! Tests for cont_to_libmprompt pass (arena-based).

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::scf as arena_scf;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::parser::parse_test_module;
use trunk_ir::arena::printer::print_module;

fn run_pass(ir: &str) -> String {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    super::lower_cont_to_libmprompt(&mut ctx, module);
    print_module(&ctx, module.op())
}

// ============================================================================
// Test 1: FFI declarations
// ============================================================================

#[test]
fn test_ffi_declarations() {
    let ir = r#"core.module @test {
  func.func @_placeholder() -> core.nil {
    func.return
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    super::lower_cont_to_libmprompt(&mut ctx, module);
    let output = print_module(&ctx, module.op());

    // Check all expected FFI function declarations are present
    let expected_ffi = [
        "@__tribute_prompt",
        "@__tribute_yield",
        "@__tribute_resume",
        "@__tribute_resume_drop",
        "@__tribute_yield_active",
        "@__tribute_get_yield_op_idx",
        "@__tribute_get_yield_continuation",
        "@__tribute_get_yield_shift_value",
        "@__tribute_reset_yield_state",
    ];
    for name in &expected_ffi {
        assert!(
            output.contains(name),
            "Expected FFI declaration {} not found in output:\n{}",
            name,
            output
        );
    }

    // Test idempotency: calling ensure_libmprompt_ffi again shouldn't duplicate
    let count_before = expected_ffi
        .iter()
        .map(|name| output.matches(name).count())
        .sum::<usize>();

    super::ffi::ensure_libmprompt_ffi(&mut ctx, module);
    let output_after = print_module(&ctx, module.op());
    let count_after = expected_ffi
        .iter()
        .map(|name| output_after.matches(name).count())
        .sum::<usize>();

    assert_eq!(
        count_before, count_after,
        "FFI declaration count changed after second ensure_libmprompt_ffi call.\nBefore: {}\nAfter: {}",
        count_before, count_after
    );
}

// ============================================================================
// Test 2: Lower cont.shift -> func.call @__tribute_yield
// ============================================================================

#[test]
fn test_lower_shift_basic() {
    let result = run_pass(
        r#"core.module @test {
  func.func @test_fn() -> core.ptr {
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
// Test 3: Lower cont.resume -> func.call @__tribute_resume
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
// Test 4: Lower cont.drop -> func.call @__tribute_resume_drop
// ============================================================================

#[test]
fn test_lower_drop_basic() {
    let result = run_pass(
        r#"core.module @test {
  func.func @test_drop(%k: core.ptr) -> core.nil {
    cont.drop %k
    func.return
  }
}"#,
    );
    insta::assert_snapshot!(result);
}

// ============================================================================
// Test 5: compute_live_ins excludes nested defs
// ============================================================================

#[test]
fn test_live_ins_excludes_nested_defs() {
    let ir = r#"core.module @test {
  func.func @test_fn(%x: core.i32) -> core.i32 {
    %cond = arith.const {value = 1} : core.i1
    %r = scf.if %cond : core.i32 {
      %inner = arith.const {value = 42} : core.i32
      scf.yield %inner
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);

    // Find the func.func op
    let body = module.body(&ctx).unwrap();
    let first_block = ctx.region(body).blocks[0];
    let mut scf_if_op = None;
    for &op in &ctx.block(first_block).ops.clone() {
        if let Ok(func_op) = arena_func::Func::from_op(&ctx, op) {
            let func_body = func_op.body(&ctx);
            let func_block = ctx.region(func_body).blocks[0];
            for &inner_op in &ctx.block(func_block).ops.clone() {
                if arena_scf::If::from_op(&ctx, inner_op).is_ok() {
                    scf_if_op = Some(inner_op);
                    break;
                }
            }
        }
    }
    let scf_if_op = scf_if_op.expect("should find scf.if op");

    let regions = ctx.op(scf_if_op).regions.clone();
    assert_eq!(
        regions.len(),
        2,
        "scf.if should have 2 regions (then, else)"
    );
    let then_region = regions[0];
    let else_region = regions[1];

    // Then region: %inner is defined inside, so 0 live-ins
    let then_live_ins = super::push_prompt::compute_live_ins(&ctx, then_region);
    assert_eq!(
        then_live_ins.len(),
        0,
        "then region should have 0 live-ins (all values defined inside), got: {:?}",
        then_live_ins
    );

    // Else region: %x is used but defined outside, so 1 live-in
    let else_live_ins = super::push_prompt::compute_live_ins(&ctx, else_region);
    assert_eq!(
        else_live_ins.len(),
        1,
        "else region should have 1 live-in (%x), got: {:?}",
        else_live_ins
    );
}

// ============================================================================
// Test 6: Nested defs not leaked in compute_live_ins
// ============================================================================

#[test]
fn test_live_ins_nested_defs_not_leaked() {
    let ir = r#"core.module @test {
  func.func @test_fn(%x: core.i32) -> core.i32 {
    %cond = arith.const {value = 1} : core.i1
    %outer_r = scf.if %cond : core.i32 {
      %inner_r = scf.if %cond : core.i32 {
        %deep = arith.const {value = 99} : core.i32
        scf.yield %deep
      } {
        scf.yield %x
      }
      scf.yield %inner_r
    } {
      scf.yield %x
    }
    func.return %outer_r
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);

    // Find the outer scf.if op
    let body = module.body(&ctx).unwrap();
    let first_block = ctx.region(body).blocks[0];
    let mut outer_scf_if = None;
    for &op in &ctx.block(first_block).ops.clone() {
        if let Ok(func_op) = arena_func::Func::from_op(&ctx, op) {
            let func_body = func_op.body(&ctx);
            let func_block = ctx.region(func_body).blocks[0];
            for &inner_op in &ctx.block(func_block).ops.clone() {
                if arena_scf::If::from_op(&ctx, inner_op).is_ok() {
                    outer_scf_if = Some(inner_op);
                    break;
                }
            }
        }
    }
    let outer_scf_if = outer_scf_if.expect("should find outer scf.if op");

    let regions = ctx.op(outer_scf_if).regions.clone();
    let then_region = regions[0];

    // Then region uses %cond (for the inner scf.if) and %x (in the inner else),
    // both defined outside. Inner-defined values (%deep, %inner_r) should NOT appear.
    let live_ins = super::push_prompt::compute_live_ins(&ctx, then_region);
    assert_eq!(
        live_ins.len(),
        2,
        "outer then region should have 2 live-ins (%cond and %x), got: {:?}",
        live_ins
    );
}

// ============================================================================
// Test 7: handler_dispatch with done only
// ============================================================================

#[test]
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
// Test 8: handler_dispatch with single suspend arm
// ============================================================================

#[test]
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
// Test 9: handler_dispatch with multiple suspend arms
// ============================================================================

#[test]
fn test_handler_dispatch_multi_suspend_arms() {
    let result = run_pass(
        r#"core.module @test {
  func.func @__tribute_resume(%k: core.ptr, %sv: core.ptr) -> core.ptr {
    func.return %k
  }
  func.func @test_multi_arms() -> core.i32 {
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
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @set} {
        ^bb0(%k2: core.ptr, %sv2: core.ptr):
          %r2 = func.call %k2, %sv2 {callee = @__tribute_resume} : core.ptr
          scf.yield %r2
      }
    }
    func.return %1
  }
}"#,
    );
    insta::assert_snapshot!(result);
}
