//! Several results in shared signatures and structured control, through
//! native and Wasm lowering and execution.
//!
//! The fixture is shared middle-end IR: two-result functions, a two-result
//! `func.call`, `scf.if` and `scf.index_switch` yielding two values, and
//! loops carrying several values. `main` traps unless every result is the
//! expected one, so a successful run checks the values.

use crate::common::{NativeTestBinary, run_wasm};
use tribute::pipeline::{
    emit_from_boundary_exit, emit_native_from_boundary_exit, run_target_to_boundary_exit,
};
use tribute_passes::abi_boundary::TargetKind;
use trunk_ir::{IrContext, Module};

const PROGRAM: &str = r#"core.module @multi_result {
  func.func @swap_if(%c: core.i1, %a: core.i32, %b: core.i32) -> (core.i32, core.i32) {
    %x, %y = scf.if %c -> (core.i32, core.i32) {
      scf.yield %b, %a
    } else {
      scf.yield %a, %b
    }
    func.return %x, %y
  }
  func.func @pick(%k: core.i32, %a: core.i32, %b: core.i32) -> (core.i32, core.i32) {
    %x, %y = scf.index_switch %k : (core.i32) -> (core.i32, core.i32)
    case 0 {
      scf.yield %a, %b
    }
    case 1 {
      scf.yield %b, %a
    }
    default {
      %s = arith.addi %a, %b : core.i32
      scf.yield %s, %s
    }
    func.return %x, %y
  }
  func.func @fib(%n: core.i32) -> (core.i32, core.i32) {
    %zero = arith.constant 0 : core.i32
    %one = arith.constant 1 : core.i32
    %a, %b = scf.loop(%n, %zero, %one) -> (core.i32, core.i32) {
      ^body(%i: core.i32, %x: core.i32, %y: core.i32):
        %done = arith.cmpi eq, %i, %zero : core.i32
        scf.if %done {
          scf.break %x, %y
        } else {
          scf.yield
        }
        %next = arith.subi %i, %one : core.i32
        %sum = arith.addi %x, %y : core.i32
        scf.continue %next, %y, %sum
    }
    func.return %a, %b
  }
  func.func @rotate(%n: core.i32, %a: core.i32, %b: core.i32) -> (core.i32, core.i32) {
    %zero = arith.constant 0 : core.i32
    %one = arith.constant 1 : core.i32
    %x, %y = scf.loop(%n, %a, %b) -> (core.i32, core.i32) {
      ^body(%i: core.i32, %p: core.i32, %q: core.i32):
        %done = arith.cmpi eq, %i, %zero : core.i32
        scf.if %done {
          scf.break %p, %q
        } else {
          scf.yield
        }
        %next = arith.subi %i, %one : core.i32
        scf.continue %next, %q, %p
    }
    func.return %x, %y
  }
  func.func @check(%ok: core.i1) {
    scf.if %ok {
      scf.yield
    } else {
      func.unreachable
    }
    func.return
  }
  func.func @expect(%x: core.i32, %y: core.i32, %ex: core.i32, %ey: core.i32) {
    %ox = arith.cmpi eq, %x, %ex : core.i32
    %oy = arith.cmpi eq, %y, %ey : core.i32
    %ok = arith.andi %ox, %oy : core.i1
    func.call @check(%ok) : (core.i1) -> ()
    func.return
  }
  func.func @main() -> core.nil {
    %true = arith.constant true : core.i1
    %false = arith.constant false : core.i1
    %c0 = arith.constant 0 : core.i32
    %c1 = arith.constant 1 : core.i32
    %c2 = arith.constant 2 : core.i32
    %c3 = arith.constant 3 : core.i32
    %c5 = arith.constant 5 : core.i32
    %c7 = arith.constant 7 : core.i32
    %c8 = arith.constant 8 : core.i32
    %c10 = arith.constant 10 : core.i32
    %c15 = arith.constant 15 : core.i32
    %s0, %s1 = func.call @swap_if(%true, %c1, %c2) : (core.i1, core.i32, core.i32) -> (core.i32, core.i32)
    func.call @expect(%s0, %s1, %c2, %c1) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %k0, %k1 = func.call @swap_if(%false, %c1, %c2) : (core.i1, core.i32, core.i32) -> (core.i32, core.i32)
    func.call @expect(%k0, %k1, %c1, %c2) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %p0, %p1 = func.call @pick(%c0, %c7, %c8) : (core.i32, core.i32, core.i32) -> (core.i32, core.i32)
    func.call @expect(%p0, %p1, %c7, %c8) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %q0, %q1 = func.call @pick(%c1, %c7, %c8) : (core.i32, core.i32, core.i32) -> (core.i32, core.i32)
    func.call @expect(%q0, %q1, %c8, %c7) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %r0, %r1 = func.call @pick(%c5, %c7, %c8) : (core.i32, core.i32, core.i32) -> (core.i32, core.i32)
    func.call @expect(%r0, %r1, %c15, %c15) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %f0, %f1 = func.call @fib(%c5) : (core.i32) -> (core.i32, core.i32)
    func.call @expect(%f0, %f1, %c5, %c8) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %t0, %t1 = func.call @rotate(%c3, %c1, %c10) : (core.i32, core.i32, core.i32) -> (core.i32, core.i32)
    func.call @expect(%t0, %t1, %c10, %c1) : (core.i32, core.i32, core.i32, core.i32) -> ()
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#;

/// One managed box returned in two result positions: each position owns a
/// unit, so the caller releases both without freeing the box twice.
const DUPLICATE_MANAGED_RETURN: &str = r#"core.module @duplicate_return {
  !Box = adt.struct<"Box",(core.i32)>
  !BoxRef = adt.struct<"Box">
  adt.struct_decl "Box" !Box ["value"]
  func.func @twice(%value: core.i32) -> (!BoxRef, !BoxRef) {
    %box = adt.struct_new(%value) : (core.i32) -> !Box
    func.return %box, %box
  }
  func.func @check(%ok: core.i1) {
    scf.if %ok {
      scf.yield
    } else {
      func.unreachable
    }
    func.return
  }
  func.func @main() -> core.nil {
    %c7 = arith.constant 7 : core.i32
    %a, %b = func.call @twice(%c7) : (core.i32) -> (!BoxRef, !BoxRef)
    %x = adt.struct_get %a {field = 0, type = !Box} : core.i32
    %y = adt.struct_get %b {field = 0, type = !Box} : core.i32
    %ok = arith.cmpi eq, %x, %y : core.i32
    func.call @check(%ok) : (core.i1) -> ()
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#;

#[test]
fn duplicate_managed_return_is_balanced_under_asan() {
    let mut ctx = IrContext::new();
    let op = trunk_ir::parser::parse_module(&mut ctx, DUPLICATE_MANAGED_RETURN)
        .unwrap_or_else(|error| panic!("fixture parses at {}: {}", error.offset, error.message));
    let module = Module::new(&ctx, op).expect("fixture is a module");
    let object = emit_native_from_boundary_exit(&mut ctx, module, true)
        .unwrap_or_else(|error| panic!("native emission: {error}"));
    let output = NativeTestBinary::from_object_bytes(&object).run_with_stdin(b"");
    assert!(
        output.status.success(),
        "ASan run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The fixture, which needs nothing the boundary adds, so it also stands
/// for the representation/ABI boundary exit, where the tests split a
/// compilation.
fn fixture() -> (IrContext, Module) {
    let mut ctx = IrContext::new();
    let op = trunk_ir::parser::parse_module(&mut ctx, PROGRAM)
        .unwrap_or_else(|error| panic!("fixture parses at {}: {}", error.offset, error.message));
    let module = Module::new(&ctx, op).expect("fixture is a module");
    (ctx, module)
}

/// The target pipeline up to the boundary exit inlines the helpers into
/// `main`, and keeps each multi-value `scf.if` and `scf.loop` with all of
/// its results.
#[test]
fn multi_value_control_reaches_the_boundary_exit() {
    for target in [TargetKind::Native, TargetKind::Wasm] {
        let (mut ctx, module) = fixture();
        run_target_to_boundary_exit(&mut ctx, module, target)
            .unwrap_or_else(|error| panic!("{target:?} boundary: {error}"));
        let ir = module.op().display(&ctx).to_string();
        assert_eq!(
            ir.matches("scf.if %0 -> (core.i32, core.i32)").count(),
            1,
            "{target:?}:\n{ir}"
        );
        assert_eq!(
            ir.lines()
                .filter(
                    |line| line.contains("scf.loop(") && line.contains("-> (core.i32, core.i32)")
                )
                .count(),
            2,
            "{target:?}: both loops keep two results:\n{ir}"
        );
    }
}

#[test]
fn multi_result_program_runs_natively() {
    let (mut ctx, module) = fixture();
    let object = emit_from_boundary_exit(&mut ctx, module, TargetKind::Native)
        .unwrap_or_else(|error| panic!("native emission: {error}"));
    let output = NativeTestBinary::from_object_bytes(&object).run_with_stdin(b"");
    assert!(
        output.status.success(),
        "native run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn multi_result_program_validates_and_runs_in_wasmtime() {
    let (mut ctx, module) = fixture();
    let binary = emit_from_boundary_exit(&mut ctx, module, TargetKind::Wasm)
        .unwrap_or_else(|error| panic!("Wasm emission: {error}"));
    let output = run_wasm(&binary);
    assert!(
        output.status.success(),
        "wasmtime failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
