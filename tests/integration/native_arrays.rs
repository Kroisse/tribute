//! Native execution of the `adt` array operations under the address
//! sanitizer.
//!
//! No source feature creates arrays yet, so these programs enter the native
//! pipeline at the representation/ABI boundary exit as textual IR.

use std::process::Output;

use tribute::pipeline::emit_native_from_boundary_exit;
use trunk_ir::context::IrContext;
use trunk_ir::parser::parse_test_module;

use crate::common::NativeTestBinary;

/// A module whose `main` runs `body` and traps unless `%sum` equals
/// `expected`.
fn program(body: &str, expected: i32) -> String {
    format!(
        r#"core.module @arrays {{
  !Node = adt.struct<Node(value: core.i32)>
  !NodeRef = adt.typeref<{{name = "Node"}}>
  !Nodes = core.array<!NodeRef>
  !Ints = core.array<core.i32>
  func.func @main() -> core.nil {{
    ^entry:
      %i0 = arith.const {{value = 0}} : core.index
      %i1 = arith.const {{value = 1}} : core.index
      %i2 = arith.const {{value = 2}} : core.index
      %i3 = arith.const {{value = 3}} : core.index
      %one = arith.const {{value = 1}} : core.i32
      %two = arith.const {{value = 2}} : core.i32
{body}
      %expected = arith.const {{value = {expected}}} : core.i32
      %ok = arith.cmpi %sum, %expected {{predicate = "eq"}} : core.i1
      cf.cond_br %ok [^done, ^wrong]
    ^done:
      %nil = core.nil_value : core.nil
      func.return %nil
    ^wrong:
      func.unreachable
  }}
}}"#
    )
}

fn run(body: &str, expected: i32) -> Output {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, &program(body, expected));
    let object = emit_native_from_boundary_exit(&mut ctx, module, true).expect("module emits");
    NativeTestBinary::from_object_bytes(&object).run_with_stdin(&[])
}

/// Builds, copies, and replaces managed and scalar elements, and sums what it
/// reads back.
const ARRAY_OPERATIONS: &str = r#"
      %a = adt.struct_new %one {type = !Node} : !NodeRef
      %b = adt.struct_new %two {type = !Node} : !NodeRef
      %pair = adt.array_of %a, %b {type = !Nodes} : !Nodes
      %filled = adt.array_new %i3, %a {type = !Nodes} : !Nodes
      %inserted = adt.array_splice %pair, %i1, %i1, %b {type = !Nodes} : !Nodes
      %trimmed = adt.array_splice %inserted, %i0, %i1 {type = !Nodes} : !Nodes
      %replaced = adt.array_splice %filled, %i0, %i2, %b, %a, %b {type = !Nodes} : !Nodes
      adt.array_set %filled, %i1, %b
      %t0 = adt.array_get %trimmed, %i0 : !NodeRef
      %v0 = adt.struct_get %t0 {field = 0, type = !Node} : core.i32
      %f1 = adt.array_get %filled, %i1 : !NodeRef
      %v1 = adt.struct_get %f1 {field = 0, type = !Node} : core.i32
      %f2 = adt.array_get %filled, %i2 : !NodeRef
      %v2 = adt.struct_get %f2 {field = 0, type = !Node} : core.i32
      %r3 = adt.array_get %replaced, %i3 : !NodeRef
      %v3 = adt.struct_get %r3 {field = 0, type = !Node} : core.i32
      %ints = adt.array_of %one, %two {type = !Ints} : !Ints
      %more = adt.array_new %i2, %two {type = !Ints} : !Ints
      %joined = adt.array_splice %ints, %i2, %i2, %one {type = !Ints} : !Ints
      adt.array_set %more, %i0, %one
      %w0 = adt.array_get %joined, %i2 : core.i32
      %w1 = adt.array_get %more, %i0 : core.i32
      %n0 = adt.array_len %replaced : core.index
      %n1 = arith.index_castui %n0 : core.i32
      %s0 = arith.addi %v0, %v1 : core.i32
      %s1 = arith.addi %s0, %v2 : core.i32
      %s2 = arith.addi %s1, %v3 : core.i32
      %s3 = arith.addi %s2, %w0 : core.i32
      %s4 = arith.addi %s3, %w1 : core.i32
      %sum = arith.addi %s4, %n1 : core.i32"#;

#[test]
fn array_operations_release_every_element_they_retain() {
    // trimmed[0] = b, filled[1] = b after the set, filled[2] = a,
    // replaced = [b, a, b, a] so replaced[3] = a, joined[2] = 1,
    // more[0] = 1 after the set, and replaced has 4 elements.
    let output = run(ARRAY_OPERATIONS, 2 + 2 + 1 + 1 + 1 + 1 + 4);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "stderr: {stderr}");
    assert!(
        !stderr.contains("detected memory leaks"),
        "stderr: {stderr}"
    );
}

fn assert_traps_without_access(body: &str) {
    let output = run(body, 0);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "stderr: {stderr}");
    assert!(output.status.code().is_none(), "stderr: {stderr}");
    assert!(!stderr.contains("ERROR"), "stderr: {stderr}");
}

#[test]
fn an_index_at_the_length_traps_before_the_access() {
    assert_traps_without_access(
        r#"
      %ints = adt.array_of %one, %two {type = !Ints} : !Ints
      %sum = adt.array_get %ints, %i2 : core.i32"#,
    );
    assert_traps_without_access(
        r#"
      %ints = adt.array_new %i2, %one {type = !Ints} : !Ints
      adt.array_set %ints, %i3, %two
      %sum = arith.const {value = 0} : core.i32"#,
    );
}

#[test]
fn a_splice_range_outside_the_source_traps() {
    assert_traps_without_access(
        r#"
      %ints = adt.array_of %one, %two {type = !Ints} : !Ints
      %spliced = adt.array_splice %ints, %i1, %i3 {type = !Ints} : !Ints
      %sum = arith.const {value = 0} : core.i32"#,
    );
    assert_traps_without_access(
        r#"
      %ints = adt.array_of %one, %two {type = !Ints} : !Ints
      %spliced = adt.array_splice %ints, %i2, %i1 {type = !Ints} : !Ints
      %sum = arith.const {value = 0} : core.i32"#,
    );
}
