//! Native execution of `clif.br_table`.

use std::fmt::Write;

use trunk_ir::context::IrContext;
use trunk_ir::parser::parse_test_module;
use trunk_ir_cranelift_backend::emit_module_to_native;

use crate::common::NativeTestBinary;

/// A module whose `main` returns 0 when `select` maps every input to its
/// expected result, and otherwise the 1-based position of the first input
/// that it does not.
fn selection_module(cases: &str, targets: usize, expected: &[(i64, i32)]) -> String {
    let mut text = String::from("core.module @test {\n");
    writeln!(
        text,
        r#"  clif.func {{sym_name = "select", type = clif.func_sig<(core.i64) -> core.i32>}} {{
    ^entry(%index: core.i64):
      clif.br_table %index [^default{}] {{cases = {cases}}}
    ^default:
      %default = clif.iconst {{value = 0}} : core.i32
      clif.return %default"#,
        (1..=targets).fold(String::new(), |mut labels, target| {
            write!(labels, ", ^target{target}").unwrap();
            labels
        })
    )
    .unwrap();
    for target in 1..=targets {
        writeln!(
            text,
            r#"    ^target{target}:
      %result{target} = clif.iconst {{value = {target}}} : core.i32
      clif.return %result{target}"#
        )
        .unwrap();
    }
    text.push_str("  }\n");
    text.push_str(r#"  clif.func {sym_name = "main", type = clif.func_sig<() -> core.i32>} {"#);
    text.push('\n');
    for (position, (input, result)) in expected.iter().enumerate() {
        let next = position + 1;
        writeln!(
            text,
            r#"    ^check{position}:
      %input{position} = clif.iconst {{value = {input}}} : core.i64
      %actual{position} = clif.call %input{position} {{callee = @select}} : core.i32
      %expected{position} = clif.iconst {{value = {result}}} : core.i32
      %matches{position} = clif.icmp %actual{position}, %expected{position} {{cond = "eq"}} : core.i8
      clif.brif %matches{position} [^check{next}, ^fail{position}]
    ^fail{position}:
      %code{position} = clif.iconst {{value = {next}}} : core.i32
      clif.return %code{position}"#
        )
        .unwrap();
    }
    writeln!(
        text,
        r#"    ^check{}:
      %success = clif.iconst {{value = 0}} : core.i32
      clif.return %success
  }}
}}"#,
        expected.len()
    )
    .unwrap();
    text
}

fn assert_selects(cases: &str, targets: usize, expected: &[(i64, i32)]) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, &selection_module(cases, targets, expected));
    let object = emit_module_to_native(&ctx, module).expect("module emits");
    let output = NativeTestBinary::from_object_bytes(&object).run_with_stdin(&[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "exit code is the 1-based position of the first input selected wrongly"
    );
}

#[test]
fn dense_cases_select_their_targets_and_others_the_default() {
    assert_selects(
        "[0, 1, 2, 3]",
        4,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (-1, 0),
            (1 << 40, 0),
        ],
    );
}

#[test]
fn sparse_and_negative_cases_select_their_targets() {
    assert_selects(
        "[7, 100, -1, 4294967296, 8]",
        5,
        &[
            (7, 1),
            (100, 2),
            (-1, 3),
            (1 << 32, 4),
            (8, 5),
            (0, 0),
            (9, 0),
            (99, 0),
            (-2, 0),
        ],
    );
}

#[test]
fn a_table_without_cases_always_selects_the_default() {
    assert_selects("[]", 0, &[(0, 0), (5, 0)]);
}
