//! Native execution of `clif.br_table`.

use std::fmt::Write;

use trunk_ir::context::IrContext;
use trunk_ir::parser::parse_test_module;
use trunk_ir_cranelift_backend::emit_module_to_native;

use crate::common::NativeTestBinary;

/// A module whose `select` branches through a table of `targets` entries,
/// returning the 1-based position of the entry taken, or 0 for the default.
/// Its `main` returns 0 when `select` maps every input to its expected result, and otherwise the 1-based position of the first input
/// that it does not.
fn selection_module(targets: usize, expected: &[(i32, i32)]) -> String {
    let mut text = String::from("core.module @test {\n");
    writeln!(
        text,
        r#"  clif.func {{sym_name = "select", type = clif.func_sig<(core.i32) -> core.i32>}} {{
    ^entry(%index: core.i32):
      clif.br_table %index [^default{}]
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
      %input{position} = clif.iconst {{value = {input}}} : core.i32
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

fn assert_selects(targets: usize, expected: &[(i32, i32)]) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, &selection_module(targets, expected));
    let object = emit_module_to_native(&ctx, module).expect("module emits");
    let output = NativeTestBinary::from_object_bytes(&object).run_with_stdin(&[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "exit code is the 1-based position of the first input selected wrongly"
    );
}

#[test]
fn an_index_in_the_table_selects_its_entry() {
    assert_selects(4, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
}

#[test]
fn an_index_out_of_bounds_selects_the_default() {
    assert_selects(4, &[(4, 0), (5, 0), (i32::MAX, 0), (-1, 0), (i32::MIN, 0)]);
}

#[test]
fn an_empty_table_always_selects_the_default() {
    assert_selects(0, &[(0, 0), (5, 0)]);
}
