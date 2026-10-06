//! Native execution of `clif.br_table` and `clif.switch`.

use std::fmt::Write;

use trunk_ir::context::IrContext;
use trunk_ir::parser::parse_test_module;
use trunk_ir_cranelift_backend::emit_module_to_native;

use crate::common::NativeTestBinary;

/// A multi-way branch of `select` and the inputs it is checked with.
struct Selection<'a> {
    /// The type of the index.
    index_ty: &'a str,
    /// The operation name and, for `clif.switch`, its attributes.
    operation: &'a str,
    attributes: &'a str,
    /// The number of successors after the default.
    targets: usize,
    /// Inputs with the 1-based position of the successor each selects, or 0
    /// for the default.
    expected: &'a [(i64, i32)],
}

/// A module whose `main` returns 0 when `select` maps every input to its
/// expected result, and otherwise the 1-based position of the first input
/// that it does not.
fn selection_module(selection: &Selection<'_>) -> String {
    let Selection {
        index_ty,
        operation,
        attributes,
        targets,
        expected,
    } = *selection;
    let mut text = String::from("core.module @test {\n");
    writeln!(
        text,
        r#"  clif.func {{sym_name = "select", type = clif.func_sig<({index_ty}) -> core.i32>}} {{
    ^entry(%index: {index_ty}):
      {operation} %index [^default{}]{attributes}
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
      %input{position} = clif.iconst {{value = {input}}} : {index_ty}
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

fn assert_selects(selection: Selection<'_>) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, &selection_module(&selection));
    let object = emit_module_to_native(&ctx, module).expect("module emits");
    let output = NativeTestBinary::from_object_bytes(&object).run_with_stdin(&[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "exit code is the 1-based position of the first input selected wrongly"
    );
}

fn br_table<'a>(targets: usize, expected: &'a [(i64, i32)]) -> Selection<'a> {
    Selection {
        index_ty: "core.i32",
        operation: "clif.br_table",
        attributes: "",
        targets,
        expected,
    }
}

#[test]
fn br_table_index_in_the_table_selects_its_entry() {
    assert_selects(br_table(4, &[(0, 1), (1, 2), (2, 3), (3, 4)]));
}

#[test]
fn br_table_index_out_of_bounds_selects_the_default() {
    let out_of_bounds = [
        (4, 0),
        (5, 0),
        (i64::from(i32::MAX), 0),
        (-1, 0),
        (i64::from(i32::MIN), 0),
    ];
    assert_selects(br_table(4, &out_of_bounds));
}

#[test]
fn br_table_without_entries_always_selects_the_default() {
    assert_selects(br_table(0, &[(0, 0), (5, 0)]));
}

#[test]
fn switch_dense_cases_select_their_targets() {
    assert_selects(Selection {
        index_ty: "core.i64",
        operation: "clif.switch",
        attributes: " {cases = [0, 1, 2, 3]}",
        targets: 4,
        expected: &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (-1, 0),
            (1 << 40, 0),
        ],
    });
}

#[test]
fn switch_sparse_and_offset_cases_select_their_targets() {
    assert_selects(Selection {
        index_ty: "core.i64",
        operation: "clif.switch",
        attributes: " {cases = [7, 100, 18446744073709551615, 4294967296, 8, 101, 102]}",
        targets: 7,
        expected: &[
            (7, 1),
            (100, 2),
            (-1, 3),
            (1 << 32, 4),
            (8, 5),
            (101, 6),
            (102, 7),
            (0, 0),
            (9, 0),
            (99, 0),
            (103, 0),
            (-2, 0),
        ],
    });
}

#[test]
fn switch_without_cases_always_selects_the_default() {
    assert_selects(Selection {
        index_ty: "core.i32",
        operation: "clif.switch",
        attributes: " {cases = []}",
        targets: 0,
        expected: &[(0, 0), (5, 0)],
    });
}
