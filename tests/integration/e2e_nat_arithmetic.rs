//! End-to-end coverage for `Nat` arithmetic and `Bool` cases on both targets.

use crate::common;

use common::assert_output_on_both_targets;

#[test]
fn nat_addition_and_multiplication_run_on_both_targets() {
    assert_output_on_both_targets(
        "nat_arithmetic.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    let a = 40 + 2
    let b = { 6 * 7 } + { 0 * 9 }
    let c = { 65535 * 65537 } + 0
    print_line(a.to_string)
    print_line(b.to_string)
    print_line(c.to_string)
    print_line({ { 3 + 4 } * { 5 + 6 } }.to_string)
}
"#,
        "42\n42\n4294967295\n77",
    );
}

#[test]
fn bool_case_runs_on_both_targets() {
    assert_output_on_both_targets(
        "bool_case.trb",
        r#"
use std::io::{Io, print_line}

fn describe(flag: Bool) -> String {
    case flag {
        True -> "yes"
        False -> "no"
    }
}

fn main() ->{Io} Nil {
    print_line(describe(True))
    print_line(describe(False))
    print_line(describe(3 < 4))
    print_line(describe({ 2 * 3 } == 7))
}
"#,
        "yes\nno\nyes\nno",
    );
}
