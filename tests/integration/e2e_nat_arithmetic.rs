//! End-to-end coverage for `Nat` arithmetic and `Bool` cases on both targets.

use crate::common;

use common::assert_output_on_both_targets;

/// The operands are read from a `Bytes` value at run time, so
/// canonicalization cannot fold `+` and `*` into constants.
#[test]
fn nat_addition_and_multiplication_run_on_both_targets() {
    assert_output_on_both_targets(
        "nat_arithmetic.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    let bs = b"\x28\x02\x06\x07\x00\x09\xff\x03\x04\x05"
    let a = bs.get_or_panic(0) + bs.get_or_panic(1)
    let b = { bs.get_or_panic(2) * bs.get_or_panic(3) } + { bs.get_or_panic(4) * bs.get_or_panic(5) }
    let x = bs.get_or_panic(6) * 257
    let c = x * { x + 2 }
    print_line(a.to_string)
    print_line(b.to_string)
    print_line(c.to_string)
    print_line({ { bs.get_or_panic(7) + bs.get_or_panic(8) } * { bs.get_or_panic(9) + 6 } }.to_string)
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
