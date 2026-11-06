//! End-to-end coverage for `case` expressions that select by integer
//! literal and lower to `scf.index_switch`, on both targets.

use crate::common;

use common::assert_output_on_both_targets;

/// The `Nat` scrutinees are read from a `Bytes` value at run time, so
/// canonicalization cannot fold a switch into one of its arms.
#[test]
fn literal_case_switch_runs_on_both_targets() {
    assert_output_on_both_targets(
        "literal_case_switch.trb",
        r#"
use std::io::{Io, print_line}

ability Ask {
    op ask() -> Nat
}

fn name(n: Nat) -> String {
    case n {
        0 -> "zero"
        1 -> "one"
        2 as two -> "two is " <> two.to_string
        other -> "many " <> other.to_string
    }
}

fn sign(x: Int) -> String {
    case x {
        -1 -> "minus one"
        0i -> "zero"
        _ -> "other"
    }
}

fn steps(n: Nat, acc: Nat) -> Nat {
    case n {
        0 -> acc
        1 -> become steps(0, acc + 1)
        _ -> become steps(n - 1, acc + 1)
    }
}

fn asked(n: Nat) ->{Ask} Nat {
    case n {
        0 -> Ask::ask()
        1 -> Ask::ask() + 1
        _ -> n
    }
}

fn answer(n: Nat) -> Nat {
    handle asked(n) {
        do value { value }
        op Ask::ask() { resume 40 }
    }
}

fn main() ->{Io} Nil {
    let bs = b"\x00\x01\x02\x07\x05\x09"
    print_line(name(bs.get_or_panic(0)))
    print_line(name(bs.get_or_panic(1)))
    print_line(name(bs.get_or_panic(2)))
    print_line(name(bs.get_or_panic(3)))
    print_line(sign(-1))
    print_line(sign(+0))
    print_line(sign(+5))
    print_line(steps(bs.get_or_panic(4), 0).to_string)
    print_line(answer(bs.get_or_panic(0)).to_string)
    print_line(answer(bs.get_or_panic(1)).to_string)
    print_line(answer(bs.get_or_panic(5)).to_string)
}
"#,
        "zero\none\ntwo is 2\nmany 7\nminus one\nzero\nother\n5\n40\n41\n9",
    );
}
