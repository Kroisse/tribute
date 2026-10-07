//! End-to-end tests for proper tail calls with `become`.
//!
//! Each program recurses far deeper than a native or Wasm stack could hold
//! if every call kept a frame, so it only finishes when `become` transfers
//! the caller's frame. They print only through `std::io` and run on both
//! the native and Wasm targets.

use crate::common;

use common::{PRINT_NAT, assert_output_on_both_targets};

/// Deep enough that a frame per call overflows any default stack.
const DEPTH: &str = "10000000";

/// A self tail call in a case arm, between Direct callables.
#[test]
fn test_become_self_recursion_runs_in_constant_stack() {
    let code = format!(
        "{PRINT_NAT}{}",
        r#"
fn count(n: Nat, acc: Nat) -> Nat {
    case n {
        0 -> acc
        _ -> become count(n - 1, acc + 1)
    }
}

fn main() ->{std::io::Io} Nil {
    print_nat(count(DEPTH, 0))
}
"#
        .replace("DEPTH", DEPTH)
    );
    assert_output_on_both_targets("become_self.trb", &code, DEPTH);
}

/// Mutual tail calls through a nested block in a case arm.
#[test]
fn test_become_mutual_recursion_runs_in_constant_stack() {
    let code = format!(
        "{PRINT_NAT}{}",
        r#"
fn is_even(n: Nat) -> Nat {
    case n {
        0 -> 1
        _ -> { become is_odd(n - 1) }
    }
}

fn is_odd(n: Nat) -> Nat {
    case n {
        0 -> 0
        _ -> become is_even(n - 1)
    }
}

fn main() ->{std::io::Io} Nil {
    print_nat(is_even(DEPTH))
}
"#
        .replace("DEPTH", DEPTH)
    );
    assert_output_on_both_targets("become_mutual.trb", &code, "1");
}

/// A tail call of a callable value, and a lambda whose body tail calls a
/// named function.
#[test]
fn test_become_callable_value_runs_in_constant_stack() {
    let code = format!(
        "{PRINT_NAT}{}",
        r#"
fn step(f: fn(Nat, fn(Nat) -> Nat) -> Nat, n: Nat) -> Nat {
    case n {
        0 -> 7
        _ -> become f(n - 1, fn(m: Nat) -> Nat { become step(f, m) })
    }
}

fn apply(n: Nat, k: fn(Nat) -> Nat) -> Nat {
    become k(n)
}

fn main() ->{std::io::Io} Nil {
    print_nat(step(apply, DEPTH))
}
"#
        .replace("DEPTH", DEPTH)
    );
    assert_output_on_both_targets("become_indirect.trb", &code, "7");
}

/// Tail calls inside an effectful callable keep the caller's continuation,
/// and a tail call to a value-returning callee completes through it.
#[test]
fn test_become_under_a_handler_runs_in_constant_stack() {
    let code = format!(
        "{PRINT_NAT}{}",
        r#"
ability Tick {
    op tick() -> Nil
}

fn finish(acc: Nat) -> Nat { acc }

fn loop(n: Nat, acc: Nat) ->{Tick} Nat {
    case n {
        0 -> become finish(acc)
        _ -> {
            Tick::tick()
            become loop(n - 1, acc + 1)
        }
    }
}

fn main() ->{std::io::Io} Nil {
    let total = handle loop(100000, 0) {
        do result { result }
        fn Tick::tick() { Nil }
    }
    print_nat(total)
}
"#
    );
    assert_output_on_both_targets("become_handler.trb", &code, "100000");
}
