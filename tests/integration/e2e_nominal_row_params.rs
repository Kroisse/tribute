//! Structs and enums with row parameters (`design/types.md`, "Row 매개변수"),
//! whose fields run in the row their type's row argument names, on both
//! targets.

use crate::common;

use common::assert_output_on_both_targets;

/// A pure and a `Tick`-performing value of the same declarations, called
/// through their fields and pattern bindings under a handler.
#[test]
fn row_parameterized_fields_run_on_both_targets() {
    assert_output_on_both_targets(
        "nominal_row_params.trb",
        r#"
use std::io::{Io, print_line}

ability Tick {
    op tick() -> Nil
}

struct Box({e}, a) { run: fn(a) ->{e} a }

enum Step({e}) { Next(fn() ->{e} Nat), Done }

fn apply(b: Box({e}, Nat), x: Nat) ->{e} Nat {
    let f = b.run
    f(x)
}

fn step(s: Step({e})) ->{e} Nat {
    case s {
        .Next(f) -> f()
        .Done -> 0
    }
}

fn count(comp: fn() ->{e, Tick} Nat) ->{e} Nat {
    handle comp() {
        do result { result }
        op Tick::tick() { 1 + count(fn() { resume Nil }) }
    }
}

fn main() ->{Io} Nil {
    let pure = Box { run: fn(x: Nat) { x + 1 } }
    print_line(apply(pure, 1).to_string)
    let ticks = count(fn() {
        let b = Box {
            run: fn(x: Nat) {
                Tick::tick()
                x
            }
        }
        let s = Step::Next(fn() {
            Tick::tick()
            10
        })
        apply(b, 5) + step(s)
    })
    print_line(ticks.to_string)
}
"#,
        "2\n17",
    );
}

/// The generated setter and modifier keep the receiver's row argument.
#[test]
fn row_parameterized_field_lenses_run_on_both_targets() {
    assert_output_on_both_targets(
        "nominal_row_param_lenses.trb",
        r#"
use std::io::{Io, print_line}

ability Tick {
    op tick() -> Nil
}

struct Box({e}, a) { run: fn(a) ->{e} a, label: Nat }

fn apply(b: Box({e}, Nat), x: Nat) ->{e} Nat {
    let f = b.run
    f(x + b.label)
}

fn count(comp: fn() ->{e, Tick} Nat) ->{e} Nat {
    handle comp() {
        do result { result }
        op Tick::tick() { 1 + count(fn() { resume Nil }) }
    }
}

fn main() ->{Io} Nil {
    let pure = Box { run: fn(x: Nat) { x + 1 }, label: 0 }
    print_line(apply(pure.label::set(10), 1).to_string)
    print_line(apply(pure.label::modify(fn(n) n + 100), 1).to_string)
    let ticks = count(fn() {
        let b = Box {
            run: fn(x: Nat) {
                Tick::tick()
                x
            },
            label: 0,
        }
        apply(b.label::set(3), 4)
    })
    print_line(ticks.to_string)
}
"#,
        "12\n102\n8",
    );
}
