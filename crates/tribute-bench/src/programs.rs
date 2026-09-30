//! Representative programs for the representation/ABI boundary benchmarks.
//!
//! The set covers Direct and CPS callables, closures with captures,
//! tail-resumptive and general handlers, the CPS root bridge, standard I/O,
//! and runtime workloads: pure recursion, a loop through a general `State`
//! handler (quadratic in its length), and a loop through a tail-resumptive
//! handler. Loop lengths stay within the default native stack.

/// One benchmark program.
pub struct Program {
    pub name: &'static str,
    pub source: &'static str,
    /// Bytes supplied to the program's standard input when it runs.
    pub stdin: &'static [u8],
}

pub const PROGRAMS: &[Program] = &[
    Program {
        name: "native_calculator",
        source: include_str!("../../../lang-examples/native_calculator.trb"),
        stdin: include_bytes!("../../../tests/fixtures/native_calculator_scripted.stdin"),
    },
    Program {
        name: "native_effects",
        source: include_str!("../../../lang-examples/native_effects.trb"),
        stdin: b"",
    },
    Program {
        name: "wasm_dynamic_output",
        source: include_str!("../../../lang-examples/wasm_dynamic_output.trb"),
        stdin: b"",
    },
    Program {
        name: "closure_capture",
        source: r#"fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() -> Nil {
    let a = +1
    let _ = apply(fn(n) { n + a }, +41)
}
"#,
        stdin: b"",
    },
    Program {
        name: "tail_resumptive_handler",
        source: r#"ability Ask {
    fn ask() -> Nat
}

fn use_ask() ->{Ask} Nat {
    Ask::ask()
}

fn main() -> Nil {
    let _ = handle use_ask() {
        do result { result }
        fn Ask::ask() { 42 }
    }
}
"#,
        stdin: b"",
    },
    Program {
        name: "state_handler",
        source: r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn set_then_get() ->{State(Int)} Int {
    State::set(+100)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Nil {
    let _ = run_state(fn() { set_then_get() }, +0)
}
"#,
        stdin: b"",
    },
    Program {
        name: "fibonacci",
        source: r#"use std::io::{Io, print_line}

fn fibonacci(n: Int) -> Int {
    case n < +2 {
        True -> n
        False -> fibonacci(n - +1) + fibonacci(n - +2)
    }
}

fn main() ->{Io} Nil {
    print_line(Int::to_string(fibonacci(+27)))
}
"#,
        stdin: b"",
    },
    Program {
        name: "state_loop",
        source: r#"use std::io::{Io, print_line}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn tick(n: Int) ->{State(Int)} Int {
    case n == +0 {
        True -> State::get()
        False -> {
            State::set(State::get() + +1)
            tick(n - +1)
        }
    }
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() ->{Io} Nil {
    print_line(Int::to_string(run_state(fn() { tick(+500) }, +0)))
}
"#,
        stdin: b"",
    },
    Program {
        name: "counter_loop",
        source: r#"use std::io::{Io, print_line}

ability Counter {
    fn next() -> Int
}

fn sum(n: Int, acc: Int) ->{Counter} Int {
    case n == +0 {
        True -> acc
        False -> sum(n - +1, acc + Counter::next())
    }
}

fn main() ->{Io} Nil {
    let total = handle sum(+50000, +0) {
        do result { result }
        fn Counter::next() { +1 }
    }
    print_line(Int::to_string(total))
}
"#,
        stdin: b"",
    },
];
