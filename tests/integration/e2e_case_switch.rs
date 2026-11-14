//! End-to-end coverage for `case` expressions that select by integer
//! literal or by variant and lower to `scf.index_switch`, on both targets.

use crate::common;

use common::assert_output_on_both_targets;
use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;

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

fn answer_ask(comp: fn() ->{e, Ask} Nat) ->{e} Nat {
    handle comp() {
        do value { value }
        op Ask::ask() { answer_ask(fn() { resume 40 }) }
    }
}

fn answer(n: Nat) -> Nat {
    answer_ask(fn() asked(n))
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

/// Literal patterns accept every value of the current 32-bit
/// representations, as literal expressions do; a `Nat` of 2^31 or more is
/// matched by its 32-bit pattern.
#[test]
fn literal_patterns_span_the_32_bit_representation_on_both_targets() {
    assert_output_on_both_targets(
        "literal_pattern_range.trb",
        r#"
use std::io::{Io, print_line}

fn nat(x: Nat) -> String {
    case x {
        1073741824 -> "2^30"
        3717208003 -> "big"
        4294967295 -> "max"
        _ -> "other"
    }
}

fn int(x: Int) -> String {
    case x {
        -2147483648 -> "min"
        2147483647i -> "max"
        _ -> "other"
    }
}

fn single(x: Nat) -> String {
    case x {
        4294967295 -> "max"
        _ -> "other"
    }
}

fn main() ->{Io} Nil {
    print_line(nat(1073741824))
    print_line(nat(3717208003))
    print_line(nat(4294967295))
    print_line(nat(7))
    print_line(int(-2147483648))
    print_line(int(2147483647i))
    print_line(int(+0))
    print_line(single(4294967295))
    print_line(single(1))
}
"#,
        "2^30\nbig\nmax\nother\nmin\nmax\nother\nmax\nother",
    );
}

/// A variant case switches on the variant's position, whichever variants
/// the program allocates: `Shape::Unused` is never built, and `Shape::Dot`
/// is built only after a variant declared after it.
#[test]
fn variant_case_switch_runs_on_both_targets() {
    assert_output_on_both_targets(
        "variant_case_switch.trb",
        r#"
use std::io::{Io, print_line}

ability Ask {
    op ask() -> Nat
}

enum Shape {
    Unused,
    Circle(Nat),
    Rect { width: Nat, height: Nat },
    Dot,
    Line(Nat, Nat),
}

fn name(shape: Shape) -> String {
    case shape {
        .Circle(r) -> "circle " <> r.to_string
        .Rect { width, height } -> "rect " <> { width * height }.to_string
        .Dot -> "dot"
        .Unused -> "unused"
        .Line(a, b) -> "line " <> { a + b }.to_string
    }
}

fn kind(shape: Shape) -> String {
    case shape {
        .Dot -> "small"
        .Line(_, _) as line -> "long " <> name(line)
        _ -> "other"
    }
}

fn count(shape: Shape, acc: Nat) -> Nat {
    case shape {
        .Line(a, _) -> become count(Shape::Circle(a), acc + 1)
        .Circle(r) -> become count(Shape::Dot, acc + r)
        .Dot -> acc
        _ -> acc + 100
    }
}

fn asked(shape: Shape) ->{Ask} Nat {
    case shape {
        .Circle(r) -> Ask::ask() + r
        .Dot -> Ask::ask()
        _ -> 0
    }
}

fn answer_ask(comp: fn() ->{e, Ask} Nat) ->{e} Nat {
    handle comp() {
        do value { value }
        op Ask::ask() { answer_ask(fn() { resume 40 }) }
    }
}

fn answer(shape: Shape) -> Nat {
    answer_ask(fn() asked(shape))
}

fn main() ->{Io} Nil {
    let line = Shape::Line(2, 3)
    let dot = Shape::Dot
    print_line(name(line))
    print_line(name(Shape::Circle(4)))
    print_line(name(Shape::Rect { width: 2, height: 5 }))
    print_line(name(dot))
    print_line(kind(dot))
    print_line(kind(line))
    print_line(kind(Shape::Circle(1)))
    print_line(count(line, 0).to_string)
    print_line(count(Shape::Rect { width: 1, height: 1 }, 0).to_string)
    print_line(answer(Shape::Circle(2)).to_string)
    print_line(answer(dot).to_string)
    print_line(answer(line).to_string)
}
"#,
        "line 5\ncircle 4\nrect 10\ndot\nsmall\nlong line 5\nother\n3\n100\n42\n40\n0",
    );
}

/// Count the `br_table` instructions in every function body of a module.
fn br_table_count(binary: &[u8]) -> usize {
    let mut count = 0;
    for payload in wasmparser::Parser::new(0).parse_all(binary) {
        let wasmparser::Payload::CodeSectionEntry(body) = payload.expect("valid Wasm payload")
        else {
            continue;
        };
        let mut reader = body.get_operators_reader().expect("function operators");
        while !reader.eof() {
            if let wasmparser::Operator::BrTable { .. } = reader.read().expect("operator") {
                count += 1;
            }
        }
    }
    count
}

/// A dense switch dispatches through one Wasm `br_table`: a variant case
/// over four variants, and a literal case whose values have a hole and
/// whose scrutinee also falls below and above them.
#[salsa_test]
fn dense_case_switches_dispatch_through_br_table_on_wasm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "dense_case_switch.trb",
        r#"
use std::io::{Io, print_line}

enum Color {
    Red,
    Green,
    Blue,
    Gray(Nat),
}

fn color(n: Nat) -> Color {
    case n {
        0 -> Color::Red
        1 -> Color::Green
        2 -> Color::Blue
        level -> Color::Gray(level)
    }
}

fn name(color: Color) -> String {
    case color {
        .Red -> "red"
        .Green -> "green"
        .Blue -> "blue"
        .Gray(level) -> "gray " <> level.to_string
    }
}

fn digit(n: Nat) -> String {
    case n {
        1 -> "one"
        2 -> "two"
        4 -> "four"
        _ -> "other"
    }
}

fn main() ->{Io} Nil {
    let bs = b"\x00\x01\x02\x03\x04\x05"
    print_line(name(color(bs.get_or_panic(0))))
    print_line(name(color(bs.get_or_panic(1))))
    print_line(name(color(bs.get_or_panic(2))))
    print_line(name(color(bs.get_or_panic(5))))
    print_line(digit(bs.get_or_panic(0)))
    print_line(digit(bs.get_or_panic(1)))
    print_line(digit(bs.get_or_panic(2)))
    print_line(digit(bs.get_or_panic(3)))
    print_line(digit(bs.get_or_panic(4)))
    print_line(digit(bs.get_or_panic(5)))
}
"#,
    );
    // The same library functions without the program's own switches, to
    // count the `br_table`s the library brings.
    let baseline = SourceCst::from_source_str(
        db,
        "dense_case_switch_baseline.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    let bs = b"\x00\x01\x02\x03\x04\x05"
    print_line("gray " <> bs.get_or_panic(5).to_string)
}
"#,
    );
    let library_tables =
        br_table_count(compile_to_wasm_binary(db, baseline).expect("baseline compiles to Wasm"));
    let binary = compile_to_wasm_binary(db, source).expect("dense switches compile to Wasm");
    // Inlining may copy a switch into each caller, so only require that
    // the program's own switches add tables.
    let tables = br_table_count(binary);
    assert!(
        tables >= library_tables + 3,
        "{tables} tables, {library_tables} from the library"
    );
    common::assert_wasm_output(
        binary,
        "red\ngreen\nblue\ngray 5\nother\none\ntwo\nother\nfour\nother\n",
    );
}
