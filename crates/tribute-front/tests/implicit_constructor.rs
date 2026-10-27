//! Tests for implicit constructors `.Name`, whose constructor the type of
//! their context gives (`new-plans/type-inference.md` § 암묵적 생성자).

mod common;

use self::common::ast_pipeline_diagnostics;
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::DiagnosticSeverity;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    let source = SourceCst::from_source_str(db, "implicit_constructor.trb", text);
    ast_pipeline_diagnostics(db, source)
        .into_iter()
        .filter(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error)
        .map(|diagnostic| diagnostic.inner.message)
        .collect()
}

const TYPES: &str = "enum Tree { Leaf(Int), Node(Tree, Tree) }\nenum Heap { Leaf(Int), Empty }\n";

#[salsa_test]
fn the_expected_type_names_the_constructor(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn tree() -> Tree { .Node(.Leaf(+1), .Leaf(+2)) }

fn value(t: Tree) -> Int {
    case t {
        .Leaf(n) -> n
        .Node(_, _) -> +0
    }
}

fn wrapped(h: Heap) -> Option(Heap) { .Some(h) }
"#
        ),
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn a_constructor_out_of_scope_is_found_through_its_type(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod shapes {
    pub enum Shape { Dot, Line(Int) }
    pub fn dot() -> Shape { .Dot }
}

fn count(s: shapes::Shape) -> Int {
    case s {
        .Dot -> +0
        .Line(n) -> n
    }
}

fn argument() -> Int { count(.Dot) }

fn later() -> Int {
    let line = .Line(+1)
    count(line)
}
"#,
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn ambiguous_constructors_wait_until_the_body_is_solved(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn take(t: Tree) -> Nil { Nil }

fn later() -> Nil {
    let t = .Leaf(+1)
    take(t)
}

fn never() -> Nil {
    let t = .Leaf(+1)
    Nil
}
"#
        ),
    );
    assert_eq!(
        errors,
        [
            "ambiguous constructor `.Leaf`: it may be `pkg::Heap::Leaf`, `pkg::Tree::Leaf`; write its type"
        ]
    );
}

#[salsa_test]
fn a_type_without_the_constructor_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn missing() -> Tree { .Empty }
fn primitive() -> Int { .Leaf(+1) }
fn nothing() -> Nil {
    let x = .Nope
    Nil
}
"#
        ),
    );
    assert_eq!(
        errors,
        [
            "type `Tree` has no constructor `Empty`",
            "`.Leaf` needs a type with constructors, found `Int`",
            "no type in scope has a constructor `Nope`",
        ]
    );
}

#[salsa_test]
fn field_counts_are_checked_against_the_selected_constructor(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn eager(t: Tree) -> Int {
    case t {
        .Leaf(a, b) -> a
        .Node(_, _) -> +0
    }
}

fn deferred() -> Nil {
    let f = fn(x) {
        case x {
            .Empty(y) -> +1
            .Leaf(n) -> n
        }
    }
    Nil
}
"#
        ),
    );
    assert_eq!(
        errors,
        [
            "constructor `pkg::Tree::Leaf` expects 1 field, but the pattern has 2",
            "constructor `pkg::Heap::Empty` expects 0 fields, but the pattern has 1",
        ]
    );
}

#[salsa_test]
fn exhaustiveness_sees_the_selected_constructors(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn incomplete(t: Tree) -> Int {
    case t {
        .Leaf(n) -> n
    }
}

fn deferred() -> Nil {
    let f = fn(x) {
        case x {
            .Empty -> +1
        }
    }
    Nil
}
"#
        ),
    );
    assert_eq!(
        errors,
        [
            "non-exhaustive case expression on `Tree`: missing patterns: .Node(_, _)",
            "non-exhaustive case expression on `pkg::Heap`: missing patterns: .Leaf(_)",
        ]
    );
}

#[salsa_test]
fn a_bare_variant_name_suggests_the_implicit_constructor(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn none() -> Option(Int) { None }
"#,
    );
    assert_eq!(errors, ["unresolved name `None`; did you mean `.None`?"]);
}

#[salsa_test]
fn nested_constructors_see_the_field_types_of_the_outer_one(db: &salsa::DatabaseImpl) {
    // `heap::Heap` is not in scope, so only the outer `.Some`'s instance
    // tells the inner `.Leaf` from the scope's `pkg::Tree::Leaf`.
    let errors = errors(
        db,
        r#"
mod heap {
    pub enum Heap(a) { Leaf(a), Empty }
}
enum Tree(a) { Leaf(a), Node(Tree(a), Tree(a)) }

fn build() -> Option(heap::Heap(Int)) { .Some(.Leaf(+1)) }

fn read(h: Option(heap::Heap(Int))) -> Int {
    case h {
        .Some(.Leaf(v)) -> v
        .Some(.Empty) -> +0
        .None -> +0
    }
}
"#,
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn deferred_records_check_their_fields(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Point { x: Int, y: Int }
mod other {
    pub enum Shape { Point { x: Int }, Blank }
}
use other::Shape

fn take(p: Point) -> Int { p.x }
fn shape(s: Shape) -> Int { +0 }

fn missing() -> Int {
    let p = .Point { x: +1 }
    take(p)
}

fn unknown() -> Int {
    let p = .Point { x: +1, z: +2 }
    shape(p)
}

fn spread(base: Shape) -> Int {
    let p = .Point { ..base, x: +1 }
    shape(p)
}
"#,
    );
    assert_eq!(
        errors,
        [
            "missing field: y",
            "unknown field `z` for variant `pkg::other::Shape::Point`",
            "record spread is not allowed for variant `pkg::other::Shape::Point`; write every field",
        ]
    );
}

#[salsa_test]
fn a_constructor_value_takes_the_result_of_its_function_type(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn apply(f: fn(Int) ->{} Tree, n: Int) -> Tree { f(n) }

fn eager() -> Tree { apply(.Leaf, +1) }

fn deferred() -> Tree {
    let f = .Leaf
    f(+1)
}
"#
        ),
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn deferred_constructors_are_selected_by_their_fields(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub enum Shape { Point { x: Int, y: Int }, Blank }
}
mod b {
    pub enum Mark { Point { x: Int }, Plain }
}
use a::Shape
use b::Mark

fn by_fields() -> Nil {
    let p = .Point { x: +1, y: +2 }
    Nil
}

fn by_pattern() -> Nil {
    let f = fn(v) {
        case v {
            .Point { x, y } -> x + y
            .Blank -> +0
        }
    }
    Nil
}
"#,
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn deferred_constructors_report_what_does_not_fit(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
fn take(t: Tree) -> Nil { Nil }

fn arity() -> Nil {
    let t = .Leaf(+1, +2)
    take(t)
}

fn braces() -> Nil {
    let t = .Leaf { value: +1 }
    take(t)
}

fn nothing_fits() -> Nil {
    let t = .Leaf("x")
    Nil
}
"#
        ),
    );
    assert_eq!(
        errors,
        [
            "constructor `pkg::Tree::Leaf` expects 1 field, but it is given 2",
            "`pkg::Tree::Leaf` has positional fields; construct it with `.Leaf(...)`",
            "no constructor `.Leaf` in scope fits here: `pkg::Heap::Leaf`, `pkg::Tree::Leaf`, `std::String::Leaf`",
        ]
    );
}

#[salsa_test]
fn a_deferred_constructor_waits_across_a_handler(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        &format!(
            "{TYPES}{}",
            r#"
ability Ask {
    op ask() -> Int
}

fn take(t: Tree) -> Int { +0 }

fn handled() -> Int {
    let t = .Leaf(+1)
    let n = handle Ask::ask() {
        do r { r }
        op Ask::ask() { resume +1 }
    }
    take(t) + n
}
"#
        ),
    );
    assert!(errors.is_empty(), "{errors:?}");
}
