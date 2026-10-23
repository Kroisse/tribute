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
        ["ambiguous constructor `.Leaf`: it may be `Heap::Leaf`, `Tree::Leaf`; write its type"]
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
            "constructor `Tree::Leaf` expects 1 field, but the pattern has 2",
            "constructor `Heap::Empty` expects 0 fields, but the pattern has 1",
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
            "non-exhaustive case expression: missing patterns: .Node(_, _)",
            "non-exhaustive case expression: missing patterns: .Leaf(_)",
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
