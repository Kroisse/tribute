//! An inline module sees its own declarations, its imports, and the prelude;
//! the items around it come in through `use super::…` or `pkg::…`. A module
//! also sees its own name, so a companion module uses the type beside it.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "scope.trb", text))
}

#[salsa_test]
fn paths_start_from_the_enclosing_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    pub mod inner {
        pub fn value() -> Int { +42 }
    }

    pub fn use_inner() -> Int {
        inner::value()
    }
}

fn main() -> Nil {
    let _ = outer::use_inner()
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn items_around_a_module_are_not_in_scope(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn base() -> Nat { 1 }

struct Point { x: Nat }

mod other {
    pub fn f() -> Nat { 2 }
}

mod m {
    pub fn value() -> Nat { base() + other::f() }

    pub fn x(p: Point) -> Nat { p.x }
}

fn main() -> Nil { }
"#,
    );
    for name in ["base", "other::f", "Point"] {
        assert!(
            errors
                .iter()
                .any(|message| message.starts_with(&format!("unresolved name `{name}`"))),
            "{name}: {errors:?}"
        );
    }
}

#[salsa_test]
fn items_around_a_module_are_imported_or_named_by_keyword(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn base() -> Nat { 1 }

struct Point { x: Nat }

mod other {
    pub fn f() -> Nat { 2 }
}

mod m {
    use super::base
    use super::Point

    pub fn value() -> Nat { base() + super::other::f() + pkg::other::f() }

    pub fn x(p: Point) -> Nat { p.x }
}

fn main() -> Nil {
    let _ = m::value() + m::x(Point { x: 3 })
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// An ability of the package root is named the same way in an effect row and
/// in the operation call, so their instances agree.
#[salsa_test]
fn imported_ability_with_arguments_checks_in_a_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
ability State(s) {
    op get() -> s
}

fn g() ->{State(Int)} Int { State::get() }

mod Nested {
    use super::State

    fn h() ->{State(Int)} Int { State::get() }
}

fn main() -> Nil { }
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn companion_module_sees_its_type(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Point { x: Nat }

pub mod Point {
    pub fn origin() -> Point { Point { x: 0 } }
    pub fn shifted(p: Point) -> Point { Point::moved(p, 1) }
    pub fn moved(p: Point, by: Nat) -> Point { Point { x: p.x + by } }
}

enum Shape {
    Circle(Nat),
    Square(Nat),
}

pub mod Shape {
    pub fn size(shape: Shape) -> Nat {
        case shape {
            Circle(r) -> r
            Shape::Square(s) -> s
        }
    }
}

fn main() -> Nil {
    let _ = Shape::size(Shape::Circle(Point::shifted(Point::origin()).x))
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn prelude_is_in_scope_in_every_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    pub fn get(o: Option(Int)) -> Int {
        case o {
            Some(x) -> x
            None -> +0
        }
    }

    pub fn text(n: Int) -> String { Int::to_string(n) }
}

fn main() -> Nil {
    let _ = m::text(m::get(Some(+1)))
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}
