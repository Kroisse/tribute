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

    pub fn read(p: Point) -> Nat { p.x }
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

    pub fn read(p: Point) -> Nat { p.x }
}

fn main() -> Nil {
    let _ = m::value() + m::read(Point { x: 3 })
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// An ability of the package root is named the same way in an ability row and
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
    pub fn shifted(p: Point) -> Point { moved(p, 1) }
    pub fn moved(p: Point, by: Nat) -> Point { Point { x: p.x + by } }
}

enum Shape {
    Circle(Nat),
    Square(Nat),
}

pub mod Shape {
    pub fn size(shape: Shape) -> Nat {
        case shape {
            .Circle(r) -> r
            .Square(s) -> s
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
            .Some(x) -> x
            .None -> +0
        }
    }

    pub fn text(n: Int) -> String { Int::to_string(n) }
}

fn main() -> Nil {
    let _ = m::text(m::get(.Some(+1)))
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// An import's path may start from another import of the same module,
/// whichever is declared first.
#[salsa_test]
fn imports_build_on_other_imports(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod other {
    pub struct Point { x: Nat }
    pub fn f() -> Nat { 2 }
}

mod m {
    use other::f
    use other::Point
    use super::other

    pub fn value(p: Point) -> Nat { f() + p.x }
}

fn main() -> Nil {
    let _ = m::value(other::Point { x: 1 })
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// An import shadows the module's own name at the start of a path.
#[salsa_test]
fn imports_take_precedence_over_the_module_name(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod other {
    pub struct Point { x: Nat }
}

mod m {
    use super::other as m

    pub fn read(p: m::Point) -> Nat { p.m::Point::x }
}

fn main() -> Nil {
    let _ = m::read(other::Point { x: 1 })
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A module's own name does not start a path inside it, companion or not;
/// its items are named directly.
#[salsa_test]
fn module_name_does_not_start_a_path(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Point { x: Nat }

pub mod Point {
    pub fn origin() -> Point { Point { x: 0 } }
    pub fn again() -> Point { Point::origin() }
}

mod m {
    pub fn one() -> Nat { 1 }
    pub fn two() -> Nat { m::one() + 1 }
}

fn main() -> Nil { }
"#,
    );
    for name in ["Point::origin", "m::one"] {
        assert!(
            errors
                .iter()
                .any(|message| message.starts_with(&format!("unresolved name `{name}`"))),
            "{name}: {errors:?}"
        );
    }
}

/// A method call finds its function from the names in scope: the function
/// of that name, or the one a module or type in scope has. A type named by
/// a path brings no methods, and a module out of scope adds no candidate.
#[salsa_test]
fn methods_come_from_the_names_in_scope(db: &salsa::DatabaseImpl) {
    const SHAPES: &str = r#"
mod shapes {
    pub struct Point { x: Nat }
    pub mod Point {
        pub fn double(p: Point) -> Nat { p.x + p.x }
    }
    pub fn x(n: Nat) -> Nat { n }
}
"#;
    let check = |module: &str| {
        errors(
            db,
            &format!("{SHAPES}mod m {{\n{module}\n}}\nfn main() -> Nil {{ }}\n"),
        )
    };
    for module in [
        "use super::shapes::Point\n    pub fn f(p: Point) -> Nat { p.double + p.x }",
        // `shapes::x` is no candidate of `p.x` here, though it is one where
        // `shapes` is in scope.
        "use super::shapes::Point\n    pub fn f(p: Point) -> Nat { p.x }",
        "use super::shapes\n    pub fn f(p: shapes::Point) -> Nat { 1.x + p.shapes::Point::x }",
        "pub fn f(p: super::shapes::Point) -> Nat { p.super::shapes::Point::double }",
    ] {
        let errors = check(module);
        assert!(errors.is_empty(), "{module}: {errors:#?}");
    }
    for (module, method) in [
        ("pub fn f(p: super::shapes::Point) -> Nat { p.x }", "x"),
        (
            "pub fn f(p: super::shapes::Point) -> Nat { p.double }",
            "double",
        ),
    ] {
        assert_eq!(
            check(module),
            [format!(
                "unresolved method `{method}`: the scope names no function `{method}`, and no \
                 module or type in it has one"
            )],
            "{module}"
        );
    }
}

/// A function in a module out of scope is no candidate of a method call, even
/// when it is the only one the receiver's type fits; the call checks against
/// the function in scope. Importing the module makes it a candidate.
#[salsa_test]
fn methods_out_of_scope_are_not_chosen_by_type(db: &salsa::DatabaseImpl) {
    let check = |import: &str, receiver: &str| {
        errors(
            db,
            &format!(
                r#"
mod outer {{
    pub mod hidden {{
        pub fn twice(x: Int) -> Int {{ x * +2 }}
    }}
}}
{import}
fn twice(x: Nat) -> Nat {{ x + x }}

fn id(x: a) -> a {{ x }}

fn main() -> Nil {{
    let n = +21
    let _ = {receiver}.twice()
}}
"#
            ),
        )
    };
    for receiver in ["n", "id(n)"] {
        let errors = check("", receiver);
        assert!(
            errors
                .iter()
                .any(|message| message.contains("expected `Nat`, found `Int`")),
            "{receiver}: {errors:#?}"
        );
        let errors = check("use outer::hidden", receiver);
        assert!(errors.is_empty(), "{receiver}: {errors:#?}");
    }
}
