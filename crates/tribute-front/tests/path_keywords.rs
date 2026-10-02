//! Path keywords: `pkg` names the package root, `self` the current module,
//! and `super` its parent.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(
        db,
        SourceCst::from_source_str(db, "path_keywords.trb", text),
    )
}

#[salsa_test]
fn imports_start_from_the_module_a_keyword_names(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn base() -> Nat { 1 }

mod outer {
    pub mod inner {
        pub fn two() -> Nat { 2 }
    }
    use super::base
    use self::inner::two
    use pkg::outer::inner::two as also_two

    pub fn sum() -> Nat { base() + two() + also_two() }
}

use self::outer::sum

fn main() -> Nil {
    let _ = sum()
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn value_paths_start_from_the_module_a_keyword_names(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn base() -> Nat { 1 }

enum Shape {
    Circle(Nat),
    Square(Nat),
}

mod outer {
    pub fn own() -> Nat { 2 }

    pub mod inner {
        pub fn sum() -> Nat {
            pkg::base() + super::own() + self::three()
        }

        fn three() -> Nat { 3 }

        pub fn size(shape: pkg::Shape) -> Nat {
            case shape {
                pkg::Shape::Circle(r) -> r
                pkg::Shape::Square(s) -> s
            }
        }

        pub fn circle() -> pkg::Shape { pkg::Shape::Circle(4) }
    }
}

fn main() -> Nil {
    let _ = outer::inner::sum() + outer::inner::size(outer::inner::circle())
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A keyword path names exactly the declaration it spells, even when the
/// current module declares the same name.
#[salsa_test]
fn type_paths_are_not_read_from_the_current_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Point { x: Nat }

mod shapes {
    struct Point { y: Nat }

    pub fn x_of(p: super::Point) -> Nat { p.x }
    pub fn origin() -> pkg::Point { pkg::Point { x: 0 } }
}

fn main() -> Nil {
    let _ = shapes::x_of(shapes::origin())
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn effect_rows_name_abilities_through_keywords(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
ability Ask {
    op ask() -> Nat
}

mod asking {
    pub fn twice() ->{super::Ask} Nat { super::Ask::ask() + super::Ask::ask() }
}

fn main() -> Nil {
    let _ = handle asking::twice() {
        do value { value }
        op pkg::Ask::ask() { resume 21 }
    }
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn invalid_keywords_are_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn base() -> Nat { 1 }

use super::base

fn f() -> Nat { super::base() }

fn g(n: pkg) -> Nat { n }

fn main() -> Nil { }
"#,
    );
    assert!(
        errors
            .iter()
            .filter(|message| *message == "`super` at the package root has no parent module")
            .count()
            >= 2,
        "{errors:?}"
    );
}
