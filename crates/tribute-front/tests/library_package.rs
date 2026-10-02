//! The prelude is the root of package `std`: every module sees its root items
//! and their namespaces under short names, and a user declaration or import
//! of the same name takes precedence.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "library.trb", text))
}

#[salsa_test]
fn builtin_list_namespace_is_a_path_start_everywhere(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    pub fn one() -> List(Nat) { List::prepend(1, []) }
}

fn main() -> Nil {
    let _ = List::prepend(2, m::one())
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn imports_may_start_from_a_library_namespace(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use Option::None as Nothing

mod m {
    use Option::Some as Just

    pub fn one() -> Option(Nat) { Just(1) }
}

fn main() -> Nil {
    let _ = case m::one() {
        Some(n) -> n
        Nothing -> 0
    }
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A package-root import of a library name hides the library item, in
/// annotations as well as expressions.
#[salsa_test]
fn root_import_shadows_a_library_name(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod foo {
    pub enum Option { Nothing }
}

use foo::Option

fn f(o: Option) -> Nat {
    case o {
        foo::Option::Nothing -> 1
    }
}

fn main() -> Nil {
    let _ = f(foo::Option::Nothing)
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}
