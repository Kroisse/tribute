//! Diagnostics for `use` paths that do not resolve.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "use_imports.trb", text))
}

#[salsa_test]
fn resolvable_imports_are_accepted(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
mod outer {
    mod inner {
        pub fn one() -> Nat { 1 }
    }
    use inner::one
}
use basic
use basic::add
use basic::add as plus
use outer::inner
use helper

fn helper() -> Nat { 1 }

fn main() {
    let _ = add(1, plus(2, 3))
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

#[salsa_test]
fn unresolved_imports_are_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
use basic::sub
use nowhere::thing
use nothing_here

fn main() {}
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved import `basic::sub`",
            "unresolved import `nowhere::thing`",
            "unresolved import `nothing_here`",
        ],
    );
}

/// Path keywords are not resolved yet. The import is reported, and a call
/// through it is rejected before lowering instead of crashing there.
#[salsa_test]
fn path_keyword_imports_are_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
use self::basic::add

fn main() {
    let _ = add(1, 2)
}
"#,
    );
    assert_eq!(
        errors,
        [
            "path keyword `self` is not supported in `use` paths yet",
            "expected a value, found module `self::basic::add`",
        ],
    );
}

/// A module named where a pattern expects a constructor is rejected before
/// lowering instead of crashing there.
#[salsa_test]
fn module_in_pattern_is_reported(db: &salsa::DatabaseImpl) {
    for (name, pattern) in [("tuple-style", "M(x)"), ("struct-style", "M { x }")] {
        let errors = errors(
            db,
            &format!(
                r#"
mod M {{ pub fn f() -> Nat {{ 1 }} }}
enum E {{ A(Nat) }}

fn g(e: E) -> Nat {{
    case e {{
        {pattern} -> x
        _ -> 0
    }}
}}
"#
            ),
        );
        assert_eq!(
            errors,
            ["expected a constructor, found module `M`"],
            "{name}"
        );
    }
}

/// A module named as a record literal's type is rejected before lowering.
#[salsa_test]
fn module_in_record_literal_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod M { pub fn f() -> Nat { 1 } }

fn main() {
    let _ = M { x: 1 }
}
"#,
    );
    assert_eq!(errors, ["expected a constructor, found module `M`"]);
}
