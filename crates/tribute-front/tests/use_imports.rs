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
use basic::add
use basic::add as plus
use outer::inner

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

fn main() {}
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved import `basic::sub`",
            "unresolved import `nowhere::thing`",
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
