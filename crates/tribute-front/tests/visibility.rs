//! Item visibility: a declaration is visible in the module that declares it
//! and its descendants, widened by `pub(super)`, `pub(pkg)`, and `pub`, and
//! only where every module around it is visible too.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "visibility.trb", text))
}

fn assert_hidden(errors: &[String], expected: &str) {
    assert!(
        errors.iter().any(|error| error == expected),
        "expected `{expected}` in {errors:#?}"
    );
}

#[salsa_test]
fn a_private_item_is_visible_in_its_module_and_below(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn helper() -> Nat { 1 }

mod outer {
    use super::helper

    fn own() -> Nat { helper() }

    pub mod inner {
        use super::own

        pub fn twice() -> Nat { own() + pkg::outer::own() }
    }

    pub fn run() -> Nat { inner::twice() }
}

fn main() -> Nil {
    let _ = outer::run() + helper()
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn a_private_item_is_hidden_outside_its_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    fn hidden() -> Nat { 1 }
    struct Secret { value: Nat }
    enum Mode { Quiet }
}

fn read(secret: m::Secret, mode: m::Mode) -> Nat {
    case mode {
        m::Mode::Quiet -> m::hidden()
    }
}

fn main() -> Nil {
    let _ = m::Quiet
}
"#,
    );
    assert_hidden(
        &errors,
        "`m::hidden` is not visible here: `m::hidden` is private",
    );
    assert_hidden(
        &errors,
        "`m::Secret` is not visible here: `m::Secret` is private",
    );
    assert_hidden(
        &errors,
        "`m::Mode` is not visible here: `m::Mode` is private",
    );
    assert_hidden(
        &errors,
        "`m::Quiet` is not visible here: `m::Mode` is private",
    );
    assert_hidden(
        &errors,
        "`m::Mode::Quiet` is not visible here: `m::Mode` is private",
    );
}

#[salsa_test]
fn a_hidden_module_hides_its_public_items(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod inner {
        pub fn shown() -> Nat { 1 }
    }
}

use outer::inner::shown

fn main() -> Nil {
    let _ = outer::inner::shown()
}
"#,
    );
    assert_hidden(
        &errors,
        "`outer::inner::shown` is not visible here: `outer::inner` is private",
    );
}

#[salsa_test]
fn restricted_visibility_widens_to_the_parent_and_the_package(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub(pkg) mod outer {
    pub mod inner {
        pub(super) fn parent() -> Nat { 1 }
        pub(pkg) fn package() -> Nat { 2 }
    }

    pub fn run() -> Nat { inner::parent() }
}

fn main() -> Nil {
    let _ = outer::run() + outer::inner::package() + outer::inner::parent()
}
"#,
    );
    assert_eq!(
        errors,
        [
            "`outer::inner::parent` is not visible here: `outer::inner::parent` is declared `pub(super)`"
        ]
    );
}

#[salsa_test]
fn pub_super_at_the_package_root_is_reported(db: &salsa::DatabaseImpl) {
    let errors = errors(db, "pub(super) fn f() -> Nat { 1 }\nfn main() -> Nil {}\n");
    assert_eq!(
        errors,
        ["`pub(super)` on an item of the package root, which has no parent module"]
    );
}

#[salsa_test]
fn a_hidden_ability_is_hidden_in_annotations_and_handler_arms(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod fx {
    ability Tick {
        op tick() -> Nat
    }
}

fn count() ->{fx::Tick} Nat { fx::Tick::tick() }

fn main() -> Nil {
    let _ = handle count() {
        do result { result }
        op fx::Tick::tick() { resume 1 }
    }
}
"#,
    );
    assert_hidden(
        &errors,
        "`fx::Tick` is not visible here: `fx::Tick` is private",
    );
    assert_hidden(
        &errors,
        "`fx::Tick::tick` is not visible here: `fx::Tick` is private",
    );
}

#[salsa_test]
fn a_hidden_method_is_no_ufcs_candidate(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Counter { n: Nat }

pub mod Counter {
    fn bump(counter: Counter) -> Nat { counter.n + 1 }
    pub fn read(counter: Counter) -> Nat { counter.n }
}

fn main() -> Nil {
    let counter = Counter { n: 1 }
    let _ = counter.read() + counter.bump()
}
"#,
    );
    assert_eq!(
        errors,
        ["`bump` is not visible here: `Counter::bump` is private"]
    );
}

#[salsa_test]
fn a_re_export_cannot_widen_its_target(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod outer {
    pub(pkg) fn internal() -> Nat { 1 }
}

pub use outer::internal
pub(pkg) use outer::internal as same

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        [
            "`outer::internal` cannot be re-exported as `pub`: `outer::internal` is declared `pub(pkg)`"
        ]
    );
}

/// The prelude is the root of package `std`, so a user package sees only its
/// `pub` items, on every resolution path.
#[salsa_test]
fn the_prelude_shows_only_its_public_items(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use std::io::ReadLineResult

fn raw(bytes: Bytes) -> Nat { __tribute_bytes_len(bytes) + std::__tribute_bytes_len(bytes) }

fn cursor(c: String::Cursor) -> Nat { 0 }

fn same(left: String, right: String) -> Bool { left.rope_equal(right) }

fn main() -> Nil {
    let _ = Option::Some(1).is_some()
    let _ = Int::to_string(+1)
}
"#,
    );
    for expected in [
        "`std::io::ReadLineResult` is not visible here: `std::io::ReadLineResult` is private",
        "`__tribute_bytes_len` is not visible here: `std::__tribute_bytes_len` is private",
        "`std::__tribute_bytes_len` is not visible here: `std::__tribute_bytes_len` is private",
        "`std::String::Cursor` is not visible here: `std::String::Cursor` is private",
        "`rope_equal` is not visible here: `std::String::rope_equal` is private",
    ] {
        assert_hidden(&errors, expected);
    }
    assert_eq!(errors.len(), 5, "{errors:#?}");
}
