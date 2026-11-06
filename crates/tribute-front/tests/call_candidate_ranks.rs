//! Tests for the ranks of call candidates: a call selects among the
//! functions its scope names directly, then `m::f` for the names `m` in
//! scope, then the prelude's functions, moving to a lower rank only when
//! no candidate of the higher ones fits.

mod common;

use self::common::{TdnrCall, TdnrSummary, ast_pipeline_error_messages, tdnr_function_summary};
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn selected(targets: &[&str]) -> TdnrSummary {
    TdnrSummary {
        method_calls: vec![],
        calls: targets
            .iter()
            .map(|target| TdnrCall {
                target: if target.starts_with("std::") {
                    (*target).to_owned()
                } else {
                    format!("pkg::{target}")
                },
                arg_count: 1,
            })
            .collect(),
    }
}

/// A user `Option` hides the prelude's short name but not its methods: on a
/// `std::Option` value, `x.map(f)` rules out the user's `Option::map` by the
/// receiver's type and selects `std::Option::map` from the prelude rank.
#[salsa_test]
fn prelude_method_is_selected_when_a_user_type_shadows_its_name(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "shadowed_option.trb",
        r#"
enum Option { Nothing }

pub mod Option {
    pub fn map(opt: Option, f: fn(Nat) -> Nat) -> Option { opt }
}

fn test(x: std::Option(Nat)) -> std::Option(Nat) {
    x.map(fn(n) n)
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, source, "test"),
        TdnrSummary {
            method_calls: vec![],
            calls: vec![TdnrCall {
                target: "std::Option::map".to_owned(),
                arg_count: 2,
            }],
        }
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
}

const NAME_AND_AUDIT: &str = r#"
struct Name { text: String }

mod audit {
    use super::Name

    pub fn text(name: Name) -> String { name.Name::text }
}
"#;

/// A function `use`d under the field's name is a direct candidate, so it
/// is selected before the getter both fit.
#[salsa_test]
fn used_function_is_selected_before_a_getter(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "used_function.trb",
        &format!(
            "{NAME_AND_AUDIT}
use audit::text

fn test(name: Name) -> String {{ name.text }}
"
        ),
    );

    assert_eq!(
        tdnr_function_summary(db, source, "test"),
        selected(&["audit::text"])
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
}

/// A function reached through a module name in scope has the getter's rank,
/// so a call both fit is ambiguous.
#[salsa_test]
fn getter_and_module_function_are_ambiguous(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "ambiguous_getter.trb",
        &format!(
            "{NAME_AND_AUDIT}
fn test(name: Name) -> String {{ name.text }}
"
        ),
    );

    assert_eq!(
        tdnr_function_summary(db, source, "test"),
        TdnrSummary {
            method_calls: vec!["text".to_owned()],
            calls: vec![],
        }
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("Name::text") && error.contains("audit::text")),
        "{errors:#?}"
    );
}

const TWICE: &str = r#"
fn twice(x: Nat) -> Nat { x + x }

mod math {
    pub fn twice(x: Int) -> Int { x * +2 }
}

fn id(x: a) -> a { x }
"#;

/// A receiver whose type is not known where the call is written rules out
/// no candidate: the call waits, and falls to a lower rank only once the
/// receiver's type rules the direct function out.
#[salsa_test]
fn undetermined_receiver_does_not_fall_through_early(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "undetermined_receiver.trb",
        &format!(
            "{TWICE}
fn by_nat() -> Nat {{ id(21).twice() }}

fn by_int() -> Int {{ id(+21).twice() }}

fn unknown() -> Nil {{
    let _ = fn(y) y.twice()
}}
"
        ),
    );

    assert_eq!(
        tdnr_function_summary(db, source, "by_nat"),
        selected(&["twice", "id"])
    );
    assert_eq!(
        tdnr_function_summary(db, source, "by_int"),
        selected(&["math::twice", "id"])
    );
    // A receiver no type ever determines leaves the direct function as the
    // highest rank that fits.
    assert_eq!(
        tdnr_function_summary(db, source, "unknown"),
        selected(&["twice"])
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "{errors:#?}");
}

/// A path that starts with a dependency package names its one function: a
/// same path through a module name in scope is no candidate, even when only
/// it fits the receiver.
#[salsa_test]
fn dependency_path_names_one_function(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "dependency_path.trb",
        r#"
mod tools {
    pub mod std {
        pub mod String {
            pub fn len(x: Int) -> Nat { 1 }
        }
    }
}
use tools

fn test(n: Int) -> Nat {
    n.std::String::len()
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, source, "test"),
        selected(&["std::String::len"])
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("expected `std::String`, found `Int`")),
        "{errors:#?}"
    );
}

/// A qualified path through a prelude name is the prelude's reading of it,
/// at the package root and in an inline module; a type the scope declares
/// under that name takes the path instead.
#[salsa_test]
fn qualified_path_through_a_prelude_name(db: &salsa::DatabaseImpl) {
    let prelude = SourceCst::from_source_str(
        db,
        "prelude_path.trb",
        r#"
fn test(x: Option(Nat)) -> Option(Nat) {
    x.Option::map(fn(n) n)
}

mod inner {
    pub fn test(x: Option(Nat)) -> Option(Nat) {
        x.Option::map(fn(n) n)
    }
}
"#,
    );
    let declared = SourceCst::from_source_str(
        db,
        "declared_path.trb",
        r#"
enum Option { Nothing }

pub mod Option {
    pub fn map(opt: Option, f: fn(Nat) -> Nat) -> Option { opt }
}

fn test(x: Option) -> Option {
    x.Option::map(fn(n) n)
}

mod inner {
    enum Option { Nothing }

    pub mod Option {
        pub fn map(opt: Option, f: fn(Nat) -> Nat) -> Option { opt }
    }

    pub fn test(x: Option) -> Option {
        x.Option::map(fn(n) n)
    }
}
"#,
    );

    let map = |target: &str| TdnrSummary {
        method_calls: vec![],
        calls: vec![TdnrCall {
            target: target.to_owned(),
            arg_count: 2,
        }],
    };
    assert_eq!(
        tdnr_function_summary(db, prelude, "test"),
        map("std::Option::map")
    );
    assert_eq!(
        tdnr_function_summary(db, declared, "test"),
        map("pkg::Option::map")
    );
    for source in [prelude, declared] {
        let errors = ast_pipeline_error_messages(db, source);
        assert!(errors.is_empty(), "{errors:#?}");
    }
}
