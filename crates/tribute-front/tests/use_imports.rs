//! Resolution of `use` paths: diagnostics for paths that do not resolve,
//! and the identity of what a resolved import names.

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

/// A type imported by a path written relative to the enclosing inline module
/// keeps the identity name resolution found; type checking does not reread the
/// path from the package root.
#[salsa_test]
fn relative_type_imports_keep_the_resolved_type(db: &salsa::DatabaseImpl) {
    for import in ["use a::P", "use outer::a::P", "use a::P as Q"] {
        let annotation = if import.ends_with("as Q") { "Q" } else { "P" };
        let errors = errors(
            db,
            &format!(
                r#"
mod outer {{
    mod a {{
        pub struct P {{ x: Nat }}
    }}
    {import}

    pub fn get(p: {annotation}) -> Nat {{
        p.x
    }}
}}

fn main() {{
    let _ = outer::get(outer::a::P {{ x: 1 }})
}}
"#
            ),
        );
        assert!(errors.is_empty(), "{import}: {errors:#?}");
    }
}

/// A path that names something from both the package root and the enclosing
/// inline module is read from the package root.
#[salsa_test]
fn root_relative_reading_wins_over_the_enclosing_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub struct P { root: Nat }
}
mod outer {
    mod a {
        pub struct P { nested: Nat }
    }
    use a::P

    pub fn get(p: P) -> Nat {
        p.root
    }
}

fn main() {
    let _ = outer::get(a::P { root: 1 })
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

/// Imports declared inside an inline module are in scope in that module's
/// body, whether the path is written from the package root or relative to the
/// module, aliased, or names a module used as a path prefix.
#[salsa_test]
fn inline_module_imports_are_in_scope_in_the_module(db: &salsa::DatabaseImpl) {
    for (import, call) in [
        ("use outer::a::one", "one()"),
        ("use a::one", "one()"),
        ("use a::one as uno", "uno()"),
        ("use a", "a::one()"),
    ] {
        let errors = errors(
            db,
            &format!(
                r#"
mod outer {{
    mod a {{
        pub fn one() -> Nat {{ 1 }}
    }}
    {import}

    pub fn get() -> Nat {{
        {call}
    }}
}}

fn main() {{
    let _ = outer::get()
}}
"#
            ),
        );
        assert!(errors.is_empty(), "{import}: {errors:#?}");
    }
}

/// Constructors and abilities imported inside an inline module resolve in
/// expressions, effect annotations, and handler arms.
#[salsa_test]
fn inline_module_imports_cover_constructors_and_abilities(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod a {
        pub struct P { x: Nat }
    }
    mod fx {
        pub ability Tick {
            op tick() -> Nat
        }
    }
    use a::P
    use fx::Tick

    pub fn make() -> P {
        P { x: 1 }
    }

    pub fn count() ->{Tick} Nat {
        Tick::tick()
    }

    pub fn run() -> Nat {
        handle count() {
            do result { result }
            op Tick::tick() { resume make().x }
        }
    }
}

fn main() {
    let _ = outer::run()
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

/// An inline module's import is not a member of the module.
#[salsa_test]
fn inline_module_imports_are_not_visible_outside_the_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod a {
        pub fn one() -> Nat { 1 }
    }
    use a::one
}

fn main() {
    let _ = outer::one()
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.starts_with("unresolved name `outer::one`")),
        "{errors:#?}"
    );
}
