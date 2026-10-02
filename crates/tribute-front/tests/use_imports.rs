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

fn main() -> Nil {
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

fn main() -> Nil {}
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

/// A `use` path may start with a path keyword.
#[salsa_test]
fn path_keyword_imports_resolve(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod basic {
    pub fn add(a: Nat, b: Nat) -> Nat { a + b }
}
use self::basic::add

fn main() -> Nil {
    let _ = add(1, 2)
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
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

fn main() -> Nil {
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
    for import in [
        "use a::P",
        "use self::a::P",
        "use pkg::outer::a::P",
        "use a::P as Q",
    ] {
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

fn main() -> Nil {{
    let _ = outer::get(outer::a::P {{ x: 1 }})
}}
"#
            ),
        );
        assert!(errors.is_empty(), "{import}: {errors:#?}");
    }
}

/// A path inside an inline module is read from that module, never from the
/// package root, even when the package root names the same path.
#[salsa_test]
fn inline_module_paths_start_from_the_module(db: &salsa::DatabaseImpl) {
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
        p.nested
    }
}

fn main() -> Nil {
    let _ = outer::get(outer::a::P { nested: 1 })
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
        ("use pkg::outer::a::one", "one()"),
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

fn main() -> Nil {{
    let _ = outer::get()
}}
"#
            ),
        );
        assert!(errors.is_empty(), "{import}: {errors:#?}");
    }
}

/// Constructors and abilities imported inside an inline module resolve in
/// expressions, effect annotations, handler arms, and unqualified operation
/// calls under an effect annotation.
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

    pub fn count_unqualified() ->{Tick} Nat {
        tick()
    }

    pub fn run() -> Nat {
        let _ = handle count_unqualified() {
            do result { result }
            op Tick::tick() { resume 2 }
        }
        handle count() {
            do result { result }
            op Tick::tick() { resume make().x }
        }
    }
}

fn main() -> Nil {
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

fn main() -> Nil {
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

/// An import of an inline module that gives a path prefix a name hides the
/// package-root namespace of that name, even for members the import lacks.
#[salsa_test]
fn inline_module_import_hides_the_root_namespace_it_shadows(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub fn one() -> Nat { 1 }
}
mod outer {
    mod b {
        pub fn two() -> Nat { 2 }
    }
    use outer::b as a

    pub fn get() -> Nat {
        a::one()
    }
}

fn main() -> Nil {
    let _ = outer::get()
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.starts_with("unresolved name `a::one`")),
        "{errors:#?}"
    );
}

/// An effect annotation may reach an ability through a module an inline
/// module imports under an alias, and its operations are callable unqualified.
#[salsa_test]
fn inline_module_import_prefixes_qualified_effect_annotations(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod fx {
        pub ability Tick {
            op tick() -> Nat
        }
    }
    use fx as effects

    pub fn count() ->{effects::Tick} Nat {
        tick()
    }

    pub fn run() -> Nat {
        handle count() {
            do result { result }
            op effects::Tick::tick() { resume 1 }
        }
    }
}

fn main() -> Nil {
    let _ = outer::run()
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}

/// Effect annotations follow the same precedence as other names: an ability
/// the inline module defines itself wins over one it imports under the same
/// name.
#[salsa_test]
fn inline_module_definition_wins_over_an_imported_ability(db: &salsa::DatabaseImpl) {
    use tribute_front::ast::{Decl, TypeAnnotationKind};
    use trunk_ir::Symbol;

    let source = SourceCst::from_source_str(
        db,
        "use_imports.trb",
        r#"
mod outer {
    mod fx {
        pub ability Tick {
            op tick() -> Nat
        }
    }
    pub ability Tick {
        op other() -> Nat
    }
    use fx::Tick

    pub fn noop() ->{Tick} Nat {
        1
    }
}
"#,
    );
    let module = tribute_front::query::resolved_module(db, source).expect("module resolves");
    let effects = module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Module(outer) => outer.body.as_ref(),
            _ => None,
        })
        .and_then(|body| {
            body.iter().find_map(|decl| match decl {
                Decl::Function(function) if function.name == Symbol::new("noop") => {
                    function.effects.clone()
                }
                _ => None,
            })
        })
        .expect("noop declares effects");
    assert!(
        matches!(&effects[0].kind, TypeAnnotationKind::Named(name) if *name == Symbol::new("Tick")),
        "{effects:?}"
    );
}

/// An inline module may import a package-root ability under an alias; the
/// alias names the package-root ability in effect annotations, unqualified
/// operation calls, and handler arms.
#[salsa_test]
fn inline_module_import_aliases_a_root_ability(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
ability Tick {
    op tick() -> Nat
}
mod outer {
    use super::Tick as T

    pub fn count() ->{T} Nat {
        tick()
    }

    pub fn run() -> Nat {
        handle count() {
            do result { result }
            op T::tick() { resume 1 }
        }
    }
}

fn main() -> Nil {
    let _ = outer::run()
}
"#,
    );
    assert!(errors.is_empty(), "{errors:#?}");
}
