//! A module's binding names it by its identity path, however deep it is
//! nested and whatever package it belongs to.

use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use tribute_front::resolve::{Binding, PackageName, build_env, build_library_env};
use trunk_ir::{Symbol, SymbolPath};

fn module_path(binding: Option<&Binding<'_>>) -> Vec<String> {
    match binding {
        Some(Binding::Module { path }) => path.iter().map(ToString::to_string).collect(),
        other => panic!("expected a module binding, got {other:?}"),
    }
}

const TREE: &str = r#"
pub mod a {
    pub mod b {
        pub mod c {
            pub fn f() -> Nat { 1 }
        }
    }
}

mod std {
    pub mod inner {
        pub fn g() -> Nat { 2 }
    }
}
"#;

#[salsa_test]
fn nested_modules_are_bound_by_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "modules.trb", TREE);
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let env = build_env(db, parsed.module(db));
    assert_eq!(env.package(), &PackageName::compiled());

    let c = env.lookup_qualified(&SymbolPath::new(["a", "b"]), &Symbol::new("c"));
    assert_eq!(module_path(c), ["a", "b", "c"]);
    assert_eq!(module_path(env.lookup(&Symbol::new("a"))), ["a"]);

    // A user module named `std` is not package `std`.
    assert_eq!(module_path(env.lookup(&Symbol::new("std"))), ["pkg", "std"]);
    let inner = env.lookup_qualified(&SymbolPath::new(["std"]), &Symbol::new("inner"));
    assert_eq!(module_path(inner), ["pkg", "std", "inner"]);
    assert!(env.declares(Symbol::new("std")));
}

#[salsa_test]
fn library_modules_are_bound_under_the_package(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "library.trb", TREE);
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let env = build_library_env(db, parsed.module(db));

    let c = env.lookup_qualified(&SymbolPath::new(["a", "b"]), &Symbol::new("c"));
    assert_eq!(module_path(c), ["std", "a", "b", "c"]);
    assert!(env.declares(Symbol::new("a")));
}

#[salsa_test]
fn a_re_exported_module_keeps_its_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "re_export.trb",
        r#"
pub mod a {
    pub mod b {
        pub fn f() -> Nat { 1 }
    }
}

pub mod facade {
    pub use pkg::a::b
}

use nowhere
"#,
    );
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let env = build_env(db, parsed.module(db));

    let b = env.lookup_qualified(&SymbolPath::new(["facade"]), &Symbol::new("b"));
    assert_eq!(module_path(b), ["a", "b"]);
    // A `use` of one segment the package root names declares it there.
    assert!(env.declares(Symbol::new("nowhere")));
}

#[salsa_test]
fn a_module_re_exported_twice_keeps_its_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "re_export_twice.trb",
        r#"
pub mod a {
    pub mod b {
        pub fn f() -> Nat { 1 }
    }
}

pub mod facade {
    pub use pkg::a::b
}

pub mod outer {
    pub use pkg::facade::b
}
"#,
    );
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let env = build_env(db, parsed.module(db));

    let b = env.lookup_qualified(&SymbolPath::new(["outer"]), &Symbol::new("b"));
    assert_eq!(module_path(b), ["a", "b"]);
}

/// The module path the body of `fn probe` refers to, when `source` is
/// resolved as the root module of package `std`.
#[salsa::tracked]
fn library_module_reference(db: &dyn salsa::Database, source: SourceCst) -> Vec<String> {
    use tribute_front::ast::{Decl, ExprKind, ResolvedRef};

    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let module = parsed.module(db);
    let env = build_library_env(db, module);
    let resolved =
        tribute_front::resolve::resolve_library_with_env(db, module, env, parsed.span_map(db));
    let body = resolved
        .decls()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name == "probe" => Some(&function.body),
            _ => None,
        })
        .expect("fn probe");
    let ExprKind::Block { value, .. } = &*body.kind else {
        panic!("a block body");
    };
    match &*value.kind {
        ExprKind::Var(ResolvedRef::Module { path }) => {
            path.segments(db).iter().map(ToString::to_string).collect()
        }
        other => panic!("expected a module reference, got {other:?}"),
    }
}

#[salsa_test]
fn a_library_one_segment_import_names_its_module_by_identity(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "library_import.trb",
        r#"
pub mod tools {
    pub fn one() -> Nat { 1 }
}
use tools

fn probe() -> Nat { tools }
"#,
    );
    assert_eq!(library_module_reference(db, source)[..], ["std", "tools"]);
}
