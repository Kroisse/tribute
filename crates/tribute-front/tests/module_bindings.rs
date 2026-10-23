//! A module's binding names it by its identity path, however deep it is
//! nested and whatever package it belongs to.

use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use tribute_front::resolve::{Binding, Package, build_env, build_library_env};
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
    assert_eq!(env.package(), &Package::compiled());

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
