//! Tests that an edit outside a function leaves its resolved and typed
//! declaration unchanged, so that results computed from it can be reused.

use salsa_test_macros::salsa_test;
use tribute_front::{
    SourceCst,
    ast::{Decl, FuncDecl, ResolvedRef, TypedRef},
    query::{resolved_module, type_check_output},
};

const PATH: &str = "test.trb";

fn resolved_function<'db>(
    db: &'db dyn salsa::Database,
    text: &str,
    name: &str,
) -> FuncDecl<ResolvedRef<'db>> {
    let source = SourceCst::from_source_str(db, PATH, text);
    let table = resolved_module(db, source).expect("source should parse");
    table
        .entries
        .iter()
        .find_map(|entry| match &entry.decl {
            Decl::Function(func) if func.name == name => Some(func.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("function `{name}` should be declared"))
}

fn typed_function<'db>(
    db: &'db dyn salsa::Database,
    text: &str,
    name: &str,
) -> FuncDecl<TypedRef<'db>> {
    let source = SourceCst::from_source_str(db, PATH, text);
    let output = type_check_output(db, source).expect("source should type check");
    output
        .module(db)
        .entries
        .iter()
        .find_map(|entry| match &entry.decl {
            Decl::Function(func) if func.name == name => Some(func.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("function `{name}` should be declared"))
}

fn local_ids(func: &FuncDecl<ResolvedRef<'_>>) -> Vec<u32> {
    let debug = format!("{func:?}");
    debug
        .split("LocalId(")
        .skip(1)
        .map(|rest| {
            let digits = rest.split(')').next().expect("LocalId debug is closed");
            digits.parse().expect("LocalId holds a number")
        })
        .collect()
}

/// Assert that `name` resolves and type checks to the same declaration in
/// `before` and `after`.
fn assert_unchanged(db: &dyn salsa::Database, before: &str, after: &str, name: &str) {
    assert_eq!(
        resolved_function(db, before, name),
        resolved_function(db, after, name)
    );
    assert_eq!(
        typed_function(db, before, name),
        typed_function(db, after, name)
    );
}

const G: &str = r#"
fn g(a: Int, b: Int) -> Int {
    let c = a + b
    let add = fn(d) { d + c }
    case add(a) {
        0 -> b
        m -> m + c
    }
}
"#;

fn module_with_f(f_body: &str) -> String {
    format!("fn f(x: Int) -> Int {{\n{f_body}\n}}\n{G}")
}

#[salsa_test]
fn an_edit_in_an_earlier_function_keeps_later_declarations(db: &salsa::DatabaseImpl) {
    let before = module_with_f("    x");
    let after = module_with_f("    let y = x + 1\n    y");

    assert_unchanged(db, &before, &after, "g");
    assert_ne!(
        resolved_function(db, &before, "f"),
        resolved_function(db, &after, "f")
    );
}

#[salsa_test]
fn comments_and_new_items_before_a_function_keep_it(db: &salsa::DatabaseImpl) {
    let before = module_with_f("    x");
    let after = format!(
        "// A comment that moves everything down.\n\nstruct Point {{ x: Int }}\n\n\
         fn h(p: Point) -> Int {{ p.x }}\n\n{}",
        module_with_f("    // inside f\n    x")
    );

    assert_unchanged(db, &before, &after, "f");
    assert_unchanged(db, &before, &after, "g");
}

#[salsa_test]
fn each_function_numbers_locals_independently(db: &salsa::DatabaseImpl) {
    let text = module_with_f("    let y = x + 1\n    y");
    let f = local_ids(&resolved_function(db, &text, "f"));
    let g = local_ids(&resolved_function(db, &text, "g"));

    assert_eq!(f.iter().min(), g.iter().min());
}
