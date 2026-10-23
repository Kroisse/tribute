//! Tests that an edit inside one function leaves the resolved identities of
//! the other functions in the module unchanged.

use salsa_test_macros::salsa_test;
use tribute_front::{
    SourceCst,
    ast::{Decl, FuncDecl, ResolvedRef},
    query::resolved_module,
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

fn local_ids_before_and_after(
    db: &dyn salsa::Database,
    before: &str,
    after: &str,
    name: &str,
) -> (Vec<u32>, Vec<u32>) {
    (
        local_ids(&resolved_function(db, before, name)),
        local_ids(&resolved_function(db, after, name)),
    )
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
fn edit_in_earlier_function_keeps_later_local_ids(db: &salsa::DatabaseImpl) {
    let before = module_with_f("    x");
    let after = module_with_f("    let y = x + 1\n    y");

    let (g_before, g_after) = local_ids_before_and_after(db, &before, &after, "g");
    assert!(!g_before.is_empty(), "g should contain local ids");
    assert_eq!(g_before, g_after);

    let (f_before, f_after) = local_ids_before_and_after(db, &before, &after, "f");
    assert_ne!(f_before, f_after);
}

#[salsa_test]
fn each_function_numbers_locals_independently(db: &salsa::DatabaseImpl) {
    let text = module_with_f("    let y = x + 1\n    y");
    let f = local_ids(&resolved_function(db, &text, "f"));
    let g = local_ids(&resolved_function(db, &text, "g"));

    assert_eq!(f.iter().min(), g.iter().min());
}
