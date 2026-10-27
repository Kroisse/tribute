//! Tests that a source type's identity depends on its declaration's path, not
//! on where the declaration is in the source text.

use salsa_test_macros::salsa_test;
use tribute_front::{
    SourceCst,
    ast::{Decl, ModuleItem, TypeDefId, TypeKind},
    query::{function_schemes, parsed_module},
};
use trunk_ir::SymbolPath;

const PATH: &str = "test.trb";

/// The nominal type of the first parameter of function `name`.
fn first_param_type<'db>(db: &'db dyn salsa::Database, text: &str, name: &str) -> TypeDefId<'db> {
    let source = SourceCst::from_source_str(db, PATH, text);
    let schemes = function_schemes(db, source).expect("source should type check");
    let (_, scheme) = schemes
        .iter()
        .find(|(path, _)| *path == ["pkg", name])
        .unwrap_or_else(|| panic!("function `{name}` should be declared"));
    let TypeKind::Func { params, .. } = scheme.body(db).kind(db) else {
        panic!("`{name}` should have a function type");
    };
    match params[0].kind(db) {
        TypeKind::Named { id, .. } => *id,
        other => panic!("expected a named parameter type, found {other:?}"),
    }
}

const POINT: &str = "struct Point { x: Int }\n\nfn get_x(p: Point) -> Int { p.x }\n";

#[salsa_test]
fn an_edit_elsewhere_keeps_type_identity(db: &salsa::DatabaseImpl) {
    let before = first_param_type(db, POINT, "get_x");
    let edited = format!("fn other(n: Int) -> Int {{\n    let m = n + 1\n    m\n}}\n\n{POINT}");
    let after = first_param_type(db, &edited, "get_x");

    assert_eq!(before, after);
    assert_eq!(
        before,
        TypeDefId::source(db, SymbolPath::new(["pkg", "Point"]))
    );
}

#[salsa_test]
fn redeclared_type_names_are_numbered_per_module(db: &salsa::DatabaseImpl) {
    let text = "\
struct A { x: Int }
enum A { One, Two }
struct B { y: Int }
mod m {
    struct A { z: Int }
}
struct A { w: Int }
";
    let source = SourceCst::from_source_str(db, PATH, text);
    let module = parsed_module(db, source).expect("source should parse");
    let disambiguators = |items: &[ModuleItem<_>]| {
        items
            .iter()
            .filter_map(|item| match item {
                ModuleItem::Decl(Decl::Struct(decl)) => {
                    Some((decl.name.to_string(), decl.disambiguator))
                }
                ModuleItem::Decl(Decl::Enum(decl)) => {
                    Some((decl.name.to_string(), decl.disambiguator))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let root = disambiguators(&module.decls);
    assert_eq!(
        root,
        [
            ("A".into(), 0),
            ("A".into(), 1),
            ("B".into(), 0),
            ("A".into(), 2)
        ]
    );
    let inner = module
        .decls
        .iter()
        .find_map(|item| match item {
            ModuleItem::Module(m) => m.body.as_deref(),
            _ => None,
        })
        .expect("inline module should have a body");
    assert_eq!(disambiguators(inner), [("A".into(), 0)]);

    let path = SymbolPath::new(["A"]);
    assert_ne!(
        TypeDefId::declared(db, path.clone(), 0),
        TypeDefId::declared(db, path, 1)
    );
}
