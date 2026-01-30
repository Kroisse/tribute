//! Salsa integration tests for AST-based compilation pipeline.

use salsa::{Database as _, Setter as _};
use salsa_test_macros::salsa_test;
use tree_sitter::Parser;
use tribute::{SourceCst, TributeDatabaseImpl, parse_and_lower_ast};
use trunk_ir::DialectOp;
use trunk_ir::dialect::func;

/// Helper to count user functions (excluding prelude) by checking for specific names.
fn find_func_by_name<'db>(
    db: &'db dyn salsa::Database,
    module: &trunk_ir::dialect::core::Module<'db>,
    name: &str,
) -> bool {
    let body = module.body(db);
    let blocks = body.blocks(db);
    if blocks.is_empty() {
        return false;
    }

    blocks[0].operations(db).iter().any(|op| {
        func::Func::from_operation(db, *op)
            .map(|f| f.name(db) == name)
            .unwrap_or(false)
    })
}

#[salsa_test]
fn test_salsa_database_examples(db: &salsa::DatabaseImpl) {
    // Example source code
    let examples = vec![
        (
            "hello.trb",
            r#"fn main() { print_line("Hello, World!") }"#,
            vec!["main"],
        ),
        ("calc.trb", r#"fn main() { 1 + 2 + 3 }"#, vec!["main"]),
        (
            "complex.trb",
            r#"
fn factorial(n) {
  case n {
    0 { 1 }
    _ { n * factorial(n - 1) }
  }
}

fn main() {
  factorial(5)
}
"#,
            vec!["factorial", "main"],
        ),
    ];

    for (filename, source_code, expected_funcs) in examples {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(source_code, None).expect("tree");
        let source_file = SourceCst::from_path(db, filename, source_code.into(), Some(tree));
        let module = parse_and_lower_ast(db, source_file);

        // Verify that expected user functions exist
        for func_name in expected_funcs {
            assert!(
                find_func_by_name(db, &module, func_name),
                "Expected function '{}' not found in {}",
                func_name,
                filename
            );
        }
    }
}

#[test]
fn test_salsa_incremental_computation_detailed() {
    // Demonstrate incremental computation
    let mut db = TributeDatabaseImpl::default();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let text = "fn main() { 1 + 2 }";
    let tree = parser.parse(text, None).expect("tree");
    let source_file = SourceCst::from_path(&db, "incremental.trb", text.into(), Some(tree));

    // Initial lowering
    let module1 = parse_and_lower_ast(&db, source_file);
    assert!(
        find_func_by_name(&db, &module1, "main"),
        "Should have main function"
    );

    // Modify the source file
    let updated_text = "fn main() { 1 + 2 + 3 + 4 }";
    let updated_tree = parser.parse(updated_text, None).expect("tree");
    source_file.set_text(&mut db).to(updated_text.into());
    source_file.set_tree(&mut db).to(Some(updated_tree));

    // Lower again - should recompute
    let module2 = parse_and_lower_ast(&db, source_file);
    assert!(
        find_func_by_name(&db, &module2, "main"),
        "Should have main function after update"
    );

    // Lower again without changes - should use cached result
    let module3 = parse_and_lower_ast(&db, source_file);

    // Verify that cached results are the same
    assert_eq!(
        module2.body(&db).blocks(&db).len(),
        module3.body(&db).blocks(&db).len(),
        "Modules should be identical (cached result)"
    );
}

#[salsa_test]
fn test_salsa_multiple_functions(db: &salsa::DatabaseImpl) {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let text = r#"
fn add(a, b) { a + b }
fn multiply(a, b) { a * b }
fn main() { print_line("test") }
"#;
    let tree = parser.parse(text, None).expect("tree");
    let source = SourceCst::from_path(db, "multi.trb", text.into(), Some(tree));
    let module = parse_and_lower_ast(db, source);

    // Verify all user functions exist
    assert!(
        find_func_by_name(db, &module, "add"),
        "Should have add function"
    );
    assert!(
        find_func_by_name(db, &module, "multiply"),
        "Should have multiply function"
    );
    assert!(
        find_func_by_name(db, &module, "main"),
        "Should have main function"
    );
}

#[test]
fn test_salsa_database_isolation() {
    // Test that different database instances are isolated
    let module1_name = TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text = "fn main() { 1 + 2 }";
        let tree = parser.parse(text, None).expect("tree");
        let source1 = SourceCst::from_path(db, "test1.trb", text.into(), Some(tree));
        let module1 = parse_and_lower_ast(db, source1);
        module1.name(db).to_string()
    });

    let module2_name = TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text = "fn main() { 3 * 4 }";
        let tree = parser.parse(text, None).expect("tree");
        let source2 = SourceCst::from_path(db, "test2.trb", text.into(), Some(tree));
        let module2 = parse_and_lower_ast(db, source2);
        module2.name(db).to_string()
    });

    // Module names are derived from file paths
    assert_eq!(module1_name, "test1");
    assert_eq!(module2_name, "test2");
}

#[salsa_test]
fn test_function_lowering(db: &salsa::DatabaseImpl) {
    let source = "fn main() { 1 + 2 }";
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parser.parse(source, None).expect("tree");
    let source_file = SourceCst::from_path(db, "func_test.trb", source.into(), Some(tree));
    let module = parse_and_lower_ast(db, source_file);

    // Verify the main function exists
    assert!(
        find_func_by_name(db, &module, "main"),
        "Should have main function"
    );
}
