//! Salsa integration tests for AST-based compilation pipeline (arena IR).

use salsa::{Database as _, Setter as _};
use salsa_test_macros::salsa_test;
use tree_sitter::Parser;
use tribute::{SourceCst, TributeDatabaseImpl, compile_frontend_to_arena};
use trunk_ir::Symbol;
use trunk_ir::arena::{IrContext, Module};

/// Helper to check whether a `func.func` with the given `sym_name` exists
/// among the top-level operations of an arena module.
fn find_func_by_name(ctx: &IrContext, module: &Module, name: &str) -> bool {
    let func_dialect = Symbol::new("func");
    let func_name = Symbol::new("func");
    let sym_name_key = Symbol::new("sym_name");
    module.ops(ctx).iter().any(|&op_ref| {
        let op_data = ctx.op(op_ref);
        if op_data.dialect == func_dialect && op_data.name == func_name {
            op_data
                .attributes
                .get(&sym_name_key)
                .map(|attr| matches!(attr, trunk_ir::arena::types::Attribute::Symbol(s) if *s == name))
                .unwrap_or(false)
        } else {
            false
        }
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
        let (ctx, module) =
            compile_frontend_to_arena(db, source_file).expect("compilation should succeed");

        // Verify that expected user functions exist
        for func_name in expected_funcs {
            assert!(
                find_func_by_name(&ctx, &module, func_name),
                "Expected function '{}' not found in {}",
                func_name,
                filename
            );
        }
    }
}

#[test]
fn test_salsa_incremental_computation_detailed() {
    // Demonstrate incremental computation via compile_frontend_to_arena
    let mut db = TributeDatabaseImpl::default();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let text = "fn main() { 1 + 2 }";
    let tree = parser.parse(text, None).expect("tree");
    let source_file = SourceCst::from_path(&db, "incremental.trb", text.into(), Some(tree));

    // Initial lowering
    let (ctx1, module1) =
        compile_frontend_to_arena(&db, source_file).expect("compilation should succeed");
    assert!(
        find_func_by_name(&ctx1, &module1, "main"),
        "Should have main function"
    );

    // Modify the source file
    let updated_text = "fn main() { 1 + 2 + 3 + 4 }";
    let updated_tree = parser.parse(updated_text, None).expect("tree");
    source_file.set_text(&mut db).to(updated_text.into());
    source_file.set_tree(&mut db).to(Some(updated_tree));

    // Lower again - should recompute with updated source
    let (ctx2, module2) = compile_frontend_to_arena(&db, source_file)
        .expect("compilation should succeed after update");
    assert!(
        find_func_by_name(&ctx2, &module2, "main"),
        "Should have main function after update"
    );

    // Lower again without changes - pipeline still works
    let (ctx3, module3) =
        compile_frontend_to_arena(&db, source_file).expect("compilation should succeed on re-run");
    assert!(
        find_func_by_name(&ctx3, &module3, "main"),
        "Should have main function on cached run"
    );

    // Verify modules have the same structure (same number of top-level ops)
    assert_eq!(
        module2.ops(&ctx2).len(),
        module3.ops(&ctx3).len(),
        "Modules should have the same number of top-level ops"
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
    let (ctx, module) = compile_frontend_to_arena(db, source).expect("compilation should succeed");

    // Verify all user functions exist
    assert!(
        find_func_by_name(&ctx, &module, "add"),
        "Should have add function"
    );
    assert!(
        find_func_by_name(&ctx, &module, "multiply"),
        "Should have multiply function"
    );
    assert!(
        find_func_by_name(&ctx, &module, "main"),
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
        let (ctx, module) =
            compile_frontend_to_arena(db, source1).expect("compilation should succeed");
        module.name(&ctx).map(|s| s.to_string())
    });

    let module2_name = TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text = "fn main() { 3 * 4 }";
        let tree = parser.parse(text, None).expect("tree");
        let source2 = SourceCst::from_path(db, "test2.trb", text.into(), Some(tree));
        let (ctx, module) =
            compile_frontend_to_arena(db, source2).expect("compilation should succeed");
        module.name(&ctx).map(|s| s.to_string())
    });

    // Module names are derived from file paths
    assert_eq!(module1_name, Some("test1".to_string()));
    assert_eq!(module2_name, Some("test2".to_string()));
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
    let (ctx, module) =
        compile_frontend_to_arena(db, source_file).expect("compilation should succeed");

    // Verify the main function exists
    assert!(
        find_func_by_name(&ctx, &module, "main"),
        "Should have main function"
    );
}
