//! Salsa integration tests for CST→TrunkIR lowering.

use salsa::{Database as _, Setter as _};
use tribute::{SourceCst, TributeDatabaseImpl, lower_source_cst};
use tree_sitter::Parser;
use trunk_ir::DialectOp;

#[test]
fn test_salsa_database_examples() {
    // Example source code
    let examples = vec![
        ("hello.trb", r#"fn main() { print_line("Hello, World!") }"#),
        ("calc.trb", r#"fn main() { 1 + 2 + 3 }"#),
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
        ),
    ];

    for (filename, source_code) in examples {
        // Use attach pattern for test isolation
        let op_count = TributeDatabaseImpl::default().attach(|db| {
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(source_code, None).expect("tree");
            let source_file = SourceCst::from_path(db, filename, source_code.into(), Some(tree));
            let module = lower_source_cst(db, source_file);

            // Count top-level operations in the module
            let body = module.body(db);
            let blocks = body.blocks(db);
            if blocks.is_empty() {
                0
            } else {
                blocks[0].operations(db).len()
            }
        });

        // Verify parsing results
        assert!(
            op_count > 0,
            "Should produce at least one operation for {}",
            filename
        );

        match filename {
            "hello.trb" | "calc.trb" => {
                assert_eq!(op_count, 1, "Simple examples should have 1 function");
            }
            "complex.trb" => {
                assert_eq!(op_count, 2, "Complex example should have 2 functions");
            }
            _ => {}
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
    let module1 = lower_source_cst(&db, source_file);
    let body1 = module1.body(&db);
    let blocks1 = body1.blocks(&db);
    assert!(!blocks1.is_empty());
    let op_count1 = blocks1[0].operations(&db).len();
    assert_eq!(op_count1, 1);

    // Modify the source file
    let updated_text = "fn main() { 1 + 2 + 3 + 4 }";
    let updated_tree = parser.parse(updated_text, None).expect("tree");
    source_file.set_text(&mut db).to(updated_text.into());
    source_file.set_tree(&mut db).to(Some(updated_tree));

    // Lower again - should recompute
    let module2 = lower_source_cst(&db, source_file);
    let body2 = module2.body(&db);
    let blocks2 = body2.blocks(&db);
    assert!(!blocks2.is_empty());
    let op_count2 = blocks2[0].operations(&db).len();
    assert_eq!(op_count2, 1);

    // Lower again without changes - should use cached result
    let module3 = lower_source_cst(&db, source_file);

    // Verify that cached results are the same
    assert_eq!(
        module2.body(&db).blocks(&db).len(),
        module3.body(&db).blocks(&db).len(),
        "Modules should be identical (cached result)"
    );
}

#[test]
fn test_salsa_multiple_functions() {
    let op_count = TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text = r#"
fn add(a, b) { a + b }
fn multiply(a, b) { a * b }
fn main() { print_line("test") }
"#
        ;
        let tree = parser.parse(text, None).expect("tree");
        let source = SourceCst::from_path(db, "multi.trb", text.into(), Some(tree));
        let module = lower_source_cst(db, source);

        let body = module.body(db);
        let blocks = body.blocks(db);
        if blocks.is_empty() {
            0
        } else {
            blocks[0].operations(db).len()
        }
    });

    assert_eq!(op_count, 3, "Should have 3 functions");
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
        let module1 = lower_source_cst(db, source1);
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
        let module2 = lower_source_cst(db, source2);
        module2.name(db).to_string()
    });

    // Both modules should have the same name "main"
    assert_eq!(module1_name, "main");
    assert_eq!(module2_name, "main");
}

#[test]
fn test_function_lowering() {
    use trunk_ir::dialect::func;

    let source = "fn main() { 1 + 2 }";
    TributeDatabaseImpl::default().attach(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(source, None).expect("tree");
        let source_file = SourceCst::from_path(db, "func_test.trb", source.into(), Some(tree));
        let module = lower_source_cst(db, source_file);

        let body = module.body(db);
        let blocks = body.blocks(db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(db);
        assert_eq!(ops.len(), 1);

        // Check that the operation is a func.func
        let func_op = func::Func::from_operation(db, ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(db), "main");
    });
}
