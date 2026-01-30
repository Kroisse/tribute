//! Example showing how to use the Salsa database for the Tribute language.
//!
//! This example demonstrates the AST-based compilation pipeline.

use salsa::Setter;
use tree_sitter::Parser;
use tribute::{SourceCst, TributeDatabaseImpl, parse_and_lower_ast};
use trunk_ir::DialectOp;
use trunk_ir::dialect::func;

fn main() {
    // Example 1: Basic database usage
    basic_database_usage();

    // Example 2: Incremental compilation demonstration
    incremental_compilation_demo();
}

fn basic_database_usage() {
    println!("=== Basic Database Usage ===");

    // Create a database instance
    let db = TributeDatabaseImpl::default();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");

    // Parse some Tribute code
    let source_code = r#"
        fn main() {
            let greeting = "Hello, Salsa!"
            print_line(greeting)
        }
    "#;

    // Lower to TrunkIR using the convenience function
    let tree = parser.parse(source_code, None).expect("tree");
    let source = SourceCst::from_path(&db, "example.tr", source_code.into(), Some(tree));
    let module = parse_and_lower_ast(&db, source);

    // Get the operations from the module
    let body = module.body(&db);
    let blocks = body.blocks(&db);

    if !blocks.is_empty() {
        let ops = blocks[0].operations(&db);
        println!("Lowered {} top-level operations", ops.len());

        // Display the lowered functions
        for (i, op) in ops.iter().enumerate() {
            if let Ok(func_op) = func::Func::from_operation(&db, *op) {
                println!("  Operation {}: func.func \"{}\"", i + 1, func_op.name(&db));
            } else {
                println!("  Operation {}: {:?}", i + 1, op);
            }
        }
    }

    println!();
}

fn incremental_compilation_demo() {
    println!("=== Incremental Compilation Demo ===");

    let mut db = TributeDatabaseImpl::default();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");

    // Create a source file
    let initial_text = "fn main() { 1 + 2 }";
    let initial_tree = parser.parse(initial_text, None).expect("tree");
    let source_file =
        SourceCst::from_path(&db, "math.trb", initial_text.into(), Some(initial_tree));

    // Lower it
    println!("Initial lowering...");
    let module1 = parse_and_lower_ast(&db, source_file);
    let body1 = module1.body(&db);
    let blocks1 = body1.blocks(&db);
    if !blocks1.is_empty() {
        println!("Lowered {} operations", blocks1[0].operations(&db).len());
    }

    // Modify the source file
    println!("Modifying source...");
    let updated_text = "fn main() { 3 * (1 + 2) }";
    let updated_tree = parser.parse(updated_text, None).expect("tree");
    source_file.set_text(&mut db).to(updated_text.into());
    source_file.set_tree(&mut db).to(Some(updated_tree));

    // Lower again - Salsa will automatically detect the change and recompute
    let module2 = parse_and_lower_ast(&db, source_file);
    let body2 = module2.body(&db);
    let blocks2 = body2.blocks(&db);
    if !blocks2.is_empty() {
        println!(
            "Lowered {} operations after modification",
            blocks2[0].operations(&db).len()
        );
    }

    // Lower again without changes - this should use the cached result
    let module3 = parse_and_lower_ast(&db, source_file);
    println!(
        "Cached result identical: {}",
        module2.body(&db).blocks(&db).len() == module3.body(&db).blocks(&db).len()
    );

    println!();
}
