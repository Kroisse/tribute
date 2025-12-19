//! Example showing how to use the Salsa database for the Tribute language.
//!
//! This example demonstrates the CST→TrunkIR lowering pipeline.

use salsa::Setter;
use tribute::{SourceFile, TributeDatabaseImpl, lower_source_file, lower_str};
use tribute_trunk_ir::DialectOp;
use tribute_trunk_ir::dialect::func;

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

    // Parse some Tribute code
    let source_code = r#"
        fn main() {
            let greeting = "Hello, Salsa!"
            print_line(greeting)
        }
    "#;

    // Lower to TrunkIR using the convenience function
    let module = lower_str(&db, "example.tr", source_code);

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

    // Create a source file
    let source_file = SourceFile::from_path(&db, "math.trb", "fn main() { 1 + 2 }".to_string());

    // Lower it
    println!("Initial lowering...");
    let module1 = lower_source_file(&db, source_file);
    let body1 = module1.body(&db);
    let blocks1 = body1.blocks(&db);
    if !blocks1.is_empty() {
        println!("Lowered {} operations", blocks1[0].operations(&db).len());
    }

    // Modify the source file
    println!("Modifying source...");
    source_file
        .set_text(&mut db)
        .to("fn main() { 3 * (1 + 2) }".to_string());

    // Lower again - Salsa will automatically detect the change and recompute
    let module2 = lower_source_file(&db, source_file);
    let body2 = module2.body(&db);
    let blocks2 = body2.blocks(&db);
    if !blocks2.is_empty() {
        println!(
            "Lowered {} operations after modification",
            blocks2[0].operations(&db).len()
        );
    }

    // Lower again without changes - this should use the cached result
    let module3 = lower_source_file(&db, source_file);
    println!(
        "Cached result identical: {}",
        module2.body(&db).blocks(&db).len() == module3.body(&db).blocks(&db).len()
    );

    println!();
}
