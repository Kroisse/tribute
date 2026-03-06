//! Example showing how to use the arena IR pipeline for the Tribute language.
//!
//! This example demonstrates the AST-based compilation pipeline that produces
//! arena IR (`IrContext` + `ArenaModule`) instead of Salsa-tracked IR.

use salsa::Setter;
use tree_sitter::Parser;
use tribute::{SourceCst, TributeDatabaseImpl, compile_frontend_to_arena};
use trunk_ir::Symbol;
use trunk_ir::arena::Attribute;

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

    // Lower to arena IR using the convenience function
    let tree = parser.parse(source_code, None).expect("tree");
    let source = SourceCst::from_path(&db, "example.tr", source_code.into(), Some(tree));

    let Some((ctx, arena_module)) = compile_frontend_to_arena(&db, source) else {
        println!("Compilation failed");
        return;
    };

    // Print module name if available
    if let Some(name) = arena_module.name(&ctx) {
        println!("Module name: {}", name);
    }

    // Get the top-level operations from the module
    let ops = arena_module.ops(&ctx);
    println!("Lowered {} top-level operations", ops.len());

    // Display the lowered functions
    for (i, op_ref) in ops.iter().enumerate() {
        let op_data = ctx.op(*op_ref);
        if op_data.dialect == Symbol::new("func") && op_data.name == Symbol::new("func") {
            if let Some(Attribute::Symbol(name)) = op_data.attributes.get(&Symbol::new("sym_name"))
            {
                println!("  Operation {}: func.func \"{}\"", i + 1, name);
            }
        } else {
            println!(
                "  Operation {}: {}.{}",
                i + 1,
                op_data.dialect,
                op_data.name
            );
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
    let Some((ctx1, m1)) = compile_frontend_to_arena(&db, source_file) else {
        println!("Initial compilation failed");
        return;
    };
    let ops1 = m1.ops(&ctx1);
    println!("Lowered {} operations", ops1.len());

    // Modify the source file
    println!("Modifying source...");
    let updated_text = "fn main() { 3 * (1 + 2) }";
    let updated_tree = parser.parse(updated_text, None).expect("tree");
    source_file.set_text(&mut db).to(updated_text.into());
    source_file.set_tree(&mut db).to(Some(updated_tree));

    // Lower again - Salsa will automatically detect the change and recompute
    // the internal tracked queries, then produce fresh arena IR.
    let Some((ctx2, m2)) = compile_frontend_to_arena(&db, source_file) else {
        println!("Recompilation failed");
        return;
    };
    let ops2 = m2.ops(&ctx2);
    println!("Lowered {} operations after modification", ops2.len());

    // Lower again without changes - internally Salsa caches the tracked
    // queries, so this recomputation is fast.
    let Some((ctx3, m3)) = compile_frontend_to_arena(&db, source_file) else {
        println!("Cached compilation failed");
        return;
    };
    let ops3 = m3.ops(&ctx3);
    println!("Cached result identical: {}", ops2.len() == ops3.len());

    // We can also inspect module names
    println!("Module name: {:?}", m3.name(&ctx3));

    println!();
}
