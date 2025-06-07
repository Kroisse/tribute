// Example showing how to use the Salsa database for the Tribute language

use salsa::Setter;
use tribute::{parse_source_file, parse_str, SourceFile, TributeDatabaseImpl};

fn main() {
    // Example 1: Basic database usage
    basic_database_usage();

    // Example 2: Incremental compilation demonstration
    incremental_compilation_demo();

    // Example 3: Error handling with diagnostics
    error_handling_demo();
}

fn basic_database_usage() {
    println!("=== Basic Database Usage ===");

    // Create a database instance
    let db = TributeDatabaseImpl::default();

    // Parse some Tribute code
    let source_code = r#"
        (define greeting "Hello, Salsa!")
        (+ 1 2 3)
        (println greeting)
    "#;

    // Method 1: Using the convenience function
    let (program, diagnostics) = parse_str(&db, "example.trb", source_code);

    println!("Parsed {} expressions", program.items(&db).len());
    println!("Found {} diagnostics", diagnostics.len());

    // Display the parsed expressions
    for (i, expr) in program.items(&db).iter().enumerate() {
        println!("  Expression {}: {}", i + 1, expr.expr(&db).0);
    }

    println!();
}

fn incremental_compilation_demo() {
    println!("=== Incremental Compilation Demo ===");

    let mut db = TributeDatabaseImpl::default();

    // Create a source file
    let source_file = SourceFile::new(&db, "math.trb".into(), "(+ 1 2)".to_string());

    // Parse it
    println!("Initial parsing...");
    let program1 = parse_source_file(&db, source_file);
    println!("Parsed {} expressions", program1.items(&db).len());

    // Modify the source file
    println!("Modifying source...");
    source_file
        .set_text(&mut db)
        .to("(* 3 (+ 1 2))".to_string());

    // Parse again - Salsa will automatically detect the change and recompute
    let program2 = parse_source_file(&db, source_file);
    println!(
        "Parsed {} expressions after modification",
        program2.items(&db).len()
    );

    // Parse again without changes - this should use the cached result
    let program3 = parse_source_file(&db, source_file);
    println!("Cached result identical: {}", program2 == program3);

    println!();
}

fn error_handling_demo() {
    println!("=== Error Handling Demo ===");

    let db = TributeDatabaseImpl::default();

    // Try to parse some invalid syntax
    let invalid_code = "this is not valid ( syntax";
    let _source_file = SourceFile::new(&db, "invalid.trb".into(), invalid_code.to_string());

    let (program, diagnostics) = parse_str(&db, "invalid.trb", invalid_code);

    println!(
        "Parsed {} expressions from invalid code",
        program.items(&db).len()
    );

    if !diagnostics.is_empty() {
        println!("Diagnostics found:");
        for diagnostic in &diagnostics {
            println!(
                "  [{}] {} (at {}..{})",
                diagnostic.severity, diagnostic.message, diagnostic.span.start, diagnostic.span.end
            );
        }
    }

    println!();
}
