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
        fn main() {
            let greeting = "Hello, Salsa!"
            print_line(greeting)
        }
    "#;

    // Method 1: Using the convenience function
    let (program, diagnostics) = parse_str(&db, "example.trb", source_code);

    println!("Parsed {} functions", program.items(&db).len());
    println!("Found {} diagnostics", diagnostics.len());

    // Display the parsed functions
    for (i, item) in program.items(&db).iter().enumerate() {
        println!("  Item {}: {:?}", i + 1, item);
    }

    println!();
}

fn incremental_compilation_demo() {
    println!("=== Incremental Compilation Demo ===");

    let mut db = TributeDatabaseImpl::default();

    // Create a source file
    let source_file = SourceFile::new(&db, "math.trb".into(), "fn main() { 1 + 2 }".to_string());

    // Parse it
    println!("Initial parsing...");
    let program1 = parse_source_file(&db, source_file);
    println!("Parsed {} functions", program1.items(&db).len());

    // Modify the source file
    println!("Modifying source...");
    source_file
        .set_text(&mut db)
        .to("fn main() { 3 * (1 + 2) }".to_string());

    // Parse again - Salsa will automatically detect the change and recompute
    let program2 = parse_source_file(&db, source_file);
    println!(
        "Parsed {} functions after modification",
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
    let invalid_code = "fn invalid { this is not valid syntax }";
    let _source_file = SourceFile::new(&db, "invalid.trb".into(), invalid_code.to_string());

    let (program, diagnostics) = parse_str(&db, "invalid.trb", invalid_code);

    println!(
        "Parsed {} functions from invalid code",
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
