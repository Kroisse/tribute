use std::path::Path;

use salsa::{Database as _, Setter as _};
use tribute::{Diagnostic, SourceFile, TributeDatabaseImpl, parse_source_file};

#[test]
fn test_salsa_database_examples() {
    // Example source code
    let examples = vec![
        (
            Path::new("hello.trb"),
            r#"fn main() { print_line("Hello, World!") }"#,
        ),
        (Path::new("calc.trb"), r#"fn main() { 1 + 2 + 3 }"#),
        (
            Path::new("complex.trb"),
            r#"
fn factorial(n) {
  case n {
    0 -> 1,
    _ -> n * factorial(n - 1)
  }
}

fn main() {
  factorial(5)
}
"#,
        ),
        (
            Path::new("invalid.trb"),
            r#"fn main() { invalid syntax here }"#,
        ),
    ];

    for (filename, source_code) in examples {
        // Use attach pattern for test isolation
        let (expr_count, diag_count) = TributeDatabaseImpl::default().attach(|db| {
            let source_file = SourceFile::new(db, filename.to_path_buf(), source_code.to_string());
            let program = parse_source_file(db, source_file);
            let diagnostics = parse_source_file::accumulated::<Diagnostic>(db, source_file);

            // Extract data that doesn't depend on database lifetime
            (program.items(db).len(), diagnostics.len())
        });

        // Verify parsing results
        assert!(
            expr_count > 0,
            "Should parse at least one expression for {}",
            filename.display()
        );

        match filename.file_name().unwrap().to_str().unwrap() {
            "hello.trb" | "calc.trb" => {
                assert_eq!(expr_count, 1, "Simple examples should have 1 function");
                assert_eq!(diag_count, 0, "Valid syntax should have no diagnostics");
            }
            "complex.trb" => {
                assert_eq!(expr_count, 2, "Complex example should have 2 functions");
                assert_eq!(diag_count, 0, "Valid syntax should have no diagnostics");
            }
            "invalid.trb" => {
                // Invalid syntax should still parse as a function with invalid content
                assert_eq!(
                    expr_count, 1,
                    "Invalid syntax should still produce 1 function"
                );
            }
            _ => {}
        }
    }
}

#[test]
fn test_salsa_incremental_computation_detailed() {
    // Demonstrate incremental computation
    let mut db = TributeDatabaseImpl::default();
    let source_file = SourceFile::new(
        &db,
        "incremental.trb".into(),
        "fn main() { 1 + 2 }".to_string(),
    );

    // Initial parse
    let program1 = parse_source_file(&db, source_file);
    assert_eq!(program1.items(&db).len(), 1);
    let _program1_str = format!("{:?}", program1);

    // Modify the source file
    source_file
        .set_text(&mut db)
        .to("fn main() { 1 + 2 + 3 + 4 }".to_string());

    // Parse again - should recompute
    let program2 = parse_source_file(&db, source_file);
    assert_eq!(program2.items(&db).len(), 1);
    let program2_str = format!("{:?}", program2);

    // Parse again without changes - should use cached result
    let program3 = parse_source_file(&db, source_file);
    let program3_str = format!("{:?}", program3);

    // Verify that incremental compilation works
    // The exact comparison depends on Salsa's caching behavior

    // Check that cached results are the same by comparing their content
    assert_eq!(
        program2_str, program3_str,
        "Programs should be identical (cached result)"
    );
}

#[test]
fn test_salsa_diagnostics_collection() {
    // Test with valid code - should have no diagnostics
    let (valid_expr_count, valid_diag_count) = TributeDatabaseImpl::default().attach(|db| {
        let valid_source =
            SourceFile::new(db, "valid.trb".into(), "fn main() { 1 + 2 }".to_string());
        let valid_program = parse_source_file(db, valid_source);
        let valid_diagnostics = parse_source_file::accumulated::<Diagnostic>(db, valid_source);

        (valid_program.items(db).len(), valid_diagnostics.len())
    });

    assert_eq!(valid_expr_count, 1);
    assert_eq!(
        valid_diag_count, 0,
        "Valid code should produce no diagnostics"
    );

    // Test multiple functions
    let (multi_expr_count, multi_diag_count) = TributeDatabaseImpl::default().attach(|db| {
        let multi_source = SourceFile::new(
            db,
            "multi.trb".into(),
            r#"
fn add(a, b) { a + b }
fn multiply(a, b) { a * b }  
fn main() { print_line("test") }
"#
            .to_string(),
        );
        let multi_program = parse_source_file(db, multi_source);
        let multi_diagnostics = parse_source_file::accumulated::<Diagnostic>(db, multi_source);

        (multi_program.items(db).len(), multi_diagnostics.len())
    });

    assert_eq!(multi_expr_count, 3);
    assert_eq!(
        multi_diag_count, 0,
        "Valid multi-function code should produce no diagnostics"
    );
}

#[test]
fn test_salsa_database_isolation() {
    // Test that different database instances are isolated
    let program1_str = TributeDatabaseImpl::default().attach(|db| {
        let source1 = SourceFile::new(db, "test1.trb".into(), "fn main() { 1 + 2 }".to_string());
        let program1 = parse_source_file(db, source1);

        assert_eq!(program1.items(db).len(), 1);
        format!("{:?}", program1)
    });

    let program2_str = TributeDatabaseImpl::default().attach(|db| {
        let source2 = SourceFile::new(db, "test2.trb".into(), "fn main() { 3 * 4 }".to_string());
        let program2 = parse_source_file(db, source2);

        assert_eq!(program2.items(db).len(), 1);
        format!("{:?}", program2)
    });

    assert_ne!(
        program1_str, program2_str,
        "Different databases should produce different results"
    );
}

#[test]
fn test_salsa_function_tracking() {
    let source = "fn main() { 1 + 2 }";
    let (item_count, program_str) = TributeDatabaseImpl::default().attach(|db| {
        let source_file = SourceFile::new(db, "span_test.trb".into(), source.to_string());
        let program = parse_source_file(db, source_file);

        assert_eq!(program.items(db).len(), 1);

        (program.items(db).len(), format!("{:?}", program))
    });

    // Verify results
    assert_eq!(item_count, 1);
    assert!(program_str.contains("main"));
}
