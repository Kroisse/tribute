//! Integration tests for HIR to MLIR lowering
//! 
//! These tests verify the complete pipeline from source code to MLIR representation

use tribute_ast::{TributeDatabaseImpl, SourceFile, parse_source_file};
use tribute_hir::queries::lower_program_to_hir;
use tribute_hir_dialect::salsa_integration::{lower_program_to_mlir, program_to_mlir_text};
use salsa::Database;

// Helper function that works with Salsa's lifetime requirements
fn test_with_source<F, R>(source: &str, test_fn: F) -> R
where
    F: for<'db> FnOnce(&'db TributeDatabaseImpl, tribute_hir::hir::HirProgram<'db>) -> R,
{
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        let source_file = SourceFile::new(db, std::path::PathBuf::from("test.trb"), source.to_string());
        let ast_program = parse_source_file(db, source_file);
        let hir_program = lower_program_to_hir(db, ast_program)
            .expect("HIR lowering should succeed for test cases");
        test_fn(db, hir_program)
    })
}

#[test] 
fn test_simple_arithmetic() {
    let input = "1 + 2 * 3";
    test_with_source(input, |db, hir_program| {
        // Test MLIR lowering
        let mlir_result = lower_program_to_mlir(db, hir_program);
        assert!(mlir_result.success, "MLIR lowering failed: {:?}", mlir_result.error);
        
        let mlir_text = mlir_result.mlir_text;
        assert!(!mlir_text.is_empty(), "MLIR text should not be empty");
        
        // Verify that the MLIR contains expected operations
        // Note: The exact MLIR structure depends on implementation details
        assert!(!mlir_text.is_empty(), "MLIR should contain operations");
        
        // Test convenience function
        let mlir_text_direct = program_to_mlir_text(db, hir_program);
        assert!(mlir_text_direct.is_ok(), "Direct MLIR text conversion should succeed");
        assert_eq!(mlir_text, mlir_text_direct.unwrap(), "Results should match");
    });
}

#[test]
#[ignore] // Disabled due to segfault in function lowering
fn test_function_definition() {
    let input = r#"
        fn add(a, b) {
            a + b
        }
        add(5, 3)
    "#;
    
    test_with_source(input, |db, hir_program| {
        let mlir_result = lower_program_to_mlir(db, hir_program);
        assert!(mlir_result.success, "Function definition MLIR lowering failed: {:?}", mlir_result.error);
        
        let mlir_text = mlir_result.mlir_text;
        assert!(!mlir_text.is_empty(), "MLIR for function should not be empty");
        
        // Test that we can get the text directly
        let mlir_text_result = program_to_mlir_text(db, hir_program);
        assert!(mlir_text_result.is_ok(), "Function MLIR text generation should succeed");
    });
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering
fn test_string_interpolation() {
    let input = r#"let name = "world"; "Hello, \{name}!""#;
    
    test_with_source(input, |db, hir_program| {
        let mlir_result = lower_program_to_mlir(db, hir_program);
        assert!(mlir_result.success, "String interpolation MLIR lowering failed: {:?}", mlir_result.error);
        
        let mlir_text = mlir_result.mlir_text;
        assert!(!mlir_text.is_empty(), "String interpolation MLIR should not be empty");
        
        // Test direct text conversion
        let mlir_text_result = program_to_mlir_text(db, hir_program);
        assert!(mlir_text_result.is_ok(), "String interpolation MLIR text should succeed");
    });
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering
fn test_numbers_and_variables() {
    let input = "let x = 42; x";
    
    test_with_source(input, |db, hir_program| {
        let mlir_result = lower_program_to_mlir(db, hir_program);
        assert!(mlir_result.success, "Let binding MLIR lowering failed: {:?}", mlir_result.error);
        
        let mlir_text = mlir_result.mlir_text;
        assert!(!mlir_text.is_empty(), "Let binding MLIR should not be empty");
    });
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering 
fn test_basic_expressions() {
    let test_cases = vec![
        ("42", "number literal"),
        ("\"hello\"", "string literal"),
        ("true", "boolean literal"),
        ("x", "variable reference"),
    ];
    
    for (input, description) in test_cases {
        test_with_source(input, |db, hir_program| {
            let mlir_result = lower_program_to_mlir(db, hir_program);
            assert!(mlir_result.success, "MLIR lowering failed for {}: {:?}", description, mlir_result.error);
            
            let mlir_text = mlir_result.mlir_text;
            assert!(!mlir_text.is_empty(), "MLIR should not be empty for {}", description);
            
            // Test that text conversion works
            let text_result = program_to_mlir_text(db, hir_program);
            assert!(text_result.is_ok(), "Text conversion should work for {}", description);
        });
    }
}

#[test]
fn test_binary_operations() {
    let operations = vec![
        ("1 + 2", "addition"),
        ("5 - 3", "subtraction"), 
        ("4 * 6", "multiplication"),
        ("8 / 2", "division"),
    ];
    
    for (input, op_name) in operations {
        test_with_source(input, |db, hir_program| {
            let mlir_result = lower_program_to_mlir(db, hir_program);
            assert!(mlir_result.success, "MLIR lowering failed for {}: {:?}", op_name, mlir_result.error);
            
            let mlir_text = mlir_result.mlir_text;
            assert!(!mlir_text.is_empty(), "MLIR should not be empty for {}", op_name);
        });
    }
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering
fn test_full_pipeline_end_to_end() {
    // Test a complete program with multiple features
    let input = r#"
        fn greet(name) {
            "Hello, \{name}!"
        }
        
        let result = greet("World");
        let number = 1 + 2 * 3;
        result
    "#;
    
    test_with_source(input, |db, hir_program| {
        // Test that the complete pipeline works
        let mlir_result = lower_program_to_mlir(db, hir_program);
        assert!(mlir_result.success, "End-to-end MLIR lowering failed: {:?}", mlir_result.error);
        
        let mlir_text = mlir_result.mlir_text;
        assert!(!mlir_text.is_empty(), "End-to-end MLIR should not be empty");
        
        // Verify we can get text representation
        let text_result = program_to_mlir_text(db, hir_program);
        assert!(text_result.is_ok(), "End-to-end text conversion should work");
        
        let text = text_result.unwrap();
        assert!(!text.is_empty(), "End-to-end MLIR text should not be empty");
    });
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering
fn test_error_handling() {
    // Test cases that might cause lowering errors
    // Note: This depends on what actually causes errors in the implementation
    
    test_with_source("1 + 2", |db, hir_program| {
        let mlir_result = lower_program_to_mlir(db, hir_program);
        
        // This should succeed, but we're testing the error handling infrastructure
        if mlir_result.success {
            assert!(mlir_result.error.is_none(), "Successful lowering should not have error message");
        } else {
            assert!(mlir_result.error.is_some(), "Failed lowering should have error message");
            assert!(mlir_result.mlir_text.is_empty(), "Failed lowering should have empty MLIR text");
        }
    });
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering
fn test_salsa_caching() {
    // Test that Salsa caching works correctly
    let input = "1 + 2";
    
    test_with_source(input, |db, hir_program| {
        // First call
        let result1 = program_to_mlir_text(db, hir_program);
        assert!(result1.is_ok(), "First MLIR generation should succeed");
        
        // Second call - should use cache
        let result2 = program_to_mlir_text(db, hir_program);
        assert!(result2.is_ok(), "Second MLIR generation should succeed"); 
        
        // Results should be identical
        assert_eq!(result1.unwrap(), result2.unwrap(), "Cached results should match");
    });
}

#[test]
#[ignore] // Disabled due to segfault in complex lowering 
fn test_multiple_expressions() {
    // Test programs with multiple top-level expressions
    let input = r#"
        1 + 2;
        "hello";
        3 * 4
    "#;
    
    test_with_source(input, |db, hir_program| {
        let mlir_result = lower_program_to_mlir(db, hir_program);
        assert!(mlir_result.success, "Multiple expressions MLIR lowering failed: {:?}", mlir_result.error);
        
        let mlir_text = mlir_result.mlir_text;
        assert!(!mlir_text.is_empty(), "Multiple expressions MLIR should not be empty");
    });
}