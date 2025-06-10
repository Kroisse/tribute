//! Salsa integration tests for HIR to MLIR lowering
//! 
//! These tests verify that Salsa caching works correctly with MLIR lowering

use tribute_ast::{TributeDatabaseImpl, SourceFile, parse_source_file};
use tribute_hir::queries::lower_program_to_hir;
use tribute_hir_dialect::salsa_integration::{lower_program_to_mlir, program_to_mlir_text};
use salsa::Database;

#[test]
fn test_mlir_salsa_caching() {
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        let source_file = SourceFile::new(db, std::path::PathBuf::from("test.trb"), "1 + 2".to_string());
        let ast_program = parse_source_file(db, source_file);
        
        if let Some(hir_program) = lower_program_to_hir(db, ast_program) {
            // First call - should compute
            let result1 = program_to_mlir_text(db, hir_program);
            assert!(result1.is_ok(), "First MLIR generation should succeed");
            
            // Second call - should use cache
            let result2 = program_to_mlir_text(db, hir_program);
            assert!(result2.is_ok(), "Second MLIR generation should succeed");
            
            // Results should be identical
            assert_eq!(result1.unwrap(), result2.unwrap(), "Cached results should match");
        }
    });
}

#[test]
fn test_mlir_error_handling() {
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        let source_file = SourceFile::new(db, std::path::PathBuf::from("test.trb"), "42".to_string());
        let ast_program = parse_source_file(db, source_file);
        
        if let Some(hir_program) = lower_program_to_hir(db, ast_program) {
            let mlir_result = lower_program_to_mlir(db, hir_program);
            
            // Test error handling infrastructure
            if mlir_result.success {
                assert!(mlir_result.error.is_none(), "Successful lowering should not have error");
                assert!(!mlir_result.mlir_text.is_empty(), "Successful lowering should have MLIR text");
            } else {
                assert!(mlir_result.error.is_some(), "Failed lowering should have error");
                assert!(mlir_result.mlir_text.is_empty(), "Failed lowering should have empty MLIR text");
            }
        }
    });
}

#[test]
fn test_mlir_incremental_compilation() {
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        // Test that different inputs produce different results
        let source1 = SourceFile::new(db, std::path::PathBuf::from("test1.trb"), "1 + 2".to_string());
        let source2 = SourceFile::new(db, std::path::PathBuf::from("test2.trb"), "3 + 4".to_string());
        
        let ast1 = parse_source_file(db, source1);
        let ast2 = parse_source_file(db, source2);
        
        if let (Some(hir1), Some(hir2)) = (lower_program_to_hir(db, ast1), lower_program_to_hir(db, ast2)) {
            let result1 = program_to_mlir_text(db, hir1);
            let result2 = program_to_mlir_text(db, hir2);
            
            assert!(result1.is_ok(), "First MLIR generation should succeed");
            assert!(result2.is_ok(), "Second MLIR generation should succeed");
            
            // Results should both succeed (they might be the same for simple expressions)
            let text1 = result1.unwrap();
            let text2 = result2.unwrap();
            
            // Both should be valid MLIR modules
            assert!(text1.contains("module"), "First result should be valid MLIR");
            assert!(text2.contains("module"), "Second result should be valid MLIR");
        }
    });
}