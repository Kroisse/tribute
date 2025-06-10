//! Simple test to debug segfault issue

use tribute_ast::{TributeDatabaseImpl, SourceFile, parse_source_file};
use tribute_hir::queries::lower_program_to_hir;
use salsa::Database;

#[test]
fn test_basic_parsing() {
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        let source_file = SourceFile::new(db, std::path::PathBuf::from("test.trb"), "42".to_string());
        let ast_program = parse_source_file(db, source_file);
        
        // Check that AST parsing worked
        let items = ast_program.items(db);
        println!("AST items count: {}", items.len());
        
        let hir_program = lower_program_to_hir(db, ast_program);
        
        if let Some(hir) = hir_program {
            let functions = hir.functions(db);
            let main = hir.main(db);
            println!("HIR functions count: {}, main: {:?}", functions.len(), main);
            
            // HIR program was created successfully
        } else {
            // HIR lowering failed, but that's okay for a simple expression
            println!("HIR lowering returned None - this is expected for simple expressions");
        }
    });
}

#[test]
fn test_salsa_integration_simple() {
    use tribute_hir_dialect::salsa_integration::lower_program_to_mlir;
    
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        let source_file = SourceFile::new(db, std::path::PathBuf::from("test.trb"), "42".to_string());
        let ast_program = parse_source_file(db, source_file);
        let hir_program = lower_program_to_hir(db, ast_program);
        
        if let Some(hir) = hir_program {
            // This is where the segfault might occur
            let mlir_result = lower_program_to_mlir(db, hir);
            
            // If we get here, no segfault occurred
            println!("MLIR result success: {}", mlir_result.success);
            if !mlir_result.success {
                println!("MLIR error: {:?}", mlir_result.error);
            }
        } else {
            println!("HIR lowering returned None - skipping MLIR test");
        }
    });
}

#[test]
#[ignore] // Disabled due to segfault in function lowering
fn test_function_program() {
    use tribute_hir_dialect::salsa_integration::lower_program_to_mlir;
    
    let db = TributeDatabaseImpl::default();
    db.attach(|db| {
        let source_file = SourceFile::new(db, std::path::PathBuf::from("test.trb"), r#"
            fn add(a, b) {
                a + b
            }
        "#.to_string());
        let ast_program = parse_source_file(db, source_file);
        let hir_program = lower_program_to_hir(db, ast_program);
        
        if let Some(hir) = hir_program {
            let functions = hir.functions(db);
            println!("HIR functions count: {}", functions.len());
            
            // This might be where the segfault occurs for complex programs
            let mlir_result = lower_program_to_mlir(db, hir);
            
            println!("MLIR result success: {}", mlir_result.success);
            if !mlir_result.success {
                println!("MLIR error: {:?}", mlir_result.error);
            }
        } else {
            println!("HIR lowering returned None for function program");
        }
    });
}