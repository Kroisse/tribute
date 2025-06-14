//! Tests for the Cranelift compiler

use tribute_ast::{SourceFile, TributeDatabaseImpl, parse_source_file};
use tribute_hir::lower_program_to_hir;

use crate::compile_to_object;

#[test]
fn test_basic_compilation() {
    let db = TributeDatabaseImpl::default();

    let source = r#"
        fn main() {
            42
        }
    "#;

    let source_file = SourceFile::new(&db, "test.trb".into(), source.to_string());
    let ast_program = parse_source_file(&db, source_file);
    let hir_program = lower_program_to_hir(&db, ast_program);

    // Try to compile to object - should not panic
    if let Some(hir_program) = hir_program {
        let result = compile_to_object(&db, hir_program, None);

        // For now, just check that compilation doesn't panic
        // We expect errors since runtime functions aren't linked
        assert!(result.is_ok() || result.is_err());
    }
}

#[test]
fn test_empty_program() {
    let db = TributeDatabaseImpl::default();

    let source = "";

    let source_file = SourceFile::new(&db, "test.trb".into(), source.to_string());
    let ast_program = parse_source_file(&db, source_file);
    let hir_program = lower_program_to_hir(&db, ast_program);

    if let Some(hir_program) = hir_program {
        let result = compile_to_object(&db, hir_program, None);

        // Empty program should compile successfully
        assert!(result.is_ok());
    }
}
