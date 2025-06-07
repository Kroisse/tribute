use insta::assert_debug_snapshot;
use salsa::{Database as Db, Setter};
use std::path::Path;
use tribute::{diagnostics, parse, parse_source_file, SourceFile, TributeDatabaseImpl};
use tribute_hir::{lower_source_to_hir, HirProgram};

pub fn parse_file(path: &Path) -> Vec<(tribute::ast::Expr, tribute::ast::SimpleSpan)> {
    let source = std::fs::read_to_string(path).expect("Failed to read file");

    parse(path, &source)
}

pub fn parse_file_to_hir<'db>(db: &'db dyn Db, path: &Path) -> HirProgram<'db> {
    let source = std::fs::read_to_string(path).expect("Failed to read file");
    let source_file = SourceFile::new(db, path.to_path_buf(), source);

    lower_source_to_hir(db, source_file).expect("Failed to lower to HIR")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hello_example() {
        let path = Path::new("lang-examples/hello.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_calc_example() {
        let path = Path::new("lang-examples/calc.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_basic_example() {
        let path = Path::new("lang-examples/basic.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_functions_example() {
        let path = Path::new("lang-examples/functions.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_let_bindings_example() {
        let path = Path::new("lang-examples/let_bindings.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_let_advanced_example() {
        let path = Path::new("lang-examples/let_advanced.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_let_simple_example() {
        let path = Path::new("lang-examples/let_simple.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_let_with_function_example() {
        let path = Path::new("lang-examples/let_with_function.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_pattern_matching_example() {
        let path = Path::new("lang-examples/pattern_matching.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_parse_function_visibility_example() {
        let path = Path::new("lang-examples/function_visibility.trb");
        let ast = parse_file(path);

        assert_debug_snapshot!(ast);
    }

    #[test]
    fn test_salsa_parse_hello_example() {
        let path = Path::new("lang-examples/hello.trb");
        let source = std::fs::read_to_string(path).expect("Failed to read file");

        let db = TributeDatabaseImpl::default();
        let source_file = SourceFile::new(&db, path.to_path_buf(), source);

        let program = parse_source_file(&db, source_file);
        let diags = diagnostics(&db, source_file);

        // Convert to the same format for comparison
        let expressions: Vec<(tribute::ast::Expr, tribute::ast::SimpleSpan)> = program
            .items(&db)
            .iter()
            .map(|item| item.expr(&db).clone())
            .collect();

        // Should have no diagnostics for valid file
        assert!(diags.is_empty());

        // Should parse the same as the legacy parser
        let legacy_ast = parse_file(path);
        assert_eq!(expressions, legacy_ast);

        assert_debug_snapshot!("salsa_hello", expressions);
    }

    #[test]
    fn test_salsa_incremental_computation() {
        let mut db = TributeDatabaseImpl::default();
        let source_file = SourceFile::new(&db, "test.trb".into(), "(+ 1 2)".to_string());

        // First parse
        let program1 = parse_source_file(&db, source_file);
        assert_eq!(program1.items(&db).len(), 1);
        let expr1_str = format!("{}", program1.items(&db)[0].expr(&db).0);

        // Modify source (simulating incremental computation)
        source_file.set_text(&mut db).to("(+ 1 2 3)".to_string());

        // Parse again - should recompute
        let program2 = parse_source_file(&db, source_file);
        assert_eq!(program2.items(&db).len(), 1);
        let expr2_str = format!("{}", program2.items(&db)[0].expr(&db).0);

        // Verify content changed
        assert_ne!(expr1_str, expr2_str);
        assert_eq!(expr1_str, "(+ 1 2)");
        assert_eq!(expr2_str, "(+ 1 2 3)");
    }

    // HIR-based tests (replacing AST tests)
    #[test]
    fn test_hir_hello_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/hello.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_calc_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/calc.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_basic_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/basic.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_functions_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/functions.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_let_bindings_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/let_bindings.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_let_advanced_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/let_advanced.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_let_simple_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/let_simple.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_let_with_function_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/let_with_function.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_pattern_matching_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/pattern_matching.trb");
            let hir = parse_file_to_hir(db, path);
            assert_debug_snapshot!(hir);
        });
    }

    #[test]
    fn test_hir_function_visibility_example() {
        TributeDatabaseImpl::default().attach(|db| {
            let path = Path::new("lang-examples/function_visibility.trb");
            let hir = parse_file_to_hir(db, path);

            assert_debug_snapshot!(hir);
        });
    }
}
