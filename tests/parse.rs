use insta::assert_ron_snapshot;
use salsa::Setter;
use std::path::Path;
use tribute::{diagnostics, parse, parse_source_file, SourceFile, TributeDatabaseImpl};

pub fn parse_file(path: &Path) -> Vec<(tribute::ast::Expr, tribute::ast::SimpleSpan)> {
    let source = std::fs::read_to_string(path).expect("Failed to read file");

    parse(path, &source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hello_example() {
        let path = Path::new("lang-examples/hello.trb");
        let ast = parse_file(path);

        assert_ron_snapshot!(ast);
    }

    #[test]
    fn test_parse_calc_example() {
        let path = Path::new("lang-examples/calc.trb");
        let ast = parse_file(path);

        assert_ron_snapshot!(ast);
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
            .expressions(&db)
            .iter()
            .map(|tracked| (tracked.expr(&db).clone(), tracked.span(&db)))
            .collect();

        // Should have no diagnostics for valid file
        assert!(diags.is_empty());

        // Should parse the same as the legacy parser
        let legacy_ast = parse_file(path);
        assert_eq!(expressions, legacy_ast);

        assert_ron_snapshot!("salsa_hello", expressions);
    }

    #[test]
    fn test_salsa_incremental_computation() {
        let mut db = TributeDatabaseImpl::default();
        let source_file = SourceFile::new(&db, "test.trb".into(), "(+ 1 2)".to_string());

        // First parse
        let program1 = parse_source_file(&db, source_file);
        assert_eq!(program1.expressions(&db).len(), 1);
        let expr1_str = format!("{}", program1.expressions(&db)[0].expr(&db));

        // Modify source (simulating incremental computation)
        source_file.set_text(&mut db).to("(+ 1 2 3)".to_string());

        // Parse again - should recompute
        let program2 = parse_source_file(&db, source_file);
        assert_eq!(program2.expressions(&db).len(), 1);
        let expr2_str = format!("{}", program2.expressions(&db)[0].expr(&db));

        // Verify content changed
        assert_ne!(expr1_str, expr2_str);
        assert_eq!(expr1_str, "(+ 1 2)");
        assert_eq!(expr2_str, "(+ 1 2 3)");
    }
}
