use tribute::TributeDatabaseImpl;
use salsa::Database;

#[test]
fn test_escape_sequences_in_evaluation() {
    TributeDatabaseImpl::default().attach(|db| {
        // Test quote escaping
        let source = r#"(fn (main) (print_line "Hello \"World\""))"#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(_) => {} // print_line returns Unit
            Err(e) => panic!("Failed to evaluate quote escape: {}", e),
        }

        // Test backslash escaping  
        let source = r#"(fn (main) (print_line "Path: C:\\Users\\name"))"#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(_) => {} // print_line returns Unit
            Err(e) => panic!("Failed to evaluate backslash escape: {}", e),
        }

        // Test whitespace escaping
        let source = r#"(fn (main) (print_line "Line1\nLine2\tTabbed"))"#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(_) => {} // print_line returns Unit
            Err(e) => panic!("Failed to evaluate whitespace escape: {}", e),
        }
    });
}

#[test]
fn test_string_literals_in_function_arguments() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            (fn (echo s) s)
            (fn (main) (echo "Test \"quotes\" and \\backslash"))
        "#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "Test \"quotes\" and \\backslash");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate string literal in function: {}", e),
        }
    });
}

#[test]
fn test_string_literals_with_let_binding() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            (fn (main) 
                (let message "Hello\n\"Escaped\" World")
                message)
        "#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "Hello\n\"Escaped\" World");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate string literal with let: {}", e),
        }
    });
}

#[test]
fn test_string_literals_in_pattern_matching() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            (fn (test_string s)
                (match s
                    (case "hello\tworld" "tab found")
                    (case "hello\"world" "quote found")
                    (case _ "other")))
            (fn (main) (test_string "hello\tworld"))
        "#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "tab found");
            }
            Ok(other) => panic!("Expected String(\"tab found\"), got {:?}", other),
            Err(e) => panic!("Failed to evaluate string literal in match: {}", e),
        }
    });
}

#[test]
fn test_edge_cases() {
    TributeDatabaseImpl::default().attach(|db| {
        // Test empty string
        let source = r#"(fn (main) "")"#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "");
            }
            Ok(other) => panic!("Expected empty String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate empty string: {}", e),
        }

        // Test string with only escape sequences
        let source = r#"(fn (main) "\"\\\n\t")"#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "\"\\\n\t");
            }
            Ok(other) => panic!("Expected escape-only String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate escape-only string: {}", e),
        }

        // Test unknown escape sequences (should be preserved)
        let source = r#"(fn (main) "Unknown \\x escape")"#;
        match tribute::eval_with_hir(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "Unknown \\x escape");
            }
            Ok(other) => panic!("Expected unknown escape String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate unknown escape string: {}", e),
        }
    });
}