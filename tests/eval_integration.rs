use tribute::{eval_with_hir, Value};
use tribute_ast::TributeDatabaseImpl;

#[test]
fn test_arithmetic_operations() {
    let db = TributeDatabaseImpl::default();

    // Test addition
    let source = "(fn (main) (+ 5 3))";
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(8)));

    // Test subtraction
    let source = "(fn (main) (- 10 4))";
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(6)));

    // Test multiplication
    let source = "(fn (main) (* 6 7))";
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(42)));

    // Test division
    let source = "(fn (main) (/ 15 3))";
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(5)));

    // Test division by zero
    let source = "(fn (main) (/ 10 0))";
    let result = eval_with_hir(&db, "test.trb", source);
    assert!(result.is_err());
}

#[test]
fn test_string_manipulation() {
    let db = TributeDatabaseImpl::default();

    // Test split
    let source = r#"(fn (main) (split " " "hello world test"))"#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    if let Value::List(items) = result {
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], Value::String(ref s) if s == "hello"));
        assert!(matches!(items[1], Value::String(ref s) if s == "world"));
        assert!(matches!(items[2], Value::String(ref s) if s == "test"));
    } else {
        panic!("Expected List value");
    }

    // Test trim_right
    let source = r#"(fn (main) (trim_right "hello world   "))"#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "hello world"));

    // Test to_number
    let source = r#"(fn (main) (to_number "42"))"#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(42)));

    // Test to_number with invalid string
    let source = r#"(fn (main) (to_number "not_a_number"))"#;
    let result = eval_with_hir(&db, "test.trb", source);
    assert!(result.is_err());
}

#[test]
fn test_collection_functions() {
    let db = TributeDatabaseImpl::default();

    // Test get with split in one expression
    let source = r#"(fn (main) (get 1 (split " " "a b c")))"#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "b"));

    // Test get with out of bounds index
    let source = r#"(fn (main) (get 10 (split " " "a b c")))"#;
    let result = eval_with_hir(&db, "test.trb", source);
    assert!(result.is_err());
}

#[test]
fn test_match_case() {
    let db = TributeDatabaseImpl::default();

    // Test match with string patterns
    let source = r#"
        (fn (main) 
          (match "+"
            (case "+" 100)
            (case "-" 200)))
    "#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(100)));

    // Test match with number patterns
    let source = r#"
        (fn (main)
          (match 42
            (case 41 "wrong")
            (case 42 "correct")))
    "#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "correct"));

    // Test match with no matching case
    let source = r#"
        (fn (main)
          (match "unknown"
            (case "known" 1)))
    "#;
    let result = eval_with_hir(&db, "test.trb", source);
    assert!(result.is_err());
}

#[test]
fn test_function_definition_and_call() {
    let db = TributeDatabaseImpl::default();

    // Define and call a function in HIR
    let source = r#"
        (fn (add_one x) (+ x 1))
        (fn (main) (add_one 5))
    "#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(6)));
}

#[test]
fn test_let_binding() {
    let db = TributeDatabaseImpl::default();

    // Test let binding and variable usage
    let source = r#"
        (fn (main) 
          (let x 42)
          x)
    "#;
    let result = eval_with_hir(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(42)));
}