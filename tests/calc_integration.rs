use tribute::{eval_str, Value};
use tribute_ast::TributeDatabaseImpl;

// Helper function to evaluate a calculator expression: a op b
fn eval_calc_expr(op: &str, a: i64, b: i64) -> i64 {
    let db = TributeDatabaseImpl::default();
    let source = format!("fn main() {{ {} {} {} }}", a, op, b);

    let result = eval_str(&db, "test.trb", &source).unwrap();
    match result {
        Value::Number(n) => n,
        _ => panic!("Expected number result"),
    }
}

#[test]
fn test_direct_addition() {
    assert_eq!(eval_calc_expr("+", 5, 3), 8);
    assert_eq!(eval_calc_expr("+", 123, 456), 579);
    assert_eq!(eval_calc_expr("+", 0, 0), 0);
}

#[test]
fn test_direct_subtraction() {
    assert_eq!(eval_calc_expr("-", 10, 4), 6);
    assert_eq!(eval_calc_expr("-", 5, 10), -5);
    assert_eq!(eval_calc_expr("-", 100, 1), 99);
}

#[test]
fn test_direct_multiplication() {
    assert_eq!(eval_calc_expr("*", 6, 7), 42);
    assert_eq!(eval_calc_expr("*", 0, 999), 0);
    assert_eq!(eval_calc_expr("*", -3, 4), -12);
}

#[test]
fn test_direct_division() {
    assert_eq!(eval_calc_expr("/", 15, 3), 5);
    assert_eq!(eval_calc_expr("/", 100, 10), 10);
    assert_eq!(eval_calc_expr("/", 7, 2), 3); // Integer division
}

#[test]
fn test_division_by_zero() {
    let db = TributeDatabaseImpl::default();
    let source = "fn main() { 10 / 0 }";

    let result = eval_str(&db, "test.trb", source);
    assert!(result.is_err(), "Division by zero should return an error");
}

#[test]
fn test_string_operations() {
    let db = TributeDatabaseImpl::default();
    let source = r#"fn main() { split(" ", "5 + 3") }"#;

    let result = eval_str(&db, "test.trb", source).unwrap();
    if let Value::List(items) = result {
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], Value::String(ref s) if s == "5"));
        assert!(matches!(items[1], Value::String(ref s) if s == "+"));
        assert!(matches!(items[2], Value::String(ref s) if s == "3"));
    } else {
        panic!("Expected List value");
    }
}

#[test]
fn test_get_function() {
    let db = TributeDatabaseImpl::default();

    // Test getting first element (index 0)
    let source0 = r#"fn main() { get(0, split(" ", "10 - 4")) }"#;
    let result = eval_str(&db, "test.trb", source0).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "10"));

    // Test getting second element (index 1)
    let source1 = r#"fn main() { get(1, split(" ", "10 - 4")) }"#;
    let result = eval_str(&db, "test.trb", source1).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "-"));

    // Test getting third element (index 2)
    let source2 = r#"fn main() { get(2, split(" ", "10 - 4")) }"#;
    let result = eval_str(&db, "test.trb", source2).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "4"));
}

#[test]
fn test_to_number_function() {
    let db = TributeDatabaseImpl::default();
    let source = r#"fn main() { to_number("42") }"#;

    let result = eval_str(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(42)));
}

#[test]
fn test_match_case_with_operators() {
    let db = TributeDatabaseImpl::default();
    let source = r#"
        fn main() {
          match "+" {
            "+" => 5 + 3,
            "-" => 5 - 3
          }
        }
    "#;

    let result = eval_str(&db, "test.trb", source).unwrap();
    assert!(matches!(result, Value::Number(8)));
}
