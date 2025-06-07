use tribute::{eval_expr, Environment, Value};
use tribute::ast::{Expr, SimpleSpan};
use tribute_ast::TributeDatabaseImpl;

type Spanned<T> = (T, SimpleSpan);

const SPAN: SimpleSpan = SimpleSpan::new(0, 0);

fn expr_number(n: i64) -> Spanned<Expr> {
    (Expr::Number(n), SPAN)
}

fn expr_string(s: &str) -> Spanned<Expr> {
    (Expr::String(s.to_string()), SPAN)
}

fn expr_ident(s: &str) -> Spanned<Expr> {
    (Expr::Identifier(s.to_string()), SPAN)
}

fn expr_list(exprs: Vec<Spanned<Expr>>) -> Spanned<Expr> {
    (Expr::List(exprs), SPAN)
}

// Helper function to evaluate a calculator expression: (op a b)
fn eval_calc_expr(op: &str, a: i64, b: i64) -> i64 {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    let calc_expr = expr_list(vec![
        expr_ident(op),
        expr_number(a),
        expr_number(b),
    ]);

    let result = eval_expr(&db, &mut env, &calc_expr.0).unwrap();
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
    let mut env = Environment::toplevel();

    let div_zero_expr = expr_list(vec![
        expr_ident("/"),
        expr_number(10),
        expr_number(0),
    ]);

    let result = eval_expr(&db, &mut env, &div_zero_expr.0);
    assert!(result.is_err(), "Division by zero should return an error");
}

#[test]
fn test_string_operations() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Test split function
    let split_expr = expr_list(vec![
        expr_ident("split"),
        expr_string(" "),
        expr_string("5 + 3"),
    ]);

    let result = eval_expr(&db, &mut env, &split_expr.0).unwrap();
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
    let mut env = Environment::toplevel();

    // First create a list
    let split_expr = expr_list(vec![
        expr_ident("split"),
        expr_string(" "),
        expr_string("10 - 4"),
    ]);
    let list_result = eval_expr(&db, &mut env, &split_expr.0).unwrap();
    env.bind("test_list".to_string(), list_result);

    // Test getting each element
    for (i, expected) in ["10", "-", "4"].iter().enumerate() {
        let get_expr = expr_list(vec![
            expr_ident("get"),
            expr_number(i as i64),
            expr_ident("test_list"),
        ]);

        let result = eval_expr(&db, &mut env, &get_expr.0).unwrap();
        assert!(matches!(result, Value::String(ref s) if s == expected));
    }
}

#[test]
fn test_to_number_function() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    let to_num_expr = expr_list(vec![
        expr_ident("to_number"),
        expr_string("42"),
    ]);

    let result = eval_expr(&db, &mut env, &to_num_expr.0).unwrap();
    assert!(matches!(result, Value::Number(42)));
}

#[test]
fn test_match_case_with_operators() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Test match with "+" operator
    let match_expr = expr_list(vec![
        expr_ident("match"),
        expr_string("+"),
        expr_list(vec![
            expr_ident("case"),
            expr_string("+"),
            expr_list(vec![
                expr_ident("+"),
                expr_number(5),
                expr_number(3),
            ]),
        ]),
        expr_list(vec![
            expr_ident("case"),
            expr_string("-"),
            expr_list(vec![
                expr_ident("-"),
                expr_number(5),
                expr_number(3),
            ]),
        ]),
    ]);

    let result = eval_expr(&db, &mut env, &match_expr.0).unwrap();
    assert!(matches!(result, Value::Number(8)));
}
