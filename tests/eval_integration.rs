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

#[test]
fn test_arithmetic_operations() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Test addition
    let add_expr = expr_list(vec![
        expr_ident("+"),
        expr_number(5),
        expr_number(3),
    ]);
    let result = eval_expr(&db, &mut env, &add_expr.0).unwrap();
    assert!(matches!(result, Value::Number(8)));

    // Test subtraction
    let sub_expr = expr_list(vec![
        expr_ident("-"),
        expr_number(10),
        expr_number(4),
    ]);
    let result = eval_expr(&db, &mut env, &sub_expr.0).unwrap();
    assert!(matches!(result, Value::Number(6)));

    // Test multiplication
    let mul_expr = expr_list(vec![
        expr_ident("*"),
        expr_number(6),
        expr_number(7),
    ]);
    let result = eval_expr(&db, &mut env, &mul_expr.0).unwrap();
    assert!(matches!(result, Value::Number(42)));

    // Test division
    let div_expr = expr_list(vec![
        expr_ident("/"),
        expr_number(15),
        expr_number(3),
    ]);
    let result = eval_expr(&db, &mut env, &div_expr.0).unwrap();
    assert!(matches!(result, Value::Number(5)));

    // Test division by zero
    let div_zero_expr = expr_list(vec![
        expr_ident("/"),
        expr_number(10),
        expr_number(0),
    ]);
    let result = eval_expr(&db, &mut env, &div_zero_expr.0);
    assert!(result.is_err());
}

#[test]
fn test_string_manipulation() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Test split
    let split_expr = expr_list(vec![
        expr_ident("split"),
        expr_string(" "),
        expr_string("hello world test"),
    ]);
    let result = eval_expr(&db, &mut env, &split_expr.0).unwrap();
    if let Value::List(items) = result {
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], Value::String(ref s) if s == "hello"));
        assert!(matches!(items[1], Value::String(ref s) if s == "world"));
        assert!(matches!(items[2], Value::String(ref s) if s == "test"));
    } else {
        panic!("Expected List value");
    }

    // Test trim_right
    let trim_expr = expr_list(vec![
        expr_ident("trim_right"),
        expr_string("hello world   "),
    ]);
    let result = eval_expr(&db, &mut env, &trim_expr.0).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "hello world"));

    // Test to_number
    let to_num_expr = expr_list(vec![
        expr_ident("to_number"),
        expr_string("42"),
    ]);
    let result = eval_expr(&db, &mut env, &to_num_expr.0).unwrap();
    assert!(matches!(result, Value::Number(42)));

    // Test to_number with invalid string
    let invalid_num_expr = expr_list(vec![
        expr_ident("to_number"),
        expr_string("not_a_number"),
    ]);
    let result = eval_expr(&db, &mut env, &invalid_num_expr.0);
    assert!(result.is_err());
}

#[test]
fn test_collection_functions() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // First create a list using split
    let split_expr = expr_list(vec![
        expr_ident("split"),
        expr_string(" "),
        expr_string("a b c"),
    ]);
    let list_result = eval_expr(&db, &mut env, &split_expr.0).unwrap();

    // Store the list in environment for get test
    env.bind("test_list".to_string(), list_result);

    // Test get
    let get_expr = expr_list(vec![
        expr_ident("get"),
        expr_number(1),
        expr_ident("test_list"),
    ]);
    let result = eval_expr(&db, &mut env, &get_expr.0).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "b"));

    // Test get with out of bounds index
    let get_oob_expr = expr_list(vec![
        expr_ident("get"),
        expr_number(10),
        expr_ident("test_list"),
    ]);
    let result = eval_expr(&db, &mut env, &get_oob_expr.0);
    assert!(result.is_err());
}

#[test]
fn test_match_case() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Test match with string patterns
    let match_expr = expr_list(vec![
        expr_ident("match"),
        expr_string("+"),
        expr_list(vec![
            expr_ident("case"),
            expr_string("+"),
            expr_number(100),
        ]),
        expr_list(vec![
            expr_ident("case"),
            expr_string("-"),
            expr_number(200),
        ]),
    ]);
    let result = eval_expr(&db, &mut env, &match_expr.0).unwrap();
    assert!(matches!(result, Value::Number(100)));

    // Test match with number patterns
    let match_num_expr = expr_list(vec![
        expr_ident("match"),
        expr_number(42),
        expr_list(vec![
            expr_ident("case"),
            expr_number(41),
            expr_string("wrong"),
        ]),
        expr_list(vec![
            expr_ident("case"),
            expr_number(42),
            expr_string("correct"),
        ]),
    ]);
    let result = eval_expr(&db, &mut env, &match_num_expr.0).unwrap();
    assert!(matches!(result, Value::String(ref s) if s == "correct"));

    // Test match with no matching case
    let no_match_expr = expr_list(vec![
        expr_ident("match"),
        expr_string("unknown"),
        expr_list(vec![
            expr_ident("case"),
            expr_string("known"),
            expr_number(1),
        ]),
    ]);
    let result = eval_expr(&db, &mut env, &no_match_expr.0);
    assert!(result.is_err());
}

#[test]
fn test_function_definition_and_call() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Define a function: (fn (add_one x) (+ x 1))
    let fn_def = expr_list(vec![
        expr_ident("fn"),
        expr_list(vec![
            expr_ident("add_one"),
            expr_ident("x"),
        ]),
        expr_list(vec![
            expr_ident("+"),
            expr_ident("x"),
            expr_number(1),
        ]),
    ]);

    let result = eval_expr(&db, &mut env, &fn_def.0).unwrap();
    assert!(matches!(result, Value::Unit));

    // Call the function: (add_one 5)
    let fn_call = expr_list(vec![
        expr_ident("add_one"),
        expr_number(5),
    ]);

    let result = eval_expr(&db, &mut env, &fn_call.0).unwrap();
    assert!(matches!(result, Value::Number(6)));
}

#[test]
fn test_let_binding() {
    let db = TributeDatabaseImpl::default();
    let mut env = Environment::toplevel();

    // Test let binding: (let x 42)
    let let_expr = expr_list(vec![
        expr_ident("let"),
        expr_ident("x"),
        expr_number(42),
    ]);

    let result = eval_expr(&db, &mut env, &let_expr.0).unwrap();
    assert!(matches!(result, Value::Number(42)));

    // Test that the binding works
    let var_expr = expr_ident("x");
    let result = eval_expr(&db, &mut env, &var_expr.0).unwrap();
    assert!(matches!(result, Value::Number(42)));
}
