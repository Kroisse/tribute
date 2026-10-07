//! Tests for `become`: its placement and operand rules and its lowering to
//! source-logical tail calls.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &salsa::DatabaseImpl, text: &str) -> Vec<String> {
    let source = SourceCst::from_source_str(db, "test.trb", text);
    ast_pipeline_error_messages(db, source)
}

const NOT_TAIL: &str = "`become` must be in tail position of the enclosing function or lambda";
const IN_HANDLER: &str = "`become` cannot be used inside a `handle` body or handler arm, \
                          which run with the handler installed";

/// Named, mutual and indirect tail calls in every tail position lower to
/// `tribute_control.tail_call` and `tail_call_indirect`.
#[salsa_test]
fn become_lowers_to_tail_call_ops(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn sum(n: Int, acc: Int) -> Int {
    case n == +0 {
        True -> acc
        False -> become sum(n - +1, acc + +1)
    }
}

fn is_even(n: Int) -> Bool {
    case n == +0 {
        True -> True
        False -> { become is_odd(n - +1) }
    }
}

fn is_odd(n: Int) -> Bool {
    case n == +0 {
        True -> False
        False -> become is_even(n - +1)
    }
}

fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    become f(x)
}

fn countdown(n: Int) -> Int {
    let step = fn(m: Int) -> Int {
        case m == +0 {
            True -> m
            False -> become countdown(m - +1)
        }
    }
    step(n)
}
"#,
    );
    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

#[salsa_test]
fn become_outside_tail_position_is_rejected(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
fn f(n: Int) -> Int { n }

fn let_value(n: Int) -> Int {
    let x = become f(n)
    x
}

fn argument(n: Int) -> Int {
    f(become f(n))
}

fn scrutinee(n: Int) -> Int {
    case become f(n) {
        _ -> n
    }
}

fn lambda_body_is_not_outer_tail(n: Int) -> Int {
    let g = fn() { become f(n) }
    g()
}
"#,
    );
    assert_eq!(messages, [NOT_TAIL, NOT_TAIL, NOT_TAIL]);
}

#[salsa_test]
fn become_inside_handler_is_rejected(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
ability Ask {
    op ask() -> Int
}

fn f(n: Int) -> Int { n }

fn in_body(n: Int) -> Int {
    handle become f(n) {
        do result { result }
        op Ask::ask() { resume n }
    }
}

fn in_completion(n: Int) -> Int {
    handle n {
        do result { become f(result) }
        op Ask::ask() { resume n }
    }
}
"#,
    );
    // A handler with no matching effect also reports its row; only the
    // `become` diagnostics matter here.
    let become_messages: Vec<_> = messages
        .iter()
        .filter(|message| message.contains("`become`"))
        .collect();
    assert_eq!(become_messages, [IN_HANDLER, IN_HANDLER]);
}

#[salsa_test]
fn become_operand_must_be_a_transferable_call(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
extern "C" fn ext(x: Int) -> Int

enum Opt { Som(Int), Non }

ability Ask {
    op ask() -> Int
}

fn constructor(n: Int) -> Opt {
    become Som(n)
}

fn foreign(n: Int) -> Int {
    become ext(n)
}

fn operation() ->{Ask} Int {
    become Ask::ask()
}

fn not_a_call(n: Int) -> Int {
    become n
}
"#,
    );
    assert_eq!(
        messages,
        [
            "`become` cannot call a constructor",
            "`become` cannot call extern function `ext`, which uses a foreign calling convention",
            "`become` cannot perform an ability operation",
            "`become` requires a function call with an argument list",
        ]
    );
}

#[salsa_test]
fn become_result_must_equal_callable_result(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
fn f(n: Int) -> Int { n }

fn g(n: Int) -> Bool {
    become f(n)
}
"#,
    );
    assert_eq!(messages.len(), 1, "{messages:?}");
}
