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
const IN_HANDLER: &str = "`become` cannot be used in a `handle` body, which runs with the \
                          handler installed, or in an `fn` arm, which returns to its `perform`";
const NOT_A_CALL: &str =
    "`become` requires a function call with an argument list, a `resume`, or a `handle`";

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

fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int {
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

/// A `become` ends an arm of a case that lowers to `scf.index_switch` in
/// place of the arm's yield.
#[salsa_test]
fn become_ends_a_literal_switch_arm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn step(n: Nat) -> Nat {
    case n {
        0 -> 0
        1 -> become step(0)
        _ -> become step(n - 1)
    }
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

const ASK: &str = r#"
ability Ask {
    op ask() -> Int
    fn peek() -> Int
}

fn f(n: Int) -> Int { n }

fn asks() ->{Ask} Int { Ask::ask() + Ask::peek() }
"#;

/// Every diagnostic of `ASK` followed by `text`.
fn ask_errors(db: &salsa::DatabaseImpl, text: &str) -> Vec<String> {
    errors(db, &format!("{ASK}{text}"))
}

/// The `become` diagnostics of `ASK` followed by `text`.
fn become_errors(db: &salsa::DatabaseImpl, text: &str) -> Vec<String> {
    ask_errors(db, text)
        .into_iter()
        .filter(|message| message.contains("`become`"))
        .collect()
}

/// A `handle` in tail position makes the results of its `do` and `op`
/// arms tail positions, through blocks and `case` arms, and in a lambda.
#[salsa_test]
fn become_in_the_arms_of_a_tail_handle_is_accepted(db: &salsa::DatabaseImpl) {
    let messages = ask_errors(
        db,
        r#"
fn arms(n: Int) -> Int {
    handle asks() {
        do result { become f(result) }
        op Ask::ask() {
            case n == +0 {
                True -> become f(n)
                False -> { become arms(n - +1) }
            }
        }
        fn Ask::peek() { n }
    }
}

fn nested(n: Int) -> Int {
    let x = n + +1
    case x == +0 {
        True -> x
        False -> handle asks() {
            do result { result }
            op Ask::ask() { become nested(n - +1) }
            fn Ask::peek() { n }
        }
    }
}

fn in_lambda(n: Int) -> Int {
    let g = fn() -> Int {
        handle asks() {
            do result { result }
            op Ask::ask() { become f(n) }
            fn Ask::peek() { n }
        }
    }
    g()
}
"#,
    );
    assert!(messages.is_empty(), "{messages:?}");
}

/// `become handle ..` and `become resume v` assert a tail position.
#[salsa_test]
fn become_applies_to_a_handle_and_a_resume_in_tail_position(db: &salsa::DatabaseImpl) {
    let messages = ask_errors(
        db,
        r#"
fn installs(n: Int) ->{Ask} Int {
    become handle asks() {
        do result { result }
        op Ask::ask() { become resume n }
        fn Ask::peek() { n }
    }
}
"#,
    );
    assert!(messages.is_empty(), "{messages:?}");

    let not_tail = become_errors(
        db,
        r#"
fn resumes(n: Int) ->{Ask} Int {
    handle asks() {
        do result { result }
        op Ask::ask() {
            let r = become resume n
            r + +1
        }
        fn Ask::peek() { n }
    }
}

fn installs(n: Int) ->{Ask} Int {
    let r = become handle asks() {
        do result { result }
        op Ask::ask() { resume n }
        fn Ask::peek() { n }
    }
    r
}
"#,
    );
    assert_eq!(not_tail, [NOT_TAIL, NOT_TAIL]);
}

/// The body of a `handle` runs with the handler installed and an `fn` arm
/// returns to its `perform`, so neither is a tail position; nor are the
/// arms of a handle out of tail position.
#[salsa_test]
fn become_in_a_handle_body_fn_arm_or_non_tail_handle_is_rejected(db: &salsa::DatabaseImpl) {
    let messages = become_errors(
        db,
        r#"
fn in_body(n: Int) -> Int {
    handle become f(n) {
        do result { result }
        op Ask::ask() { resume n }
        fn Ask::peek() { n }
    }
}

fn in_fn_arm(n: Int) -> Int {
    handle asks() {
        do result { result }
        op Ask::ask() { resume n }
        fn Ask::peek() { become f(n) }
    }
}

fn not_tail(n: Int) -> Int {
    let r = handle asks() {
        do result { become f(result) }
        op Ask::ask() { become f(n) }
        fn Ask::peek() { n }
    }
    r
}
"#,
    );
    assert_eq!(messages, [IN_HANDLER, IN_HANDLER, NOT_TAIL, NOT_TAIL]);
}

/// A `become` in an arm of a handle in a lambda produces the lambda's
/// result, not the enclosing function's.
#[salsa_test]
fn become_in_a_lambda_arm_has_the_lambdas_result_type(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        &format!(
            "{ASK}{}",
            r#"
fn is_zero(n: Int) -> Bool { n == +0 }

fn count(n: Int) -> Int {
    let g = fn() -> Bool {
        handle asks() {
            do result { result == +0 }
            op Ask::ask() { become is_zero(n) }
            fn Ask::peek() { n }
        }
    }
    case g() {
        True -> +1
        False -> +0
    }
}
"#
        ),
    );
    assert!(messages.is_empty(), "{messages:?}");
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
    become .Som(n)
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
            NOT_A_CALL,
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

/// A callee still to be selected among several functions is checked once
/// the call's types have selected it.
#[salsa_test]
fn become_checks_a_callee_selected_late(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
mod a {
    pub extern "C" fn make() -> Int
}
mod b {
    pub fn make() -> Bool { True }
}
use a::make
use b::make

fn late() -> Int {
    become make()
}
"#,
    );
    assert_eq!(
        messages,
        ["`become` cannot call extern function `make`, which uses a foreign calling convention",]
    );
}

/// A name that stands for several functions is not a call either.
#[salsa_test]
fn become_requires_a_call_of_a_function_value(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
mod a {
    pub fn step(n: Int) -> Int { n }
}
mod b {
    pub fn step(n: Bool) -> Bool { n }
}
use a::step
use b::step

fn value() -> fn(Int) -> Int {
    become step
}
"#,
    );
    assert_eq!(messages, [NOT_A_CALL]);
}

const STATE: &str = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}
"#;

const BECOME_SELECTION: &str = "`become` cannot be used for this call: the callee receives a \
                                different selection of handlers than the enclosing callable, \
                                which must be kept until the callee finishes; call it without \
                                `become`";

/// A `become` whose callee takes a different selection of the caller's
/// handlers is rejected: `select`, `tails` and `mask` selections.
#[salsa_test]
fn become_with_a_different_handler_selection_is_rejected(db: &salsa::DatabaseImpl) {
    let select = errors(
        db,
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    become g()
}
"#,
    );
    assert_eq!(select, [BECOME_SELECTION]);

    let tails = errors(
        db,
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil, n: Int) ->{e1, e2} Nil {
    f()
    case n == +0 {
        True -> Nil
        False -> become both(g, f, n - +1)
    }
}
"#,
    );
    assert_eq!(tails, [BECOME_SELECTION]);

    let mask = errors(
        db,
        &format!(
            "{STATE}{}",
            r#"
fn f(n: Int, k: fn(Int) ->{e} Int) ->{State(Int), e} Int {
    State::set(n)
    become k(n - +1)
}
"#
        ),
    );
    assert_eq!(mask, [BECOME_SELECTION]);
}

/// A `become resume` whose resumed computation takes a different selection
/// of handlers is rejected like a call: here the lambda's own `Other`
/// handler must stay hidden from the resumed body.
#[salsa_test]
fn become_resume_with_a_different_handler_selection_is_rejected(db: &salsa::DatabaseImpl) {
    let messages = errors(
        db,
        r#"
ability Ping { op ping() -> Nat }
ability Other { op other() -> Nil }

fn handled(comp: fn() ->{e, Ping} Nat) ->{e, Ping} Nat {
    handle comp() {
        do n { n }
        op Ping::ping() {
            let k = fn() {
                Other::other()
                become resume 0
            }
            handle k() {
                op Other::other() { 0 }
            }
        }
    }
}
"#,
    );
    assert_eq!(
        messages,
        [
            "`become` cannot be used for this `resume`: the resumed computation receives a \
             different selection of handlers than the enclosing callable, which must be kept \
             until it finishes; resume without `become`"
        ]
    );
}

/// A `become` of a callee that takes the caller's handlers unchanged is
/// accepted, including a callee whose tails are the caller's own.
#[salsa_test]
fn become_with_the_callers_own_handlers_is_accepted(db: &salsa::DatabaseImpl) {
    let same_order = errors(
        db,
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil, n: Int) ->{e1, e2} Nil {
    f()
    case n == +0 {
        True -> Nil
        False -> become both(f, g, n - +1)
    }
}
"#,
    );
    assert!(same_order.is_empty(), "{same_order:?}");

    let lambda = errors(
        db,
        &format!(
            "{STATE}{}",
            r#"
fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    let go = fn(c: fn() ->{e, State(s)} a, i: s) ->{e} a {
        handle c() {
            do result { result }
            op State::get() { run_state(fn() { resume i }, i) }
            op State::set(v) { run_state(fn() { resume Nil }, v) }
        }
    }
    become go(comp, init)
}
"#
        ),
    );
    assert!(lambda.is_empty(), "{lambda:?}");
}
