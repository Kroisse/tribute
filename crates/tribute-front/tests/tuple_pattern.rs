//! Tests for tuple pattern type inference and IR lowering.
//!
//! These tests verify correct UniVar resolution in tuple patterns,
//! and that tuple destructuring in `let` bindings produces correct IR.

mod common;

use self::common::{run_ast_pipeline, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// Test basic tuple pattern matching in case expression.
/// This should infer types correctly without UniVar leakage.
#[salsa_test]
fn test_tuple_pattern_basic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test(x: Bool, y: Bool) -> Int {
    case #(x, y) {
        #(True, True) -> +1
        #(True, False) -> +2
        #(False, _) -> +3
    }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test tuple pattern with nested structure.
#[salsa_test]
fn test_tuple_pattern_nested(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test(a: Bool, b: Bool, c: Bool) -> Int {
    case #(a, #(b, c)) {
        #(True, #(True, True)) -> +1
        #(True, #(_, False)) -> +2
        #(True, #(False, True)) -> +4
        #(False, _) -> +3
    }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test tuple pattern in generic function context.
#[salsa_test]
fn test_tuple_pattern_generic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn first(pair: #(a, b)) -> a {
    case pair {
        #(x, _) -> x
    }
}

fn test() -> Int {
    first(#(+42, True))
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// Tuple Let Destructuring Tests
// ========================================================================

/// Test basic tuple let destructuring lowers to adt.struct_get with correct types.
#[salsa_test]
fn test_tuple_let_destructure_basic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test() -> Nat {
    let t = #(1, 2)
    let #(a, b) = t
    a + b
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test tuple let destructuring with nested tuples.
#[salsa_test]
fn test_tuple_let_destructure_nested(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test() -> Nat {
    let t = #(1, #(2, 3))
    let #(a, #(b, c)) = t
    a + b + c
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test tuple let destructuring with wildcard elements.
#[salsa_test]
fn test_tuple_let_destructure_wildcard(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test() -> Nat {
    let t = #(1, 2, 3)
    let #(a, _, c) = t
    a + c
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test tuple let destructuring used in function arguments.
#[salsa_test]
fn test_tuple_let_destructure_from_function(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn make_pair(x: Nat, y: Nat) -> #(Nat, Nat) {
    #(x, y)
}

fn test() -> Nat {
    let #(a, b) = make_pair(10, 20)
    a + b
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Destructuring a `let`-bound call result projects the tuple layout the call
/// returns: the binding's uses share the call's rows.
#[salsa_test]
fn test_tuple_let_destructure_shares_call_rows(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair(f: fn() ->{e} Nat) -> #(fn() ->{e} Nat, Nat) { #(f, 1) }

fn test() -> Nat {
    let p = pair(fn() { 2 })
    let #(f, n) = p
    f() + n
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let test_body = ir_text
        .split("@\"pkg::test\"")
        .nth(1)
        .expect("the module defines test");
    let call = test_body
        .lines()
        .find(|line| line.contains("@\"pkg::pair\""))
        .expect("test calls pair");
    let result_ty = call.rsplit("-> ").next().unwrap().trim();
    for line in test_body
        .lines()
        .filter(|line| line.contains("adt.struct_get"))
    {
        let layout = line
            .split("type = ")
            .nth(1)
            .and_then(|rest| rest.split('}').next())
            .expect("struct_get names its layout");
        assert!(
            result_ty.starts_with(layout),
            "projection layout {layout} differs from the call result {result_ty}"
        );
    }
    assert_snapshot!(ir_text);
}

/// A constructor pattern inside a tuple projects the tuple field at the
/// layout's field type, not at the constructor's callable type.
#[salsa_test]
fn test_tuple_pattern_constructor_element(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
enum Item { Number(Nat), Empty }

fn first(p: #(Item, Nat)) -> Nat {
    case p {
        #(.Number(x), y) -> x + y
        _ -> 0
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    for line in ir_text
        .lines()
        .filter(|line| line.contains("adt.struct_get"))
    {
        assert!(
            !line.contains("func"),
            "tuple field projection must not use a callable type: {line}"
        );
    }
    assert_snapshot!(ir_text);
}
