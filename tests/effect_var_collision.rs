//! Tests for EffectVar collision detection.
//!
//! This test verifies that effect variables from function annotation collection
//! and fresh effect variables from lambda inference don't collide.
//!
//! ## Background
//!
//! In `collect.rs`, function signatures with effect annotations use a placeholder
//! `EffectVar { id: 0 }`. During function checking, `FunctionInferenceContext`
//! creates fresh effect variables starting from id=0. If these collide, a pure
//! lambda inside an effectful function might be incorrectly unified with the
//! function's effect row.

use ropey::Rope;
use salsa_test_macros::salsa_test;
use tree_sitter::Parser;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::SourceCst;

/// Helper to create SourceCst from string
fn source_from_str(path: &str, text: &str) -> SourceCst {
    salsa::with_attached_database(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(text, None).expect("tree");
        SourceCst::from_path(db, path, Rope::from_str(text), Some(tree))
    })
    .expect("attached db")
}

/// Test that pure lambdas inside effectful functions have distinct effect variables.
///
/// This test verifies that:
/// 1. A function with `->{State(Int)}` annotation has State(Int) in its effect row
/// 2. A lambda `fn(x: Int) { x + 1 }` inside that function should be pure
/// 3. The lambda's effect variable should NOT be the same as the function's effect variable
#[salsa_test]
#[ignore = "Ability operation name resolution fails in test environment (#317)"]
fn test_lambda_effect_var_independence(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn effectful_with_lambda() ->{State(Int)} Int {
    let f = fn(x: Int) { x + 1 }
    let n = State::get()
    f(n)
}

fn main() -> Nat { 0 }
"#;

    let source = source_from_str("effect_collision.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    assert!(
        result.diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        result.diagnostics.len()
    );
}

/// Test that multiple lambdas in the same function get independent effect variables.
#[salsa_test]
#[ignore = "Ability operation name resolution fails in test environment (#317)"]
fn test_multiple_lambdas_independence(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn effectful_with_multiple_lambdas() ->{State(Int)} Int {
    let f1 = fn(x: Int) { x + 1 }
    let f2 = fn(x: Int) { x * 2 }
    let n = State::get()
    f2(f1(n))
}

fn main() -> Nat { 0 }
"#;

    let source = source_from_str("multiple_lambdas.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    assert!(
        result.diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        result.diagnostics.len()
    );
}

/// Test effect variable collision scenario: passing pure lambda where pure is required.
///
/// If effect variables collide, a "pure" lambda inside an effectful function
/// might get incorrectly typed as effectful, causing this to fail.
///
/// This is the KEY test for the bug:
/// - `apply_pure` requires `fn(Int) ->{} Int` (pure function)
/// - `effectful_using_pure` is `->{State(Int)}`
/// - The lambda `fn(x: Int) { x * 2 }` inside should be inferred as pure
/// - If EffectVar { id: 0 } collision occurs, the lambda might get typed as effectful
/// - This would cause a type error when passing to `apply_pure`
#[salsa_test]
fn test_pure_lambda_in_effectful_context(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn apply_pure(f: fn(Int) ->{} Int, x: Int) ->{} Int {
    f(x)
}

fn effectful_using_pure(init: Int) ->{State(Int)} Int {
    // This lambda should be pure (empty effect row)
    let pure_fn = fn(x: Int) { x * 2 }

    // Calling apply_pure requires a pure function
    // If effect vars collide, pure_fn might be typed as ->{State(Int)} Int
    // which would make this call fail to typecheck
    apply_pure(pure_fn, init)
}

fn main() -> Nat { 0 }
"#;

    let source = source_from_str("pure_in_effectful.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    // This should succeed - the lambda should be inferred as pure
    // If effect variables collide, this might fail because pure_fn
    // would be incorrectly typed as effectful
    assert!(
        result.diagnostics.is_empty(),
        "Type checking should succeed - pure lambda should remain pure even in effectful context. \
         Got {} diagnostics. If this fails with a type mismatch, it indicates EffectVar collision.",
        result.diagnostics.len()
    );
}

/// Test the inverse: effectful lambda should NOT be usable where pure is required.
///
/// This validates that the type system correctly distinguishes effectful from pure.
/// If this test passes but `test_pure_lambda_in_effectful_context` fails,
/// it strongly suggests an EffectVar collision bug.
#[salsa_test]
#[ignore = "Regression in #317: ability operations cause ICE in evidence pass (#319)"]
fn test_effectful_lambda_rejected_for_pure(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn apply_pure(f: fn(Int) ->{} Int, x: Int) ->{} Int {
    f(x)
}

fn should_fail() ->{State(Int)} Int {
    // This lambda uses State::get(), so it's effectful
    let effectful_fn = fn(x: Int) ->{State(Int)} {
        let n = State::get()
        x + n
    }

    // This should fail - effectful_fn is NOT pure
    apply_pure(effectful_fn, 0)
}

fn main() -> Nat { 0 }
"#;

    let source = source_from_str("effectful_rejected.trb", code);
    let result = compile_with_diagnostics(db, source);

    // This SHOULD fail with a type error
    assert!(
        !result.diagnostics.is_empty(),
        "Expected type error when passing effectful lambda to pure-only function"
    );

    eprintln!("Correctly rejected effectful lambda for pure parameter");
}

/// Test nested lambdas - each should have independent effect variables.
#[salsa_test]
#[ignore = "Ability operation name resolution fails in test environment (#317)"]
fn test_nested_lambda_effects(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn nested_lambdas() ->{State(Int)} Int {
    let outer = fn(x: Int) {
        let inner = fn(y: Int) { y + 1 }
        inner(x)
    }
    let n = State::get()
    outer(n)
}

fn main() -> Nat { 0 }
"#;

    let source = source_from_str("nested_lambdas.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    assert!(
        result.diagnostics.is_empty(),
        "Nested lambdas should typecheck correctly. Got {} diagnostics",
        result.diagnostics.len()
    );
}

// =============================================================================
// Pure Lambda Tests (without abilities - should work now)
// =============================================================================

/// Test basic lambda type inference without abilities.
/// This verifies that lambda effect inference works in the simple case.
#[salsa_test]
fn test_pure_lambda_basic(db: &salsa::DatabaseImpl) {
    // Use negative numbers to ensure Int inference (positive literals are Nat)
    let code = r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn test_lambda() -> Int {
    let double = fn(x: Int) { x * 2 }
    apply(double, -21)
}

fn main() -> Int { test_lambda() }
"#;

    let source = source_from_str("pure_lambda_basic.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    assert!(
        result.diagnostics.is_empty(),
        "Basic pure lambda should typecheck. Got {} diagnostics",
        result.diagnostics.len()
    );
}

/// Test multiple lambdas without abilities.
#[salsa_test]
fn test_multiple_pure_lambdas(db: &salsa::DatabaseImpl) {
    // Use negative numbers to ensure Int inference
    let code = r#"
fn compose(f: fn(Int) -> Int, g: fn(Int) -> Int, x: Int) -> Int {
    f(g(x))
}

fn test_compose() -> Int {
    let add_one = fn(x: Int) { x + 1 }
    let double = fn(x: Int) { x * 2 }
    compose(add_one, double, -10)
}

fn main() -> Int { test_compose() }
"#;

    let source = source_from_str("multiple_pure_lambdas.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    assert!(
        result.diagnostics.is_empty(),
        "Multiple pure lambdas should typecheck. Got {} diagnostics",
        result.diagnostics.len()
    );
}

/// Test nested lambdas without abilities.
#[salsa_test]
fn test_nested_pure_lambdas(db: &salsa::DatabaseImpl) {
    // Use negative numbers to ensure Int inference
    let code = r#"
fn test_nested() -> Int {
    let outer = fn(x: Int) {
        let inner = fn(y: Int) { y + 1 }
        inner(x) * 2
    }
    outer(-20)
}

fn main() -> Int { test_nested() }
"#;

    let source = source_from_str("nested_pure_lambdas.trb", code);
    let result = compile_with_diagnostics(db, source);

    for diag in &result.diagnostics {
        eprintln!("Diagnostic: {:?}", diag);
    }

    assert!(
        result.diagnostics.is_empty(),
        "Nested pure lambdas should typecheck. Got {} diagnostics",
        result.diagnostics.len()
    );
}
