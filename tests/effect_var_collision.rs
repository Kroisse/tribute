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

mod common;

use salsa_test_macros::salsa_test;
use tribute::pipeline::{
    OptimizationOptions, SharedPipelineStage, compile_with_diagnostics, dump_shared_ir_at_stage,
};
use tribute_front::SourceCst;

fn logical_function<'a>(ir: &'a str, name: &str) -> &'a str {
    let markers = [
        format!("tribute_control.func @{name}"),
        format!("func.func @{name}"),
    ];
    let start = markers
        .iter()
        .filter_map(|marker| ir.find(marker))
        .min()
        .unwrap_or_else(|| panic!("missing logical function {name}:\n{ir}"));
    let function = &ir[start..];
    let end = ["\n  tribute_control.func @", "\n  func.func @"]
        .iter()
        .filter_map(|marker| function[1..].find(marker).map(|offset| offset + 1))
        .min()
        .unwrap_or(function.len());
    &function[..end]
}

/// Test that pure lambdas inside effectful functions have distinct effect variables.
///
/// This test verifies that:
/// 1. A function with `->{State(Int)}` annotation has State(Int) in its effect row
/// 2. A lambda `fn(x: Int) { x + 1 }` inside that function should be pure
/// 3. The lambda's effect variable should NOT be the same as the function's effect variable
#[salsa_test]
fn test_lambda_effect_var_independence(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn effectful_with_lambda() ->{State(Int)} Int {
    let f = fn(x: Int) { x + +1 }
    let n = State::get()
    f(n)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "effect_collision.trb", code);
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
fn test_multiple_lambdas_independence(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn effectful_with_multiple_lambdas() ->{State(Int)} Int {
    let f1 = fn(x: Int) { x + +1 }
    let f2 = fn(x: Int) { x * +2 }
    let n = State::get()
    f2(f1(n))
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "multiple_lambdas.trb", code);
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
/// - `apply_pure` requires `fn(Int) ->{} Int` (explicit pure function)
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
    let pure_fn = fn(x: Int) { x * +2 }

    // Calling apply_pure requires a pure function
    // If effect vars collide, pure_fn might be typed as ->{State(Int)} Int
    // which would make this call fail to typecheck
    apply_pure(pure_fn, init)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "pure_in_effectful.trb", code);
    let logical = dump_shared_ir_at_stage(
        db,
        source,
        SharedPipelineStage::AfterFrontend,
        OptimizationOptions::production(),
    )
    .expect("source-logical dump must succeed");
    let worker = logical_function(&logical, "effectful_using_pure");
    assert!(
        worker.contains("tribute_control.lambda(") && worker.contains("convention(direct)"),
        "a pure callback must remain Direct inside an effectful worker:\n{worker}"
    );
    assert!(
        !worker.contains("core.unrealized_conversion_cast"),
        "the exact Direct callback parameter must not require a control cast:\n{worker}"
    );
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

/// A CPS worker sequences its own operation through a continuation, but a
/// closed pure callback value remains Direct at the independently typed call.
#[salsa_test]
fn test_direct_callback_remains_direct_inside_cps_worker(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "direct_callback_in_cps.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn apply_pure(f: fn(Int) ->{} Int, x: Int) ->{} Int {
    f(x)
}

fn effectful_using_pure(init: Int) ->{State(Int)} Int {
    let pure_fn = fn(x: Int) { x * +2 }
    let _ = State::get()
    apply_pure(pure_fn, init)
}

fn main() { }
"#,
    );
    let logical = dump_shared_ir_at_stage(
        db,
        source,
        SharedPipelineStage::AfterFrontend,
        OptimizationOptions::production(),
    )
    .expect("source-logical dump must succeed");
    let worker = logical_function(&logical, "effectful_using_pure");

    assert!(
        worker.contains("convention(cps)"),
        "the enclosing operation worker must use CPS:\n{worker}"
    );
    assert!(
        worker.contains("tribute_control.lambda(") && worker.contains("convention(direct)"),
        "the closed callback must keep its Direct callable convention:\n{worker}"
    );
    assert!(
        worker.contains("tribute_control.call") && worker.contains("callee = @apply_pure"),
        "the Direct callback must feed the exact Direct parameter:\n{worker}"
    );
    assert!(
        !worker.contains("core.unrealized_conversion_cast"),
        "the logical worker must not cast control callable values:\n{worker}"
    );

    let post_cps = dump_shared_ir_at_stage(
        db,
        source,
        SharedPipelineStage::AfterControlLegalization,
        OptimizationOptions::production(),
    )
    .expect("CPS legalization must preserve the direct callback");
    let worker = logical_function(&post_cps, "effectful_using_pure");
    assert!(
        worker.contains("tribute.calling_convention = 2"),
        "the enclosing worker must remain CPS after legalization:\n{worker}"
    );
    assert!(
        worker.contains("closure.lambda") && worker.contains("tribute.calling_convention = 0"),
        "the callback closure must remain an ordinary Direct closure:\n{worker}"
    );
    assert!(
        worker.contains("func.tail_call_indirect"),
        "the enclosing CPS worker must end through its continuation:\n{worker}"
    );
    assert!(
        worker.contains("callee = @apply_pure, tribute.calling_convention = 0"),
        "the Direct callback call must remain an ordinary Direct call:\n{worker}"
    );
}

/// A lambda that evaluates an open effect-polymorphic callback remains Cps;
/// an empty known row prefix alone is not proof that the body is pure.
#[salsa_test]
fn test_open_effect_callback_body_remains_cps(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "open_effect_lambda.trb",
        r#"
fn invoke(f: fn(Int) ->{e} Int, value: Int) ->{e} Int { f(value) }

fn relay(g: fn(Int) ->{e} Int, value: Int) ->{e} Int {
    invoke(fn(inner) { g(inner) }, value)
}

fn main() { }
"#,
    );
    let logical = dump_shared_ir_at_stage(
        db,
        source,
        SharedPipelineStage::AfterFrontend,
        OptimizationOptions::production(),
    )
    .expect("open-effect lambda must lower as a Cps callable");
    let relay = logical_function(&logical, "relay");
    assert!(
        relay.contains("tribute_control.lambda(") && relay.contains("convention(cps)"),
        "an open effect row evaluated by the lambda body must retain Cps:\n{relay}"
    );
    assert!(
        !relay.contains("core.unrealized_conversion_cast"),
        "the open-effect callable already has the exact Cps convention:\n{relay}"
    );
}

/// Test the inverse: effectful lambda should NOT be usable where pure is required.
///
/// This validates that the type system correctly distinguishes effectful from pure.
/// If this test passes but `test_pure_lambda_in_effectful_context` fails,
/// it strongly suggests an EffectVar collision bug.
#[salsa_test]
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
    apply_pure(effectful_fn, +0)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "effectful_rejected.trb", code);
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
fn test_nested_lambda_effects(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn nested_lambdas() ->{State(Int)} Int {
    let outer = fn(x: Int) {
        let inner = fn(y: Int) { y + +1 }
        inner(x)
    }
    let n = State::get()
    outer(n)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "nested_lambdas.trb", code);
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
    let double = fn(x: Int) { x * +2 }
    apply(double, -21)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "pure_lambda_basic.trb", code);
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
    let add_one = fn(x: Int) { x + +1 }
    let double = fn(x: Int) { x * +2 }
    compose(add_one, double, -10)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "multiple_pure_lambdas.trb", code);
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
        let inner = fn(y: Int) { y + +1 }
        inner(x) * +2
    }
    outer(-20)
}

fn main() { }
"#;

    let source = SourceCst::from_source_str(db, "nested_pure_lambdas.trb", code);
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
