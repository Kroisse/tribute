//! Tests for lambda effect type propagation to IR.
//!
//! These tests verify that lambda expressions with effects have their
//! effect types correctly propagated to the lifted IR functions.
//!
//! This is critical for the evidence pass to correctly identify which
//! lifted lambdas need evidence parameters.

use insta::assert_debug_snapshot;
use ropey::Rope;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use trunk_ir::dialect::core::Module;

fn source_from_str(path: &str, text: &str) -> SourceCst {
    use tree_sitter::Parser;
    salsa::with_attached_database(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text: Rope = text.into();
        let tree = parser.parse(text.to_string().as_str(), None);
        SourceCst::new(
            db,
            fluent_uri::Uri::parse_from(format!("test:///{}", path)).unwrap(),
            text,
            tree,
        )
    })
    .expect("attached db")
}

/// Helper tracked function to run the AST pipeline and return the IR module.
#[salsa::tracked]
fn run_ast_pipeline_with_ir<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let parsed = tribute_front::query::parsed_ast(db, source);
    assert!(parsed.is_some(), "Should parse successfully");

    let parsed = parsed.unwrap();
    let ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();

    let env = tribute_front::resolve::build_env(db, &ast);
    let resolved = tribute_front::resolve::resolve_with_env(db, ast, env, span_map.clone());

    let checker = tribute_front::typeck::TypeChecker::new(db, span_map.clone());
    let result = checker.check_module(resolved);

    let tdnr_ast = tribute_front::tdnr::resolve_tdnr(db, result.module);

    let function_types_map: std::collections::HashMap<_, _> =
        result.function_types.into_iter().collect();
    let node_types_map: std::collections::HashMap<_, _> = result.node_types.into_iter().collect();
    tribute_front::ast_to_ir::lower_ast_to_ir(
        db,
        tdnr_ast,
        span_map,
        source.uri(db).as_str(),
        function_types_map,
        node_types_map,
    )
}

// ========================================================================
// Pure Lambda Tests - No Effect Expected
// ========================================================================

/// Test that a pure lambda (no effects) is lifted without effect type.
#[salsa_test]
fn test_pure_lambda_no_effect(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }

fn main() -> Int {
    apply(fn(n) { n + 1 }, 41)
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test that a pure lambda capturing a variable has no effect type.
#[salsa_test]
fn test_pure_lambda_with_capture_no_effect(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }

fn main() -> Int {
    let offset = 10;
    apply(fn(n) { n + offset }, 32)
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

// ========================================================================
// Effectful Lambda Tests - Effect Type Expected
// ========================================================================

/// Test that a lambda directly calling an ability operation has effect type.
///
/// The lambda `fn() { State::get() }` should have `State(Int)` effect
/// in its lifted function type.
#[salsa_test]
fn test_effectful_lambda_direct_ability_call(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        { result } -> result
        { State::get() -> k } -> k(42)
        { State::set(v) -> k } -> k(Nil)
    }
}

fn main() -> Int {
    run_with_state(fn() { State::get() })
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test that a lambda calling an effectful function inherits the effect.
///
/// The lambda `fn() { counter() }` should have `State(Int)` effect
/// because `counter` has that effect.
#[salsa_test]
fn test_effectful_lambda_indirect_effect_call(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        { result } -> result
        { State::get() -> k } -> k(0)
        { State::set(v) -> k } -> k(Nil)
    }
}

fn main() -> Int {
    run_with_state(fn() { counter() })
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test that multiple ability operations in lambda accumulate effects.
#[salsa_test]
fn test_effectful_lambda_multiple_operations(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        { result } -> result
        { State::get() -> k } -> k(0)
        { State::set(v) -> k } -> k(Nil)
    }
}

fn main() -> Int {
    run_with_state(fn() {
        let n = State::get()
        State::set(n + 1)
        n
    })
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

// ========================================================================
// Handler Arm Lambda Tests - Core Ability Pattern
// ========================================================================

/// Test handler arm lambdas that call continuations.
///
/// This is the core pattern from ability_core.trb:
/// `{ State::get() -> k } -> run_state(fn() { k(init) }, init)`
///
/// The lambda `fn() { k(init) }` should preserve the effect row variable `e`
/// from the outer handler context.
#[salsa_test]
fn test_handler_arm_continuation_lambda(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int { 0 }
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test the full ability_core pattern with multiple counter calls.
#[salsa_test]
fn test_ability_core_full_pattern(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

// ========================================================================
// Nested Lambda Tests
// ========================================================================

/// Test nested lambdas where inner lambda has effect.
#[salsa_test]
fn test_nested_lambda_inner_effectful(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        { result } -> result
        { State::get() -> k } -> k(99)
    }
}

fn apply_thunk(f: fn() -> Int) -> Int { f() }

fn main() -> Int {
    apply_thunk(fn() {
        run_with_state(fn() { State::get() })
    })
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

// ========================================================================
// Effect Row Variable Tests
// ========================================================================

/// Test that effect row variables are properly unified.
///
/// When a lambda is passed to a function expecting `fn() ->{e, State(s)} a`,
/// the lambda's effect should include State(s) with the row variable e.
#[salsa_test]
fn test_lambda_effect_row_unification(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn with_state(f: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle f() {
        { result } -> result
        { State::get() -> k } -> with_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> with_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Nat {
    with_state(fn() {
        State::get()
    }, 42)
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}
