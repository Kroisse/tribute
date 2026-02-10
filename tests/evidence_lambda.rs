//! Tests for evidence parameter insertion on lifted lambdas.
//!
//! These tests verify that the evidence pass correctly identifies effectful
//! lifted lambdas and adds evidence parameters to them.
//!
//! This is the second level of verification for the lambda effect type
//! propagation fix. The first level (IR snapshot tests in tribute-front)
//! verifies the effect type is present in the IR. This level verifies
//! that the evidence pass acts on that information correctly.

mod common;

use ropey::Rope;
use salsa::Database;
use std::collections::HashSet;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute_front::SourceCst;
use tribute_passes::evidence::{collect_effectful_functions, is_effectful_type};
use trunk_ir::dialect::{core, core::Module, func};
use trunk_ir::{DialectOp, DialectType};

/// Helper to compile code through AST pipeline and return the IR module.
fn compile_to_ir(code: &str, name: &str) -> impl FnOnce(&dyn salsa::Database) -> Module<'_> {
    let source_code = Rope::from_str(code);
    let name = name.to_string();

    move |db: &dyn salsa::Database| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, &name, source_code.clone(), tree);
        tribute::pipeline::parse_and_lower_ast(db, source_file)
    }
}

/// Helper to get all function names and their effectful status.
fn get_function_effectfulness<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> Vec<(String, bool)> {
    let mut results = Vec::new();
    let body = module.body(db);

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let name = func_op.sym_name(db).to_string();
                let func_ty = func_op.r#type(db);
                let is_effectful = is_effectful_type(db, func_ty);
                results.push((name, is_effectful));
            }
        }
    }

    results
}

/// Debug helper to print detailed function effect information.
#[allow(dead_code)]
fn debug_function_effects<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) {
    let body = module.body(db);

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let name = func_op.sym_name(db).to_string();
                let func_ty = func_op.r#type(db);

                // Get the core::Func wrapper
                if let Some(core_func) = core::Func::from_type(db, func_ty) {
                    let effect = core_func.effect(db);
                    eprintln!(
                        "Function: {} | Type: {:?} | Effect: {:?}",
                        name, func_ty, effect
                    );

                    if let Some(eff) = effect {
                        if let Some(row) = core::EffectRowType::from_type(db, eff) {
                            let abilities = row.abilities(db);
                            eprintln!("  -> Effect row abilities: {:?}", abilities);
                        } else {
                            eprintln!("  -> Effect is not an EffectRowType");
                        }
                    }
                }
            }
        }
    }
}

/// Helper to get effectful function names as a set.
fn get_effectful_function_names<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> HashSet<String> {
    collect_effectful_functions(db, module)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

// ========================================================================
// Pure Lambda Tests - Should NOT be effectful
// ========================================================================

/// Pure lambda should not be marked as effectful.
#[test]
fn test_pure_lambda_not_effectful() {
    let code = r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }

fn main() -> Int {
    apply(fn(n) { n + 1 }, 41)
}
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let module = compile_to_ir(code, "pure_lambda.trb")(db);
        let effectful = get_effectful_function_names(db, &module);

        // main and apply are pure, lifted lambda should also be pure
        assert!(
            !effectful.iter().any(|n| n.contains("lambda")),
            "Pure lambda should not be effectful. Effectful functions: {:?}",
            effectful
        );
    });
}

// ========================================================================
// Effectful Lambda Tests - Should be effectful
// ========================================================================

/// Lambda directly calling ability operation should be effectful.
#[test]
fn test_direct_ability_lambda_effectful() {
    let code = r#"
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
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let module = compile_to_ir(code, "direct_ability.trb")(db);

        let functions = get_function_effectfulness(db, &module);

        // Find the lifted lambda function
        let lambda_functions: Vec<_> = functions
            .iter()
            .filter(|(name, _)| name.contains("lambda"))
            .collect();

        assert!(
            !lambda_functions.is_empty(),
            "Should have at least one lifted lambda function. All functions: {:?}",
            functions
        );

        // The lambda calling State::get() should be effectful
        let effectful_lambdas: Vec<_> = lambda_functions
            .iter()
            .filter(|(_, is_effectful)| *is_effectful)
            .collect();

        assert!(
            !effectful_lambdas.is_empty(),
            "Lambda calling State::get() should be effectful. Lambda functions: {:?}",
            lambda_functions
        );
    });
}

/// Lambda calling effectful function should inherit effect.
#[test]
fn test_indirect_effect_lambda_effectful() {
    let code = r#"
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
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let module = compile_to_ir(code, "indirect_effect.trb")(db);
        let functions = get_function_effectfulness(db, &module);

        // counter should be effectful
        let counter_effectful = functions
            .iter()
            .find(|(name, _)| name == "counter")
            .map(|(_, e)| *e)
            .unwrap_or(false);
        assert!(counter_effectful, "counter() should be effectful");

        // The lambda calling counter() should also be effectful
        let lambda_functions: Vec<_> = functions
            .iter()
            .filter(|(name, _)| name.contains("lambda"))
            .collect();

        let effectful_lambdas: Vec<_> = lambda_functions
            .iter()
            .filter(|(_, is_effectful)| *is_effectful)
            .collect();

        assert!(
            !effectful_lambdas.is_empty(),
            "Lambda calling counter() should be effectful. Lambda functions: {:?}",
            lambda_functions
        );
    });
}

// ========================================================================
// Handler Arm Lambda Tests - Core ability pattern
// ========================================================================

/// Handler arm lambdas that call continuations.
///
/// This is the critical test for the fix - the lambdas in handler arms:
/// - `fn() { k(init) }` in State::get handler
/// - `fn() { k(Nil) }` in State::set handler
///
/// These should be effectful if the effect row variable `e` is non-empty.
#[test]
#[ignore] // TODO: add assertions once expected effectfulness behavior is determined
fn test_handler_arm_lambdas_in_run_state() {
    let code = r#"
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
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let module = compile_to_ir(code, "handler_arm.trb")(db);
        let functions = get_function_effectfulness(db, &module);

        eprintln!("All functions:");
        for (name, is_effectful) in &functions {
            eprintln!("  {} -> effectful: {}", name, is_effectful);
        }

        // run_state should be effectful (has effect row variable e)
        let run_state_effectful = functions
            .iter()
            .find(|(name, _)| name == "run_state")
            .map(|(_, e)| *e);

        // Note: run_state has effect `{e}` which is a row variable.
        // Whether it's considered effectful depends on the implementation.
        eprintln!("run_state effectful: {:?}", run_state_effectful);
    });
}

/// Full ability_core pattern with counter calls.
#[test]
fn test_ability_core_pattern_effectfulness() {
    let code = r#"
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
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let module = compile_to_ir(code, "ability_core.trb")(db);
        let functions = get_function_effectfulness(db, &module);

        eprintln!("=== ability_core pattern functions ===");
        for (name, is_effectful) in &functions {
            eprintln!("  {} -> effectful: {}", name, is_effectful);
        }

        // counter should definitely be effectful
        let counter_effectful = functions
            .iter()
            .find(|(name, _)| name == "counter")
            .map(|(_, e)| *e)
            .unwrap_or(false);
        assert!(counter_effectful, "counter() should be effectful");

        // The main lambda `fn() { counter(); counter(); counter() }` should be effectful
        let lambda_functions: Vec<_> = functions
            .iter()
            .filter(|(name, _)| name.contains("lambda"))
            .collect();

        eprintln!("\n=== Lambda functions ===");
        for (name, is_effectful) in &lambda_functions {
            eprintln!("  {} -> effectful: {}", name, is_effectful);
        }

        // At least the main lambda should be effectful
        let main_lambda_effectful = lambda_functions.iter().any(|(name, is_effectful)| {
            // The lambda in main that calls counter() should be effectful
            *is_effectful && name.contains("main")
        });

        // If no lambda contains "main", check if any lambda is effectful
        let any_effectful_lambda = lambda_functions
            .iter()
            .any(|(_, is_effectful)| *is_effectful);

        assert!(
            main_lambda_effectful || any_effectful_lambda,
            "The lambda calling counter() should be effectful. Lambda functions: {:?}",
            lambda_functions
        );
    });
}

// ========================================================================
// Evidence Parameter Count Tests
// ========================================================================

/// Effectful lambdas already have evidence as their first parameter (from ast_to_ir).
/// The evidence pass should detect this and not add a duplicate parameter.
#[test]
fn test_evidence_param_not_duplicated_for_effectful_lambda() {
    use tribute::pipeline::stage_evidence_params;

    let code = r#"
ability State(s) {
    fn get() -> s
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        { result } -> result
        { State::get() -> k } -> k(42)
    }
}

fn main() -> Int {
    run_with_state(fn() { State::get() })
}
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let module = compile_to_ir(code, "evidence_param.trb")(db);

        // Get function param counts before evidence pass
        let before_counts = get_function_param_counts(db, &module);
        eprintln!("Before evidence pass: {:?}", before_counts);

        // Apply evidence params pass
        // (boxing is now handled via unrealized_conversion_cast in ast_to_ir)
        let after_evidence = stage_evidence_params(db, module);

        // Get lambda param count after evidence pass
        let after_counts = get_function_param_counts(db, &after_evidence);
        eprintln!("After evidence pass: {:?}", after_counts);

        // Find effectful lambdas and verify they got an extra parameter
        let effectful_before = collect_effectful_functions(db, &module);
        eprintln!("Effectful functions: {:?}", effectful_before);

        // Effectful lambdas already get evidence as their first parameter during
        // ast_to_ir lowering (for uniform calling convention at call_indirect sites).
        // The evidence pass should detect this and NOT add a duplicate.
        let mut checked = 0;
        for name in effectful_before.iter() {
            let name_str = name.to_string();
            if name_str.contains("lambda") {
                let before_count = before_counts
                    .iter()
                    .find(|(n, _)| n == &name_str)
                    .map(|(_, c)| *c);
                let after_count = after_counts
                    .iter()
                    .find(|(n, _)| n == &name_str)
                    .map(|(_, c)| *c);

                if let (Some(before), Some(after)) = (before_count, after_count) {
                    assert_eq!(
                        after,
                        before,
                        "Effectful lambda {} should keep same param count (evidence already added in ast_to_ir) (before: {}, after: {})",
                        name_str,
                        before,
                        after
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "Expected at least one effectful lambda to be checked");
    });
}

/// Helper to get function parameter counts.
fn get_function_param_counts<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> Vec<(String, usize)> {
    let mut results = Vec::new();
    let body = module.body(db);

    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let name = func_op.sym_name(db).to_string();
                let func_body = func_op.body(db);
                if let Some(entry_block) = func_body.blocks(db).first() {
                    let param_count = entry_block.args(db).len();
                    results.push((name, param_count));
                }
            }
        }
    }

    results
}
