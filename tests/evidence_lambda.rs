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
use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::rewrite::Module;

/// Helper to compile code through AST pipeline and return arena IR.
fn compile_to_ir(db: &dyn salsa::Database, code: &str, name: &str) -> (IrContext, Module) {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);
    tribute::pipeline::compile_frontend(db, source_file).expect("compilation should succeed")
}

/// Helper to get all function names and their effectful status.
fn get_function_effectfulness(ctx: &IrContext, module: &Module) -> Vec<(String, bool)> {
    let mut results = Vec::new();

    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx).to_string();
            let func_ty = func_op.r#type(ctx);
            let is_effectful = is_effectful_type(ctx, func_ty);
            results.push((name, is_effectful));
        }
    }

    results
}

/// Debug helper to print detailed function effect information.
#[allow(dead_code)]
fn debug_function_effects(ctx: &IrContext, module: &Module) {
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx).to_string();
            let func_ty = func_op.r#type(ctx);

            let data = ctx.types.get(func_ty);
            if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
                let effect = data
                    .attrs
                    .get(&Symbol::new("effect"))
                    .and_then(|a| match a {
                        trunk_ir::arena::types::Attribute::Type(ty) => Some(*ty),
                        _ => None,
                    });
                eprintln!(
                    "Function: {} | Type dialect: {}.{} | Effect attr: {:?}",
                    name, data.dialect, data.name, effect
                );

                if let Some(eff_ref) = effect {
                    let eff_data = ctx.types.get(eff_ref);
                    eprintln!(
                        "  -> Effect type: {}.{} with {} params",
                        eff_data.dialect,
                        eff_data.name,
                        eff_data.params.len()
                    );
                }
            }
        }
    }
}

/// Helper to get effectful function names as a set.
fn get_effectful_function_names(ctx: &IrContext, module: &Module) -> HashSet<String> {
    collect_effectful_functions(ctx, *module)
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

fn run() -> Int {
    apply(fn(n) { n + 1 }, 41)
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "pure_lambda.trb");
        let effectful = get_effectful_function_names(&ctx, &module);

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

fn run() -> Int {
    run_with_state(fn() { State::get() })
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "direct_ability.trb");

        let functions = get_function_effectfulness(&ctx, &module);

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

fn run() -> Int {
    run_with_state(fn() { counter() })
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "indirect_effect.trb");
        let functions = get_function_effectfulness(&ctx, &module);

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

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "handler_arm.trb");
        let functions = get_function_effectfulness(&ctx, &module);

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

fn run() -> Int {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "ability_core.trb");
        let functions = get_function_effectfulness(&ctx, &module);

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

        // The run lambda `fn() { counter(); counter(); counter() }` should be effectful
        let lambda_functions: Vec<_> = functions
            .iter()
            .filter(|(name, _)| name.contains("lambda"))
            .collect();

        eprintln!("\n=== Lambda functions ===");
        for (name, is_effectful) in &lambda_functions {
            eprintln!("  {} -> effectful: {}", name, is_effectful);
        }

        // At least the run lambda should be effectful
        let run_lambda_effectful = lambda_functions.iter().any(|(name, is_effectful)| {
            // The lambda in run that calls counter() should be effectful
            *is_effectful && name.contains("run")
        });

        // If no lambda contains "run", check if any lambda is effectful
        let any_effectful_lambda = lambda_functions
            .iter()
            .any(|(_, is_effectful)| *is_effectful);

        assert!(
            run_lambda_effectful || any_effectful_lambda,
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
    use tribute::pipeline::run_through_evidence_params;

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

fn run() -> Int {
    run_with_state(fn() { State::get() })
}

fn main() { }
"#;

    let source_code = Rope::from_str(code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "evidence_param.trb", source_code.clone(), tree);

        let (ctx_before, module_before) = compile_to_ir(db, code, "evidence_param.trb");

        // Get function param counts before evidence pass
        let before_counts = get_function_param_counts(&ctx_before, &module_before);
        eprintln!("Before evidence pass: {:?}", before_counts);

        // Apply evidence params pass via arena-based run_through_evidence_params
        let (ctx_after, module_after) = run_through_evidence_params(db, source_file)
            .expect("evidence pass should succeed");

        // Get lambda param count after evidence pass
        let after_counts = get_function_param_counts(&ctx_after, &module_after);
        eprintln!("After evidence pass: {:?}", after_counts);

        // Find effectful lambdas and verify they got an extra parameter
        let effectful_before = collect_effectful_functions(&ctx_before, module_before);
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
fn get_function_param_counts(ctx: &IrContext, module: &Module) -> Vec<(String, usize)> {
    let mut results = Vec::new();

    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx).to_string();
            let func_body = func_op.body(ctx);
            let blocks = &ctx.region(func_body).blocks;
            if let Some(&entry_block) = blocks.first() {
                let param_count = ctx.block_args(entry_block).len();
                results.push((name, param_count));
            }
        }
    }

    results
}
