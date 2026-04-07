//! Tests for evidence parameter handling on lifted lambdas.
//!
//! Note: The `effect` attribute has been removed from `core.func` types.
//! `is_effectful_type` now always returns `false`, and `collect_effectful_functions`
//! always returns an empty set. The evidence pass (`add_evidence_params`) is
//! therefore a no-op based on effect attributes. Evidence insertion is now
//! handled through other mechanisms (e.g., `resolve_evidence`).
//!
//! These tests verify that the pure-lambda classification still works and
//! that the evidence pass does not spuriously modify function signatures.

mod common;

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute_front::SourceCst;
use tribute_passes::evidence::{collect_effectful_functions, is_effectful_type};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;

/// Helper to compile code through AST pipeline and return arena IR.
fn compile_to_ir(db: &dyn salsa::Database, code: &str, name: &str) -> (IrContext, Module) {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);
    let (mut ctx, m) =
        tribute::pipeline::compile_frontend(db, source_file).expect("compilation should succeed");
    tribute_passes::lower_closure_lambda::lower_closure_lambda(&mut ctx, m);
    (ctx, m)
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

// ========================================================================
// Pure Lambda Tests
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
        let effectful = collect_effectful_functions(&ctx, module);

        // No functions should be considered effectful (effect attribute removed)
        assert!(
            effectful.is_empty(),
            "No functions should be effectful since effect attribute was removed. Got: {:?}",
            effectful
        );
    });
}

// ========================================================================
// Effect attribute removal verification
// ========================================================================

/// Verify that is_effectful_type returns false for all functions,
/// since the effect attribute has been removed from core.func.
#[test]
fn test_no_functions_are_effectful_after_removal() {
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

        // All functions should have is_effectful = false
        for (name, is_effectful) in &functions {
            assert!(
                !is_effectful,
                "Function {} should not be effectful (effect attribute removed from core.func)",
                name
            );
        }

        // collect_effectful_functions should return empty set
        let effectful = collect_effectful_functions(&ctx, module);
        assert!(
            effectful.is_empty(),
            "collect_effectful_functions should return empty set. Got: {:?}",
            effectful
        );
    });
}

/// Evidence pass should be a no-op when no functions are effectful.
#[test]
fn test_evidence_pass_noop_after_effect_removal() {
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
        let before_counts = get_function_param_counts(&ctx_before, &module_before);

        // Apply evidence params pass
        let (ctx_after, module_after) = run_through_evidence_params(db, source_file)
            .expect("evidence pass should succeed");
        let after_counts = get_function_param_counts(&ctx_after, &module_after);

        // No function should have its param count changed by the evidence pass,
        // since is_effectful_type always returns false
        for (name, before_count) in &before_counts {
            if let Some((_, after_count)) = after_counts.iter().find(|(n, _)| n == name) {
                assert_eq!(
                    before_count, after_count,
                    "Function {} param count should not change (evidence pass is no-op). Before: {}, After: {}",
                    name, before_count, after_count
                );
            }
        }
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
