//! Tests for evidence parameter presence on lifted lambdas.
//!
//! These tests verify that effectful lifted lambdas receive evidence as their
//! first parameter during ast_to_ir lowering, and pure lambdas do not.

mod common;

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute_front::SourceCst;
use tribute_passes::evidence::has_evidence_first_param;
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

/// Helper to check which functions have evidence as first parameter.
fn get_functions_with_evidence(ctx: &IrContext, module: &Module) -> Vec<(String, bool)> {
    let mut results = Vec::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = arena_func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx).to_string();
            let func_ty = func_op.r#type(ctx);
            let has_evidence = has_evidence_first_param(ctx, func_ty);
            results.push((name, has_evidence));
        }
    }
    results
}

// ========================================================================
// Pure Lambda Tests
// ========================================================================

/// Pure top-level functions should not have evidence parameter.
/// Note: lifted lambdas always get evidence as part of the closure calling
/// convention (added by lower_closure_lambda), regardless of effectfulness.
#[test]
fn test_pure_toplevel_function_no_evidence() {
    let code = r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }

fn run() -> Int {
    apply(fn(n) { n + 1 }, 41)
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "pure_toplevel.trb");
        let functions = get_functions_with_evidence(&ctx, &module);

        // Pure user-defined top-level functions should not have evidence
        for (name, has_ev) in &functions {
            if !name.contains("clam") && !name.contains("lambda") && !name.contains("::") {
                assert!(
                    !has_ev,
                    "Pure top-level function '{}' should not have evidence parameter",
                    name
                );
            }
        }
    });
}

// ========================================================================
// Effectful Lambda Tests
// ========================================================================

/// Lambda directly calling ability operation should have evidence.
#[test]
fn test_direct_ability_lambda_has_evidence() {
    let code = r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        fn State::get() { 42 }
        fn State::set(v) { Nil }
    }
}

fn run() -> Int {
    run_with_state(fn() { State::get() })
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "direct_ability.trb");
        let functions = get_functions_with_evidence(&ctx, &module);

        // counter should have evidence
        let effectful_lambdas: Vec<_> = functions
            .iter()
            .filter(|(name, has_ev)| name.contains("clam") && *has_ev)
            .collect();

        assert!(
            !effectful_lambdas.is_empty(),
            "Lambda calling State::get() should have evidence. All functions: {:?}",
            functions
        );
    });
}

/// Lambda calling effectful function should have evidence.
#[test]
fn test_indirect_effect_lambda_has_evidence() {
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
        do result { result }
        fn State::get() { 0 }
        fn State::set(v) { Nil }
    }
}

fn run() -> Int {
    run_with_state(fn() { counter() })
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "indirect_effect.trb");
        let functions = get_functions_with_evidence(&ctx, &module);

        // counter should have evidence
        let counter_has_evidence = functions
            .iter()
            .any(|(name, has_ev)| name == "counter" && *has_ev);
        assert!(counter_has_evidence, "counter() should have evidence");

        // The lambda calling counter() should also have evidence
        let effectful_lambdas: Vec<_> = functions
            .iter()
            .filter(|(name, has_ev)| name.contains("clam") && *has_ev)
            .collect();

        assert!(
            !effectful_lambdas.is_empty(),
            "Lambda calling counter() should have evidence. All functions: {:?}",
            functions
        );
    });
}

// ========================================================================
// Evidence Parameter Stability Test
// ========================================================================

/// Evidence params inserted in ast_to_ir should survive through lower_closure_lambda
/// without duplication.
#[test]
fn test_evidence_param_count_stable_after_lambda_lifting() {
    let code = r#"
ability State(s) {
    fn get() -> s
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        fn State::get() { 42 }
    }
}

fn run() -> Int {
    run_with_state(fn() { State::get() })
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "evidence_stable.trb");

        // Count evidence params per function via block args
        for op in module.ops(&ctx) {
            if let Ok(func_op) = arena_func::Func::from_op(&ctx, op) {
                let name = func_op.sym_name(&ctx).to_string();
                let body = func_op.body(&ctx);
                let blocks = &ctx.region(body).blocks;
                if let Some(&entry) = blocks.first() {
                    let args = ctx.block_args(entry);
                    let evidence_count = args
                        .iter()
                        .filter(|&&arg| {
                            tribute_ir::dialect::ability::is_evidence_type_ref(
                                &ctx,
                                ctx.value_ty(arg),
                            )
                        })
                        .count();
                    assert!(
                        evidence_count <= 1,
                        "Function '{}' has {} evidence parameters, expected at most 1",
                        name,
                        evidence_count
                    );
                }
            }
        }
    });
}
