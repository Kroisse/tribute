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
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;

/// Helper to compile code through AST pipeline and return arena IR.
fn compile_to_ir(db: &dyn salsa::Database, code: &str, name: &str) -> (IrContext, Module) {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);
    let (mut ctx, m) =
        tribute::pipeline::compile_frontend(db, source_file).expect("compilation should succeed");
    let core_module = trunk_ir::dialect::core::Module::from_op(&ctx, m.op())
        .expect("frontend output must be a core.module");
    let mut pm = trunk_ir::pass::PassManager::new();
    pm.add_pass(tribute_passes::lower_closure_lambda::LowerClosureLambda);
    pm.run(&mut ctx, core_module).unwrap();
    (ctx, m)
}

/// Helper to check which functions have evidence as first parameter.
fn get_functions_with_evidence(ctx: &IrContext, module: &Module) -> Vec<(String, bool)> {
    let mut results = Vec::new();
    for op in module.ops(ctx) {
        if let Ok(func_op) = func::Func::from_op(ctx, op) {
            let name = func_op.sym_name(ctx).to_string();
            let func_ty = func_op.r#type(ctx);
            let has_evidence = has_evidence_first_param(ctx, func_ty);
            results.push((name, has_evidence));
        }
    }
    results
}

/// Return `(source parameter count, has evidence, has done_k, returns anyref)`.
fn function_abi(ctx: &IrContext, module: &Module, target: &str) -> (usize, bool, bool, bool) {
    let func_op = module
        .ops(ctx)
        .into_iter()
        .find_map(|op| {
            let func_op = func::Func::from_op(ctx, op).ok()?;
            (func_op.sym_name(ctx) == target).then_some(func_op)
        })
        .unwrap_or_else(|| panic!("missing function '{target}'"));
    let func_ty = func_op.r#type(ctx);
    let data = ctx.types.get(func_ty);
    let params = &data.params[1..];
    let has_evidence = params
        .first()
        .is_some_and(|ty| tribute_ir::dialect::ability::is_evidence_type_ref(ctx, *ty));
    let hidden_count = usize::from(has_evidence)
        + usize::from(
            has_evidence
                && params.get(1).is_some_and(|ty| {
                    let ty = ctx.types.get(*ty);
                    ty.dialect == trunk_ir::Symbol::new("tribute_rt")
                        && ty.name == trunk_ir::Symbol::new("anyref")
                }),
        );
    let has_done_k = hidden_count == 2;
    let result = ctx.types.get(data.params[0]);
    let returns_anyref = result.dialect == trunk_ir::Symbol::new("tribute_rt")
        && result.name == trunk_ir::Symbol::new("anyref");
    (
        params.len() - hidden_count,
        has_evidence,
        has_done_k,
        returns_anyref,
    )
}

fn function_param_names(ctx: &IrContext, module: &Module, target: &str) -> Vec<String> {
    let func_op = module
        .ops(ctx)
        .into_iter()
        .find_map(|op| {
            let func_op = func::Func::from_op(ctx, op).ok()?;
            (func_op.sym_name(ctx) == target).then_some(func_op)
        })
        .unwrap_or_else(|| panic!("missing function '{target}'"));
    let entry = ctx.region(func_op.body(ctx)).blocks[0];
    ctx.block(entry)
        .args
        .iter()
        .map(|arg| match arg.attrs.get("bind_name") {
            Some(trunk_ir::Attribute::Symbol(name)) => name.to_string(),
            _ => "_".to_owned(),
        })
        .collect()
}

#[test]
fn test_toplevel_calling_conventions_follow_ability_upper_bound() {
    let code = r#"
ability Logger {
    fn log(message: String) -> Nil
}

ability State {
    op get() -> Int
}

fn direct() -> Int { +1 }

fn explicit_pure() ->{} Int { +1 }

fn evidence_direct() ->{Logger} Int { +2 }

fn inferred_evidence_direct() {
    Logger::log("inferred")
}

fn call_evidence_direct() ->{Logger} Int {
    evidence_direct()
}

fn cps() ->{State} Int {
    State::get()
}

fn inferred_cps() -> Int {
    State::get()
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "calling_conventions.trb");

        assert_eq!(
            function_abi(&ctx, &module, "direct"),
            (0, false, false, false)
        );
        assert_eq!(
            function_abi(&ctx, &module, "evidence_direct"),
            (0, true, false, false)
        );
        assert_eq!(
            function_abi(&ctx, &module, "explicit_pure"),
            (0, false, false, false)
        );
        assert_eq!(
            function_abi(&ctx, &module, "call_evidence_direct"),
            (0, true, false, false)
        );
        assert_eq!(
            function_abi(&ctx, &module, "inferred_evidence_direct"),
            (0, true, false, false)
        );
        assert_eq!(function_abi(&ctx, &module, "cps"), (0, true, true, true));
        assert_eq!(
            function_abi(&ctx, &module, "inferred_cps"),
            (0, true, true, true)
        );
    });
}

#[test]
fn test_lifted_closure_physical_abi_interposes_environment() {
    let code = r#"
ability Logger {
    fn log(message: String) -> Nil
}

ability State {
    op get() -> Int
}

fn call_direct(f: fn(Int) ->{} Int) -> Int {
    f(+1)
}

fn call_logger(f: fn(Int) ->{Logger} Int) ->{Logger} Int {
    f(+1)
}

fn call_state(f: fn() ->{State} Int) ->{State} Int {
    f()
}

fn direct_closure() -> Int {
    call_direct(fn(x: Int) { x })
}

fn evidence_direct_closure() ->{Logger} Int {
    call_logger(fn(x: Int) {
        Logger::log("called")
        x
    })
}

fn cps_closure() ->{State} Int {
    call_state(fn() { State::get() })
}

fn main() { }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "closure_calling_conventions.trb");

        assert_eq!(
            function_param_names(&ctx, &module, "direct_closure::__clam_0"),
            ["__env", "x"]
        );
        assert_eq!(
            function_param_names(&ctx, &module, "evidence_direct_closure::__clam_0"),
            ["__evidence", "__env", "x"]
        );
        assert_eq!(
            function_param_names(&ctx, &module, "cps_closure::__clam_0"),
            ["__evidence", "__env", "__done_k"]
        );
    });
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
    apply(fn(n) { n + +1 }, +41)
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
        fn State::get() { +42 }
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
    State::set(n + +1)
    n
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        fn State::get() { +0 }
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
        fn State::get() { +42 }
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
            if let Ok(func_op) = func::Func::from_op(&ctx, op) {
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
