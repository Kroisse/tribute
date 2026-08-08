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
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::printer::print_module;
use trunk_ir::rewrite::Module;

/// Helper to compile code through AST pipeline and return arena IR.
fn compile_to_ir(db: &dyn salsa::Database, code: &str, name: &str) -> (IrContext, Module) {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);
    let tribute::pipeline::FrontendCompilation {
        context: mut ctx,
        module: m,
        operation_declarations,
    } = tribute::pipeline::compile_frontend(db, source_file).expect("compilation should succeed");
    let core_module = trunk_ir::dialect::core::Module::from_op(&ctx, m.op())
        .expect("frontend output must be a core.module");
    let mut pm = trunk_ir::pass::PassManager::new();
    pm.add_pass(
        tribute_passes::tribute_control_to_cps::TributeControlToCps::new(operation_declarations),
    )
    .add_pass(tribute_passes::lower_closure_lambda::LowerClosureLambda);
    pm.run(&mut ctx, core_module).unwrap();
    (ctx, m)
}

/// Compile only to the source-logical boundary.
fn compile_source_logical_ir(
    db: &dyn salsa::Database,
    code: &str,
    name: &str,
) -> (IrContext, Module) {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    let source_file = SourceCst::from_path(db, name, source_code, tree);
    let tribute::pipeline::FrontendCompilation {
        context, module, ..
    } = tribute::pipeline::compile_frontend(db, source_file).expect("compilation should succeed");
    (context, module)
}

fn logical_function_text<'a>(ir: &'a str, name: &str) -> &'a str {
    let unquoted = format!("tribute_control.func @{name}(");
    let quoted = format!("tribute_control.func @\"{name}\"(");
    let start = ir
        .find(&unquoted)
        .or_else(|| ir.find(&quoted))
        .unwrap_or_else(|| panic!("missing logical function `{name}`:\n{ir}"));
    let tail = &ir[start..];
    let end = tail[1..]
        .find("\n  tribute_control.func ")
        .map_or(tail.len(), |offset| offset + 1);
    &tail[..end]
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

fn is_type(ctx: &IrContext, ty: trunk_ir::refs::TypeRef, dialect: &str, name: &str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == trunk_ir::Symbol::from_dynamic(dialect)
        && data.name == trunk_ir::Symbol::from_dynamic(name)
}

/// Whether a Cps worker owns the complete result-indexed callback ABI.
///
/// The worker receives `Evidence, Done<R>, Dispatch<R>` and returns
/// `core.never`.  The Dispatch's Resume closure in turn proves the exact
/// immutable `Parent<R>` layout, rather than relying on an erased carrier.
fn has_cps_callback_params(
    ctx: &IrContext,
    result: trunk_ir::refs::TypeRef,
    evidence: trunk_ir::refs::TypeRef,
    done: trunk_ir::refs::TypeRef,
    dispatch: trunk_ir::refs::TypeRef,
) -> bool {
    if !is_type(ctx, result, "core", "never")
        || !tribute_ir::dialect::ability::is_evidence_type_ref(ctx, evidence)
    {
        return false;
    }

    let Some(done_func) = tribute_core::cps_closure_function_type(ctx, done) else {
        return false;
    };
    let done_params = &ctx.types.get(done_func).params;
    if done_params.len() != 2 || !is_type(ctx, done_params[0], "core", "never") {
        return false;
    }
    let source_result = done_params[1];

    let Some(dispatch_func) = tribute_core::cps_closure_function_type(ctx, dispatch) else {
        return false;
    };
    let dispatch_params = &ctx.types.get(dispatch_func).params;
    let [
        dispatch_result,
        dispatch_evidence,
        resume,
        prompt,
        ability,
        operation,
        payload,
    ] = dispatch_params.as_slice()
    else {
        return false;
    };
    if !is_type(ctx, *dispatch_result, "core", "never")
        || *dispatch_evidence != evidence
        || !is_type(ctx, *prompt, "core", "i32")
        || !is_type(ctx, *ability, "core", "i32")
        || !is_type(ctx, *operation, "core", "i32")
        || !is_type(ctx, *payload, "tribute_rt", "anyref")
    {
        return false;
    }

    let Some(resume_func) = tribute_core::cps_closure_function_type(ctx, *resume) else {
        return false;
    };
    let resume_params = &ctx.types.get(resume_func).params;
    let [resume_result, resume_evidence, parent, input] = resume_params.as_slice() else {
        return false;
    };
    is_type(ctx, *resume_result, "core", "never")
        && *resume_evidence == evidence
        && is_type(ctx, *input, "tribute_rt", "anyref")
        && tribute_core::cps_parent_result_type(ctx, *parent) == Some(source_result)
        && tribute_core::has_canonical_cps_parent_layout(ctx, *parent, dispatch)
}

fn has_cps_callback_abi(ctx: &IrContext, func_ty: trunk_ir::refs::TypeRef) -> bool {
    let params = &ctx.types.get(func_ty).params;
    let [result, evidence, done, dispatch, ..] = params.as_slice() else {
        return false;
    };
    has_cps_callback_params(ctx, *result, *evidence, *done, *dispatch)
}

/// Return `(source parameter count, has evidence, has exact CPS callbacks, returns never)`.
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
    let has_cps_callbacks = has_cps_callback_abi(ctx, func_ty);
    let hidden_count = if has_cps_callbacks {
        3
    } else {
        usize::from(has_evidence)
    };
    let returns_never = is_type(ctx, data.params[0], "core", "never");
    (
        params.len() - hidden_count,
        has_evidence,
        has_cps_callbacks,
        returns_never,
    )
}

fn function_type(ctx: &IrContext, module: &Module, target: &str) -> trunk_ir::refs::TypeRef {
    let func_op = module
        .ops(ctx)
        .into_iter()
        .find_map(|op| {
            let func_op = func::Func::from_op(ctx, op).ok()?;
            (func_op.sym_name(ctx) == target).then_some(func_op)
        })
        .unwrap_or_else(|| panic!("missing function '{target}'"));
    func_op.r#type(ctx)
}

fn function_param_types(
    ctx: &IrContext,
    module: &Module,
    target: &str,
) -> Vec<trunk_ir::refs::TypeRef> {
    ctx.types.get(function_type(ctx, module, target)).params[1..].to_vec()
}

fn assert_no_control_unrealized_casts(ctx: &IrContext, module: &Module) {
    fn is_control_type(ctx: &IrContext, ty: trunk_ir::refs::TypeRef) -> bool {
        let data = ctx.types.get(ty);
        (data.dialect == trunk_ir::Symbol::new("adt")
            && data.name == trunk_ir::Symbol::new("typeref")
            && data.attrs.get_type("tribute.cps_parent_result").is_some())
            || (data.dialect == trunk_ir::Symbol::new("closure")
                && data.name == trunk_ir::Symbol::new("closure")
                && data.attrs.get_i128("tribute.calling_convention") == Some(2))
    }

    fn visit(ctx: &IrContext, op: trunk_ir::refs::OpRef) {
        if core::UnrealizedConversionCast::matches(ctx, op) {
            let is_control = ctx
                .op_operands(op)
                .iter()
                .any(|value| is_control_type(ctx, ctx.value_ty(*value)))
                || ctx
                    .op_results(op)
                    .iter()
                    .any(|value| is_control_type(ctx, ctx.value_ty(*value)));
            assert!(
                !is_control,
                "Parent and CPS closure values must not cross an unrealized conversion cast"
            );
        }
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child);
                }
            }
        }
    }

    for op in module.ops(ctx) {
        visit(ctx, op);
    }
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

        let direct = function_param_types(&ctx, &module, "direct_closure::__clam_0");
        assert!(
            matches!(direct.as_slice(), [environment, value]
                if is_type(&ctx, *environment, "tribute_rt", "anyref")
                    && is_type(&ctx, *value, "core", "i32")),
            "Direct closure must interpose only its environment: {direct:?}"
        );

        let evidence_direct =
            function_param_types(&ctx, &module, "evidence_direct_closure::__clam_0");
        assert!(
            matches!(evidence_direct.as_slice(), [evidence, environment, value]
                if tribute_ir::dialect::ability::is_evidence_type_ref(&ctx, *evidence)
                    && is_type(&ctx, *environment, "tribute_rt", "anyref")
                    && is_type(&ctx, *value, "core", "i32")),
            "EvidenceDirect closure must interpose env after evidence: {evidence_direct:?}"
        );

        let cps_type = function_type(&ctx, &module, "cps_closure::__clam_0");
        let cps = &ctx.types.get(cps_type).params;
        assert!(
            matches!(cps.as_slice(), [result, evidence, environment, done, dispatch]
                if is_type(&ctx, *environment, "tribute_rt", "anyref")
                    && has_cps_callback_params(&ctx, *result, *evidence, *done, *dispatch)),
            "CPS closure must interpose env after evidence and preserve exact Done/Dispatch: {cps:?}"
        );
    });
}

#[test]
fn fn_handler_dispatchers_use_the_erased_payload_abi_after_environment_interposition() {
    let code = r#"
ability Ask {
    fn ask() -> Nat
}

ability Pair {
    fn add(left: Nat, right: Nat) -> Nat
}

fn call_ask() ->{Ask} Nat { Ask::ask() }
fn call_pair() ->{Pair} Nat { Pair::add(15, 27) }

fn one() -> Nat {
    handle call_ask() {
        do value { value }
        fn Ask::ask() { 42 }
    }
}

fn two() -> Nat {
    handle call_pair() {
        do value { value }
        fn Pair::add(left, right) { left + right }
    }
}
"#;

    fn is_type(ctx: &IrContext, ty: trunk_ir::TypeRef, dialect: &str, name: &str) -> bool {
        let data = ctx.types.get(ty);
        data.dialect == trunk_ir::Symbol::from_dynamic(dialect)
            && data.name == trunk_ir::Symbol::from_dynamic(name)
    }

    fn count_struct_gets(ctx: &IrContext, op: trunk_ir::refs::OpRef) -> usize {
        let mut count = usize::from(adt::StructGet::from_op(ctx, op).is_ok());
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    count += count_struct_gets(ctx, child);
                }
            }
        }
        count
    }

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "fn_handler_dispatcher_abi.trb");
        let mut payload_field_counts = Vec::new();
        for op in module.ops(&ctx) {
            let Ok(function) = func::Func::from_op(&ctx, op) else {
                continue;
            };
            if ctx.op(op).attributes.get_i128("tribute.calling_convention") != Some(1) {
                continue;
            }
            let ty = ctx.types.get(function.r#type(&ctx));
            let params = &ty.params[1..];
            let is_dispatcher = ty.params.len() == 5
                && is_type(&ctx, ty.params[0], "tribute_rt", "anyref")
                && tribute_ir::dialect::ability::is_evidence_type_ref(&ctx, params[0])
                && is_type(&ctx, params[1], "tribute_rt", "anyref")
                && is_type(&ctx, params[2], "core", "i32")
                && is_type(&ctx, params[3], "tribute_rt", "anyref");
            if is_dispatcher {
                payload_field_counts.push(count_struct_gets(&ctx, op));
            }
        }

        assert_eq!(payload_field_counts.len(), 2, "{payload_field_counts:?}");
        assert!(
            payload_field_counts.iter().any(|count| *count >= 3),
            "the two-argument arm must unpack the canonical payload product: {payload_field_counts:?}"
        );

        let printed = print_module(&ctx, module.op());
        let mut reparsed = IrContext::new();
        trunk_ir::parser::parse_test_module(&mut reparsed, &printed);
    });
}

#[test]
fn test_generic_recursive_handler_call_specializes_without_control_casts() {
    let code = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn set_then_get() ->{State(Int)} Int {
    State::set(+100)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(value) { run_state(fn() { resume Nil }, value) }
    }
}

fn main() {
    let _ = run_state(fn() { set_then_get() }, +0)
}
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_to_ir(db, code, "generic_recursive_handler.trb");
        let canonical = print_module(&ctx, module.op());
        let names = get_functions_with_evidence(&ctx, &module)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert!(
            names.iter().any(|name| name == "run_state$Int$Int"),
            "the recursive concrete handler call must use its exact specialization: {names:?}"
        );
        assert!(
            canonical.contains("func.func @\"run_state$Int$Int\"(")
                || canonical.contains("func.func @run_state$Int$Int("),
            "canonical post-CPS IR must retain the exact specialized worker:\n{canonical}"
        );
        assert_no_control_unrealized_casts(&ctx, &module);
    });
}

#[test]
fn test_transitive_generic_handler_specialization_retains_exact_control_types() {
    let code = r#"
ability A {
    op do_a() -> Nat
}

fn run_b(value: a) -> a { value }

fn run_a(comp: fn() ->{e, A} a) ->{e} a {
    handle comp() {
        do result { run_b(result) }
        op A::do_a() { run_a(fn() { resume 10 }) }
    }
}

fn main() {
    let _ = run_a(fn() { A::do_a() })
}
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let (ctx, module) = compile_source_logical_ir(db, code, "transitive_generic_handler.trb");
        let logical = print_module(&ctx, module.op());
        for name in ["run_a", "run_b", "run_a$Nat", "run_b$Nat"] {
            assert_eq!(
                logical
                    .lines()
                    .filter(|line| {
                        let line = line.trim_start();
                        line.starts_with(&format!("tribute_control.func @{name}("))
                            || line.starts_with(&format!("tribute_control.func @\"{name}\"("))
                    })
                    .count(),
                1,
                "expected one retained/generated `{name}`:\n{logical}"
            );
        }
        let concrete_run_a = logical_function_text(&logical, "run_a$Nat");
        assert!(
            concrete_run_a.contains("callee = @\"run_b$Nat\"")
                && concrete_run_a.contains(": core.i32"),
            "the generated handler must rewrite its deferred call to run_b$Nat:\n{logical}"
        );
        let concrete_run_b = logical_function_text(&logical, "run_b$Nat");
        assert!(
            concrete_run_b.contains("tribute_control.return"),
            "the deferred helper must have one concrete Nat body:\n{concrete_run_b}"
        );
        let generic_run_a = logical_function_text(&logical, "run_a");
        assert!(
            generic_run_a.contains("tribute_rt.anyref"),
            "the retained generic body may erase only source data:\n{logical}"
        );
        for (name, function) in [("run_a", generic_run_a), ("run_a$Nat", concrete_run_a)] {
            assert!(
                function.contains("tribute_control.lambda") && function.contains("convention(cps)"),
                "the resumptive nested lambda in `{name}` must be CPS directly:\n{function}"
            );
            assert!(
                !function.contains("convention(direct)"),
                "`{name}` must not wrap a Direct resumptive lambda in CPS:\n{function}"
            );
        }
        assert!(
            !generic_run_a.contains("core.unrealized_conversion_cast")
                && !concrete_run_a.contains("core.unrealized_conversion_cast")
                && !concrete_run_b.contains("core.unrealized_conversion_cast"),
            "generic lowering must never cast CPS control identity:\n{logical}"
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
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }

fn run() ->{} Int {
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
                if ctx.op(op).regions.is_empty() {
                    continue;
                }
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
