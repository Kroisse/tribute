//\! Shared test helpers for tribute-front integration tests.
#![allow(dead_code)]

use itertools::Itertools;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;
use tribute_front::analysis::{self, Environment};
use tribute_front::ast::{Decl, DeclTable, Expr, ExprKind, ResolvedRef, Stmt, TypedRef};
use trunk_ir::context::IrContext;
use trunk_ir::printer::print_module;

/// Prelude source, loaded at compile time relative to this file.
const PRELUDE_SOURCE: &str = include_str!("../../../../lib/std/prelude.trb");

#[salsa::tracked(returns(copy))]
fn prelude_source(db: &dyn salsa::Database) -> SourceCst {
    SourceCst::from_source_str(db, "prelude:///std/prelude", PRELUDE_SOURCE)
}

/// The environment the sources under test are analyzed in: package `std`
/// from the prelude, which declares the operators (`Int::(+)` etc.).
fn environment(db: &dyn salsa::Database) -> Environment {
    Environment::Std(prelude_source(db))
}

/// Inner tracked function that runs the full pipeline.
///
/// Diagnostics emitted during the pipeline are accumulated as Salsa
/// accumulators and can be collected by the caller.
#[salsa::tracked(returns(deref))]
fn run_ast_pipeline_inner(db: &dyn salsa::Database, source: SourceCst) -> String {
    let parsed = tribute_front::query::parsed_ast(db, source);
    assert!(parsed.is_some(), "Should parse successfully");

    let span_map = parsed.unwrap().span_map(db);

    let checked = analysis::analyze(db, source, environment(db)).expect("the source parses");
    // Specialization looks package std's declarations up but specializes only
    // the module under test, so its output holds no prelude bodies.
    let prelude = analysis::checked_std(db, prelude_source(db)).expect("the prelude parses");
    let compiler_intrinsics =
        tribute_front::ast_to_ir::registered_compiler_intrinsics(prelude.checked.module(db))
            .expect("prelude intrinsic directives must be supported");
    let input =
        tribute_front::monomorphize::SpecializationInput::new(db, checked, Some(prelude.checked))
            .with_compiler_intrinsics(compiler_intrinsics);
    let specialized = tribute_front::monomorphize::monomorphize_functions(db, input)
        .expect("checked instances must specialize");
    let mut ir = IrContext::new();
    let module = tribute_front::ast_to_ir::LoweringInput::from_specialization(
        db,
        specialized,
        tribute_front::ast_to_ir::LoweringEnvironment {
            span_map: tribute_front::ast::SpanMaps::new([span_map]),
            ..Default::default()
        },
    )
    .lower_to_ir(db, &mut ir, source.uri(db).as_str());
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        module.module,
        &module.compiler_intrinsics,
        &mut Default::default(),
    );
    assert!(
        validation.is_ok(),
        "logical frontend output must validate: {validation}"
    );
    print_module(&ir, module.module.op())
}

#[salsa::tracked]
fn run_frontend_pipeline_inner(db: &dyn salsa::Database, source: SourceCst) {
    let checked = analysis::analyze(db, source, environment(db));
    assert!(checked.is_some(), "Should parse successfully");
}

#[salsa::tracked]
fn tdnr_function_summary_inner(
    db: &dyn salsa::Database,
    source: SourceCst,
    function_name: String,
) -> TdnrSummary {
    let checked =
        analysis::analyze(db, source, environment(db)).expect("Should parse successfully");
    let body = function_body(checked.module(db), &function_name);
    let mut summary = TdnrSummary::default();
    collect_tdnr_summary(db, body, &mut summary);
    summary
}

/// Run the AST pipeline through TDNR and summarize one function body.
///
/// This intentionally stops before IR lowering so tests can inspect calls
/// left without a selected function.
pub fn tdnr_function_summary<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    function_name: &str,
) -> &'db TdnrSummary {
    tdnr_function_summary_inner(db, source, function_name.to_owned())
}

fn function_body<'db>(
    module: &'db DeclTable<TypedRef<'db>>,
    name: &str,
) -> &'db Expr<TypedRef<'db>> {
    module
        .decls()
        .find_map(|decl| match decl {
            Decl::Function(func) if func.name == name => Some(&func.body),
            _ => None,
        })
        .expect("function should exist")
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TdnrSummary {
    pub method_calls: Vec<String>,
    pub calls: Vec<TdnrCall>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TdnrCall {
    pub target: String,
    pub arg_count: usize,
}

fn collect_tdnr_summary<'db>(
    db: &'db dyn salsa::Database,
    expr: &Expr<TypedRef<'db>>,
    summary: &mut TdnrSummary,
) {
    match &*expr.kind {
        ExprKind::Call { callee, args, .. } => {
            if let ExprKind::Var(TypedRef {
                resolved: ResolvedRef::Function { id },
                ..
            }) = &*callee.kind
            {
                summary.calls.push(TdnrCall {
                    target: id.qualified(db).to_string(),
                    arg_count: args.len(),
                });
            }
            collect_tdnr_summary(db, callee, summary);
            for arg in args {
                collect_tdnr_summary(db, arg, summary);
            }
        }
        ExprKind::Functions { name, .. } => summary.method_calls.push(name.to_string()),
        ExprKind::Record { fields, spread, .. } => {
            for field in fields {
                collect_tdnr_summary(db, &field.value, summary);
            }
            if let Some(spread) = spread {
                collect_tdnr_summary(db, spread, summary);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Let { value, .. } | Stmt::Expr { expr: value, .. } => {
                        collect_tdnr_summary(db, value, summary);
                    }
                }
            }
            collect_tdnr_summary(db, value, summary);
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_tdnr_summary(db, lhs, summary);
            collect_tdnr_summary(db, rhs, summary);
        }
        ExprKind::Case { scrutinee, arms } => {
            collect_tdnr_summary(db, scrutinee, summary);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_tdnr_summary(db, guard, summary);
                }
                collect_tdnr_summary(db, &arm.body, summary);
            }
        }
        ExprKind::Lambda { body, .. } => collect_tdnr_summary(db, body, summary),
        ExprKind::Handle { body, handlers } => {
            collect_tdnr_summary(db, body, summary);
            for handler in handlers {
                collect_tdnr_summary(db, &handler.body, summary);
            }
        }
        ExprKind::Become { call } => collect_tdnr_summary(db, call, summary),
        ExprKind::Resume { arg, .. } => collect_tdnr_summary(db, arg, summary),
        ExprKind::Tuple(elements) | ExprKind::List(elements) => {
            for element in elements {
                collect_tdnr_summary(db, element, summary);
            }
        }
        ExprKind::Cons { args, .. } => {
            for arg in args {
                collect_tdnr_summary(db, arg, summary);
            }
        }
        ExprKind::Var(_)
        | ExprKind::NatLit(_)
        | ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::Nil
        | ExprKind::RuneLit(_)
        | ExprKind::Error => {}
    }
}

/// Run the full AST pipeline (parse -> resolve -> typecheck -> TDNR -> IR)
/// with the prelude loaded, and return the IR text.
///
/// Panics if any parse error diagnostics were emitted for the test source.
///
/// Uses `parsed_ast` to check only the test source for parse errors,
/// avoiding false positives from prelude diagnostics.
pub fn run_ast_pipeline_with_ir(db: &dyn salsa::Database, source: SourceCst) -> &str {
    // Check for parse errors in the test source (not prelude)
    let parse_diagnostics = tribute_front::query::parsed_ast::accumulated::<Diagnostic>(db, source);
    let parse_errors: Vec<_> = parse_diagnostics
        .iter()
        .filter(|d| {
            d.inner.severity == tribute_core::diagnostic::DiagnosticSeverity::Error
                && matches!(
                    d.phase,
                    tribute_core::diagnostic::CompilationPhase::Parsing
                        | tribute_core::diagnostic::CompilationPhase::AstGeneration
                )
        })
        .collect();
    assert!(
        parse_errors.is_empty(),
        "Expected no parse errors, but found {}:\n{}",
        parse_errors.len(),
        parse_errors
            .iter()
            .format_with("\n", |d, f| f(&format_args!(
                "  - [{}] {} (span: {}..{})",
                d.phase, d.inner.message, d.inner.span.start, d.inner.span.end
            )))
    );

    let ir = run_ast_pipeline_inner(db, source);
    // Production lowers only a module without errors.
    let errors: Vec<_> = run_ast_pipeline_inner::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|d| d.inner.severity == tribute_core::diagnostic::DiagnosticSeverity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "Expected no errors, but found {}:\n{}",
        errors.len(),
        errors.iter().format_with("\n", |d, f| f(&format_args!(
            "  - [{}] {}",
            d.phase, d.inner.message
        )))
    );
    ir
}

/// Run the full AST pipeline without returning IR text.
///
/// Panics if any error diagnostics were emitted during the pipeline.
pub fn run_ast_pipeline(db: &dyn salsa::Database, source: SourceCst) {
    let _ = run_ast_pipeline_with_ir(db, source);
}

/// Run the frontend through TDNR, without lowering invalid ASTs to IR.
pub fn ast_pipeline_diagnostics(db: &dyn salsa::Database, source: SourceCst) -> Vec<Diagnostic> {
    run_frontend_pipeline_inner(db, source);
    run_frontend_pipeline_inner::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect()
}

/// Run the frontend and return error diagnostic messages.
pub fn ast_pipeline_error_messages(db: &dyn salsa::Database, source: SourceCst) -> Vec<String> {
    ast_pipeline_diagnostics(db, source)
        .into_iter()
        .filter(|diagnostic| {
            diagnostic.inner.severity == tribute_core::diagnostic::DiagnosticSeverity::Error
        })
        .map(|diagnostic| diagnostic.inner.message)
        .collect()
}
