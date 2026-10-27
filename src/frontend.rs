//! Frontend assembly for Tribute.
//!
//! Tracked queries take a user source through parsing, name resolution, and
//! type checking against package std, merge std's bodies into the typed
//! module, check compiler intrinsics and runtime items, specialize, and lower
//! the result to source-logical arena IR. The stage chain in
//! [`crate::pipeline`] runs on that IR.

use crate::SourceCst;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tribute_core::diagnostic::{Diagnostic, DiagnosticSeverity};
use tribute_front::ast::SpanMaps;
use tribute_front::ast_to_ir;
use tribute_front::monomorphize::{SpecializationInput, SpecializationOutput};
use tribute_front::query as ast_query;
use tribute_front::resolve as ast_resolve;
use tribute_front::typeck as ast_typeck;
use trunk_ir::{IrContext, Module};

/// Parse source and run the frontend pipeline (parse → resolve → typecheck).
///
/// Returns the typed AST with span map, ready for `ast_to_ir` lowering.
/// Does NOT call `ast_to_ir` — that is the caller's responsibility.
///
/// The user package depends on package `std`:
/// 1. Parse user code to AST
/// 2. Build the user's PackageEnv, which depends on package `std`
/// 3. Resolve names through it and package `std`'s environment
/// 4. Type check, reading package `std`'s declarations through the
///    dependency table; the output holds only the user's declarations
#[salsa::tracked(returns(copy))]
pub fn parse_and_lower_ast<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<ast_typeck::TypeCheckOutput<'db>> {
    // Phase 1: Parse user code to AST
    let parsed = ast_query::parsed_ast(db, source)?;

    let user_ast = parsed.module(db);
    let span_map = parsed.span_map(db);
    tracing::debug!(
        "Phase 1: parsed AST has {} declarations",
        user_ast.decls.len()
    );

    // Phase 2: Build user env, which depends on package std
    let dependencies = crate::prelude::prelude_dependencies(db);
    let user_env = ast_resolve::build_env(db, user_ast, &dependencies);

    // Phase 3: Name resolution
    let resolved_ast =
        ast_resolve::resolve_with_env(db, user_ast, &user_env, dependencies, span_map);

    // Phase 4: Type checking against package std
    let mut checker = ast_typeck::TypeChecker::new(db, span_map);
    if let Some(p_exports) = crate::prelude::prelude_exports(db) {
        checker.depend_on(&p_exports);
    }
    let result = checker.check_module(&resolved_ast);

    tracing::debug!(
        "Phase 4: after typecheck, {} declarations, {} function_types, {} node_types",
        result.module.entries.len(),
        result.function_types.len(),
        result.node_types.len()
    );

    Some(ast_typeck::TypeCheckOutput::from_check(db, result))
}

/// Merge prelude decls into user's typed AST and lower to arena IR.
///
/// This performs AST-level prelude merge (prepending prelude decls to user decls),
/// then runs `ast_to_ir` on the merged module.
///
/// Returns the logical frontend result, including the metadata required by the
/// shared CPS boundary.
pub(crate) fn merge_and_lower_to_ir<'db>(
    db: &'db dyn salsa::Database,
    typed: &ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> FrontendCompilation {
    let prepared = prepare_frontend_details(db, *typed, source)
        .expect("frontend instances must be checked before lowering");
    let mut context = IrContext::new();
    let frontend = ast_to_ir::LoweringInput::from_specialization(
        db,
        prepared.specialized,
        ast_to_ir::LoweringEnvironment {
            span_map: frontend_spans(db, source),
            runtime_types: prepared.runtime_types.clone(),
            merged_sources: vec![crate::prelude::PRELUDE_URI.to_owned()],
        },
    )
    .lower_to_ir(db, &mut context, source.uri(db).as_str());
    FrontendCompilation {
        context,
        module: frontend.module,
        compiler_intrinsics: frontend.compiler_intrinsics,
    }
}

/// Specialized frontend output, with the runtime types root-crate
/// preparation supplies to lowering.
#[derive(Clone, PartialEq, Eq, salsa::SalsaValue)]
struct PreparedFrontend<'db> {
    specialized: SpecializationOutput<'db>,
    runtime_types: ast_to_ir::runtime_items::RuntimeTypes,
}

/// Merge and specialize inside a tracked query so specialization failures
/// become source diagnostics before either public IR lowering route runs.
#[salsa::tracked(returns(copy))]
pub fn prepare_frontend_for_lowering<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<SpecializationOutput<'db>> {
    prepare_frontend_details(db, typed, source).map(|prepared| prepared.specialized)
}

#[salsa::tracked(returns(as_ref))]
fn prepare_frontend_details<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<PreparedFrontend<'db>> {
    // Type checking reads package std's declarations without copying them.
    // Specialization takes them as one table with the user's, and std's
    // bodies and per-expression metadata ahead of the user's.
    let prelude = crate::prelude::prelude_module(db);
    let mut input = SpecializationInput::new(db, typed, prelude);
    if let Some(prelude) = prelude {
        input = input.merge_bodies(db, prelude);
    }
    let span_map = frontend_spans(db, source);

    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(input.module()) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                directive.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };
    // Without a prelude there are no runtime declarations to check.
    let runtime_items = prelude.map_or(&[][..], |_| tribute_ir::runtime_items::RUNTIME_ITEMS);
    let function_types: HashMap<_, _> = input
        .declarations()
        .function_types
        .iter()
        .cloned()
        .collect();
    let runtime_types = match ast_to_ir::runtime_items::check_runtime(
        db,
        &ast_to_ir::runtime_items::RuntimeInputs {
            module: input.module(),
            function_types: &function_types,
        },
        tribute_ir::runtime_items::RUNTIME_MODULE,
        runtime_items,
    ) {
        Ok(types) => types,
        Err(errors) => {
            for error in errors {
                error.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };

    let input = input.with_compiler_intrinsics(compiler_intrinsics);
    let specialized = match tribute_front::monomorphize::monomorphize_functions(db, input) {
        Ok(specialized) => specialized,
        Err(errors) => {
            for error in errors {
                error.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };
    Some(PreparedFrontend {
        specialized,
        runtime_types,
    })
}

/// The span maps of a user source and of package std, which locate the
/// nodes of the module merged from both.
pub(crate) fn frontend_spans(db: &dyn salsa::Database, source: SourceCst) -> SpanMaps<'_> {
    let prelude = crate::prelude::prelude_span_map(db);
    SpanMaps::new(ast_query::span_map(db, source).into_iter().chain(prelude))
}

/// Arena IR together with the registered compiler intrinsics the shared CPS
/// conversion checks. They stay private; [`run_shared_middle_end`](crate::pipeline::run_shared_middle_end) is their
/// only consumer.
#[derive(Clone)]
pub struct FrontendCompilation {
    pub(crate) context: IrContext,
    pub(crate) module: Module,
    pub(crate) compiler_intrinsics:
        Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
}

/// Run frontend (parse → typecheck) and lower to arena IR.
///
/// Returns `None` if parsing fails. Otherwise returns arena IR ready
/// for in-place passes, avoiding unnecessary Salsa↔Arena round-trips.
pub fn compile_frontend(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<(IrContext, Module)> {
    let compilation = compile_frontend_for_shared_route(db, source)?;
    Some((compilation.context, compilation.module))
}

/// Run the frontend for the shared middle-end.
///
/// Returns `None` if parsing fails or the frontend reports an error.
pub fn compile_frontend_for_shared_route(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<FrontendCompilation> {
    let typed = parse_and_lower_ast(db, source)?;
    let has_frontend_errors = parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
        .iter()
        .any(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error);
    if has_frontend_errors {
        return None;
    }
    prepare_frontend_for_lowering(db, typed, source)?;
    Some(merge_and_lower_to_ir(db, &typed, source))
}
