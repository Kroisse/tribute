//! Compilation pipeline for Tribute.
//!
//! This module orchestrates the compilation stages with centralized control flow.
//! Tracked queries prepare the typed frontend. Shared and target passes then
//! mutate one arena session under this module's pass ordering.
//!
//! ## Architecture Principles
//!
//! 1. **Explicit Boundaries**: Verify source-logical, shared CPS, and target contracts;
//!    the representation/ABI boundary exit verifies the physical contracts later
//!    passes consume, and native validation checks direct-call argument and
//!    result compatibility later, before object emission
//! 2. **Centralized Orchestration**: Pass sequencing is managed here, not in passes
//! 3. **Scoped Caching**: Salsa caches frontend queries; arena passes own IR analyses
//! 4. **Separation of Concerns**: Pass implementation vs pipeline composition
//!
//! ## Pipeline Stages
//!
//! ```text
//! SourceCst
//!     │
//!     ▼ parse_cst + lower_cst
//! Module (tribute.* ops)
//!     │
//!     ▼ merge_with_prelude
//! Module (with prelude definitions)
//!     │
//!     ├─────────────── Frontend Passes ───────────────┤
//!     ▼ resolve
//! Module (resolved names)
//!     │
//!     ▼ typecheck
//! Typed Module
//!     │
//!     ▼ tdnr
//! Module (UFCS resolved)
//!     │
//!     ├─── Shared Middle-End (single arena session) ──┤
//!     ▼ ast_to_ir
//! Module (source-logical callable/control IR)
//!     │
//!     ▼ global DCE (artifact compiles only: unreachable source-logical
//!     │             functions skip CPS legalization)
//!     ▼ tribute_control_to_cps → lower_closure_lambda → intrinsic/list/io lowering
//! Module (CPS callable contracts and explicit evidence)
//!     │
//!     ▼ lower_ability_perform (CPS tail-call)
//! Module (ability.perform/call lowered to effect.dispatch_*)
//!     │
//!     ▼ resolve_evidence
//! Module (handler evidence setup lowered to effect.extend)
//!     │
//!     ▼ lower_handle_dispatch
//! Module (ability.handle_dispatch lowered)
//!     │
//!     ├─── Representation/ABI Boundary (run_target_to_boundary_exit) ──┤
//!     ▼ global DCE (unreachable functions skip target lowering)
//!     ▼ inline_functions
//!     ▼ target ABI validation → CPS signature physicalization
//!     ▼ root entry bridge
//!     ▼ lower-prepared-closures
//!     ▼ target evidence lowering ([wasm] evidence_to_wasm, [native] evidence_to_native)
//!     ▼ bytes intrinsic bridge
//!     ▼ closure storage layout finalization
//!     ▼ cleanup (global DCE, canonicalize, DCE, cast materialization)
//!     ▼ boundary exit verification (enforced in every build)
//! Module (physical contracts verified)  ◄── dump_ir (`--dump-ir`)
//!     │
//!     ├─── After the Boundary Exit (emit_from_boundary_exit) ──┤
//!     ├─► [wasm]   compile_to_wasm: lower_to_wasm → cast legalization → emit
//!     └─► [native] prepare_module_to_native: entrypoint, clif lowering,
//!                  RTTI/RC (◄── dump_native_ir_at_stage)
//!                  → validate_clif_ir (direct-call argument/result types) → emit
//! ```
//!
//! ## Diagnostics
//!
//! Diagnostics are collected using Salsa accumulators. Each stage can emit
//! diagnostics via `Diagnostic::new(...).accumulate(db)`, which are then
//! collected at the end of compilation.

use crate::SourceCst;
use itertools::Itertools;
use ropey::Rope;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tree_sitter::Parser;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::source_file::parse_with_rope;
use tribute_passes::generic_type_converter;
use trunk_ir::Span;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::conversion::{
    UnrealizedCastConversionPattern, materialize_unrealized_casts, reconcile_unrealized_casts,
};
use trunk_ir::dialect::{core as core_dialect, func as func_dialect};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{PassError, PassManager, PassResult, PassRunError};
use trunk_ir::rewrite::PatternApplicator;
use trunk_ir::{IrContext, Module};
use trunk_ir_wasm_backend::passes::reference_upcast_elision::ReferenceUpcastElisionPattern;

/// Error returned while dumping shared or target-specific IR.
#[derive(Debug, Clone, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("{message}")]
pub struct DumpIrError {
    message: String,
}

impl From<PassError> for DumpIrError {
    fn from(error: PassError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<tribute_passes::bytes_intrinsic::BytesIntrinsicError> for DumpIrError {
    fn from(error: tribute_passes::bytes_intrinsic::BytesIntrinsicError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<tribute_passes::target_abi::TargetAbiError> for DumpIrError {
    fn from(error: tribute_passes::target_abi::TargetAbiError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

// =============================================================================
// Compilation configuration
// =============================================================================

/// Compilation options threaded through the pipeline via Salsa.
#[salsa::input]
pub struct CompilationConfig {
    /// Enable AddressSanitizer instrumentation.
    #[returns(copy)]
    pub sanitize_address: bool,
    /// Stage-specific optimization policies.
    #[returns(copy)]
    pub optimizations: OptimizationOptions,
}

/// Optimization policies for the source-logical production pipeline.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OptimizationOptions {
    pub native: NativeOptimizationOptions,
}

impl OptimizationOptions {
    pub const fn production() -> Self {
        Self {
            native: NativeOptimizationOptions::production(),
        }
    }

    /// Disable optional native optimizations. Source-logical legalization is
    /// unchanged. Source-logical CPS legalization always runs.
    pub const fn baseline() -> Self {
        Self {
            native: NativeOptimizationOptions::baseline(),
        }
    }
}

/// Independently selectable native-backend optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeOptimizationOptions {
    pub paired_rc_elimination: PairedRcEliminationPolicy,
    pub borrowed_parameters: BorrowedParameterPolicy,
    pub temporary_borrows: TemporaryBorrowPolicy,
}

impl NativeOptimizationOptions {
    pub const fn production() -> Self {
        Self {
            paired_rc_elimination: PairedRcEliminationPolicy::Enabled,
            borrowed_parameters: BorrowedParameterPolicy::ElideProvenBorrowed,
            temporary_borrows: TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        }
    }

    pub const fn baseline() -> Self {
        Self {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
        }
    }
}

/// Policy for local retain/release pair elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PairedRcEliminationPolicy {
    Disabled,
    Enabled,
}

/// Policy for eliding ownership of proven borrowed native parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorrowedParameterPolicy {
    Preserve,
    ElideProvenBorrowed,
}

/// Policy for eliding ownership of proven field-derived native temporaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporaryBorrowPolicy {
    Preserve,
    ElideProvenFieldBorrows,
}

impl Default for OptimizationOptions {
    fn default() -> Self {
        Self::production()
    }
}

/// Stable native-pipeline boundaries available to optimization tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativePipelineStage {
    /// Immediately after reference-counting operations are inserted.
    AfterRcInsertion,
    /// After borrowed-parameter-aware insertion, before paired elimination.
    AfterBorrowedParameterOptimization,
    /// After temporary field-borrow-aware insertion, before paired elimination.
    AfterTemporaryBorrowOptimization,
    /// After optional local RC optimization, before cast resolution and lowering.
    AfterRcOptimization,
}

// AST-based pipeline imports
use tribute_front::ast::visit::{Visit, walk_expr, walk_module};
use tribute_front::ast::{Expr, ExprKind, SpanMap, TypedRef};
use tribute_front::ast_to_ir;
use tribute_front::astgen::ParsedAst;
use tribute_front::query as ast_query;
use tribute_front::resolve as ast_resolve;
use tribute_front::resolve::ModuleEnv;
use tribute_front::tdnr as ast_tdnr;
use tribute_front::typeck as ast_typeck;
use tribute_front::typeck::PreludeExports;
use trunk_ir_cranelift_backend::passes::{arith_to_clif, cf_to_clif, func_to_clif, mem_to_clif};
use trunk_ir_cranelift_backend::{
    CompilationResult as NativeCompilationResult, emit_module_to_native,
};
use trunk_ir_wasm_backend::{
    CompilationError, CompilationResult as WasmCompilationResult, WasmBinary,
};

// =============================================================================
// Standard Library Prelude
// =============================================================================

/// The prelude source code, embedded at compile time.
const PRELUDE_SOURCE: &str = include_str!("../lib/std/prelude.trb");

/// The URI the prelude source is registered under.
const PRELUDE_URI: &str = "prelude:///std/prelude";

/// Parse the prelude source.
///
/// This is the first stage of prelude processing, shared by all prelude-related functions.
/// Returns both the parsed AST and the SourceCst to avoid redundant creation.
#[salsa::tracked(returns(copy))]
fn parse_prelude<'db>(db: &'db dyn salsa::Database) -> Option<(ParsedAst<'db>, crate::SourceCst)> {
    let prelude_source = create_prelude_source(db)?;
    let parsed = ast_query::parsed_ast_with_module_path(
        db,
        prelude_source,
        trunk_ir::Symbol::new("prelude"),
    )?;
    Some((parsed, prelude_source))
}

/// Parse, resolve, type check, and apply TDNR to the prelude once.
///
/// Returns both the typed prelude, which user modules import methods from and
/// merge before lowering, and the exports injected into their type checking.
/// Salsa caches the result for all subsequent compilations.
#[salsa::tracked(returns(copy))]
fn checked_prelude<'db>(
    db: &'db dyn salsa::Database,
) -> Option<(ast_typeck::TypeCheckOutput<'db>, PreludeExports<'db>)> {
    let (parsed, _) = parse_prelude(db)?;
    let prelude_ast = &ast_resolve::library_package_module(parsed.module(db));
    let span_map = parsed.span_map(db);
    let prelude_env = ast_resolve::build_env(db, prelude_ast);
    let resolved =
        ast_resolve::resolve_library_with_env(db, prelude_ast, prelude_env, span_map.clone());

    // Typecheck with independent TypeContext (all UniVars resolved)
    let checker = ast_typeck::TypeChecker::new(db, span_map.clone());
    let (result, exports) = checker.check_prelude(&resolved);

    // TDNR for remaining MethodCall → Call AST transformations
    let mut tdnr_ast = result.module;
    ast_tdnr::resolve_tdnr(db, &mut tdnr_ast, std::iter::empty());

    let typed = ast_typeck::TypeCheckOutput::new(
        db,
        tdnr_ast,
        result.function_types,
        result.constructor_types.into(),
        ast_typeck::ExpressionTypeMetadata {
            node_types: result.node_types,
            function_instances: result.function_instances,
            local_instances: result.local_instances,
            evidence_plans: result.evidence_plans,
        },
        result.ability_conventions,
        ast_typeck::ability_schemas(&result.ability_definitions),
        result.handler_operations,
        result.perform_operations,
        result.lambda_signatures,
        result.exhaustive_cases,
        result.well_known_types,
        span_map,
    );
    Some((typed, exports))
}

/// The prelude's typed AST (parse → resolve → typecheck → TDNR), without
/// lowering to TrunkIR. The caller is responsible for `ast_to_ir`.
fn prelude_module(db: &dyn salsa::Database) -> Option<ast_typeck::TypeCheckOutput<'_>> {
    checked_prelude(db).map(|(typed, _)| typed)
}

/// Create a SourceCst for the prelude.
fn create_prelude_source(db: &dyn salsa::Database) -> Option<crate::SourceCst> {
    let uri = fluent_uri::Uri::parse_from(PRELUDE_URI.to_owned()).expect("valid prelude URI");
    let text: Rope = PRELUDE_SOURCE.into();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None)?;
    Some(crate::SourceCst::new(db, uri, text, Some(tree)))
}

/// Get prelude's ModuleEnv for name resolution.
///
/// This parses the prelude to AST and builds its module environment.
/// Cached by Salsa - computed once and reused.
#[salsa::tracked(returns(as_ref))]
fn prelude_env<'db>(db: &'db dyn salsa::Database) -> Option<ModuleEnv<'db>> {
    let (parsed, _) = parse_prelude(db)?;
    let prelude_ast = ast_resolve::library_package_module(parsed.module(db));
    Some(ast_resolve::build_env(db, &prelude_ast))
}

/// The prelude's type exports (TypeSchemes only, no UniVars), injected into
/// user module type checking.
fn prelude_exports(db: &dyn salsa::Database) -> Option<PreludeExports<'_>> {
    checked_prelude(db).map(|(_, exports)| exports)
}

/// Merge prelude decls into user's typed AST and lower to arena IR.
///
/// This performs AST-level prelude merge (prepending prelude decls to user decls),
/// then runs `ast_to_ir` on the merged module.
///
/// Returns the logical frontend result, including the metadata required by the
/// shared CPS boundary.
fn merge_and_lower_to_ir<'db>(
    db: &'db dyn salsa::Database,
    typed: &ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> FrontendCompilation {
    let (context, frontend) =
        merge_and_lower_to_ir_with(db, typed, source, |typed, db, ir, source_uri| {
            typed.lower_to_ir(db, ir, source_uri)
        });
    FrontendCompilation {
        context,
        module: frontend.module,
        operation_declarations: frontend.operation_declarations,
        compiler_intrinsics: frontend.compiler_intrinsics,
    }
}

/// Specialized frontend output together with the exact compiler-intrinsic
/// identities the shared CPS boundary needs.
#[derive(Clone, PartialEq, Eq, salsa::SalsaValue)]
struct PreparedFrontend<'db> {
    typed: ast_typeck::TypeCheckOutput<'db>,
    compiler_intrinsics: HashMap<tribute_front::ast::NodeId, trunk_ir::Symbol>,
}

/// Merge and specialize inside a tracked query so specialization failures
/// become source diagnostics before either public IR lowering route runs.
#[salsa::tracked(returns(copy))]
pub fn prepare_frontend_for_lowering<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<ast_typeck::TypeCheckOutput<'db>> {
    prepare_frontend_details(db, typed, source).map(|prepared| prepared.typed)
}

#[salsa::tracked(returns(as_ref))]
fn prepare_frontend_details<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    _source: SourceCst,
) -> Option<PreparedFrontend<'db>> {
    use tribute_front::ast::TypedRef;

    let user_module = typed.module(db);
    let user_fn_types = typed.function_types(db);
    let user_node_types = &typed.expression_types(db).node_types;
    let user_span_map = typed.span_map(db);

    // Merge prelude at AST level
    let user_ability_conventions = typed.ability_conventions(db);
    let (
        merged_module,
        merged_fn_types,
        merged_node_types,
        merged_ability_conventions,
        merged_span_map,
    ) = if let Some(prelude) = prelude_module(db) {
        let prelude_module_ast = prelude.module(db);
        let prelude_fn_types = prelude.function_types(db);
        let prelude_node_types = &prelude.expression_types(db).node_types;
        let prelude_ability_conventions = prelude.ability_conventions(db);
        let prelude_span_map = prelude.span_map(db);

        // Prepend prelude decls before user decls
        let mut merged_decls = prelude_module_ast.decls.clone();
        merged_decls.extend(user_module.decls.iter().cloned());

        let merged_ast = tribute_front::ast::Module::<TypedRef<'db>>::new(
            user_module.id,
            user_module.name.clone(),
            merged_decls,
        );

        // Merge function_types: prelude first, user overrides
        let mut fn_types: HashMap<_, _> = prelude_fn_types.iter().cloned().collect();
        fn_types.extend(user_fn_types.iter().cloned());

        // Merge node_types: prelude first, user overrides
        let mut node_types: HashMap<_, _> = prelude_node_types.iter().cloned().collect();
        node_types.extend(user_node_types.iter().cloned());

        let mut ability_conventions: HashMap<_, _> =
            prelude_ability_conventions.iter().cloned().collect();
        ability_conventions.extend(user_ability_conventions.iter().cloned());

        // Merge span maps (user overrides prelude on conflict)
        let merged_span_map = user_span_map.merge(&prelude_span_map);

        (
            merged_ast,
            fn_types,
            node_types,
            ability_conventions,
            merged_span_map,
        )
    } else {
        let fn_types: HashMap<_, _> = user_fn_types.iter().cloned().collect();
        let node_types: HashMap<_, _> = user_node_types.iter().cloned().collect();
        let ability_conventions: HashMap<_, _> = user_ability_conventions.iter().cloned().collect();
        (
            user_module.clone(),
            fn_types,
            node_types,
            ability_conventions,
            user_span_map,
        )
    };

    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(&merged_module) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                Diagnostic::new(
                    format!(
                        "unsupported compiler intrinsic directive `{}`",
                        directive.identity
                    ),
                    merged_span_map.get_or_default(directive.node),
                    DiagnosticSeverity::Error,
                    CompilationPhase::Lowering,
                )
                .accumulate(db);
            }
            return None;
        }
    };

    let mut function_instances: HashMap<_, _> = prelude_module(db)
        .map(|prelude| {
            prelude
                .expression_types(db)
                .function_instances
                .iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    function_instances.extend(
        typed
            .expression_types(db)
            .function_instances
            .iter()
            .cloned(),
    );
    // Monomorphize generic functions
    let mono_result = tribute_front::monomorphize::monomorphize_functions(
        db,
        merged_module,
        merged_fn_types,
        tribute_front::monomorphize::MonomorphizeMetadata {
            constructor_types: typed
                .constructor_types(db)
                .schemes
                .iter()
                .cloned()
                .collect(),
            specialized_enum_variants: typed
                .constructor_types(db)
                .specialized_enum_variants
                .iter()
                .cloned()
                .collect(),
            node_types: merged_node_types,
            local_instances: prelude_module(db)
                .into_iter()
                .flat_map(|prelude| prelude.expression_types(db).local_instances.clone())
                .chain(typed.expression_types(db).local_instances.iter().cloned())
                .collect(),
            function_instances,
            evidence_plans: prelude_module(db)
                .into_iter()
                .flat_map(|prelude| prelude.expression_types(db).evidence_plans.clone())
                .chain(typed.expression_types(db).evidence_plans.iter().cloned())
                .collect(),
            handler_operations: prelude_module(db)
                .into_iter()
                .flat_map(|prelude| prelude.handler_operations(db).iter().cloned())
                .chain(typed.handler_operations(db).iter().cloned())
                .collect(),
            perform_operations: prelude_module(db)
                .into_iter()
                .flat_map(|prelude| prelude.perform_operations(db).iter().cloned())
                .chain(typed.perform_operations(db).iter().cloned())
                .collect(),
            lambda_signatures: prelude_module(db)
                .into_iter()
                .flat_map(|prelude| prelude.lambda_signatures(db).iter().cloned())
                .chain(typed.lambda_signatures(db).iter().cloned())
                .collect(),
            exhaustive_cases: prelude_module(db)
                .into_iter()
                .flat_map(|prelude| prelude.exhaustive_cases(db).iter().copied())
                .chain(typed.exhaustive_cases(db).iter().copied())
                .collect(),
            compiler_intrinsics,
        },
    );
    let mono_result = match mono_result {
        Ok(result) => result,
        Err(errors) => {
            for error in errors {
                Diagnostic::new(
                    format!(
                        "invalid specialization instance at {}: {:?}",
                        error.node, error.kind
                    ),
                    merged_span_map.get_or_default(error.node),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(db);
            }
            return None;
        }
    };
    let mut exhaustive_cases: Vec<_> = mono_result.metadata.exhaustive_cases.into_iter().collect();
    exhaustive_cases.sort();
    let compiler_intrinsics = mono_result.metadata.compiler_intrinsics;
    let typed = ast_typeck::TypeCheckOutput::new(
        db,
        mono_result.module,
        mono_result.function_types,
        ast_typeck::ConstructorTypeMetadata {
            schemes: mono_result.metadata.constructor_types.into_iter().collect(),
            specialized_enum_variants: mono_result
                .metadata
                .specialized_enum_variants
                .into_iter()
                .collect(),
        },
        ast_typeck::ExpressionTypeMetadata {
            node_types: mono_result.metadata.node_types.into_iter().collect(),
            function_instances: mono_result
                .metadata
                .function_instances
                .into_iter()
                .collect(),
            local_instances: mono_result.metadata.local_instances.into_iter().collect(),
            evidence_plans: mono_result.metadata.evidence_plans.into_iter().collect(),
        },
        merged_ability_conventions.into_iter().collect::<Vec<_>>(),
        typed.ability_definitions(db).to_vec(),
        mono_result
            .metadata
            .handler_operations
            .into_iter()
            .collect(),
        mono_result
            .metadata
            .perform_operations
            .into_iter()
            .collect(),
        mono_result.metadata.lambda_signatures.into_iter().collect(),
        exhaustive_cases,
        *typed.well_known_types(db),
        merged_span_map,
    );
    Some(PreparedFrontend {
        typed,
        compiler_intrinsics,
    })
}

fn merge_and_lower_to_ir_with<'db, M>(
    db: &'db dyn salsa::Database,
    typed: &ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
    lower: impl FnOnce(ast_to_ir::TypedModule<'db>, &'db dyn salsa::Database, &mut IrContext, &str) -> M,
) -> (IrContext, M) {
    let prepared = prepare_frontend_details(db, *typed, source)
        .expect("frontend instances must be checked before lowering");
    let typed = prepared.typed;
    let compiler_intrinsics = prepared.compiler_intrinsics.clone();
    let mut ir = IrContext::new();
    let module = lower(
        ast_to_ir::TypedModule {
            ast: typed.module(db).clone(),
            local_instances: typed.expression_types(db).local_instances.clone(),
            span_map: typed.span_map(db).clone(),
            function_types: typed.function_types(db).iter().cloned().collect(),
            constructor_types: typed
                .constructor_types(db)
                .schemes
                .iter()
                .cloned()
                .collect(),
            specialized_enum_variants: typed
                .constructor_types(db)
                .specialized_enum_variants
                .clone(),
            node_types: typed.expression_types(db).node_types.clone(),
            ability_conventions: typed.ability_conventions(db).iter().cloned().collect(),
            ability_definitions: ast_typeck::ability_definitions_from_schemas(
                typed.ability_definitions(db),
            ),
            handler_operations: typed.handler_operations(db).clone(),
            perform_operations: typed.perform_operations(db).clone(),
            lambda_signatures: typed.lambda_signatures(db).clone(),
            exhaustive_cases: typed.exhaustive_cases(db).iter().copied().collect(),
            evidence_plans: typed.expression_types(db).evidence_plans.clone(),
            well_known_types: *typed.well_known_types(db),
            compiler_intrinsics,
            merged_sources: vec![PRELUDE_URI.to_owned()],
        },
        db,
        &mut ir,
        source.uri(db).as_str(),
    );
    (ir, module)
}

/// Arena IR together with the exact semantic metadata required by the shared
/// CPS conversion. The metadata stays private; [`run_shared_middle_end`] is
/// its only consumer.
#[derive(Clone)]
pub struct FrontendCompilation {
    context: IrContext,
    module: Module,
    operation_declarations: Vec<tribute_ir::dialect::tribute_control::OperationDeclaration>,
    compiler_intrinsics: Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
}

/// Run frontend (parse → typecheck → TDNR) and lower to arena IR.
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

/// Result of diagnostic compilation through the shared pipeline.
pub struct CompilationResult {
    /// Whether the shared pipeline produced a module.
    pub produced_module: bool,
    /// Diagnostics collected during compilation.
    pub diagnostics: Vec<Diagnostic>,
}

// =============================================================================
// Target Pipeline Entry Points
// =============================================================================
//
// Target entry points continue the shared arena session and apply backend
// transformations in the required order.

/// Compile a TrunkIR module to WebAssembly binary (arena-based).
///
/// Runs all WASM backend passes in a single arena session:
/// 1. Lowers the module from func/scf/arith dialects to wasm dialect operations
/// 2. Converts and reconciles unrealized conversion casts using the WASM type converter
/// 3. Validates and emits the wasm binary (delegated to trunk-ir-wasm-backend)
fn compile_to_wasm(ctx: &mut IrContext, module: Module) -> WasmCompilationResult<WasmBinary> {
    let _span = tracing::info_span!("compile_to_wasm").entered();
    let mut analyses = AnalysisCache::new();

    // Phase 1 - Lower to wasm dialect (Tribute-specific)
    {
        let _span = tracing::info_span!("lower_to_wasm").entered();
        tribute_passes::wasm::lower::lower_to_wasm(ctx, module, &mut analyses)
            .map_err(wasm_lowering_failure)?;
    }

    // Phase 2 - Legalize unrealized_conversion_cast operations (WASM type
    // converter): convert their result types, materialize real
    // representation changes, elide WasmGC reference upcasts, and reconcile
    // the identities left behind. A remaining cast is rejected by the
    // emission boundary in `finalize_wasm_gc_types`.
    {
        let _span = tracing::info_span!("legalize_unrealized_casts").entered();
        let tc = tribute_passes::wasm::type_converter::wasm_type_converter(ctx);
        let result = PatternApplicator::new(tc)
            .add_pattern(UnrealizedCastConversionPattern)
            .add_pattern(ReferenceUpcastElisionPattern)
            .apply_partial(ctx, module);
        if !result.reached_fixpoint {
            tracing::warn!("wasm cast legalization did not reach a fixpoint");
        }
        reconcile_unrealized_casts(ctx, module);
    }

    // Materialization may introduce semantic WasmGC operations after the main
    // lowering pipeline. Assign their module-local indices before emission.
    {
        let _span = tracing::info_span!("wasm_gc_to_wasm_after_casts").entered();
        tribute_passes::wasm::lower::finalize_wasm_gc_types(ctx, module)
            .map_err(tribute_passes::wasm::lower::WasmLowerError::from)
            .map_err(wasm_lowering_failure)?;
    }

    // Phase 3 - Emit WASM binary
    let _span = tracing::info_span!("emit_module_to_wasm").entered();
    trunk_ir_wasm_backend::emit_module_to_wasm(ctx, module)
}

// =============================================================================
// Pipeline Entry Points (SourceCst → Module)
// =============================================================================
//
// These functions take SourceCst and run the pipeline up to a specific stage.
// Useful for testing individual stages or for tools that need intermediate results.

/// Run frontend, source-logical CPS conversion, and lambda lifting (for testing).
///
/// Evidence params are introduced by the physical CPS conversion. Keep frontend
/// metadata in the same arena so the conversion can authenticate declarations.
pub fn run_through_cps_lowering(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    let Some(FrontendCompilation {
        context,
        module: m,
        operation_declarations,
        compiler_intrinsics,
    }) = compile_frontend_for_shared_route(db, source)
    else {
        return Ok(None);
    };
    let mut ctx = context;
    let core_module =
        core_dialect::Module::from_op(&ctx, m.op()).expect("frontend output must be a core.module");
    let mut pm = structural_pass_pipeline(operation_declarations, compiler_intrinsics);
    pm.run(&mut ctx, core_module, &mut Default::default())?;
    Ok(Some((ctx, m)))
}

// =============================================================================
// Full Pipeline (Orchestration)
// =============================================================================

/// Drop the source-logical functions nothing reachable from the roots
/// references, before CPS legalization.
///
/// The prelude and every struct's field functions are lowered whole; most
/// programs use little of either. Removing the rest here keeps every later
/// pass from converting functions the target pipeline would discard anyway.
///
/// Bodyless declarations stay: CPS legalization checks the registered
/// compiler intrinsics against them, and they cost no lowering.
fn eliminate_unreferenced_source_functions(ctx: &mut IrContext, m: Module) {
    trunk_ir::transforms::eliminate_dead_definitions(ctx, m, |ctx, op| {
        tribute_ir::dialect::tribute_control::Func::matches(ctx, op) && ctx.op_has_regions(op)
    });
}

/// Build the shared structural pass pipeline that legalizes source-logical
/// callable/control IR into CPS with explicit evidence.
fn structural_pass_pipeline(
    operation_declarations: Vec<tribute_ir::dialect::tribute_control::OperationDeclaration>,
    compiler_intrinsics: Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
) -> PassManager {
    let mut pm = PassManager::new();
    pm.add_pass(
        tribute_passes::tribute_control_to_cps::TributeControlToCps::new(operation_declarations)
            .with_compiler_intrinsics(compiler_intrinsics),
    )
    .add_pass(tribute_passes::lower_closure_lambda::LowerClosureLambda)
    .add_pass(tribute_passes::intrinsic_to_arith::LowerIntrinsicToArith)
    .add_pass(tribute_passes::list_intrinsics::LowerListIntrinsics)
    .add_pass(tribute_passes::io_lowering::LowerIoIntrinsics);
    pm.with_debug_verifier();
    pm
}

/// Run the shared middle-end pipeline (backend-independent) in an arena session.
///
/// Runs frontend passes and continuation lowering in a **single arena session**,
/// avoiding Salsa↔Arena round-trips.
///
/// Returns the arena session so that backend-specific pipelines can continue
/// without a Salsa↔Arena round-trip.
fn run_shared_pipeline(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    let Some(frontend) = compile_frontend_for_shared_route(db, source) else {
        return Ok(None);
    };
    run_shared_middle_end(frontend).map(Some)
}

/// Run the shared pipeline on every function of the program, reachable or
/// not, so that a compilation for diagnostics checks all of them.
fn check_shared_pipeline(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    let Some(frontend) = compile_frontend_for_shared_route(db, source) else {
        return Ok(None);
    };
    shared_middle_end(frontend, SourceFunctions::All).map(Some)
}

/// Which source-logical functions the shared middle-end legalizes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceFunctions {
    /// Those reachable from the program's roots; an artifact has no others.
    Reachable,
    All,
}

/// Run the shared middle-end on a frontend result.
///
/// The result is ready for [`run_target_to_boundary_exit`].
pub fn run_shared_middle_end(frontend: FrontendCompilation) -> PassResult<(IrContext, Module)> {
    shared_middle_end(frontend, SourceFunctions::Reachable)
}

fn shared_middle_end(
    frontend: FrontendCompilation,
    functions: SourceFunctions,
) -> PassResult<(IrContext, Module)> {
    let FrontendCompilation {
        context,
        module: m,
        operation_declarations,
        compiler_intrinsics,
    } = frontend;
    let mut ctx = context;

    // Middle-end passes, sequenced through the PassManager (#268).
    // Registration order == execution order.
    let core_module =
        core_dialect::Module::from_op(&ctx, m.op()).expect("frontend output must be a core.module");
    // One cache for the phase: every pass and boundary check reuses analyses
    // of unchanged IR, and any IR change discards them.
    if functions == SourceFunctions::Reachable {
        eliminate_unreferenced_source_functions(&mut ctx, m);
    }
    let mut analyses = AnalysisCache::new();
    let mut structural_pm = structural_pass_pipeline(operation_declarations, compiler_intrinsics);
    structural_pm.run(&mut ctx, core_module, &mut analyses)?;

    // CPS effect handling, function-local phase: lower_ability_perform produces
    // explicit effect dispatches; evidence resolution then extends handler scopes.
    let mut ability_pm = PassManager::new();
    ability_pm
        .nest::<func_dialect::Func>()
        .add_pass(tribute_passes::lower_ability_perform::LowerAbilityPerform);
    ability_pm.with_debug_verifier();
    ability_pm.run(&mut ctx, core_module, &mut analyses)?;

    let mut evidence_pm = PassManager::new();
    evidence_pm.add_pass(tribute_passes::resolve_evidence::ResolveEvidenceDispatch);
    evidence_pm.with_debug_verifier();
    evidence_pm.run(&mut ctx, core_module, &mut analyses)?;

    // Final function-local ability conversion. This consumes handle_dispatch ops
    // after resolve_evidence expands evidence setup.
    let mut ability_boundary_pm = PassManager::new();
    ability_boundary_pm
        .nest::<func_dialect::Func>()
        .add_pass(tribute_passes::lower_handle_dispatch::LowerHandleDispatch);
    ability_boundary_pm.with_debug_verifier();
    ability_boundary_pm.run(&mut ctx, core_module, &mut analyses)?;

    Ok((ctx, m))
}

/// Dump native IR at a named RC optimization boundary.
///
/// Optimization options apply to the native portion of the pipeline.
/// Native emission is intentionally skipped.
#[salsa::tracked(returns(as_deref))]
pub fn dump_native_ir_at_stage(
    db: &dyn salsa::Database,
    source: SourceCst,
    stage: NativePipelineStage,
    options: OptimizationOptions,
) -> Result<String, DumpIrError> {
    let Some((mut ctx, module)) = run_shared_pipeline(db, source)? else {
        return Ok(String::new());
    };
    validate_and_report_arity(db, &ctx, module);
    run_native_target_pipeline(&mut ctx, module)?;
    prepare_module_to_native(&mut ctx, module, false, options.native, Some(stage)).map_err(
        |error| DumpIrError {
            message: error.to_string(),
        },
    )?;
    Ok(trunk_ir::printer::print_module(&ctx, module.op()))
}

/// Validate call arity and report mismatches as diagnostics.
///
/// Arity mismatches are collected into `IrContext` diagnostics by the
/// validation function and then converted to Salsa `Diagnostic` accumulators.
fn validate_and_report_arity(db: &dyn salsa::Database, ctx: &IrContext, m: Module) {
    trunk_ir::validation::validate_call_arity(ctx, m);
    for diag in ctx.diagnostics().iter() {
        Diagnostic::from_ir(diag.clone(), CompilationPhase::Lowering).accumulate(db);
    }
}

fn report_pass_error(db: &dyn salsa::Database, error: &PassError) {
    Diagnostic::new(
        format!("Pipeline failed: {error}"),
        Span::new(0, 0),
        DiagnosticSeverity::Error,
        CompilationPhase::Lowering,
    )
    .accumulate(db);
}

/// Debug-only value-integrity check at a pipeline boundary.
///
/// The shared pipeline owns control legalization; the target pipelines only
/// verify its output before their own lowering runs.
fn debug_validate_value_integrity(ctx: &IrContext, m: Module, boundary: &str) {
    if !cfg!(debug_assertions) {
        return;
    }
    let result = trunk_ir::validation::validate_value_integrity(ctx, m);
    if !result.is_ok() {
        tracing::warn!("Value integrity errors {boundary}: {:?}", result.errors);
    }
}

/// Run inlining + DCE + cast materialization (shared cleanup after all lowering).
///
fn run_cleanup_passes(ctx: &mut IrContext, m: Module, analyses: &mut AnalysisCache) {
    trunk_ir::transforms::global_dce::eliminate_dead_functions(ctx, m, analyses);
    if let Ok(core_module) = core_dialect::Module::from_op(ctx, m.op()) {
        let mut pm = PassManager::new();
        pm.nest::<func_dialect::Func>()
            .add_pass(trunk_ir::transforms::canonicalize_pass())
            .add_pass(trunk_ir::transforms::dce_pass(
                trunk_ir::transforms::DceConfig::default(),
            ));
        pm.with_debug_verifier();
        if let Err(error) = pm.run(ctx, core_module, analyses) {
            tracing::warn!("cleanup function passes failed: {error}");
        }
    } else {
        tracing::warn!("cleanup skipped function passes: root op is not core.module");
    }
    // Target type conversion has not run yet: materialize only the casts that
    // need real operations and keep the ones that only retype a value, so
    // every use still sees the type its operation declares.
    let tc = generic_type_converter(ctx);
    materialize_unrealized_casts(ctx, m, &tc);
}

/// Run the WASM target pipeline: lowering + cleanup.
fn run_wasm_target_pipeline(ctx: &mut IrContext, m: Module) -> Result<(), DumpIrError> {
    debug_validate_value_integrity(ctx, m, "before Wasm target lowering");

    let mut analyses = AnalysisCache::new();

    // Drop unreachable functions first, so that no later pass lowers them.
    trunk_ir::transforms::global_dce::eliminate_dead_functions(ctx, m, &mut analyses);

    // General function inlining. The pass is single-block-only and cf-free,
    // so its output stays within dialects WASM lowering already handles.
    trunk_ir::transforms::inline::inline_functions(ctx, m, &mut analyses);

    enter_target_closure_storage_boundary(ctx, m, &mut analyses)?;

    let core_module = core_dialect::Module::from_op(ctx, m.op())
        .expect("Wasm evidence lowering requires a core.module");
    tribute_passes::wasm::evidence_to_wasm::prepare_wasm_evidence_runtime(ctx, m);
    let mut pm = PassManager::new();
    pm.nest::<func_dialect::Func>()
        .add_pass(tribute_passes::wasm::evidence_to_wasm::LowerEvidenceToWasm);
    pm.with_debug_verifier();
    pm.run(ctx, core_module, &mut analyses)?;
    // Complete the supported bytes intrinsic bridge inside the boundary; the
    // lowering consumes its verified compiler intrinsic identity.
    tribute_passes::wasm::bytes::lower(ctx, m)?;
    tribute_passes::closure_lower::finalize_closure_storage_layout(ctx, m);
    debug_validate_value_integrity(ctx, m, "after evidence_to_wasm");

    run_cleanup_passes(ctx, m, &mut analyses);
    enforce_boundary_exit(ctx, m, tribute_passes::abi_boundary::TargetKind::Wasm)
}

/// Run the native target pipeline: lowering + evidence_to_native + cleanup.
fn run_native_target_pipeline(ctx: &mut IrContext, m: Module) -> Result<(), DumpIrError> {
    debug_validate_value_integrity(ctx, m, "before native target lowering");

    let mut analyses = AnalysisCache::new();

    // Drop unreachable functions first, so that no later pass lowers them.
    trunk_ir::transforms::global_dce::eliminate_dead_functions(ctx, m, &mut analyses);

    // General function inlining. Single-block-only (no `cf` dialect
    // dependency), so it preserves the caller's block structure. That
    // keeps `evidence_to_native`'s per-block producer/consumer correlation
    // assumptions intact, and lets the same pass work on both backend paths.
    trunk_ir::transforms::inline::inline_functions(ctx, m, &mut analyses);

    enter_target_closure_storage_boundary(ctx, m, &mut analyses)?;

    if let Ok(core_module) = core_dialect::Module::from_op(ctx, m.op()) {
        tribute_passes::native::evidence::prepare_native_evidence_runtime(ctx, m);
        let mut pm = PassManager::new();
        pm.nest::<func_dialect::Func>()
            .add_pass(tribute_passes::native::evidence::LowerEvidenceToNative);
        pm.with_debug_verifier();
        pm.run(ctx, core_module, &mut analyses)?;
    } else {
        tribute_passes::native::evidence::lower_evidence_to_native(ctx, m);
    }
    // Complete the supported bytes intrinsic bridge inside the boundary; the
    // lowering consumes its verified compiler intrinsic identity.
    tribute_passes::native::intrinsic_to_native::lower(ctx, m)?;
    tribute_passes::closure_lower::finalize_closure_storage_layout(ctx, m);
    debug_validate_value_integrity(ctx, m, "after evidence_to_native");

    run_cleanup_passes(ctx, m, &mut analyses);
    enforce_boundary_exit(ctx, m, tribute_passes::abi_boundary::TargetKind::Native)
}

/// Run `target`'s pipeline from the shared middle-end output to the
/// representation/ABI boundary exit.
pub fn run_target_to_boundary_exit(
    ctx: &mut IrContext,
    m: Module,
    target: tribute_passes::abi_boundary::TargetKind,
) -> Result<(), DumpIrError> {
    match target {
        tribute_passes::abi_boundary::TargetKind::Native => run_native_target_pipeline(ctx, m),
        tribute_passes::abi_boundary::TargetKind::Wasm => run_wasm_target_pipeline(ctx, m),
    }
}

/// Error returned by target lowering and emission after the boundary exit.
#[derive(Debug, Clone, derive_more::Display, derive_more::Error)]
pub enum EmitError {
    #[display("native compilation failed: {_0}")]
    Native(trunk_ir_cranelift_backend::CompilationError),
    #[display("WebAssembly compilation failed: {_0}")]
    Wasm(CompilationError),
}

/// Lower and emit a module at `target`'s boundary exit.
///
/// Native output is an unlinked object file built with production
/// optimizations and no sanitizer; Wasm output is a module binary.
pub fn emit_from_boundary_exit(
    ctx: &mut IrContext,
    m: Module,
    target: tribute_passes::abi_boundary::TargetKind,
) -> Result<Vec<u8>, EmitError> {
    match target {
        tribute_passes::abi_boundary::TargetKind::Native => {
            compile_module_to_native(ctx, m, false, NativeOptimizationOptions::production())
                .map_err(EmitError::Native)
        }
        tribute_passes::abi_boundary::TargetKind::Wasm => compile_to_wasm(ctx, m)
            .map(|binary| binary.bytes)
            .map_err(EmitError::Wasm),
    }
}

/// Reject a module whose representation/ABI boundary exit violates the
/// contract.
fn enforce_boundary_exit(
    ctx: &IrContext,
    m: Module,
    target: tribute_passes::abi_boundary::TargetKind,
) -> Result<(), DumpIrError> {
    let violations = tribute_passes::abi_boundary::verify_boundary_exit(ctx, m, target);
    if violations.is_empty() {
        return Ok(());
    }
    Err(DumpIrError {
        message: format!(
            "{target:?} representation/ABI boundary exit violations: {}",
            violations.iter().format("; ")
        ),
    })
}

/// Enter the sole target-side closure storage boundary. Exact ABI validation
/// observes semantic closure types first; `LowerPreparedClosures` then consumes
/// any remaining closure operations before whole-module storage finalization.
fn enter_target_closure_storage_boundary(
    ctx: &mut IrContext,
    m: Module,
    analyses: &mut AnalysisCache,
) -> Result<(), DumpIrError> {
    tribute_passes::target_abi::lower_cps_signatures_to_physical(ctx, m)?;
    tribute_passes::target_abi::compose_root_entry_bridge(ctx, m)?;
    let core_module = core_dialect::Module::from_op(ctx, m.op())
        .expect("target closure lowering requires a core.module");
    let mut pm = PassManager::new();
    pm.add_pass(tribute_passes::closure_lower::LowerPreparedClosures);
    pm.with_debug_verifier();
    pm.run(ctx, core_module, analyses)?;
    Ok(())
}

/// Dump IR text at the target's representation/ABI boundary exit.
///
/// If `native` is true, runs the native pipeline; otherwise runs the WASM pipeline.
/// A module that violates the exit contract is an error.
/// Returns the IR text borrowed from the database, or an error. Diagnostics are
/// accumulated.
#[salsa::tracked(returns(as_deref))]
pub fn dump_ir(
    db: &dyn salsa::Database,
    source: SourceCst,
    native: bool,
) -> Result<String, DumpIrError> {
    let Some((mut ctx, m)) = run_shared_pipeline(db, source)? else {
        return Ok(String::new());
    };
    validate_and_report_arity(db, &ctx, m);

    if native {
        run_native_target_pipeline(&mut ctx, m)?;
    } else {
        run_wasm_target_pipeline(&mut ctx, m)?;
    }

    Ok(trunk_ir::printer::print_module(&ctx, m.op()))
}

#[salsa::tracked(returns(as_deref))]
fn compile_to_wasm_binary_tracked(db: &dyn salsa::Database, source: SourceCst) -> Option<Vec<u8>> {
    let (mut ctx, m) = match run_shared_pipeline(db, source) {
        Ok(Some(result)) => result,
        Ok(None) => return None,
        Err(error) => {
            report_pass_error(db, &error);
            return None;
        }
    };
    validate_and_report_arity(db, &ctx, m);

    if let Err(e) = run_wasm_target_pipeline(&mut ctx, m) {
        Diagnostic::new(
            format!("Pipeline failed: {}", e),
            Span::new(0, 0),
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
        .accumulate(db);
        return None;
    }

    // WASM backend lowering + emit
    match compile_to_wasm(&mut ctx, m) {
        Ok(binary) => Some(binary.bytes),
        Err(e) => {
            Diagnostic::new(
                format!("WebAssembly compilation failed: {}", e),
                Span::new(0, 0),
                DiagnosticSeverity::Error,
                CompilationPhase::Lowering,
            )
            .accumulate(db);
            None
        }
    }
}

/// Compile to WebAssembly binary bytes.
///
/// Runs the full pipeline (frontend → shared passes → WASM lowering → emit)
/// in a single arena session, avoiding Salsa↔Arena round-trips after ast_to_ir.
///
/// Returns the raw WASM bytes on success, borrowed from the database, or the
/// accumulated diagnostics on failure.
pub fn compile_to_wasm_binary(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Result<&[u8], Vec<&Diagnostic>> {
    compile_to_wasm_binary_tracked(db, source)
        .ok_or_else(|| compile_to_wasm_binary_tracked::accumulated::<Diagnostic>(db, source))
}

/// Diagnostics accumulated by [`compile_to_wasm_binary`], including the
/// warnings of a successful compilation.
pub fn wasm_binary_diagnostics(db: &dyn salsa::Database, source: SourceCst) -> Vec<&Diagnostic> {
    compile_to_wasm_binary_tracked::accumulated::<Diagnostic>(db, source)
}

// =============================================================================
// Native Pipeline (Cranelift)
// =============================================================================

/// Select ownership-plan policy for each observable native pipeline boundary.
///
/// The staged dump boundaries deliberately suppress later elisions so their
/// output remains comparable with the corresponding optimization policy.
fn native_ownership_plan_options(
    stop_after: Option<NativePipelineStage>,
    optimizations: NativeOptimizationOptions,
) -> tribute_passes::native::ownership_plan::NativeOwnershipPlanOptions {
    let configured = tribute_passes::native::ownership_plan::NativeOwnershipPlanOptions {
        elide_proven_borrowed_parameters: matches!(
            optimizations.borrowed_parameters,
            BorrowedParameterPolicy::ElideProvenBorrowed
        ),
        elide_proven_field_borrows: matches!(
            optimizations.temporary_borrows,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows
        ),
    };

    match stop_after {
        Some(NativePipelineStage::AfterRcInsertion) => {
            tribute_passes::native::ownership_plan::NativeOwnershipPlanOptions {
                elide_proven_borrowed_parameters: false,
                elide_proven_field_borrows: false,
            }
        }
        Some(NativePipelineStage::AfterBorrowedParameterOptimization) => {
            tribute_passes::native::ownership_plan::NativeOwnershipPlanOptions {
                elide_proven_borrowed_parameters: configured.elide_proven_borrowed_parameters,
                elide_proven_field_borrows: false,
            }
        }
        Some(
            NativePipelineStage::AfterTemporaryBorrowOptimization
            | NativePipelineStage::AfterRcOptimization,
        )
        | None => configured,
    }
}

/// Compile a TrunkIR module to a native object file (arena-based).
///
/// This function runs all backend passes in a single arena session:
/// 1. Generates native entrypoint
/// 2. Lowers scf/func/cf/adt/arith dialects to `clif.*`
/// 3. Runs RTTI, RC insertion, and RC lowering passes
/// 4. Converts and reconciles unrealized conversion casts
/// 5. Validates and emits the native object file via Cranelift
fn prepare_module_to_native(
    ctx: &mut IrContext,
    module: Module,
    sanitize: bool,
    optimizations: NativeOptimizationOptions,
    stop_after: Option<NativePipelineStage>,
) -> NativeCompilationResult<()> {
    let _span = tracing::info_span!("prepare_module_to_native").entered();
    let mut analyses = AnalysisCache::new();
    let core_module = core_dialect::Module::from_op(ctx, module.op()).map_err(|_| {
        trunk_ir_cranelift_backend::CompilationError::ir_validation(
            "native lowering requires a core.module".to_owned(),
        )
    })?;
    for mut stage in native_lowering_passes(sanitize, optimizations, stop_after) {
        stage
            .run(ctx, core_module, &mut analyses)
            .map_err(native_pass_failure)?;
    }
    Ok(())
}

/// The native lowering stages, in order, as passes of [`PassManager`]s that
/// run one after another.
///
/// Stages share one analysis cache and, except in the middle segment, the debug
/// verifier. That segment starts with `func-to-clif`, whose type conversion
/// retypes values before their users are converted, so its IR is not
/// schema-clean until the clif lowerings finish; the next segment's entry check verifies its result. The ownership plan, which
/// several stages read, travels in a cell captured by those stages. A stage
/// that fails is named in the resulting [`PassError`].
fn native_lowering_passes(
    sanitize: bool,
    optimizations: NativeOptimizationOptions,
    stop_after: Option<NativePipelineStage>,
) -> Vec<PassManager> {
    use std::cell::RefCell;
    use std::rc::Rc;

    use tribute_passes::native::ownership_plan::NativeOwnershipPlan;
    use trunk_ir::pass::{Pass, pass_fn};

    fn failure(error: impl std::error::Error + Send + Sync + 'static) -> PassRunError {
        Box::new(error)
    }

    fn missing_plan() -> PassRunError {
        "the ownership plan has not been built".into()
    }

    let shared_plan: Rc<RefCell<Option<NativeOwnershipPlan>>> = Rc::default();

    let mut pm = PassManager::new();
    pm.with_debug_verifier();

    pm.add_pass(pass_fn(
        "native-entrypoint",
        move |ctx, m: core_dialect::Module, _| {
            tribute_passes::native::entrypoint::generate_native_entrypoint(ctx, m.into(), sanitize);
            Ok(())
        },
    ))
    // Declare a clif.data object per string/bytes payload, then lower
    // adt.string_const to adt.variant_new + bytes alloc and adt.bytes_const
    // to a clif alloc + data reference.
    .add_pass(pass_fn(
        "const-to-native",
        |ctx, m: core_dialect::Module, _| {
            let const_analysis =
                tribute_passes::native::const_to_native::analyze_consts(ctx, m.into());
            tribute_passes::native::const_to_native::lower(ctx, m.into(), &const_analysis)
                .map_err(failure)
        },
    ))
    // Lower target-independent I/O to the native runtime ABI.
    .add_pass(pass_fn(
        "io-to-native",
        |ctx, m: core_dialect::Module, _| {
            tribute_passes::native::io::lower(ctx, m.into()).map_err(failure)
        },
    ))
    // Select the private native representation for opaque lists.
    .add_pass(pass_fn(
        "list-to-native",
        |ctx, m: core_dialect::Module, _| {
            tribute_passes::native::list::lower(ctx, m.into()).map_err(failure)
        },
    ));

    // Lower structured control flow to CFG-based control flow.
    pm.nest::<func_dialect::Func>()
        .add_pass(trunk_ir::transforms::scf_to_cf_pass());

    // Plan ownership and RTTI, then lower func dialect to clif dialect.
    let plan_options = native_ownership_plan_options(stop_after, optimizations);
    pm.add_pass(pass_fn("plan-ownership", {
        let plan = Rc::clone(&shared_plan);
        move |ctx, m: core_dialect::Module, analyses| {
            let built = tribute_passes::native::ownership_plan::build_native_ownership_plan(
                ctx,
                m.into(),
                plan_options,
                analyses,
            )
            .map_err(failure)?;
            *plan.borrow_mut() = Some(built);
            Ok(())
        }
    }))
    .add_pass(pass_fn("materialize-rc", {
        let plan = Rc::clone(&shared_plan);
        move |ctx, m: core_dialect::Module, _| {
            let plan = plan.borrow();
            let plan = plan.as_ref().ok_or_else(missing_plan)?;
            tribute_passes::native::rc_materialization::materialize(ctx, m.into(), plan)
                .map_err(failure)
        }
    }))
    // Record the planned RTTI layouts in the IR, then adapt semantic closure
    // allocations and their declaration to the native layout.
    .add_pass(pass_fn("declare-rtti-layouts", {
        let plan = Rc::clone(&shared_plan);
        move |ctx, m: core_dialect::Module, _| {
            let plan = plan.borrow();
            let plan = plan.as_ref().ok_or_else(missing_plan)?;
            tribute_passes::native::rtti::declare_rtti_layouts(ctx, m.into(), plan.rtti_types());
            Ok(())
        }
    }))
    .add_pass(pass_fn(
        "adapt-closure-layout",
        |ctx, m: core_dialect::Module, _| {
            tribute_passes::native::adapt_closure_layout::lower(ctx, m.into());
            Ok(())
        },
    ))
    // Field accesses need only the structural `mem.struct` layout, and
    // variant tests only the variant's descriptor number. The nominal layouts
    // stay on allocations, which RC header lowering resolves to descriptors.
    .add_pass(pass_fn("struct-to-mem", {
        let plan = Rc::clone(&shared_plan);
        move |ctx, m: core_dialect::Module, analyses| {
            let owned = plan.borrow_mut().take().ok_or_else(missing_plan)?;
            tribute_passes::native::struct_to_mem::StructToMem::new(owned).run(ctx, m, analyses)
        }
    }))
    // A descriptor read takes a managed reference, so it is lowered before
    // type conversion turns its operand into a pointer.
    .add_pass(tribute_passes::native::descriptor_to_clif::DescriptorToClif);

    let mut lowering = PassManager::new();
    lowering
        .add_pass(pass_fn(
            "func-to-clif",
            |ctx, m: core_dialect::Module, _| {
                let (type_converter, _) =
                    tribute_passes::native::type_converter::native_type_converter(ctx);
                func_to_clif::lower(ctx, m.into(), type_converter).map_err(failure)
            },
        ))
        .add_pass(pass_fn("cf-to-clif", |ctx, m: core_dialect::Module, _| {
            let (type_converter, _) =
                tribute_passes::native::type_converter::native_type_converter(ctx);
            cf_to_clif::lower(ctx, m.into(), type_converter).map_err(failure)
        }))
        .add_pass(pass_fn(
            "generate-rtti",
            |ctx, m: core_dialect::Module, _| {
                let (type_converter, _) =
                    tribute_passes::native::type_converter::native_type_converter(ctx);
                tribute_passes::native::rtti::generate_rtti(ctx, m.into(), &type_converter)
                    .map_err(failure)
            },
        ))
        .add_pass(pass_fn(
            "adt-rc-header",
            |ctx, m: core_dialect::Module, _| {
                let (type_converter, _) =
                    tribute_passes::native::type_converter::native_type_converter(ctx);
                tribute_passes::native::adt_rc_header::lower(ctx, m.into(), type_converter)
                    .map_err(failure)
            },
        ))
        .add_pass(pass_fn("adt-to-clif", |ctx, m: core_dialect::Module, _| {
            let (type_converter, _) =
                tribute_passes::native::type_converter::native_type_converter(ctx);
            tribute_passes::native::adt_to_clif::lower(ctx, m.into(), type_converter)
                .map_err(failure)
        }))
        .add_pass(pass_fn(
            "arith-to-clif",
            |ctx, m: core_dialect::Module, _| {
                let (type_converter, _) =
                    tribute_passes::native::type_converter::native_type_converter(ctx);
                arith_to_clif::lower(ctx, m.into(), type_converter).map_err(failure)
            },
        ))
        .add_pass(pass_fn("mem-to-clif", |ctx, m: core_dialect::Module, _| {
            let (type_converter, _) =
                tribute_passes::native::type_converter::native_type_converter(ctx);
            mem_to_clif::lower(ctx, m.into(), type_converter).map_err(failure)
        }))
        // Lower non-RC tribute runtime operations. The explicit RC operations were
        // materialized from typed ownership actions before erasure.
        .add_pass(pass_fn(
            "tribute-rt-to-clif",
            |ctx, m: core_dialect::Module, _| {
                let (type_converter, _) =
                    tribute_passes::native::type_converter::native_type_converter(ctx);
                tribute_passes::native::tribute_rt_to_clif::lower(ctx, m.into(), type_converter)
                    .map_err(failure)
            },
        ));

    if matches!(
        stop_after,
        Some(
            NativePipelineStage::AfterRcInsertion
                | NativePipelineStage::AfterBorrowedParameterOptimization
                | NativePipelineStage::AfterTemporaryBorrowOptimization
        )
    ) {
        return vec![pm, lowering];
    }

    let mut finish = PassManager::new();
    finish.with_debug_verifier();

    // Eliminate local retain/release pairs while they are still directly
    // observable tribute_rt operations.
    if optimizations.paired_rc_elimination == PairedRcEliminationPolicy::Enabled {
        finish.add_pass(pass_fn(
            "eliminate-paired-rc",
            |ctx, m: core_dialect::Module, _| {
                tribute_passes::native::rc_optimization::eliminate_paired_rc(ctx, m.into());
                Ok(())
            },
        ));
    }

    if stop_after == Some(NativePipelineStage::AfterRcOptimization) {
        return vec![pm, lowering, finish];
    }

    // Legalize unrealized_conversion_cast operations: convert their result
    // types, materialize real representation changes, and reconcile the
    // identities left behind. A remaining cast is rejected by
    // `validate_clif_ir` before emission.
    finish
        .add_pass(pass_fn(
            "legalize-casts",
            |ctx, m: core_dialect::Module, _| {
                let (type_converter, _) =
                    tribute_passes::native::type_converter::native_type_converter(ctx);
                PatternApplicator::new(type_converter)
                    .add_pattern(UnrealizedCastConversionPattern)
                    .apply_partial(ctx, Module::from(m));
                reconcile_unrealized_casts(ctx, m.into());
                Ok(())
            },
        ))
        // Lower RC operations (retain/release) to inline clif code.
        .add_pass(pass_fn("rc-lowering", |ctx, m: core_dialect::Module, _| {
            tribute_passes::native::rc_lowering::lower_rc(ctx, m.into());
            Ok(())
        }));

    vec![pm, lowering, finish]
}

fn compile_module_to_native(
    ctx: &mut IrContext,
    module: Module,
    sanitize: bool,
    optimizations: NativeOptimizationOptions,
) -> NativeCompilationResult<Vec<u8>> {
    prepare_module_to_native(ctx, module, sanitize, optimizations, None)?;
    let _emit_span = tracing::info_span!("emit_module_to_native").entered();
    emit_module_to_native(ctx, module)
}

fn native_pass_failure(error: PassError) -> trunk_ir_cranelift_backend::CompilationError {
    trunk_ir_cranelift_backend::CompilationError::ir_validation(error.to_string())
}

fn wasm_lowering_failure(error: tribute_passes::wasm::lower::WasmLowerError) -> CompilationError {
    CompilationError::ir_validation(error.to_string())
}

/// Compile to native object bytes.
///
/// Runs the full pipeline (frontend → shared passes → native lowering → emit)
/// in a single arena session, avoiding Salsa↔Arena round-trips after ast_to_ir.
///
/// Returns `None` if compilation fails, with diagnostics accumulated.
#[salsa::tracked(returns(as_deref))]
pub fn compile_to_native_binary(
    db: &dyn salsa::Database,
    source: SourceCst,
    config: CompilationConfig,
) -> Option<Vec<u8>> {
    let options = config.optimizations(db);
    let (mut ctx, m) = match run_shared_pipeline(db, source) {
        Ok(Some(result)) => result,
        Ok(None) => return None,
        Err(error) => {
            report_pass_error(db, &error);
            return None;
        }
    };
    validate_and_report_arity(db, &ctx, m);

    if let Err(e) = run_native_target_pipeline(&mut ctx, m) {
        Diagnostic::new(
            format!("Pipeline failed: {}", e),
            Span::new(0, 0),
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
        .accumulate(db);
        return None;
    }

    // Native backend lowering + emit
    let sanitize = config.sanitize_address(db);
    match compile_module_to_native(&mut ctx, m, sanitize, options.native) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            Diagnostic::new(
                format!("Native compilation failed: {}", e),
                Span::new(0, 0),
                DiagnosticSeverity::Error,
                CompilationPhase::Lowering,
            )
            .accumulate(db);
            None
        }
    }
}

// =============================================================================
// AST-Based Pipeline (New)
// =============================================================================
//
// The AST-based pipeline provides better type safety and separation of concerns.
// It transforms: CST → AST → resolve → typecheck → tdnr → ast_to_ir → TrunkIR
//

/// Parse source and run the frontend pipeline (parse → resolve → typecheck → TDNR).
///
/// Returns the typed AST with span map, ready for `ast_to_ir` lowering.
/// Does NOT call `ast_to_ir` — that is the caller's responsibility.
///
/// Uses the Type Info Injection approach to make prelude types available:
/// 1. Parse user code to AST
/// 2. Merge prelude bindings into user's ModuleEnv
/// 3. Resolve names with merged environment
/// 4. Inject prelude TypeSchemes into user's ModuleTypeEnv
/// 5. Type check with injected types
/// 6. Run TDNR
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

    // Phase 2: Build user env and merge prelude bindings
    let mut user_env = ast_resolve::build_env(db, user_ast);
    if let Some(p_env) = prelude_env(db) {
        // Prelude bindings injected, user definitions take precedence
        ast_resolve::merge_library(&mut user_env, p_env);
    }

    // Phase 3: Name resolution with merged environment
    let resolved_ast = ast_resolve::resolve_with_env(db, user_ast, user_env, span_map.clone());

    // Phase 4: Type checking with prelude types injected
    let mut checker = ast_typeck::TypeChecker::new(db, span_map.clone());
    if let Some(p_exports) = prelude_exports(db) {
        checker.inject_prelude(&p_exports); // Prelude TypeSchemes injected (no UniVars)
    }
    let result = checker.check_module(&resolved_ast);

    tracing::debug!(
        "Phase 4: after typecheck, {} declarations, {} function_types, {} node_types",
        result.module.decls.len(),
        result.function_types.len(),
        result.node_types.len()
    );

    // TDNR for remaining MethodCall → Call AST transformations
    let mut tdnr_ast = result.module;
    ast_tdnr::resolve_tdnr(
        db,
        &mut tdnr_ast,
        prelude_module(db).iter().map(|p| p.module(db)),
    );
    report_unresolved_methods(db, &tdnr_ast, &span_map);

    Some(ast_typeck::TypeCheckOutput::new(
        db,
        tdnr_ast,
        result.function_types,
        result.constructor_types.into(),
        ast_typeck::ExpressionTypeMetadata {
            node_types: result.node_types,
            function_instances: result.function_instances,
            local_instances: result.local_instances,
            evidence_plans: result.evidence_plans,
        },
        result.ability_conventions,
        ast_typeck::ability_schemas(&result.ability_definitions),
        result.handler_operations,
        result.perform_operations,
        result.lambda_signatures,
        result.exhaustive_cases,
        result.well_known_types,
        span_map,
    ))
}

/// Report method calls that remain unresolved after TDNR has had the final
/// opportunity to use types propagated through the enclosing function.
fn report_unresolved_methods<'db>(
    db: &'db dyn salsa::Database,
    module: &tribute_front::ast::Module<TypedRef<'db>>,
    span_map: &SpanMap,
) {
    struct Report<'a, 'db> {
        db: &'db dyn salsa::Database,
        span_map: &'a SpanMap,
    }
    impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Report<'_, 'db> {
        fn visit_expr(&mut self, expr: &'ast Expr<TypedRef<'db>>) {
            // Type checking reports a path call it cannot resolve.
            if let ExprKind::MethodCall {
                method, path: None, ..
            } = &*expr.kind
            {
                Diagnostic::new(
                    format!("unresolved method '{}' for this receiver type", method),
                    self.span_map.get_or_default(expr.id),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db);
            }
            walk_expr(self, expr);
        }
    }
    walk_module(&mut Report { db, span_map }, module);
}

/// Run the shared pipeline for diagnostic collection only.
///
/// This is a `#[salsa::tracked]` function so that diagnostics accumulated
/// during compilation can be collected via `compile_ast_tracked::accumulated`.
/// Returns whether the shared pipeline produced a module.
#[salsa::tracked(returns(copy))]
fn compile_ast_tracked(db: &dyn salsa::Database, source: SourceCst) -> bool {
    // Run the shared pipeline; diagnostics are accumulated as side effects
    match check_shared_pipeline(db, source) {
        Ok(Some((ctx, m))) => {
            validate_and_report_arity(db, &ctx, m);
            true
        }
        Ok(None) => false,
        Err(error) => {
            report_pass_error(db, &error);
            false
        }
    }
}

/// Compile using the AST-based pipeline.
///
/// Runs the shared pipeline (frontend + evidence/closure/evidence-calls/
/// resolve-evidence passes) and returns arena IR. Does NOT run
/// target-specific passes (WASM/native), making it suitable for
/// diagnostic-only compilation ("none" target).
pub fn compile_ast(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    check_shared_pipeline(db, source)
}

/// Run compilation and return detailed results including diagnostics.
///
/// Diagnostics are collected using Salsa accumulators from all compilation stages.
pub fn compile_with_diagnostics(db: &dyn salsa::Database, source: SourceCst) -> CompilationResult {
    // Run the tracked function to collect diagnostics
    let produced_module = compile_ast_tracked(db, source);

    // Collect all accumulated diagnostics from the compilation
    let mut diagnostics: Vec<Diagnostic> =
        compile_ast_tracked::accumulated::<Diagnostic>(db, source)
            .into_iter()
            .cloned()
            .collect();
    diagnostics.sort_by(compare_diagnostics);

    CompilationResult {
        produced_module,
        diagnostics,
    }
}

/// Compare diagnostics using a stable user-facing order.
///
/// Compilation phase is primary, followed by source location and message as
/// deterministic tie-breakers. This is also used by the CLI for target-specific
/// accumulator results.
pub fn compare_diagnostics(left: &Diagnostic, right: &Diagnostic) -> std::cmp::Ordering {
    fn phase_rank(phase: &CompilationPhase) -> u8 {
        match phase {
            CompilationPhase::Parsing => 0,
            CompilationPhase::AstGeneration => 1,
            CompilationPhase::TirGeneration => 2,
            CompilationPhase::NameResolution => 3,
            CompilationPhase::TypeChecking => 4,
            CompilationPhase::Lowering => 5,
            CompilationPhase::Optimization => 6,
        }
    }

    phase_rank(&left.phase)
        .cmp(&phase_rank(&right.phase))
        .then_with(|| left.inner.span.cmp(&right.inner.span))
        .then_with(|| left.inner.message.cmp(&right.inner.message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::link_native_binary;
    use rustc_hash::FxHashSet as HashSet;
    use salsa_test_macros::salsa_test;
    use std::ops::ControlFlow;
    use trunk_ir::dialect::clif;
    use trunk_ir::ops::DialectType;
    use trunk_ir::walk::{WalkAction, walk_region};

    fn source_logical_cps_root_module(body: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let source = r#"core.module @test {
            !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
            !Frame = adt.typeref<{name = "__tribute_continuation_frame_root_nil", tribute.cps_continuation_frame_result = core.nil}>
            !Done = closure.closure<func.func_sig<(core.nil) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
            !Resume = closure.closure<func.func_sig<(!Evidence, !Frame, tribute_rt.anyref) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 0}>
            !Dispatch = closure.closure<func.func_sig<(!Evidence, !Resume, core.i32, core.i32, core.i32, tribute_rt.anyref) -> core.never>, {tribute.calling_convention = 2, tribute.closure_environment_index = 1}>
            !__tribute_continuation_frame_root_nil = adt.struct<__tribute_continuation_frame_root_nil(done: !Done, dispatch: !Dispatch), {tribute.cps_continuation_frame_result = core.nil}>
            !Payload = adt.struct<__tribute_ability_payload_7590c57e()>
            func.func @main(%evidence: !Evidence, %frame: !Frame) -> core.never attributes {tribute.calling_convention = 2, tribute.root_source_result = core.nil} {
                BODY
            }
        }"#.replace("BODY", body);
        let module = trunk_ir::parser::parse_test_module(&mut ctx, &source);
        (ctx, module)
    }

    #[test]
    fn source_logical_root_defers_closure_storage_until_target_finalization() {
        let (mut ctx, module) = source_logical_cps_root_module("func.unreachable");
        enter_target_closure_storage_boundary(&mut ctx, module, &mut Default::default()).unwrap();
        let after_abi = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(after_abi.contains("closure.closure"), "{after_abi}");

        tribute_passes::closure_lower::finalize_closure_storage_layout(&mut ctx, module);

        let physical = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!physical.contains("closure.closure"), "{physical}");
    }

    #[test]
    fn empty_result_cps_root_executes_native_done_continuation() {
        let (mut ctx, module) = source_logical_cps_root_module(
            r#"
            %done = adt.struct_get %frame {field = 0, type = !__tribute_continuation_frame_root_nil} : !Done
            %nil = core.nil_value : core.nil
            func.tail_call_indirect %done, %nil {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
        "#,
        );

        run_native_target_pipeline(&mut ctx, module)
            .expect("logical CPS root crosses native boundary");
        for name in ["__tribute_main", "__tribute_done_k", "__tribute_unhandled"] {
            let function = module
                .ops(&ctx)
                .iter()
                .copied()
                .find_map(|op| {
                    let function = func_dialect::Func::from_op(&ctx, op).ok()?;
                    (function.sym_name(&ctx) == name).then_some(function)
                })
                .expect("root bridge function remains present");
            // The convention is consumed inside the boundary; the empty
            // physical result list is what identifies a Cps callable here.
            assert_eq!(
                tribute_core::get_calling_convention(&ctx, function.op_ref()),
                None
            );
            assert!(
                func_dialect::FuncSig::from_type_ref(&ctx, function.r#type(&ctx))
                    .unwrap()
                    .results(&ctx)
                    .is_empty()
            );
        }
        let object = compile_module_to_native(
            &mut ctx,
            module,
            false,
            NativeOptimizationOptions::production(),
        )
        .expect("empty-result root emits native code");
        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join("empty-result-cps-root");
        link_native_binary(&object, &executable, None).expect("empty-result root links");
        let output = std::process::Command::new(executable)
            .output()
            .expect("empty-result root starts");
        assert!(
            output.status.success(),
            "native root failed: {:?}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn evidence_lookup_preserves_marker_type_in_emitted_binary() {
        let mut ctx = trunk_ir::IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
                func.func @__tribute_evidence_lookup(%ev: core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>, %id: core.i32) -> core.i32 attributes {abi = "C"}
            }"#,
        );
        tribute_passes::wasm::evidence_to_wasm::bind_wasm_evidence_runtime(&mut ctx, module);
        tribute_passes::wasm::lower::finalize_wasm_gc_types(&mut ctx, module).unwrap();
        let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module).unwrap();
        wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
            .validate_all(&binary.bytes)
            .expect("evidence lookup must retain concrete Marker locals and return type");
    }

    #[test]
    fn root_dispatch_definition_matches_fixed_wasm_tail_signature_and_validates_binary() {
        use trunk_ir::dialect::wasm;
        let (mut ctx, module) = source_logical_cps_root_module(
            r#"
            %dispatch = adt.struct_get %frame {field = 1, type = !__tribute_continuation_frame_root_nil} : !Dispatch
            %resume = adt.ref_null {type = !Resume} : !Resume
            %product = adt.struct_new {type = !Payload} : !Payload
            %payload = core.unrealized_conversion_cast %product : tribute_rt.anyref
            effect.dispatch_cps %evidence, %dispatch, %resume, %payload {ability_ref = core.ability_ref<{name = "State"}>, op_name = "get", answer_type = core.nil}
        "#,
        );
        run_wasm_target_pipeline(&mut ctx, module).unwrap();
        let binary = compile_to_wasm(&mut ctx, module).unwrap_or_else(|error| {
            panic!(
                "{error}\n{}",
                trunk_ir::printer::print_module(&ctx, module.op())
            )
        });
        let dispatch_definition = module
            .ops(&ctx)
            .iter()
            .copied()
            .find_map(|op| {
                let function = wasm::Func::from_op(&ctx, op).ok()?;
                (function.sym_name(&ctx) == "__tribute_unhandled").then_some(function.r#type(&ctx))
            })
            .unwrap();
        let mut signatures = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            if wasm::ReturnCallIndirect::matches(&ctx, op) {
                signatures.push(
                    trunk_ir::op_interface::IndirectCallLikeOps::exact_signature(&ctx, op).unwrap(),
                );
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert!(signatures.contains(&dispatch_definition));
        let canonical_closure = tribute_passes::wasm::type_converter::closure_adt_type(&mut ctx);
        let signature = wasm::FuncSig::from_type_ref(&ctx, dispatch_definition).unwrap();
        assert_eq!(signature.inputs(&ctx)[2], canonical_closure);
        assert!(signature.results(&ctx).is_empty());
        let mut binary_dispatch_signatures = 0;
        for payload in wasmparser::Parser::new(0).parse_all(&binary.bytes) {
            if let wasmparser::Payload::TypeSection(types) = payload.unwrap() {
                for group in types {
                    for ty in group.unwrap().into_types() {
                        if let wasmparser::CompositeInnerType::Func(signature) =
                            ty.composite_type.inner
                            && signature.params().len() == 7
                        {
                            use wasmparser::{RefType, ValType};
                            let concrete = |index| {
                                ValType::Ref(
                                    RefType::new(
                                        true,
                                        wasmparser::HeapType::Concrete(
                                            wasmparser::UnpackedIndex::Module(index),
                                        ),
                                    )
                                    .unwrap(),
                                )
                            };
                            assert_eq!(
                                signature.params(),
                                [
                                    concrete(trunk_ir_wasm_backend::gc_types::EVIDENCE_IDX),
                                    ValType::Ref(RefType::ANYREF),
                                    concrete(trunk_ir_wasm_backend::gc_types::CLOSURE_STRUCT_IDX),
                                    ValType::I32,
                                    ValType::I32,
                                    ValType::I32,
                                    ValType::Ref(RefType::ANYREF)
                                ]
                            );
                            assert!(signature.results().is_empty());
                            binary_dispatch_signatures += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(binary_dispatch_signatures, 1);
        wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
            .validate_all(&binary.bytes)
            .expect("canonical root dispatch binary validates");
    }

    #[test]
    fn native_ownership_plan_options_follow_stage_policy_table() {
        let parameter_elision_only = NativeOptimizationOptions {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::ElideProvenBorrowed,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
        };
        let field_elision_only = NativeOptimizationOptions {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        };
        let both_elisions = NativeOptimizationOptions::production();

        let cases = [
            (
                Some(NativePipelineStage::AfterRcInsertion),
                both_elisions,
                false,
                false,
            ),
            (
                Some(NativePipelineStage::AfterBorrowedParameterOptimization),
                both_elisions,
                true,
                false,
            ),
            (
                Some(NativePipelineStage::AfterBorrowedParameterOptimization),
                field_elision_only,
                false,
                false,
            ),
            (
                Some(NativePipelineStage::AfterTemporaryBorrowOptimization),
                parameter_elision_only,
                true,
                false,
            ),
            (
                Some(NativePipelineStage::AfterRcOptimization),
                field_elision_only,
                false,
                true,
            ),
            (None, both_elisions, true, true),
        ];

        for (stage, optimizations, borrowed_parameters, field_borrows) in cases {
            assert_eq!(
                native_ownership_plan_options(stage, optimizations),
                tribute_passes::native::ownership_plan::NativeOwnershipPlanOptions {
                    elide_proven_borrowed_parameters: borrowed_parameters,
                    elide_proven_field_borrows: field_borrows,
                },
                "unexpected ownership plan options at {stage:?}",
            );
        }
    }

    fn source_from_str(path: &str, text: &str) -> SourceCst {
        salsa::with_attached_database(|db| SourceCst::from_source_str(db, path, text))
            .expect("attached db")
    }

    /// Programs whose boundary exit is observed on both targets. They cover
    /// Direct and CPS callables, closures with captures, tail-resumptive and
    /// general handlers, the CPS root bridge, and standard I/O.
    const BOUNDARY_EXIT_PROGRAMS: &[(&str, &str)] = &[
        (
            "native_calculator.trb",
            include_str!("../lang-examples/native_calculator.trb"),
        ),
        (
            "native_effects.trb",
            include_str!("../lang-examples/native_effects.trb"),
        ),
        (
            "wasm_dynamic_output.trb",
            include_str!("../lang-examples/wasm_dynamic_output.trb"),
        ),
        (
            "closure_capture.trb",
            r#"fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() -> Nil {
    let a = +1
    let _ = apply(fn(n) { n + a }, +41)
}
"#,
        ),
        (
            "tail_resumptive_handler.trb",
            r#"ability Ask {
    fn ask() -> Nat
}

fn use_ask() ->{Ask} Nat {
    Ask::ask()
}

fn main() -> Nil {
    let _ = handle use_ask() {
        do result { result }
        fn Ask::ask() { 42 }
    }
}
"#,
        ),
        (
            "state_handler.trb",
            r#"ability State(s) {
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
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Nil {
    let _ = run_state(fn() { set_then_get() }, +0)
}
"#,
        ),
    ];

    /// Every program reaches `target`'s boundary exit, whose enforcement
    /// rejects any violation.
    fn assert_boundary_exit_programs_meet_the_contract(
        db: &crate::TributeDatabaseImpl,
        target: tribute_passes::abi_boundary::TargetKind,
    ) {
        for (path, text) in BOUNDARY_EXIT_PROGRAMS {
            let source = source_from_str(path, text);
            let (mut ctx, module) = run_shared_pipeline(db, source)
                .expect("shared pipeline must succeed")
                .unwrap_or_else(|| panic!("{path} must lower"));
            run_target_to_boundary_exit(&mut ctx, module, target)
                .unwrap_or_else(|error| panic!("{path}: target boundary failed: {error}"));
        }
    }

    #[salsa_test]
    fn native_boundary_exit_programs_meet_the_contract(db: &crate::TributeDatabaseImpl) {
        assert_boundary_exit_programs_meet_the_contract(
            db,
            tribute_passes::abi_boundary::TargetKind::Native,
        );
    }

    #[salsa_test]
    fn wasm_boundary_exit_programs_meet_the_contract(db: &crate::TributeDatabaseImpl) {
        assert_boundary_exit_programs_meet_the_contract(
            db,
            tribute_passes::abi_boundary::TargetKind::Wasm,
        );
    }

    #[test]
    fn boundary_exit_enforcement_rejects_any_violation() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil attributes {tribute.calling_convention = 0} {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#,
        );
        for target in [
            tribute_passes::abi_boundary::TargetKind::Native,
            tribute_passes::abi_boundary::TargetKind::Wasm,
        ] {
            let error = enforce_boundary_exit(&ctx, module, target)
                .expect_err("a forbidden attribute must fail the boundary exit");
            assert!(
                error
                    .to_string()
                    .contains("forbidden attribute tribute.calling_convention"),
                "{error}"
            );
        }
    }

    /// Print `module` and parse the text into a fresh context.
    fn reparse_module(ctx: &IrContext, module: Module, path: &str) -> (IrContext, Module) {
        let text = trunk_ir::printer::print_module(ctx, module.op());
        let mut reparsed = IrContext::new();
        let op = trunk_ir::parser::parse_module(&mut reparsed, &text).unwrap_or_else(|error| {
            panic!(
                "{path}: printed boundary-exit IR must parse at offset {}: {}",
                error.offset, error.message
            )
        });
        let module = Module::new(&reparsed, op)
            .unwrap_or_else(|| panic!("{path}: parsed boundary-exit IR must be a module"));
        (reparsed, module)
    }

    /// Lowering and emission after the boundary exit need nothing but the
    /// printed IR: a program parsed back from its boundary-exit text emits
    /// the same binary as the original.
    fn assert_boundary_exit_round_trips(
        db: &crate::TributeDatabaseImpl,
        target: tribute_passes::abi_boundary::TargetKind,
    ) {
        for (path, text) in BOUNDARY_EXIT_PROGRAMS {
            let source = source_from_str(path, text);
            let (mut ctx, module) = run_shared_pipeline(db, source)
                .expect("shared pipeline must succeed")
                .unwrap_or_else(|| panic!("{path} must lower"));
            run_target_to_boundary_exit(&mut ctx, module, target)
                .unwrap_or_else(|error| panic!("{path}: target boundary failed: {error}"));
            let (mut reparsed, reparsed_module) = reparse_module(&ctx, module, path);

            // Parsing registers every alias explicitly, which can change how
            // the first reprint spells types; the text is stable from then on.
            let reprinted = trunk_ir::printer::print_module(&reparsed, reparsed_module.op());
            let (again, again_module) = reparse_module(&reparsed, reparsed_module, path);
            assert_eq!(
                trunk_ir::printer::print_module(&again, again_module.op()),
                reprinted,
                "{path}: reparsed boundary-exit IR must print stably"
            );

            let direct = emit_from_boundary_exit(&mut ctx, module, target);
            let round_trip = emit_from_boundary_exit(&mut reparsed, reparsed_module, target);
            match (direct, round_trip) {
                (Ok(direct), Ok(round_trip)) => assert!(
                    round_trip == direct,
                    "{path}: the round-tripped IR must emit the same binary"
                ),
                // A program the target cannot emit yet, such as `read_line` on
                // Wasm, fails either way.
                (Err(_), Err(_)) => {}
                (direct, round_trip) => panic!(
                    "{path}: emission outcome changed through text: direct {:?}, round trip {:?}",
                    direct.map(|bytes| bytes.len()),
                    round_trip.map(|bytes| bytes.len())
                ),
            }
        }
    }

    /// Every physical tail signature at the boundary exit states that the
    /// callee consumes each input: definitions, function references, and
    /// indirect calls, including the bridge and dispatch signatures the
    /// boundary builds itself.
    fn assert_tail_signatures_consume_every_input(
        db: &crate::TributeDatabaseImpl,
        target: tribute_passes::abi_boundary::TargetKind,
    ) {
        use trunk_ir::op_interface::IndirectCallLikeOps;
        let consumed = |ctx: &IrContext, signature: func_dialect::FuncSig| {
            signature
                .input_attrs(ctx)
                .all(|attrs| attrs.get_str(ctx, "tribute.ownership") == Some("consumed"))
        };
        let mut tail_signatures = 0;
        for (path, text) in BOUNDARY_EXIT_PROGRAMS {
            let source = source_from_str(path, text);
            let (mut ctx, module) = run_shared_pipeline(db, source)
                .expect("shared pipeline must succeed")
                .unwrap_or_else(|| panic!("{path} must lower"));
            run_target_to_boundary_exit(&mut ctx, module, target)
                .unwrap_or_else(|error| panic!("{path}: target boundary failed: {error}"));
            let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
                let signature = if let Ok(function) = func_dialect::Func::from_op(&ctx, op) {
                    Some(function.r#type(&ctx))
                } else if func_dialect::Constant::matches(&ctx, op) {
                    ctx.op_result_types(op).first().copied()
                } else {
                    IndirectCallLikeOps::exact_signature(&ctx, op)
                };
                if let Some(signature) =
                    signature.and_then(|ty| func_dialect::FuncSig::from_type_ref(&ctx, ty))
                    && signature.call_conv(&ctx) == Some(func_dialect::CallConv::Tail)
                {
                    tail_signatures += 1;
                    assert!(
                        consumed(&ctx, signature),
                        "{path}: {target:?} tail signature lacks the consumed contract: {}",
                        trunk_ir::printer::print_type(&ctx, signature.as_type_ref())
                    );
                }
                ControlFlow::Continue(WalkAction::Advance)
            });
        }
        assert!(tail_signatures > 0, "no physical tail signature observed");
    }

    #[salsa_test]
    fn native_tail_signatures_consume_every_input(db: &crate::TributeDatabaseImpl) {
        assert_tail_signatures_consume_every_input(
            db,
            tribute_passes::abi_boundary::TargetKind::Native,
        );
    }

    #[salsa_test]
    fn wasm_tail_signatures_consume_every_input(db: &crate::TributeDatabaseImpl) {
        assert_tail_signatures_consume_every_input(
            db,
            tribute_passes::abi_boundary::TargetKind::Wasm,
        );
    }

    #[salsa_test]
    fn native_boundary_exit_round_trips_through_text(db: &crate::TributeDatabaseImpl) {
        assert_boundary_exit_round_trips(db, tribute_passes::abi_boundary::TargetKind::Native);
    }

    #[salsa_test]
    fn wasm_boundary_exit_round_trips_through_text(db: &crate::TributeDatabaseImpl) {
        assert_boundary_exit_round_trips(db, tribute_passes::abi_boundary::TargetKind::Wasm);
    }

    fn prepare_native_fixture(
        db: &crate::TributeDatabaseImpl,
        path: &str,
        text: &str,
    ) -> (IrContext, Module) {
        let source = source_from_str(path, text);
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .expect("fixture must lower");
        validate_and_report_arity(db, &ctx, module);
        run_native_target_pipeline(&mut ctx, module).expect("native target lowering must succeed");
        prepare_module_to_native(
            &mut ctx,
            module,
            false,
            NativeOptimizationOptions::production(),
            None,
        )
        .expect("native preparation must succeed");
        trunk_ir_cranelift_backend::validate_clif_ir(&ctx, module)
            .expect("prepared native IR must satisfy exact callable slot contracts");
        (ctx, module)
    }

    /// Indirect calls of the compiled program, excluding the RTTI release
    /// dispatch that native lowering generates.
    fn clif_indirect_calls(ctx: &IrContext, module: Module) -> Vec<trunk_ir::OpRef> {
        let body = module.body(ctx).expect("module must have a body");
        let mut calls = Vec::new();
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if clif::Func::from_op(ctx, op).is_ok_and(|function| {
                function.sym_name(ctx) == tribute_passes::native::rtti::DEEP_RELEASE_FN
            }) {
                return ControlFlow::Continue(WalkAction::Skip);
            }
            if clif::CallIndirect::matches(ctx, op) || clif::ReturnCallIndirect::matches(ctx, op) {
                calls.push(op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        calls
    }

    fn clif_indirect_signature(ctx: &IrContext, call: trunk_ir::OpRef) -> clif::FuncSig {
        let signature = ctx
            .op(call)
            .attributes
            .get_type("sig")
            .and_then(|ty| clif::FuncSig::from_type_ref(ctx, ty))
            .expect("indirect call must have an exact signature");
        let operands = ctx.op_operands(call);
        assert_eq!(
            operands[1..]
                .iter()
                .map(|&operand| ctx.value_ty(operand))
                .collect::<Vec<_>>(),
            signature.inputs(ctx),
            "indirect call operands must match its exact signature"
        );
        assert_eq!(
            ctx.op_result_types(call),
            signature.results(ctx),
            "indirect call results must match its exact signature"
        );
        signature
    }

    #[salsa_test]
    fn native_preparation_direct_closure_call_slots_are_exact(db: &crate::TributeDatabaseImpl) {
        let (mut ctx, module) = prepare_native_fixture(
            db,
            "closure_exec_simple.trb",
            r#"
extern "C" fn __tribute_print_nat(value: Nat) -> Nil

fn main() -> Nil {
    let f = fn(x) { x + 1 }
    __tribute_print_nat(f(41))
}
"#,
        );
        let pointer_type = core_dialect::ptr(&mut ctx).as_type_ref();
        let i32_type = ctx
            .types()
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == trunk_ir::Symbol::new("core")
                    && data.name == trunk_ir::Symbol::new("i32"))
                .then_some(ty)
            })
            .expect("fixture must contain core.i32");
        let calls = clif_indirect_calls(&ctx, module);
        assert_eq!(
            calls.len(),
            1,
            "pure closure fixture must have one Direct indirect call"
        );
        let call = calls[0];
        let signature = clif_indirect_signature(&ctx, call);
        assert_eq!(signature.inputs(&ctx), [pointer_type, i32_type]);
        assert_eq!(signature.results(&ctx), [i32_type]);
    }

    #[salsa_test]
    fn native_preparation_cps_closure_call_slots_are_exact(db: &crate::TributeDatabaseImpl) {
        let (mut ctx, module) = prepare_native_fixture(
            db,
            "closure_exec_callback.trb",
            r#"
extern "C" fn __tribute_print_nat(value: Nat) -> Nil

fn apply(f: fn(Nat) ->{e} Nat, value: Nat) ->{e} Nat { f(value) }

fn main() -> Nil {
    __tribute_print_nat(apply(fn(x) { x + 1 }, 41))
}
"#,
        );
        let pointer_type = core_dialect::ptr(&mut ctx).as_type_ref();
        let i32_type = ctx
            .types()
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == trunk_ir::Symbol::new("core")
                    && data.name == trunk_ir::Symbol::new("i32"))
                .then_some(ty)
            })
            .expect("fixture must contain core.i32");
        let calls: Vec<_> = clif_indirect_calls(&ctx, module)
            .into_iter()
            .filter(|&call| {
                ctx.op_operands(call).last().is_some_and(|&argument| {
                    matches!(
                        ctx.value_def(argument),
                        trunk_ir::refs::ValueDef::OpResult(producer, _)
                            if clif::Iadd::matches(&ctx, producer)
                    )
                })
            })
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "open callback fixture must have one CPS call with the addition result"
        );
        let call = calls[0];
        let [callee, continuation_environment, value] = ctx.op_operands(call) else {
            panic!("CPS Done tail must have one callee, its environment, and the value");
        };
        assert!(clif::ReturnCallIndirect::matches(&ctx, call));
        assert_eq!(
            ctx.op(call).attributes.get("tribute.calling_convention"),
            None
        );
        let trunk_ir::refs::ValueDef::OpResult(environment_load, 0) =
            ctx.value_def(*continuation_environment)
        else {
            panic!("the CPS continuation environment must be loaded from its closure");
        };
        let environment_load = clif::Load::from_op(&ctx, environment_load)
            .expect("the CPS continuation environment must come from clif.load");
        assert_eq!(environment_load.offset(&ctx), 8);
        let done = ctx.op_operands(environment_load.op_ref())[0];
        let trunk_ir::refs::ValueDef::OpResult(done_load, 0) = ctx.value_def(done) else {
            panic!("Done closure must be loaded from its continuation frame");
        };
        let done_load =
            clif::Load::from_op(&ctx, done_load).expect("Done must come from clif.load");
        assert_eq!(done_load.offset(&ctx), 0);
        let frame = ctx.op_operands(done_load.op_ref())[0];
        assert!(matches!(
            ctx.value_def(frame),
            trunk_ir::refs::ValueDef::BlockArg(_, 2)
        ));
        let trunk_ir::refs::ValueDef::OpResult(callee_load, 0) = ctx.value_def(*callee) else {
            panic!("Done callee must be loaded from the same closure");
        };
        let callee_load = clif::Load::from_op(&ctx, callee_load).expect("callee load");
        assert_eq!(callee_load.offset(&ctx), 0);
        assert_eq!(ctx.op_operands(callee_load.op_ref()), [done]);
        assert!(matches!(
            ctx.value_def(*value),
            trunk_ir::refs::ValueDef::OpResult(producer, 0)
                if clif::Iadd::matches(&ctx, producer)
        ));
        let signature = clif_indirect_signature(&ctx, call);
        assert_eq!(signature.inputs(&ctx), [pointer_type, i32_type]);
        assert!(signature.results(&ctx).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn native_one_shot_wrapper_traps_on_second_invocation() {
        use std::os::unix::process::ExitStatusExt;

        let input = r#"core.module @one_shot {
  !state = adt.struct<OneShotState(consumed: core.i1)>

  func.func @one_shot_wrapper(%state: !state, %value: core.i32) -> core.i32 attributes {tribute.calling_convention = 0} {
    %consumed = adt.struct_get %state {field = 0, type = !state} : core.i1
    %answer = scf.if %consumed : core.i32 {
      func.unreachable
    } {
      %consumed_true = arith.const {value = 1} : core.i1
      adt.struct_set %state, %consumed_true {field = 0, type = !state}
      scf.yield %value
    }
    func.return %answer
  }

  func.func @main() -> core.i32 attributes {tribute.calling_convention = 0} {
    %not_consumed = arith.const {value = 0} : core.i1
    %state = adt.struct_new %not_consumed {type = !state} : !state
    %input = arith.const {value = 0} : core.i32
    %first = func.call %state, %input {callee = @one_shot_wrapper, tribute.calling_convention = 0} : core.i32
    %second = func.call %state, %first {callee = @one_shot_wrapper, tribute.calling_convention = 0} : core.i32
    func.return %second
  }
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, input);

        let object = compile_module_to_native(
            &mut ctx,
            module,
            false,
            NativeOptimizationOptions::production(),
        )
        .expect("one-shot wrapper must compile to native code");
        let temp = tempfile::tempdir().expect("temporary executable directory");
        let executable = temp.path().join("one-shot-wrapper");
        link_native_binary(&object, &executable, None).expect("one-shot wrapper must link");

        let status = std::process::Command::new(executable)
            .status()
            .expect("one-shot wrapper executable must start");
        assert!(
            status.signal().is_some(),
            "the second invocation of the same wrapper must terminate by signal, got {status}"
        );
    }

    /// A physical module carries its callable contract in signatures alone:
    /// no semantic calling convention, root, or frame metadata is present.
    fn physical_tail_chain_module(done_call_conv: &str, reference_call_conv: &str) -> String {
        format!(
            r#"core.module @physical {{
  func.func @fail() {{
    func.unreachable
  }}

  func.func @done(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (){done_call_conv}>}} {{
    %expected = arith.const {{value = 3}} : core.i32
    %ok = arith.cmpi %value, %expected {{predicate = "eq"}} : core.i1
    scf.if %ok {{
      scf.yield
    }} {{
      func.call {{callee = @fail}}
      scf.yield
    }}
    func.return
  }}

  func.func @step(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}} {{
    %one = arith.const {{value = 1}} : core.i32
    %next = arith.addi %value, %one : core.i32
    %done = func.constant {{func_ref = @done}} : func.func_sig<(core.i32) -> (){reference_call_conv}>
    func.tail_call_indirect %done, %next {{signature = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}}
  }}

  func.func @start(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}} {{
    %one = arith.const {{value = 1}} : core.i32
    %next = arith.addi %value, %one : core.i32
    func.tail_call %next {{callee = @step}}
  }}

  func.func @main() -> core.i32 {{
    %input = arith.const {{value = 1}} : core.i32
    func.call %input {{callee = @start}}
    %exit = arith.const {{value = 0}} : core.i32
    func.return %exit
  }}
}}"#
        )
    }

    #[test]
    fn physical_module_without_control_metadata_runs_native_tail_calls() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            &physical_tail_chain_module(", {call_conv = \"tail\"}", ", {call_conv = \"tail\"}"),
        );
        let verified = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
        assert!(verified.is_ok(), "{verified}");

        let object = compile_module_to_native(
            &mut ctx,
            module,
            false,
            NativeOptimizationOptions::production(),
        )
        .expect("a physical module must lower without semantic control metadata");
        let temp = tempfile::tempdir().expect("temporary executable directory");
        let executable = temp.path().join("physical-tail-chain");
        link_native_binary(&object, &executable, None).expect("physical module must link");

        let status = std::process::Command::new(executable)
            .status()
            .expect("physical module executable must start");
        assert!(
            status.success(),
            "direct and indirect tail transfers must deliver the value, got {status}"
        );
    }

    #[test]
    fn physical_module_with_mismatched_tail_convention_is_rejected() {
        // A reference typed like its platform target cannot flow into a
        // tail-convention indirect call: the typed-callee verifier rejects it.
        let mut ctx = IrContext::new();
        let module =
            trunk_ir::parser::parse_test_module(&mut ctx, &physical_tail_chain_module("", ""));
        let verified = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
        assert!(
            verified
                .to_string()
                .contains("exact indirect signature differs from typed callee"),
            "{verified}"
        );

        // A reference typed with the wrong convention for its target passes
        // the typed-callee check, so lowering must refuse to erase it.
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            &physical_tail_chain_module("", ", {call_conv = \"tail\"}"),
        );
        let verified = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
        assert!(verified.is_ok(), "{verified}");
        let error = compile_module_to_native(
            &mut ctx,
            module,
            false,
            NativeOptimizationOptions::production(),
        )
        .expect_err("a tail transfer to a platform-convention function must not be emitted");
        assert!(error.to_string().contains("func.constant"), "{error}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_use_chain_verifier_reports_offending_pass() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %y = arith.addi %x, %x : core.i32
    func.return %y
  }
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, input);
        let core_module = core_dialect::Module::from_op(&ctx, module.op())
            .expect("test input must parse a core.module");

        let mut pm = PassManager::new();
        pm.add_pass(trunk_ir::pass::pass_fn(
            "break-use-chain",
            |ctx: &mut IrContext, module: core_dialect::Module, _analyses: &mut AnalysisCache| {
                let module_block = ctx.region(module.body(ctx)).blocks[0];
                let func_op = ctx.block(module_block).ops[0];
                let func_region = ctx.op_region(func_op, 0).unwrap();
                let func_block = ctx.region(func_region).blocks[0];
                let add_op = ctx
                    .block(func_block)
                    .ops
                    .iter()
                    .copied()
                    .find(|&op| {
                        let data = ctx.op(op);
                        data.dialect == trunk_ir::Symbol::new("arith")
                            && data.name == trunk_ir::Symbol::new("addi")
                    })
                    .expect("test input must contain arith.addi");

                // Deliberately mutate operands without updating use-chains to
                // exercise the graph-wide pass-manager verifier.
                let _old_operands = std::mem::take(&mut ctx.op_mut(add_op).operands);
                Ok(())
            },
        ));
        pm.with_debug_verifier();

        let error = pm
            .run(&mut ctx, core_module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "break-use-chain");
        assert!(
            error
                .to_string()
                .contains("pass `break-use-chain` broke an IR invariant: use-chain regression"),
            "{error}"
        );
        assert!(
            error.to_string().contains("no such operand exists"),
            "{error}"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_verifier_reports_schema_violations_after_a_pass() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %y = arith.addi %x, %x : core.i32
    func.return %y
  }
}"#;
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, input);
        let core_module = core_dialect::Module::from_op(&ctx, module.op())
            .expect("test input must parse a core.module");

        let mut pm = PassManager::new();
        pm.add_pass(trunk_ir::pass::pass_fn(
            "break-schema",
            |ctx: &mut IrContext, module: core_dialect::Module, _analyses: &mut AnalysisCache| {
                let module_block = ctx.region(module.body(ctx)).blocks[0];
                let func_op = ctx.block(module_block).ops[0];
                let func_block = ctx.region(ctx.op_region(func_op, 0).unwrap()).blocks[0];
                let add_op = ctx.block(func_block).ops[0];
                // Retype the result without touching use-chains, so only the
                // `T` binding of `arith.addi` is violated.
                let i64_ty =
                    ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("core", "i64").build());
                ctx.set_op_result_type(add_op, 0, i64_ty);
                Ok(())
            },
        ));
        pm.with_debug_verifier();

        let error = pm
            .run(&mut ctx, core_module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "break-schema");
        let message = error.to_string();
        assert!(message.contains("schema regression"), "{message}");
        assert!(message.contains("arith.addi"), "{message}");
    }

    #[salsa_test]
    fn test_full_pipeline(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            "fn compute() -> Int { +42 }\nfn main() -> Nil { }",
        );

        let result = compile_ast(db, source).expect("pipeline should not fail");
        assert!(result.is_some(), "Should compile successfully");
        let (ctx, m) = result.unwrap();
        assert_eq!(m.name(&ctx), Some(trunk_ir::Symbol::new("test")));
    }

    #[salsa_test]
    fn production_route_legalizes_source_control_with_exact_declarations(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "production-route.trb",
            r#"
ability Counter {
    op next() -> Int
}

fn count() ->{Counter} Int {
    Counter::next()
}

fn main() -> Nil {
    let _ = handle count() {
        do result { result }
        op Counter::next() { resume +1 }
    }
}
"#,
        );
        let frontend = compile_frontend_for_shared_route(db, source)
            .expect("production frontend route should lower");
        let logical = trunk_ir::printer::print_module(&frontend.context, frontend.module.op());
        assert!(
            logical.contains("tribute_control.perform"),
            "frontend must retain source logical perform:\n{logical}"
        );
        let throw = frontend
            .operation_declarations
            .iter()
            .position(|declaration| frontend.context.str(declaration.op_name) == "throw")
            .expect("prelude Throw declaration");
        let next = frontend
            .operation_declarations
            .iter()
            .position(|declaration| frontend.context.str(declaration.op_name) == "next")
            .expect("source Counter declaration");
        assert!(
            throw < next,
            "operation declarations must preserve prelude-before-source order"
        );
        let (ctx, module) = compile_ast(db, source)
            .expect("shared lowering should succeed")
            .expect("shared lowering should produce a module");
        let output = trunk_ir::printer::print_module(&ctx, module.op());
        for forbidden in [
            "tribute_control.",
            "tribute_control.func_sig",
            "resume_token",
            "ability.legacy_",
        ] {
            assert!(
                !output.contains(forbidden),
                "shared route leaked `{forbidden}`:\n{output}"
            );
        }
    }

    #[salsa_test]
    fn test_std_io_lowers_to_shared_dialect(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "io.trb",
            r#"
use std::io::print
use std::io::print_line
use std::io::read_line

fn input() ->{std::io::Io, abilities::Throw(std::io::Error)} String {
    read_line()
}

fn main() ->{std::io::Io} Nil {
    print("hello")
    print_line("hello")
}
"#,
        );

        let (ctx, module) = compile_ast(db, source)
            .expect("pipeline should not fail")
            .expect("pipeline should produce a module");
        let output = trunk_ir::printer::print_module(&ctx, module.op());

        assert!(output.contains("tribute_io.write"), "{output}");
        assert!(output.contains("tribute_io.read_line"), "{output}");
        assert!(!output.contains("__tribute_io_"), "{output}");

        let mut newline_values = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            if let Ok(write) = tribute_ir::dialect::tribute_io::Write::from_op(&ctx, op)
                && let trunk_ir::refs::ValueDef::OpResult(producer, _) =
                    ctx.value_def(write.newline(&ctx))
                && let Ok(constant) = trunk_ir::dialect::arith::Const::from_op(&ctx, producer)
                && let trunk_ir::Attribute::Bool(value) = constant.value(&ctx)
            {
                newline_values.push(value);
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        newline_values.sort_unstable();
        assert_eq!(newline_values, [false, true], "{output}");
    }

    #[salsa_test]
    fn test_compile_with_diagnostics(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without errors
        assert!(
            result.diagnostics.is_empty(),
            "Expected no diagnostics, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_unresolved_reference_diagnostic(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn main() -> Int { undefined_var }");

        let result = compile_with_diagnostics(db, source);
        // Should have an unresolved reference error
        assert!(
            !result.diagnostics.is_empty(),
            "Expected diagnostic for unresolved reference"
        );

        // Check that the diagnostic message mentions the unresolved name
        let has_unresolved_error = result.diagnostics.iter().any(|d| {
            d.inner.message.contains("unresolved")
                && d.inner.severity == DiagnosticSeverity::Error
                && d.phase == CompilationPhase::NameResolution
        });
        assert!(
            has_unresolved_error,
            "Expected unresolved name error, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_parse_error_diagnostic(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn identity(a)(x: a) -> a { x }");
        let result = compile_with_diagnostics(db, source);

        let has_parse_error = result.diagnostics.iter().any(|d| {
            d.inner.severity == DiagnosticSeverity::Error
                && d.phase == CompilationPhase::Parsing
                && d.inner.message.contains("syntax error")
        });
        assert!(
            has_parse_error,
            "Expected parse error diagnostic, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_prelude_loads(db: &salsa::DatabaseImpl) {
        let prelude = prelude_module(db);
        assert!(prelude.is_some(), "Prelude should load successfully");
    }

    #[salsa_test]
    fn test_prelude_exports_share_typed_declaration_identity(db: &salsa::DatabaseImpl) {
        let exports = prelude_exports(db).expect("prelude exports");
        let typed = prelude_module(db).expect("typed prelude");
        for (id, exported) in exports.function_types(db) {
            let (_, definition) = typed
                .function_types(db)
                .iter()
                .find(|(name, _)| *name == id.qualified(db))
                .expect("exported function has a typed declaration");
            assert_eq!(exported, definition, "{}", id.qualified(db));
        }
    }

    #[salsa_test]
    fn test_prelude_option_type(db: &salsa::DatabaseImpl) {
        // Use Option type from prelude
        let source = source_from_str("test.trb", "fn maybe() -> Option(Int) { None }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without "unresolved" errors for Option or None
        let has_option_error = result
            .diagnostics
            .iter()
            .any(|d| d.inner.message.contains("Option") || d.inner.message.contains("None"));
        assert!(
            !has_option_error,
            "Option and None should be available from prelude, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_prelude_result_type(db: &salsa::DatabaseImpl) {
        // Use Result type from prelude
        let source = source_from_str("test.trb", "fn success() -> Result(Int, String) { Ok(42) }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without "unresolved" errors for Result or Ok
        let has_result_error = result
            .diagnostics
            .iter()
            .any(|d| d.inner.message.contains("Result") || d.inner.message.contains("Ok"));
        assert!(
            !has_result_error,
            "Result and Ok should be available from prelude, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_expression_pattern_binding(db: &salsa::DatabaseImpl) {
        // Simple case expression with identifier pattern binding
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Int) -> Int {
                case x {
                    y -> y
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        // Pattern binding `y` should be resolved in the case arm body
        let has_unresolved_y = result
            .diagnostics
            .iter()
            .any(|d| d.inner.message.contains("unresolved") && d.inner.message.contains("y"));
        assert!(
            !has_unresolved_y,
            "Pattern binding `y` should be resolved, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_lowering_exhaustive(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Nat) -> Nat {
                case x {
                    0 -> 1
                    _ -> 2
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        assert!(
            result.diagnostics.is_empty(),
            "Expected no diagnostics, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_lowering_non_exhaustive(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Nat) -> Nat {
                case x {
                    0 -> 1
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        let has_non_exhaustive = result.diagnostics.iter().any(|d| {
            d.inner.message.contains("non-exhaustive")
                && d.inner.severity == DiagnosticSeverity::Error
        });
        assert!(
            has_non_exhaustive,
            "Expected non-exhaustive case diagnostic, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_lowering_nested_non_exhaustive(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn pick(o: Option(Bool)) -> Nat {
                case o {
                    None -> 0
                    Some(True) -> 1
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        let has_non_exhaustive = result.diagnostics.iter().any(|d| {
            d.inner.message.contains("missing patterns: Some(False)")
                && d.inner.severity == DiagnosticSeverity::Error
        });
        assert!(
            has_non_exhaustive,
            "Expected non-exhaustive case diagnostic, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_lowering_bool_exhaustive(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Bool) -> Nat {
                case x {
                    True -> 1
                    False -> 0
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        assert!(
            result.diagnostics.is_empty(),
            "Expected no diagnostics, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_tdnr_struct_field_access(db: &salsa::DatabaseImpl) {
        // Test that TDNR resolves user.name to User::name(user)
        let source = source_from_str(
            "test.trb",
            r#"
            struct User {
                name: String,
                age: Int,
            }

            fn get_name(user: User) -> String {
                user.name
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);

        // Should compile without unresolved errors - TDNR should resolve user.name
        let has_unresolved_name = result
            .diagnostics
            .iter()
            .any(|d| d.inner.message.contains("unresolved") && d.inner.message.contains("name"));
        assert!(
            !has_unresolved_name,
            "TDNR should resolve user.name to User::name(user), got: {:?}",
            result.diagnostics
        );
    }

    // =========================================================================
    // AST-based Pipeline Tests
    // =========================================================================

    #[salsa_test]
    fn test_ast_pipeline_simple_function(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            "fn compute() -> Int { +42 }\nfn main() -> Nil { }",
        );

        let result = compile_frontend(db, source);
        assert!(result.is_some());
        let (ctx, m) = result.unwrap();
        assert_eq!(m.name(&ctx), Some(trunk_ir::Symbol::new("test")));
    }

    #[salsa_test]
    fn test_ast_pipeline_with_params(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

        let result = compile_frontend(db, source);
        assert!(result.is_some());
        let (ctx, m) = result.unwrap();
        assert_eq!(m.name(&ctx), Some(trunk_ir::Symbol::new("test")));
    }

    #[test]
    fn generic_intrinsic_specialization_reaches_concrete_lowering() {
        salsa::Database::attach(&salsa::DatabaseImpl::default(), |db| {
            let source = source_from_str(
                "prelude_list_prepend.trb",
                r#"
fn main() -> Nil {
    let _ = List::prepend("token", [])
}
"#,
            );
            let typed = parse_and_lower_ast(db, source).expect("frontend output");
            let (_, monomorphized) =
                merge_and_lower_to_ir_with(db, &typed, source, |typed, _, _, _| typed);
            let ast = format!("{:#?}", monomorphized.ast);
            let concrete = "std::collections::List::__tribute_list_prepend_intrinsic$std::String";

            assert!(
                monomorphized
                    .function_types
                    .contains_key(&trunk_ir::Symbol::new(concrete)),
                "monomorphization must retain the concrete intrinsic scheme"
            );
            assert!(
                ast.contains(concrete),
                "specialized List body must call the concrete intrinsic:\n{ast}"
            );
            assert!(
                !ast.contains("std::collections::List::__tribute_list_prepend_intrinsic$T"),
                "specialized List body must not retain generic intrinsic binders:\n{ast}"
            );

            let frontend = compile_frontend_for_shared_route(db, source)
                .expect("source-logical lowering must retain the specialization");
            let concrete_symbol = trunk_ir::Symbol::new(concrete);
            assert!(
                frontend.compiler_intrinsics.iter().any(|declaration| {
                    declaration.symbol == concrete_symbol
                        && declaration.identity
                            == trunk_ir::Symbol::new(
                                "std::collections::List::__tribute_list_prepend_intrinsic",
                            )
                }),
                "the concrete specialization must retain its base intrinsic identity"
            );
            let logical = trunk_ir::printer::print_module(&frontend.context, frontend.module.op());
            assert!(
                logical.contains(
                    r#"tribute.compiler_intrinsic = "std::collections::List::__tribute_list_prepend_intrinsic""#
                ),
                "the concrete declaration must carry its exact intrinsic identity:\n{logical}"
            );

            let (ctx, module) = compile_ast(db, source)
                .expect("shared pipeline must lower the concrete intrinsic specialization")
                .expect("shared pipeline must produce a module");
            let lowered = trunk_ir::printer::print_module(&ctx, module.op());
            assert!(lowered.contains("list.prepend"), "{lowered}");
            assert!(
                !lowered.contains("std::collections::List::__tribute_list_prepend_intrinsic"),
                "list intrinsic lowering must consume the concrete declaration:\n{lowered}"
            );
        });
    }

    #[salsa_test]
    fn list_import_paths_share_canonical_specialization(db: &salsa::DatabaseImpl) {
        use tribute_front::ast::Decl;
        use trunk_ir::Symbol;

        let source = source_from_str(
            "list_import_paths.trb",
            r#"
use std::collections::List as Sequence
use std::collections::List::prepend as push

fn main() -> Nil {
    let first = List::prepend(1, [])
    let second = std::collections::List::prepend(2, first)
    let third = Sequence::prepend(3, second)
    let _ = push(4, third)
}
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        let instances = &typed.expression_types(db).function_instances;
        let canonical = Symbol::new("std::collections::List::prepend");
        assert_eq!(instances.len(), 4);
        let declaration = instances[0].1.function;
        assert_eq!(declaration.qualified(db), canonical);
        assert!(
            instances
                .iter()
                .all(|(_, instance)| instance.function == declaration),
            "{instances:#?}"
        );

        let (_, prepared) = merge_and_lower_to_ir_with(db, &typed, source, |typed, _, _, _| typed);
        let specialized = Symbol::new("std::collections::List::prepend$Nat");
        assert_eq!(
            prepared
                .ast
                .decls
                .iter()
                .filter(|decl| {
                    matches!(decl, Decl::Function(function) if function.name == specialized)
                })
                .count(),
            1,
            "all import paths must share one specialized definition"
        );
        assert!(prepared.function_types.contains_key(&specialized));
        assert!(
            !prepared
                .function_types
                .contains_key(&Symbol::new("List::prepend$Nat"))
        );
    }

    #[salsa_test]
    fn logical_enum_layout_rejects_missing_or_wrong_specialized_schema(db: &salsa::DatabaseImpl) {
        use tribute_front::ast::{Decl, Type, TypeKind, TypeScheme};
        let source = source_from_str(
            "invalid_enum_schema.trb",
            "enum Boxed(a) { Box(a), Empty }\nfn keep(value: Boxed(Int)) -> Boxed(Int) { value }\nfn main() -> Nil {}",
        );
        let typed = parse_and_lower_ast(db, source).unwrap();
        for (missing, expected) in [
            (true, "missing specialized enum variant schema"),
            (false, "specialized enum variant schema has wrong owner"),
        ] {
            let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                merge_and_lower_to_ir_with(db, &typed, source, |mut input, db, ir, uri| {
                    let variant = input
                        .ast
                        .decls
                        .iter()
                        .find_map(|d| match d {
                            Decl::Enum(e) if e.name == trunk_ir::Symbol::new("Boxed$Int") => {
                                Some(e.variants[0].id)
                            }
                            _ => None,
                        })
                        .unwrap();
                    if missing {
                        input.specialized_enum_variants.remove(&variant);
                    } else {
                        input
                            .specialized_enum_variants
                            .insert(variant, TypeScheme::mono(db, Type::new(db, TypeKind::Int)));
                    }
                    input.lower_to_ir(db, ir, uri)
                })
            }));
            let Err(error) = failure else {
                panic!("invalid specialized schema was accepted")
            };
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(message.contains(expected), "{message}");
        }
    }

    #[salsa_test]
    fn specialized_enum_schemas_and_dependencies_reach_logical_cps(db: &salsa::DatabaseImpl) {
        use tribute_front::ast::{Decl, TypeKind};
        use tribute_ir::dialect::adt::layout::{get_enum_variants, get_struct_fields};
        use trunk_ir::{Symbol, TypeRef};

        fn alias(ir: &IrContext, name: &str) -> TypeRef {
            let name = Symbol::new(name);
            let ty = ir
                .type_alias_by_name(&name)
                .expect("dependency layout is published");
            assert!(
                ir.get_type(ty)
                    .attrs
                    .get_str(ir, "name")
                    .is_some_and(|text| name == text)
            );
            ty
        }
        fn target(ir: &IrContext, ty: TypeRef) -> TypeRef {
            let data = ir.get_type(ty);
            assert_eq!(data.name, Symbol::new("typeref"));
            ir.type_alias_by_text(data.attrs.get_str(ir, "name").unwrap())
                .expect("referenced layout")
        }
        let source = source_from_str(
            "specialized_enum_dependencies.trb",
            r#"
struct Inner(a) { value: a }
enum Outer(a) { Wrap(Inner(a)), Pair(#(a, fn(a) -> a)), Empty, Both(a, Bool), Label(String) }
struct Envelope(a) { outer: Outer(a) }
struct Link(a) { tail: Loop(a) }
enum Loop(a) { Next(Link(a)), Done(a) }
pub mod A { pub enum Token(a) { Item(a), Empty } }
pub mod B { pub enum Token(a) { Item(a), Empty } }
enum Boxed(a) { Box(a), EmptyBox }
fn keep(value: Envelope(Int)) -> Envelope(Int) { value }
fn keep_bool(value: Outer(Bool)) -> Outer(Bool) { value }
fn keep_loop(value: Loop(Int)) -> Loop(Int) { value }
fn keep_a(value: A::Token(Int)) -> A::Token(Int) { value }
fn keep_b(value: B::Token(Bool)) -> B::Token(Bool) { value }
fn int_payload(value: Boxed(Int)) -> Int { case value { Box(x) -> x, EmptyBox -> +0 } }
fn bool_payload(value: Boxed(Bool)) -> Bool { case value { Box(x) -> x, EmptyBox -> False } }
fn main() -> Nil {}
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("typed fixture");
        let prepared =
            prepare_frontend_for_lowering(db, typed, source).expect("closed nominal dependencies");
        let schemas = &prepared.constructor_types(db).specialized_enum_variants;
        let mut seen = HashSet::default();
        for declaration in &prepared.module(db).decls {
            if let Decl::Enum(e) = declaration
                && e.id.variant().is_some()
            {
                for v in &e.variants {
                    assert!(seen.insert(v.id));
                    let matching: Vec<_> = schemas.iter().filter(|(id, _)| *id == v.id).collect();
                    assert_eq!(matching.len(), 1);
                    let scheme = matching[0].1;
                    assert!(scheme.type_params(db).is_empty());
                    let result = match scheme.body(db).kind(db) {
                        TypeKind::Func { params, result, .. } => {
                            assert_eq!(params.len(), v.fields.len());
                            *result
                        }
                        _ => {
                            assert!(v.fields.is_empty());
                            scheme.body(db)
                        }
                    };
                    assert!(matches!(result.kind(db), TypeKind::Named { id, args, .. }
                        if id.qualified(db) == e.name && args.is_empty()));
                }
            }
        }
        assert_eq!(seen.len(), schemas.len());
        for (id, source_scheme) in &typed.constructor_types(db).schemes {
            if id.qualified(db) == Symbol::new("Box") {
                let retained = prepared
                    .constructor_types(db)
                    .schemes
                    .iter()
                    .find(|(key, _)| key == id)
                    .unwrap()
                    .1;
                assert_eq!(
                    retained, *source_scheme,
                    "source generic schema is preserved"
                );
            }
        }
        let (mut ir, logical) =
            merge_and_lower_to_ir_with(db, &typed, source, |typed, db, ir, uri| {
                typed.lower_to_ir(db, ir, uri)
            });
        for after_cps in [false, true] {
            if after_cps {
                tribute_passes::tribute_control_to_cps::tribute_control_to_cps(
                    &mut ir,
                    logical.module,
                    &logical.operation_declarations,
                    &logical.compiler_intrinsics,
                    &mut Default::default(),
                )
                .unwrap();
            }
            for (suffix, primitive) in [("Int", "i32"), ("Bool", "i1")] {
                let inner = alias(&ir, &format!("Inner${suffix}"));
                let value = get_struct_fields(&ir, inner).unwrap()[0].1;
                assert_eq!(ir.get_type(value).name, Symbol::new(primitive));
                let outer = get_enum_variants(&ir, alias(&ir, &format!("Outer${suffix}"))).unwrap();
                assert_eq!(ir.str(outer[0].0), "Wrap");
                assert_eq!(target(&ir, outer[0].1[0]), inner);
                assert!(outer[2].1.is_empty());
                assert_eq!(outer[3].1.len(), 2);
                assert_eq!(outer[3].1[0], value);
                assert_eq!(ir.get_type(outer[3].1[1]).name, Symbol::new("i1"));
                assert_eq!(target(&ir, outer[4].1[0]), alias(&ir, "std::String"));
                let tuple = target(&ir, outer[1].1[0]);
                let fields = get_struct_fields(&ir, tuple).unwrap();
                assert_eq!(fields[0].1, value);
                let callback = ir.get_type(fields[1].1);
                assert_eq!(
                    callback.dialect,
                    Symbol::new(if after_cps {
                        "closure"
                    } else {
                        "tribute_control"
                    })
                );
                let boxed = get_enum_variants(&ir, alias(&ir, &format!("Boxed${suffix}"))).unwrap();
                assert_eq!(boxed[0].1, vec![value]);
            }
            let envelope = get_struct_fields(&ir, alias(&ir, "Envelope$Int")).unwrap();
            assert_eq!(target(&ir, envelope[0].1), alias(&ir, "Outer$Int"));
            let link = get_struct_fields(&ir, alias(&ir, "Link$Int")).unwrap();
            assert_eq!(target(&ir, link[0].1), alias(&ir, "Loop$Int"));
            let recursive = get_enum_variants(&ir, alias(&ir, "Loop$Int")).unwrap();
            assert_eq!(target(&ir, recursive[0].1[0]), alias(&ir, "Link$Int"));
            for (name, primitive) in [("A::Token$Int", "i32"), ("B::Token$Bool", "i1")] {
                let variants = get_enum_variants(&ir, alias(&ir, name)).unwrap();
                assert_eq!(ir.str(variants[0].0), "Item");
                assert_eq!(ir.get_type(variants[0].1[0]).name, Symbol::new(primitive));
            }
            let output = trunk_ir::printer::print_module(&ir, logical.module.op());
            for (name, primitive) in [("Boxed$Int", "core.i32"), ("Boxed$Bool", "core.i1")] {
                assert!(
                    output.lines().any(|line| line.contains("adt.variant_get")
                        && line.contains(name)
                        && line.ends_with(primitive)),
                    "{output}"
                );
            }
        }
    }

    #[salsa_test]
    fn logical_nominal_layouts_are_published_through_cps(db: &salsa::DatabaseImpl) {
        use tribute_ir::dialect::adt::layout::get_struct_fields;
        use tribute_ir::dialect::tribute_control;
        use trunk_ir::{Symbol, TypeRef};

        fn layout(ir: &IrContext, name: &Symbol, kind: &str) -> TypeRef {
            let ty = ir
                .type_alias_by_name(name)
                .expect("published nominal layout");
            let data = ir.get_type(ty);
            assert_eq!(data.dialect, Symbol::new("adt"));
            assert_eq!(data.name, Symbol::new(kind));
            assert!(
                data.attrs
                    .get_str(ir, "name")
                    .is_some_and(|text| name == text)
            );
            assert_eq!(
                ir.type_aliases()
                    .iter()
                    .filter(|(_, ty)| {
                        ir.get_type(*ty)
                            .attrs
                            .get_str(ir, "name")
                            .is_some_and(|text| name == text)
                    })
                    .count(),
                1,
                "one published layout for {name}"
            );
            ty
        }

        fn reference_name(ir: &IrContext, ty: TypeRef) -> Symbol {
            let data = ir.get_type(ty);
            assert_eq!(data.dialect, Symbol::new("adt"));
            assert_eq!(data.name, Symbol::new("typeref"));
            Symbol::new(data.attrs.get_str(ir, "name").expect("nominal identity"))
        }

        // These specializations occur only in signatures, never in allocations.
        let source = source_from_str(
            "published_nominal_layouts.trb",
            r#"
pub mod A { pub struct Token(a) {} }
pub mod B { pub struct Token(a) {} }
enum Choice(a) { Item(a), Empty }
struct Holder { pair: #(Nat, fn(Nat) -> Nat) }
struct Node { next: Node }
struct First { second: Second }
struct Second { first: First }

fn keep_a(value: A::Token(Nat)) -> A::Token(Nat) { value }
fn keep_b(value: B::Token(Nat)) -> B::Token(Nat) { value }
fn keep_choice(value: Choice(Nat)) -> Choice(Nat) { value }
fn keep_holder(value: Holder) -> Holder { value }
fn keep_node(value: Node) -> Node { value }
fn keep_first(value: First) -> First { value }
fn main() -> Nil {}
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        let (mut ir, logical) =
            merge_and_lower_to_ir_with(db, &typed, source, |typed, db, ir, uri| {
                typed.lower_to_ir(db, ir, uri)
            });
        let mut source_tuple = None;
        for after_cps in [false, true] {
            if after_cps {
                tribute_passes::tribute_control_to_cps::tribute_control_to_cps(
                    &mut ir,
                    logical.module,
                    &logical.operation_declarations,
                    &logical.compiler_intrinsics,
                    &mut Default::default(),
                )
                .expect("CPS converts published layout fields");
            }
            for (function, nominal, kind) in [
                ("keep_a", "A::Token$Nat", "struct"),
                ("keep_b", "B::Token$Nat", "struct"),
                ("keep_choice", "Choice$Nat", "enum"),
            ] {
                let parameter = logical
                    .module
                    .ops(&ir)
                    .iter()
                    .copied()
                    .find_map(|op| {
                        if after_cps {
                            let f = func_dialect::Func::from_op(&ir, op).ok()?;
                            (f.sym_name(&ir) == function).then(|| {
                                func_dialect::FuncSig::from_type_ref(&ir, f.r#type(&ir))
                                    .unwrap()
                                    .inputs(&ir)[0]
                            })
                        } else {
                            let f = tribute_control::Func::from_op(&ir, op).ok()?;
                            (f.sym_name(&ir) == function).then(|| {
                                tribute_control::FuncSig::from_type_ref(&ir, f.r#type(&ir))
                                    .unwrap()
                                    .inputs(&ir)[0]
                            })
                        }
                    })
                    .expect("signature-only specialization");
                let name = reference_name(&ir, parameter);
                assert_eq!(name, Symbol::new(nominal));
                layout(&ir, &name, kind);
            }
            assert_ne!(
                layout(&ir, &Symbol::new("A::Token$Nat"), "struct"),
                layout(&ir, &Symbol::new("B::Token$Nat"), "struct")
            );
            for (owner, target) in [("Node", "Node"), ("First", "Second"), ("Second", "First")] {
                let owner = layout(&ir, &Symbol::new(owner), "struct");
                let field = get_struct_fields(&ir, owner).unwrap()[0].1;
                let name = reference_name(&ir, field);
                assert_eq!(name, Symbol::new(target));
                layout(&ir, &name, "struct");
            }
            let holder = layout(&ir, &Symbol::new("Holder"), "struct");
            let pair = get_struct_fields(&ir, holder).unwrap()[0].1;
            let tuple = layout(&ir, &reference_name(&ir, pair), "struct");
            let fields = get_struct_fields(&ir, tuple).unwrap();
            let callback = ir.get_type(fields[1].1);
            if after_cps {
                assert_ne!(Some(tuple), source_tuple);
                assert_eq!(callback.dialect, Symbol::new("closure"));
                assert_eq!(callback.name, Symbol::new("closure"));
            } else {
                source_tuple = Some(tuple);
                assert_eq!(callback.dialect, Symbol::new("tribute_control"));
                assert_eq!(callback.name, Symbol::new("func_sig"));
            }
        }
    }

    #[salsa_test]
    fn source_list_does_not_change_logical_prelude_signatures(db: &salsa::DatabaseImpl) {
        use tribute_ir::dialect::tribute_control;
        use trunk_ir::{Symbol, ops::DialectOp};

        let source = source_from_str(
            "unused_source_list.trb",
            "enum List(a) { SourceList(a), }\nfn main() -> Nil {}",
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let (ir, logical) = merge_and_lower_to_ir_with(db, &typed, source, |typed, db, ir, uri| {
            typed.lower_to_ir(db, ir, uri)
        });
        let mut checked = 0;
        for &operation in logical.module.ops(&ir) {
            let Ok(function) = tribute_control::Func::from_op(&ir, operation) else {
                continue;
            };
            if function.sym_name(&ir) == "std::collections::List::prepend" {
                let signature = tribute_control::FuncSig::from_type_ref(&ir, function.r#type(&ir))
                    .expect("logical signature");
                let result = ir.get_type(signature.result(&ir));
                assert_eq!(result.dialect, Symbol::new("tribute_rt"));
                assert_eq!(result.name, Symbol::new("anyref"));
                assert_eq!(signature.inputs(&ir)[1], signature.result(&ir));
                checked += 1;
            }
        }
        assert_eq!(checked, 1);
        let validation = tribute_control::validate(
            &ir,
            logical.module,
            &logical.operation_declarations,
            &logical.compiler_intrinsics,
            &mut Default::default(),
        );
        assert!(validation.is_ok(), "{validation}");
    }

    #[test]
    fn recursive_generic_handler_collects_a_concrete_specialization() {
        salsa::Database::attach(&salsa::DatabaseImpl::default(), |db| {
            let source = source_from_str(
                "recursive_generic_handler.trb",
                r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(value) { run_state(fn() { resume Nil }, value) }
    }
}

fn consume(value: Nat) -> Nil { Nil }

fn main() -> Nil {
    consume(run_state(fn() { State::get() }, 0))
}
"#,
            );
            let typed = parse_and_lower_ast(db, source).expect("frontend output");
            let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
            assert!(
                diagnostics.is_empty(),
                "recursive handler fixture must type-check without diagnostics: {diagnostics:#?}"
            );
            let (_, monomorphized) =
                merge_and_lower_to_ir_with(db, &typed, source, |typed, _, _, _| typed);
            let specialized = trunk_ir::Symbol::new("run_state$Nat$Nat");
            let scheme = monomorphized
                .function_types
                .get(&specialized)
                .expect("recursive handler call must collect a concrete run_state specialization");

            assert!(
                scheme.type_params(db).is_empty(),
                "specialized run_state must not retain generic binders: {scheme:?}"
            );
            let specialized_body = monomorphized
                .ast
                .decls
                .iter()
                .find_map(|decl| match decl {
                    tribute_front::ast::Decl::Function(function)
                        if function.name == specialized =>
                    {
                        Some(function)
                    }
                    _ => None,
                })
                .expect("monomorphization must materialize the concrete run_state body");
            assert!(
                !format!("{specialized_body:#?}").contains("BoundVar"),
                "specialized handler body must not retain BoundVar metadata"
            );
        });
    }

    #[salsa_test]
    fn wasm_managed_c_ffi_omits_unreferenced_declaration(db: &salsa::DatabaseImpl) {
        let source = SourceCst::from_source_str(
            db,
            "unused_c_ffi.trb",
            r#"extern "C" fn user_bridge(value: String) -> String
fn main() -> Nil { Nil }"#,
        );
        let binary = compile_to_wasm_binary(db, source)
            .expect("unused managed C declaration may be omitted");
        wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
            .validate_all(binary)
            .expect("unused C declaration must not leave an invalid Wasm definition");
    }

    #[salsa_test]
    fn wasm_managed_c_ffi_requires_a_binding_when_referenced(db: &salsa::DatabaseImpl) {
        let source = SourceCst::from_source_str(
            db,
            "used_c_ffi.trb",
            r#"extern "C" fn user_bridge(value: String) -> String
fn main() -> Nil {
    let _ = user_bridge("hello")
    Nil
}"#,
        );
        let errors = compile_to_wasm_binary(db, source)
            .expect_err("C ABI alone must not create a Wasm import");
        // Reaching the boundary exit proves shared lowering accepted the
        // managed FFI signature, without running the shared pipeline again
        // just to check it.
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0].phase, CompilationPhase::Lowering);
        let message = &errors[0].inner.message;
        assert!(
            message.contains("unsatisfiable runtime binding user_bridge"),
            "{message}"
        );
    }

    #[salsa_test]
    fn unknown_intrinsic_directives_are_rejected_before_specialization(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "unknown_intrinsics.trb",
            r#"extern "intrinsic" fn user_intrinsic(value: Int) -> Int
mod Nested {
    extern "intrinsic" fn unused_generic(value: a) -> a
}"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
        assert!(prepare_frontend_for_lowering(db, typed, source).is_none());
        let diagnostics =
            prepare_frontend_for_lowering::accumulated::<Diagnostic>(db, typed, source);
        assert_eq!(diagnostics.len(), 2, "{diagnostics:?}");
        for (diagnostic, identity) in diagnostics
            .iter()
            .zip(["user_intrinsic", "Nested::unused_generic"])
        {
            assert_eq!(
                diagnostic.inner.message,
                format!("unsupported compiler intrinsic directive `{identity}`")
            );
            assert_eq!(diagnostic.inner.severity, DiagnosticSeverity::Error);
            assert_eq!(diagnostic.phase, CompilationPhase::Lowering);
            assert_ne!(diagnostic.inner.span, trunk_ir::Span::default());
        }
    }

    /// An intrinsic identity is the declaration's package path, so a user
    /// declaration cannot claim one the `std` package owns.
    #[salsa_test]
    fn user_declaration_cannot_claim_a_library_intrinsic(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "library_intrinsic.trb",
            r#"extern "intrinsic" fn __bytes_get_or_panic(bytes: Bytes, index: Nat) -> Nat"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
        assert!(prepare_frontend_for_lowering(db, typed, source).is_none());
        let diagnostics =
            prepare_frontend_for_lowering::accumulated::<Diagnostic>(db, typed, source);
        assert_eq!(
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.inner.message.as_str())
                .collect::<Vec<_>>(),
            ["unsupported compiler intrinsic directive `__bytes_get_or_panic`"]
        );
    }

    #[salsa_test]
    fn arithmetic_intrinsic_lowering_consumes_its_identities(db: &salsa::DatabaseImpl) {
        use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
        use trunk_ir::dialect::func;

        let source = source_from_str(
            "native_calculator.trb",
            include_str!("../lang-examples/native_calculator.trb"),
        );
        let (ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .expect("fixture must lower");
        let identity = |name: &'static str| {
            module.ops(&ctx).iter().copied().find_map(|op| {
                let function = func::Func::from_op(&ctx, op).ok()?;
                (function.sym_name(&ctx) == name).then(|| {
                    assert_eq!(
                        ctx.op(op).attributes.get_str(&ctx, "abi"),
                        Some("intrinsic")
                    );
                    ctx.op(op)
                        .attributes
                        .get_str(&ctx, COMPILER_INTRINSIC_ATTR)
                        .map(str::to_owned)
                })
            })
        };
        assert_eq!(
            identity("std::Int::+"),
            None,
            "arithmetic lowering is the last reader of its identities and removes \
             declarations nothing references"
        );
        assert_eq!(
            identity("std::__bytes_get_or_panic"),
            Some(Some("std::__bytes_get_or_panic".to_owned())),
            "bytes lowering consumes its identity inside the target boundary"
        );
    }

    #[salsa_test]
    fn both_targets_consume_the_bytes_intrinsic_inside_the_boundary(db: &salsa::DatabaseImpl) {
        use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
        use tribute_passes::abi_boundary::TargetKind;
        use trunk_ir::dialect::func;

        for target in [TargetKind::Native, TargetKind::Wasm] {
            let source = source_from_str(
                "native_calculator.trb",
                include_str!("../lang-examples/native_calculator.trb"),
            );
            let (mut ctx, module) = run_shared_pipeline(db, source)
                .expect("shared pipeline must succeed")
                .expect("fixture must lower");
            match target {
                TargetKind::Native => run_native_target_pipeline(&mut ctx, module),
                TargetKind::Wasm => run_wasm_target_pipeline(&mut ctx, module),
            }
            .unwrap_or_else(|error| panic!("{target:?} boundary failed: {error}"));

            for &op in module.ops(&ctx) {
                assert!(
                    ctx.op(op).attributes.get(COMPILER_INTRINSIC_ATTR).is_none(),
                    "{target:?}: the identity is consumed at the boundary exit"
                );
                let declares_intrinsic = func::Func::from_op(&ctx, op)
                    .is_ok_and(|function| function.sym_name(&ctx) == "__bytes_get_or_panic");
                assert!(
                    !declares_intrinsic,
                    "{target:?}: the intrinsic declaration is removed"
                );
            }
        }
    }

    /// Literal patterns compare through the prelude's `==` for their type,
    /// resolved by receiver type, even beside a user `String` lookalike with
    /// its own `==`.
    #[salsa_test]
    fn literal_pattern_equalities_are_the_prelude_methods(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "lookalike_equality.trb",
            r#"
mod shadow {
    pub enum String { Leaf(Bytes) }

    pub mod String {
        pub fn (==)(left: shadow::String, right: shadow::String) -> Bool { True }
    }
}

fn main() -> Nil { }
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let well_known = typed.well_known_types(db);
        let qualified = |id: Option<tribute_front::ast::FuncDefId<'_>>| {
            id.map(|id| id.qualified(db).to_string())
        };
        assert_eq!(
            qualified(well_known.string_equality).as_deref(),
            Some("std::String::==")
        );
        assert_eq!(
            qualified(well_known.bytes_equality).as_deref(),
            Some("std::Bytes::==")
        );
    }

    /// Every type reachable from `ty`, through type parameters and
    /// type-bearing attributes, satisfies `ok`.
    fn type_tree_all(
        ctx: &IrContext,
        ty: trunk_ir::TypeRef,
        ok: &impl Fn(&IrContext, trunk_ir::TypeRef) -> bool,
    ) -> bool {
        let mut pending = vec![ty];
        let mut seen = HashSet::default();
        while let Some(ty) = pending.pop() {
            if !seen.insert(ty) {
                continue;
            }
            if !ok(ctx, ty) {
                return false;
            }
            let data = ctx.get_type(ty);
            pending.extend(data.params.iter().copied());
            for value in data.attrs.values() {
                value.visit_types(&mut |nested| pending.push(nested));
            }
        }
        true
    }

    /// The places where a type failing `ok` survives Wasm lowering of `text`:
    /// type aliases and operations, including nested type parameters,
    /// attributes, results, and block arguments.
    fn types_surviving_wasm_lowering(
        db: &salsa::DatabaseImpl,
        path: &str,
        text: &str,
        ok: &impl Fn(&IrContext, trunk_ir::TypeRef) -> bool,
    ) -> std::collections::BTreeSet<String> {
        let source = source_from_str(path, text);
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .expect("fixture must lower");
        run_wasm_target_pipeline(&mut ctx, module).expect("Wasm boundary");
        tribute_passes::wasm::lower::lower_to_wasm(&mut ctx, module, &mut AnalysisCache::new())
            .expect("Wasm lowering");

        let mut sites = std::collections::BTreeSet::new();
        for (name, ty) in ctx.type_aliases().iter().cloned() {
            if !type_tree_all(&ctx, ty, ok) {
                sites.insert(format!("alias !{name}"));
            }
        }
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            let site = format!("{}.{}", data.dialect, data.name);
            let mut types: Vec<_> = ctx.op_result_types(op).to_vec();
            for value in data.attributes.values() {
                value.visit_types(&mut |ty| types.push(ty));
            }
            for region in ctx.op_regions(op) {
                for &block in &ctx.region(region).blocks {
                    for argument in ctx.block(block).args.iter() {
                        types.push(argument.ty);
                        for value in argument.attrs.values() {
                            value.visit_types(&mut |ty| types.push(ty));
                        }
                    }
                }
            }
            if types.into_iter().any(|ty| !type_tree_all(&ctx, ty, ok)) {
                sites.insert(site);
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        sites
    }

    #[salsa_test]
    fn wasm_lowering_leaves_no_core_bytes(db: &salsa::DatabaseImpl) {
        let not_core_bytes = |ctx: &IrContext, ty| {
            let data = ctx.get_type(ty);
            !(data.dialect == trunk_ir::Symbol::new("core")
                && data.name == trunk_ir::Symbol::new("bytes"))
        };
        for (path, text) in [
            (
                "wasm_dynamic_output.trb",
                include_str!("../lang-examples/wasm_dynamic_output.trb"),
            ),
            (
                "case_bytes.trb",
                r#"fn pick(x: Bytes, y: Bytes, f: Bool) -> Bytes {
    case f {
        True -> x
        False -> y
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(String::from_bytes(pick(b"ab", b"cd", True)))
}
"#,
            ),
        ] {
            let sites = types_surviving_wasm_lowering(db, path, text, &not_core_bytes);
            assert!(
                sites.is_empty(),
                "{path}: core.bytes survives Wasm lowering at {sites:#?}"
            );
        }
    }

    #[salsa_test]
    fn wasm_lowering_leaves_no_tribute_rt_types(db: &salsa::DatabaseImpl) {
        let not_tribute_rt =
            |ctx: &IrContext, ty| ctx.get_type(ty).dialect != trunk_ir::Symbol::new("tribute_rt");
        for (path, text) in [
            (
                "native_effects.trb",
                include_str!("../lang-examples/native_effects.trb"),
            ),
            ("lambda.trb", include_str!("../lang-examples/lambda.trb")),
            ("float.trb", include_str!("../lang-examples/float.trb")),
            (
                "record-patterns.trb",
                include_str!("../lang-examples/record-patterns.trb"),
            ),
        ] {
            let sites = types_surviving_wasm_lowering(db, path, text, &not_tribute_rt);
            assert!(
                sites.is_empty(),
                "{path}: a tribute_rt type survives Wasm lowering at {sites:#?}"
            );
        }
    }

    #[salsa_test]
    fn wasm_lowering_leaves_adt_types_only_as_builtin_layouts(db: &salsa::DatabaseImpl) {
        // The backend identifies a builtin layout by its `layout` attribute
        // and knows no other `adt` type.
        let builtin_layout_or_not_adt = |ctx: &IrContext, ty| {
            let data = ctx.get_type(ty);
            data.dialect != trunk_ir::Symbol::new("adt")
                || data.attrs.get(trunk_ir::types::LAYOUT_ATTR).is_some()
        };
        for (path, text) in [
            (
                "native_effects.trb",
                include_str!("../lang-examples/native_effects.trb"),
            ),
            ("lambda.trb", include_str!("../lang-examples/lambda.trb")),
            ("float.trb", include_str!("../lang-examples/float.trb")),
            ("tuples.trb", include_str!("../lang-examples/tuples.trb")),
            (
                "record-patterns.trb",
                include_str!("../lang-examples/record-patterns.trb"),
            ),
        ] {
            let mut sites =
                types_surviving_wasm_lowering(db, path, text, &builtin_layout_or_not_adt);
            // Source layouts stay declared as aliases and module metadata,
            // which no operation or value refers to.
            sites.retain(|site| !site.starts_with("alias !") && site != "core.module");
            assert!(
                sites.is_empty(),
                "{path}: a source adt type survives Wasm lowering at {sites:#?}"
            );
        }
    }

    #[salsa_test]
    fn wasm_lowering_carries_only_preserved_language_metadata(db: &salsa::DatabaseImpl) {
        use tribute_passes::abi_boundary::is_preserved_attribute;

        let source = source_from_str(
            "wasm_dynamic_output.trb",
            include_str!("../lang-examples/wasm_dynamic_output.trb"),
        );
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .expect("fixture must lower");
        run_wasm_target_pipeline(&mut ctx, module).expect("Wasm boundary");
        tribute_passes::wasm::lower::lower_to_wasm(&mut ctx, module, &mut AnalysisCache::new())
            .expect("Wasm lowering");

        // Wasm lowering copies operation attributes without interpreting
        // them, so what it carries is exactly what the boundary let through.
        let mut unexpected = std::collections::BTreeSet::new();
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            for name in ctx.op(op).attributes.keys() {
                let name = name.to_string();
                if name.starts_with("tribute.") && !is_preserved_attribute(&name) {
                    unexpected.insert(format!("{}.{} {name}", ctx.op(op).dialect, ctx.op(op).name));
                }
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        assert!(unexpected.is_empty(), "{unexpected:#?}");
    }

    #[salsa_test]
    fn well_known_string_metadata_uses_prelude_declaration_identity(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "lookalike.trb",
            r#"
enum String {
    Leaf(Bytes),
    Branch(String, String, Nat),
}

fn main() -> String { "hello" }
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let canonical = typed
            .well_known_types(db)
            .string
            .expect("prelude String identity");
        let FrontendCompilation {
            context: ctx,
            module,
            ..
        } = merge_and_lower_to_ir(db, &typed, source);
        let string_ty = tribute_ir::metadata::WellKnownTypes::from_module(&ctx, module.op())
            .string
            .expect("String IR metadata");
        let string_data = ctx.get_type(string_ty);

        let Some(trunk_ir::Attribute::Location(definition)) =
            string_data.attrs.get("tribute.definition")
        else {
            panic!("String IR type records its definition location");
        };
        assert_eq!(ctx.paths().get(definition.path), PRELUDE_URI);
        assert_eq!(
            (definition.span.start, definition.span.end),
            (canonical.definition.start, canonical.definition.end)
        );

        let user_lookalike = ctx.types().iter().find_map(|(ty, data)| {
            (ty != string_ty
                && data.dialect == "adt"
                && data.name == "enum"
                && data.attrs.get_str(&ctx, "name") == Some("String"))
            .then_some(ty)
        });
        assert!(
            user_lookalike.is_some(),
            "compatible user String must remain distinct from prelude String"
        );
    }

    #[salsa_test]
    fn test_ast_pipeline_let_binding(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn main() -> Nil {
                let x = 10
                let y = 20
                let _ = x + y
            }
            "#,
        );

        let result = compile_frontend(db, source);
        assert!(
            result.is_some(),
            "{:?}",
            parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
        );
        let (ctx, m) = result.unwrap();
        assert_eq!(m.name(&ctx), Some(trunk_ir::Symbol::new("test")));
    }

    #[salsa_test]
    fn test_ast_pipeline_struct(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            struct Point {
                x: Int,
                y: Int,
            }

            fn main() -> Nil { }
            "#,
        );

        let result = compile_frontend(db, source);
        assert!(result.is_some());
        let (ctx, m) = result.unwrap();
        assert_eq!(m.name(&ctx), Some(trunk_ir::Symbol::new("test")));
    }

    // =========================================================================
    // main function return type validation
    // =========================================================================

    #[salsa_test]
    fn test_main_returns_nil_ok(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn main() -> Nil { }");

        let result = compile_with_diagnostics(db, source);
        let has_main_error = result
            .diagnostics
            .iter()
            .any(|d| d.inner.message.contains("must return Nil"));
        assert!(
            !has_main_error,
            "main() returning Nil should not produce an error, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_main_returns_int_error(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn main() -> Int { 42 }");

        let result = compile_with_diagnostics(db, source);
        let has_main_error = result.diagnostics.iter().any(|d| {
            d.inner.message.contains("must return Nil")
                && d.inner.severity == DiagnosticSeverity::Error
                && d.phase == CompilationPhase::TypeChecking
        });
        assert!(
            has_main_error,
            "main() returning Int should produce 'must return Nil' error, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_non_main_returns_int_ok(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn foo() -> Int { 42 }");

        let result = compile_with_diagnostics(db, source);
        let has_main_error = result
            .diagnostics
            .iter()
            .any(|d| d.inner.message.contains("must return Nil"));
        assert!(
            !has_main_error,
            "non-main function returning Int should not produce main-specific error, got: {:?}",
            result.diagnostics
        );
    }
}
