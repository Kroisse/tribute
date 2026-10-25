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
//! Typed Module (UFCS resolved)
//!     │
//!     ├─── Shared Middle-End (single arena session) ──┤
//!     ▼ ast_to_ir
//! Module (source-logical callable/control IR)
//!     │
//!     ▼ eliminate_unreferenced_source_functions (artifact compiles only:
//!     │   unreachable source-logical functions skip CPS legalization;
//!     │   runtime functions stay where a reachable perform, handle, or
//!     │   evidence plan can make target lowering call them)
//!     ▼ dce (unused pure operations, such as dead function references)
//!     ▼ specialize_convention_classes, then
//!     │   eliminate_unreferenced_source_functions again (artifact compiles only)
//!     ▼ check_fn_arm_calls (calls in `fn` arms that reach a `Cps` callee)
//!     ▼ tribute_control_to_cps → lower_continuation_frames →
//!       lower_closure_lambda → intrinsic/list/io lowering
//! Module (CPS callable contracts, explicit evidence, and general
//!         operations lowered to effect.dispatch_cps)
//!     │
//!     ▼ lower_ability_call
//! Module (ability.call lowered to effect.dispatch_tail)
//!     │
//!     ▼ resolve_evidence
//! Module (handler evidence setup lowered to effect.extend)
//!     │
//!     ▼ lower_handle_dispatch
//! Module (ability.handle_dispatch lowered)
//!     │
//!     ├─── Representation/ABI Boundary (run_target_to_boundary_exit) ──┤
//!     ▼ global DCE (unreachable functions skip target lowering; runtime
//!     │             items stay until target evidence lowering calls them)
//!     ▼ inline_functions → canonicalize, DCE
//!     ▼ target ABI validation → CPS signature physicalization
//!     ▼ root entry bridge
//!     ▼ lower-prepared-closures
//!     ▼ evidence lowering (shared by the targets, each with its type converter)
//!     ▼ bytes intrinsic bridge
//!     ▼ closure storage layout finalization
//!     ▼ cleanup (global DCE removes uncalled runtime items, canonicalize, DCE)
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
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Span;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::conversion::{ReconcileUnrealizedCasts, UnrealizedCastConversionPattern};
use trunk_ir::dialect::{core as core_dialect, func as func_dialect};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{PassError, PassManager, PassResult};
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

impl Default for OptimizationOptions {
    fn default() -> Self {
        Self::production()
    }
}

pub use tribute_passes::native::optimization::{
    BorrowedParameterPolicy, FieldMovePolicy, LastUseMovePolicy, NativeOptimizationOptions,
    NativePipelineStage, PairedRcEliminationPolicy, TemporaryBorrowPolicy,
};

// AST-based pipeline imports
use tribute_front::ast_to_ir;
use tribute_front::monomorphize::{SpecializationInput, SpecializationOutput};
use tribute_front::query as ast_query;
use tribute_front::resolve as ast_resolve;
use tribute_front::typeck as ast_typeck;
use trunk_ir_cranelift_backend::{
    CompilationResult as NativeCompilationResult, emit_module_to_native,
};
use trunk_ir_wasm_backend::{
    CompilationError, CompilationResult as WasmCompilationResult, WasmBinary,
};

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
    _source: SourceCst,
) -> Option<PreparedFrontend<'db>> {
    // Type checking reads package std's declarations without copying them.
    // Specialization takes them as one table with the user's, and std's
    // bodies and per-expression metadata ahead of the user's.
    let prelude = crate::prelude::prelude_module(db);
    let mut input = SpecializationInput::new(db, typed, prelude);
    if let Some(prelude) = prelude {
        input = input.merge_bodies(db, prelude);
    }
    let span_map = input.span_map().clone();

    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(input.module()) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                Diagnostic::new(
                    format!(
                        "unsupported compiler intrinsic directive `{}`",
                        directive.identity
                    ),
                    span_map.get_or_default(directive.node),
                    DiagnosticSeverity::Error,
                    CompilationPhase::Lowering,
                )
                .accumulate(db);
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
                Diagnostic::new(
                    format!(
                        "invalid specialization instance at {}: {:?}",
                        error.node, error.kind
                    ),
                    span_map.get_or_default(error.node),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(db);
            }
            return None;
        }
    };
    Some(PreparedFrontend {
        specialized,
        runtime_types,
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
    let typed = prepared.specialized;
    let compiler_intrinsics = typed.compiler_intrinsics(db).iter().cloned().collect();
    let runtime_types = prepared.runtime_types.clone();
    let mut ir = IrContext::new();
    let module = lower(
        ast_to_ir::TypedModule {
            ast: typed.module(db).clone(),
            local_instances: typed.expression_types(db).local_instances.clone(),
            function_instances: typed.expression_types(db).function_instances.clone(),
            span_map: typed.span_map(db).clone(),
            function_types: typed
                .declarations(db)
                .function_types
                .iter()
                .cloned()
                .collect(),
            constructor_types: typed
                .declarations(db)
                .constructor_types
                .schemes
                .iter()
                .cloned()
                .collect(),
            specialized_enum_variants: typed
                .declarations(db)
                .constructor_types
                .specialized_enum_variants
                .clone(),
            node_types: typed.expression_types(db).node_types.clone(),
            ability_definitions: ast_typeck::ability_definitions_from_schemas(
                &typed.declarations(db).ability_definitions,
            ),
            handler_operations: typed.expression_types(db).handler_operations.clone(),
            perform_operations: typed.expression_types(db).perform_operations.clone(),
            lambda_signatures: typed.expression_types(db).lambda_signatures.clone(),
            exhaustive_cases: typed
                .expression_types(db)
                .exhaustive_cases
                .iter()
                .copied()
                .collect(),
            evidence_plans: typed.expression_types(db).evidence_plans.clone(),
            well_known_types: typed.declarations(db).well_known_types,
            compiler_intrinsics,
            runtime_types,
            merged_sources: vec![crate::prelude::PRELUDE_URI.to_owned()],
        },
        db,
        &mut ir,
        source.uri(db).as_str(),
    );
    (ir, module)
}

/// Arena IR together with the registered compiler intrinsics the shared CPS
/// conversion checks. They stay private; [`run_shared_middle_end`] is their
/// only consumer.
#[derive(Clone)]
pub struct FrontendCompilation {
    context: IrContext,
    module: Module,
    compiler_intrinsics: Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
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

/// Run the frontend and class specialization, and remove the definitions
/// nothing reachable references: the source-logical IR CPS legalization
/// receives.
///
/// Returns `None` if parsing fails or the frontend reports an error.
pub fn compile_specialized_frontend(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    use tribute_passes::unreferenced_source_functions::EliminateUnreferencedSourceFunctions;

    let Some(FrontendCompilation {
        context, module: m, ..
    }) = compile_frontend_for_shared_route(db, source)
    else {
        return Ok(None);
    };
    let mut ctx = context;
    let core_module =
        core_dialect::Module::from_op(&ctx, m.op()).expect("frontend output must be a core.module");
    let mut pm = PassManager::new();
    pm.add_pass(EliminateUnreferencedSourceFunctions::new())
        .add_pass(trunk_ir::transforms::EliminateDeadCode::<
            core_dialect::Module,
        >::new())
        .add_pass(tribute_passes::specialize_convention_classes::SpecializeConventionClasses)
        .add_pass(EliminateUnreferencedSourceFunctions::new());
    pm.run(&mut ctx, core_module, &mut AnalysisCache::new())?;
    Ok(Some((ctx, m)))
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
        let core_module = core_dialect::Module::from_op(ctx, module.op())
            .expect("wasm cast reconciliation requires a core.module");
        let mut reconcile = PassManager::new();
        reconcile.add_pass(ReconcileUnrealizedCasts);
        reconcile
            .run(ctx, core_module, &mut analyses)
            .map_err(tribute_passes::wasm::lower::WasmLowerError::from)
            .map_err(wasm_lowering_failure)?;
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
/// Evidence params are introduced by the physical CPS conversion.
pub fn run_through_cps_lowering(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    let Some(FrontendCompilation {
        context,
        module: m,
        compiler_intrinsics,
    }) = compile_frontend_for_shared_route(db, source)
    else {
        return Ok(None);
    };
    let mut ctx = context;
    let core_module =
        core_dialect::Module::from_op(&ctx, m.op()).expect("frontend output must be a core.module");
    let mut pm = structural_pass_pipeline(compiler_intrinsics, SourceFunctions::All);
    pm.run(&mut ctx, core_module, &mut Default::default())?;
    Ok(Some((ctx, m)))
}

// =============================================================================
// Full Pipeline (Orchestration)
// =============================================================================

/// Build the shared structural pass pipeline that legalizes source-logical
/// callable/control IR into CPS with explicit evidence. For
/// [`SourceFunctions::Reachable`], it first removes the functions nothing
/// reachable references, and again after class specialization.
fn structural_pass_pipeline(
    compiler_intrinsics: Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
    functions: SourceFunctions,
) -> PassManager {
    use tribute_passes::unreferenced_source_functions::EliminateUnreferencedSourceFunctions;

    let mut pm = PassManager::new();
    if functions == SourceFunctions::Reachable {
        pm.add_pass(EliminateUnreferencedSourceFunctions::new());
    }
    pm.add_pass(trunk_ir::transforms::EliminateDeadCode::<
        core_dialect::Module,
    >::new())
        .add_pass(tribute_passes::specialize_convention_classes::SpecializeConventionClasses);
    if functions == SourceFunctions::Reachable {
        pm.add_pass(EliminateUnreferencedSourceFunctions::new());
    }
    pm.add_pass(tribute_passes::check_fn_arm_calls::CheckFnArmCalls);
    if cfg!(debug_assertions) {
        pm.add_pass(tribute_passes::check_convention_classes::CheckConventionClasses);
    }
    pm.add_pass(
        tribute_passes::tribute_control_to_cps::TributeControlToCps::new()
            .with_compiler_intrinsics(compiler_intrinsics),
    )
    .add_pass(tribute_passes::lower_continuation_frames::LowerContinuationFrames)
    .add_pass(tribute_passes::lower_closure_lambda::LowerClosureLambda)
    .add_pass(tribute_passes::intrinsic_to_arith::LowerIntrinsicToArith)
    .add_pass(tribute_passes::list_intrinsics::LowerListIntrinsics)
    .add_pass(tribute_passes::slots_intrinsics::LowerSlotsIntrinsics)
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
        compiler_intrinsics,
    } = frontend;
    let mut ctx = context;

    // Middle-end passes, sequenced through the PassManager (#268).
    // Registration order == execution order.
    let core_module =
        core_dialect::Module::from_op(&ctx, m.op()).expect("frontend output must be a core.module");
    // One cache for the phase: every pass and boundary check reuses analyses
    // of unchanged IR, and any IR change discards them.
    let mut analyses = AnalysisCache::new();
    let mut structural_pm = structural_pass_pipeline(compiler_intrinsics, functions);
    structural_pm.run(&mut ctx, core_module, &mut analyses)?;

    // CPS effect handling, function-local phase: lower_ability_call produces
    // the tail-resumptive effect dispatches; evidence resolution then extends
    // handler scopes.
    let mut ability_pm = PassManager::new();
    ability_pm
        .nest::<func_dialect::Func>()
        .add_pass(tribute_passes::lower_ability_call::LowerAbilityCall);
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
    // A `fn` arm call that may suspend is a source error at that call.
    if let trunk_ir::pass::PassErrorKind::Execution(source) = error.kind()
        && let Some(error) =
            source.downcast_ref::<tribute_passes::check_fn_arm_calls::FnArmCallError>()
    {
        for call in &error.calls {
            call.origins
                .iter()
                .fold(
                    Diagnostic::builder(
                        call.message.clone(),
                        call.location.span,
                        DiagnosticSeverity::Error,
                        CompilationPhase::Lowering,
                    ),
                    |diagnostic, origin| {
                        diagnostic.label(
                            origin.span,
                            "this use passes effects that include an `op` operation",
                        )
                    },
                )
                .build()
                .accumulate(db);
        }
        return;
    }
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

/// Forward known struct reads module-wide, then run canonicalization and DCE
/// on every function.
///
/// `stage` names the caller in diagnostics. A failed pass is returned so the
/// caller stops lowering.
fn run_canonicalize_passes(
    ctx: &mut IrContext,
    m: Module,
    analyses: &mut AnalysisCache,
    stage: &str,
) -> Result<(), PassError> {
    let Ok(core_module) = core_dialect::Module::from_op(ctx, m.op()) else {
        tracing::warn!("{stage} skipped function passes: root op is not core.module");
        return Ok(());
    };
    tribute_ir::dialect::adt::forward_struct_gets(ctx, m);
    let mut pm = PassManager::new();
    pm.nest::<func_dialect::Func>()
        .add_pass(trunk_ir::transforms::canonicalize_pass())
        .add_pass(trunk_ir::transforms::EliminateDeadCode::new());
    pm.with_debug_verifier();
    pm.run(ctx, core_module, analyses)
}

/// Run global DCE + canonicalization (shared cleanup after all lowering).
///
/// The passes that created conversion casts already emitted their boxing, so
/// the casts left here only retype values for the target type conversion.
fn run_cleanup_passes(
    ctx: &mut IrContext,
    m: Module,
    analyses: &mut AnalysisCache,
) -> Result<(), PassError> {
    trunk_ir::transforms::global_dce::eliminate_dead_functions(ctx, m, analyses);
    run_canonicalize_passes(ctx, m, analyses, "cleanup")
}

/// Run the WASM target pipeline: lowering + cleanup.
fn run_wasm_target_pipeline(ctx: &mut IrContext, m: Module) -> Result<(), DumpIrError> {
    let mut analyses = AnalysisCache::new();
    enter_target_lowering(
        ctx,
        m,
        &mut analyses,
        "Wasm",
        tribute_ir::runtime_items::RUNTIME_ITEMS,
    )?;
    lower_wasm_runtime_calls(ctx, m)?;
    finish_target_lowering(
        ctx,
        m,
        &mut analyses,
        tribute_passes::abi_boundary::TargetKind::Wasm,
    )
}

/// Run the native target pipeline: lowering + cleanup.
fn run_native_target_pipeline(ctx: &mut IrContext, m: Module) -> Result<(), DumpIrError> {
    let mut analyses = AnalysisCache::new();
    enter_target_lowering(
        ctx,
        m,
        &mut analyses,
        "native",
        tribute_ir::runtime_items::RUNTIME_ITEMS,
    )?;
    lower_native_runtime_calls(ctx, m)?;
    finish_target_lowering(
        ctx,
        m,
        &mut analyses,
        tribute_passes::abi_boundary::TargetKind::Native,
    )
}

/// The target-independent start of the representation/ABI boundary, through
/// evidence lowering, up to the target-specific stages that call runtime
/// functions.
fn enter_target_lowering(
    ctx: &mut IrContext,
    m: Module,
    analyses: &mut AnalysisCache,
    target: &str,
    runtime: &[tribute_ir::runtime_items::RuntimeItem],
) -> Result<(), DumpIrError> {
    debug_validate_value_integrity(ctx, m, &format!("before {target} target lowering"));

    // Drop unreachable functions first, so that no later pass lowers them.
    // Runtime functions stay until the target lowering has called them.
    trunk_ir::transforms::global_dce::eliminate_dead_functions_with_config(
        ctx,
        m,
        tribute_ir::runtime_items::kept_alive(runtime),
        analyses,
    );

    // General function inlining. Single-block-only (no `cf` dialect
    // dependency), so it preserves the caller's block structure. That keeps
    // WASM lowering within the dialects it handles and lets the same pass
    // work on both backend paths.
    trunk_ir::transforms::inline::inline_functions(ctx, m, analyses);
    // Fold what inlining exposed (struct reads of known constructions, box
    // round trips, constant selectors) before closure storage and evidence
    // lowering see the IR.
    run_canonicalize_passes(ctx, m, analyses, "post-inline canonicalization")?;

    enter_target_closure_storage_boundary(ctx, m, analyses)?;
    lower_evidence(ctx, m, analyses)
}

/// The Wasm bytes intrinsic bridge, the target-specific stage that calls
/// runtime items.
fn lower_wasm_runtime_calls(ctx: &mut IrContext, m: Module) -> Result<(), DumpIrError> {
    // Complete the supported bytes intrinsic bridge inside the boundary; the
    // lowering consumes its verified compiler intrinsic identity.
    tribute_passes::wasm::bytes::lower(ctx, m)?;
    Ok(())
}

/// The native bytes intrinsic bridge, the target-specific stage that calls
/// runtime items.
fn lower_native_runtime_calls(ctx: &mut IrContext, m: Module) -> Result<(), DumpIrError> {
    // Complete the supported bytes intrinsic bridge inside the boundary; the
    // lowering consumes its verified compiler intrinsic identity.
    tribute_passes::native::intrinsic_to_native::lower(ctx, m)?;
    Ok(())
}

/// Lower `effect.*` to calls of the evidence runtime and drop the ability
/// declarations.
fn lower_evidence(
    ctx: &mut IrContext,
    m: Module,
    analyses: &mut AnalysisCache,
) -> Result<(), DumpIrError> {
    use tribute_passes::evidence_lowering::{LowerEvidence, prepare_evidence_runtime};

    let core_module = core_dialect::Module::from_op(ctx, m.op())
        .expect("evidence lowering requires a core.module");
    let runtime = prepare_evidence_runtime(ctx, m).map_err(|error| DumpIrError {
        message: error.to_string(),
    })?;
    let mut pm = PassManager::new();
    pm.nest::<func_dialect::Func>()
        .add_pass(LowerEvidence::new(runtime));
    pm.with_debug_verifier();
    pm.run(ctx, core_module, analyses)?;
    tribute_ir::dialect::ability::remove_declarations(ctx, m);
    Ok(())
}

/// The end of the representation/ABI boundary, after every stage that calls
/// runtime items: closure storage layout, the cleanup that removes the
/// runtime functions nothing calls, and the boundary exit verification.
fn finish_target_lowering(
    ctx: &mut IrContext,
    m: Module,
    analyses: &mut AnalysisCache,
    target: tribute_passes::abi_boundary::TargetKind,
) -> Result<(), DumpIrError> {
    tribute_passes::closure_lower::finalize_closure_storage_layout(ctx, m);
    debug_validate_value_integrity(ctx, m, "after target evidence lowering");

    run_cleanup_passes(ctx, m, analyses)?;
    enforce_boundary_exit(ctx, m, target)
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
            emit_native_from_boundary_exit(ctx, m, false)
        }
        tribute_passes::abi_boundary::TargetKind::Wasm => compile_to_wasm(ctx, m)
            .map(|binary| binary.bytes)
            .map_err(EmitError::Wasm),
    }
}

/// Lower and emit a native object at the boundary exit with production
/// optimizations, instrumented by the address sanitizer when `sanitize` is
/// set.
pub fn emit_native_from_boundary_exit(
    ctx: &mut IrContext,
    m: Module,
    sanitize: bool,
) -> Result<Vec<u8>, EmitError> {
    compile_module_to_native(ctx, m, sanitize, NativeOptimizationOptions::production())
        .map_err(EmitError::Native)
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
    tribute_passes::target_abi::compose_value_delimiters(ctx, m)?;
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
    native_lowering_passes(sanitize, optimizations, stop_after)
        .run(ctx, core_module, &mut analyses)
        .map_err(native_pass_failure)
}

/// The native lowering stages, in order, as the passes of one
/// [`PassManager`] with the debug verifier.
///
/// A stage that fails is named in the resulting [`PassError`].
fn native_lowering_passes(
    sanitize: bool,
    optimizations: NativeOptimizationOptions,
    stop_after: Option<NativePipelineStage>,
) -> PassManager {
    use tribute_passes::native::{
        adt_rc_header::AdtRcHeader,
        adt_to_clif::AdtToClif,
        cast_legalization::LegalizeCasts,
        clif_lowering::{ArithToClif, CfToClif, FuncToClif, MemToClif},
        const_to_native::ConstToNative,
        descriptor_to_clif::DescriptorToClif,
        entrypoint::GenerateNativeEntrypoint,
        io::IoToNative,
        list::ListToNative,
        ownership_lowering::LowerNativeOwnership,
        rc_lowering::RcLowering,
        rc_optimization::EliminatePairedRc,
        rtti::GenerateRtti,
        sanitize_access::{DeclareAccessChecks, InstrumentMemoryAccesses},
        tribute_rt_to_clif::TributeRtToClif,
    };

    let mut pm = PassManager::new();
    pm.with_debug_verifier();

    pm.add_pass(GenerateNativeEntrypoint { sanitize })
        // Declare a clif.data object per string/bytes payload, then lower
        // adt.string_const to adt.variant_new + bytes alloc and
        // adt.bytes_const to a clif alloc + data reference.
        .add_pass(ConstToNative)
        // Lower target-independent I/O to the native runtime ABI.
        .add_pass(IoToNative)
        // Select the private native representation for opaque lists.
        .add_pass(ListToNative);

    // Lower structured control flow to CFG-based control flow.
    pm.nest::<func_dialect::Func>()
        .add_pass(trunk_ir::transforms::scf_to_cf_pass());

    pm.add_pass(LowerNativeOwnership {
        options: tribute_passes::native::optimization::ownership_plan_options(
            stop_after,
            optimizations,
        ),
    })
    // A descriptor read takes a managed reference, so it is lowered before
    // type conversion turns its operand into a pointer.
    .add_pass(DescriptorToClif)
    .add_pass(FuncToClif)
    .add_pass(CfToClif)
    .add_pass(GenerateRtti)
    .add_pass(AdtRcHeader)
    .add_pass(AdtToClif)
    .add_pass(ArithToClif)
    .add_pass(MemToClif)
    // Lower non-RC tribute runtime operations. The explicit RC operations
    // were materialized from typed ownership actions before erasure.
    .add_pass(TributeRtToClif);

    if matches!(
        stop_after,
        Some(
            NativePipelineStage::AfterRcInsertion
                | NativePipelineStage::AfterBorrowedParameterOptimization
                | NativePipelineStage::AfterTemporaryBorrowOptimization
        )
    ) {
        return pm;
    }

    // Eliminate local retain/release pairs while they are still directly
    // observable tribute_rt operations.
    if optimizations.paired_rc_elimination == PairedRcEliminationPolicy::Enabled {
        pm.add_pass(EliminatePairedRc);
    }

    if stop_after == Some(NativePipelineStage::AfterRcOptimization) {
        return pm;
    }

    pm.add_pass(LegalizeCasts)
        .add_pass(ReconcileUnrealizedCasts)
        // Lower RC operations (retain/release) to inline clif code.
        .add_pass(RcLowering);

    // A sanitized build checks every access, including the refcount accesses
    // RC lowering just produced.
    if sanitize {
        pm.add_pass(DeclareAccessChecks);
        pm.nest::<trunk_ir::dialect::clif::Func>()
            .add_pass(InstrumentMemoryAccesses);
    }

    pm
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
// It transforms: CST → AST → resolve → typecheck → ast_to_ir → TrunkIR
//

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
        ast_resolve::resolve_with_env(db, user_ast, user_env, &dependencies, span_map);

    // Phase 4: Type checking against package std
    let mut checker = ast_typeck::TypeChecker::new(db, span_map.clone());
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

    Some(ast_typeck::TypeCheckOutput::from_check(
        db,
        result,
        span_map.clone(),
    ))
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
    diagnostics.sort_by(tribute_core::diagnostic::compare_diagnostics);

    CompilationResult {
        produced_module,
        diagnostics,
    }
}

#[cfg(test)]
mod runtime_item_tests;

#[cfg(test)]
mod tests;
