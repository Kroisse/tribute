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
//!     ▼ ast_to_ir
//! Module (source-logical callable/control IR)
//!     │
//!     ├─── Stage chain: one PassManager per entry point, over a range ──┤
//!     │    (`Stage`, built by `stage_passes`; one arena session)
//!     ▼ Specialize
//!     │   eliminate_unreferenced_source_functions (artifact compiles only:
//!     │   unreachable source-logical functions skip CPS legalization;
//!     │   runtime functions stay where a reachable perform, handle, or
//!     │   evidence plan can make target lowering call them) → dce →
//!     │   specialize_convention_classes → eliminate_unreferenced_source_functions
//!     │                                    ◄── compile_specialized_frontend
//!     ▼ LegalizeCps
//!     │   check_fn_arm_calls (calls in `fn` arms that reach a `Cps` callee) →
//!     │   tribute_control_to_cps → lower_continuation_frames →
//!     │   lower_closure_lambda → intrinsic/list/io lowering
//!     │                                    ◄── run_through_cps_lowering
//!     ▼ LowerEffects
//!     │   lower_ability_call → resolve_evidence → lower_handle_dispatch
//!     │                                    ◄── run_shared_middle_end,
//!     │                                        compile_ast (all functions)
//!     ▼ CheckArity (warnings into the context's diagnostics)
//!     │                                    ◄── compile_with_diagnostics
//!     ├─── Representation/ABI Boundary (run_target_to_boundary_exit) ──┤
//!     ▼ EnterTarget
//!     │   global DCE (unreachable functions skip target lowering; runtime
//!     │   items stay until target lowering calls them) → inline_functions →
//!     │   canonicalize, DCE → CPS callable physicalization (target ABI
//!     │   validation, root entry bridge, value delimiters) →
//!     │   lower-prepared-closures → evidence lowering
//!     ▼ LowerRuntimeCalls
//!     │   bytes intrinsic bridge (target-specific)
//!     ▼ ExitBoundary
//!     │   closure storage layout finalization → cleanup (global DCE removes
//!     │   uncalled runtime items, canonicalize, DCE) → boundary exit check
//!     │   (enforced in every build)        ◄── dump_ir (`--dump-ir`)
//!     │
//!     ├─── After the Boundary Exit (emit_from_boundary_exit) ──┤
//!     ▼ LowerTarget
//!     ├─► [wasm]   lower_to_wasm → cast legalization (reference upcasts as
//!     │            wasm.ref_upcast) → reconcile → GC index assignment
//!     └─► [native] entrypoint, clif lowering, RTTI/RC, cast legalization
//!                  (◄── dump_native_ir_at_stage)
//!     │
//!     ▼ emission (outside the chain)
//!     ├─► [wasm]   emit
//!     └─► [native] validate_clif_ir (direct-call argument/result types) → emit
//! ```
//!
//! ## Diagnostics
//!
//! Diagnostics are collected using Salsa accumulators. Each stage can emit
//! diagnostics via `Diagnostic::new(...).accumulate(db)`, which are then
//! collected at the end of compilation.

use crate::SourceCst;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use std::ops::RangeInclusive;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Span;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::conversion::ReconcileUnrealizedCasts;
use trunk_ir::dialect::{core as core_dialect, func as func_dialect};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{PassError, PassManager, PassResult};
use trunk_ir::{IrContext, Module};

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
use tribute_front::ast::SpanMaps;
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
fn frontend_spans(db: &dyn salsa::Database, source: SourceCst) -> SpanMaps<'_> {
    let prelude = crate::prelude::prelude_span_map(db);
    SpanMaps::new(ast_query::span_map(db, source).into_iter().chain(prelude))
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
// Pipeline Stages
// =============================================================================

/// A stage of the pipeline after the frontend, in execution order.
///
/// [`stage_passes`] builds every stage, so its body is the stage order. An
/// entry point runs a contiguous range of stages as one [`PassManager`] on
/// one arena session; the ranges split the pipeline where tools and tests
/// observe it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    /// Remove the source-logical functions nothing reachable references
    /// (artifact compiles only), specialize convention classes, and remove
    /// unreferenced functions again.
    Specialize,
    /// Legalize source-logical callable/control IR into CPS with explicit
    /// evidence, and lower the intrinsics it leaves.
    LegalizeCps,
    /// Lower ability calls, resolve handler evidence, and lower handle
    /// dispatch: the shared middle-end output.
    LowerEffects,
    /// Check call arity. A mismatch is a warning in the context's
    /// diagnostics, which the reporting entry points pass on to Salsa.
    CheckArity,
    /// Enter the representation/ABI boundary: global DCE (runtime functions
    /// stay), inlining, canonicalization, CPS callable physicalization,
    /// prepared closure lowering, and evidence lowering.
    EnterTarget,
    /// The target-specific stage that calls runtime items: the bytes
    /// intrinsic bridge.
    LowerRuntimeCalls,
    /// Leave the boundary: closure storage layout, the cleanup that removes
    /// the runtime functions nothing calls, and the boundary exit check.
    ExitBoundary,
    /// Lower to the target after the boundary exit, up to emission.
    LowerTarget,
}

/// Which source-logical functions the shared middle-end legalizes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceFunctions {
    /// Those reachable from the program's roots; an artifact has no others.
    Reachable,
    All,
}

/// How the stages of one run are configured.
struct StageConfig {
    functions: SourceFunctions,
    /// The registered compiler intrinsics CPS legalization checks.
    compiler_intrinsics: Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
    /// The target of the target stages.
    target: Option<tribute_passes::abi_boundary::TargetKind>,
    /// The runtime items target entry keeps alive until they are called.
    runtime: &'static [tribute_ir::runtime_items::RuntimeItem],
    native: NativeLowering,
}

/// How native lowering after the boundary exit is configured.
#[derive(Clone, Copy)]
struct NativeLowering {
    sanitize: bool,
    optimizations: NativeOptimizationOptions,
    /// The observable native stage to stop after, for dumps.
    stop_after: Option<NativePipelineStage>,
}

impl StageConfig {
    /// A run of the stages before the boundary that legalizes `functions`.
    fn shared(functions: SourceFunctions) -> Self {
        Self {
            functions,
            compiler_intrinsics: Vec::new(),
            target: None,
            runtime: tribute_ir::runtime_items::RUNTIME_ITEMS,
            native: NativeLowering {
                sanitize: false,
                optimizations: NativeOptimizationOptions::production(),
                stop_after: None,
            },
        }
    }

    /// A run that reaches `target`'s stages, on reachable functions only.
    fn target(target: tribute_passes::abi_boundary::TargetKind) -> Self {
        Self {
            target: Some(target),
            ..Self::shared(SourceFunctions::Reachable)
        }
    }

    fn target_kind(&self) -> tribute_passes::abi_boundary::TargetKind {
        self.target.expect("target stages need a target")
    }
}

/// The passes of `stages`, in order, as one [`PassManager`] with the debug
/// verifier.
fn stage_passes(mut config: StageConfig, stages: RangeInclusive<Stage>) -> PassManager {
    use tribute_passes::abi_boundary::TargetKind;
    use tribute_passes::unreferenced_source_functions::EliminateUnreferencedSourceFunctions;

    let mut pm = PassManager::new();
    let reachable = config.functions == SourceFunctions::Reachable;

    if stages.contains(&Stage::Specialize) {
        if reachable {
            pm.add_pass(EliminateUnreferencedSourceFunctions::new());
        }
        pm.add_pass(trunk_ir::transforms::EliminateDeadCode::<
            core_dialect::Module,
        >::new())
            .add_pass(tribute_passes::specialize_convention_classes::SpecializeConventionClasses);
        if reachable {
            pm.add_pass(EliminateUnreferencedSourceFunctions::new());
        }
    }

    if stages.contains(&Stage::LegalizeCps) {
        pm.add_pass(tribute_passes::check_fn_arm_calls::CheckFnArmCalls);
        if cfg!(debug_assertions) {
            pm.add_pass(tribute_passes::check_convention_classes::CheckConventionClasses);
        }
        pm.add_pass(
            tribute_passes::tribute_control_to_cps::TributeControlToCps::new()
                .with_compiler_intrinsics(std::mem::take(&mut config.compiler_intrinsics)),
        )
        .add_pass(tribute_passes::lower_continuation_frames::LowerContinuationFrames)
        .add_pass(tribute_passes::lower_closure_lambda::LowerClosureLambda)
        .add_pass(tribute_passes::intrinsic_to_arith::LowerIntrinsicToArith)
        .add_pass(tribute_passes::list_intrinsics::LowerListIntrinsics)
        .add_pass(tribute_passes::slots_intrinsics::LowerSlotsIntrinsics)
        .add_pass(tribute_passes::io_lowering::LowerIoIntrinsics);
    }

    if stages.contains(&Stage::LowerEffects) {
        // lower_ability_call produces the tail-resumptive effect dispatches;
        // evidence resolution then extends handler scopes, and handle
        // dispatch lowering consumes what it expanded.
        pm.nest::<func_dialect::Func>()
            .add_pass(tribute_passes::lower_ability_call::LowerAbilityCall);
        pm.add_pass(tribute_passes::resolve_evidence::ResolveEvidenceDispatch);
        pm.nest::<func_dialect::Func>()
            .add_pass(tribute_passes::lower_handle_dispatch::LowerHandleDispatch);
    }

    if stages.contains(&Stage::CheckArity) {
        pm.add_pass(trunk_ir::validation::CheckCallArity);
    }

    if stages.contains(&Stage::EnterTarget) {
        pm.add_pass(trunk_ir::validation::CheckValueIntegrity {
            boundary: "before target lowering",
        })
        // Drop unreachable functions first, so that no later pass lowers
        // them. Runtime functions stay until the target lowering has called
        // them.
        .add_pass(trunk_ir::transforms::GlobalDce {
            config: tribute_ir::runtime_items::kept_alive(config.runtime),
        })
        // General function inlining. Single-block-only (no `cf` dialect
        // dependency), so it preserves the caller's block structure. That
        // keeps Wasm lowering within the dialects it handles and lets the
        // same pass work on both backend paths.
        .add_pass(trunk_ir::transforms::InlineFunctions);
        // Fold what inlining exposed (struct reads of known constructions,
        // box round trips, constant selectors) before closure storage and
        // evidence lowering see the IR.
        add_canonicalize_passes(&mut pm);
        // Exact ABI validation observes semantic closure types first;
        // `LowerPreparedClosures` then consumes any remaining closure
        // operations before whole-module storage finalization.
        pm.add_pass(tribute_passes::target_abi::PhysicalizeCpsCallables)
            .add_pass(tribute_passes::closure_lower::LowerPreparedClosures)
            .add_pass(tribute_passes::evidence_lowering::LowerEvidence);
    }

    if stages.contains(&Stage::LowerRuntimeCalls) {
        // Complete the supported bytes intrinsic bridge inside the boundary;
        // the lowering consumes its verified compiler intrinsic identity.
        match config.target_kind() {
            TargetKind::Native => {
                pm.add_pass(tribute_passes::native::intrinsic_to_native::LowerBytesIntrinsics)
            }
            TargetKind::Wasm => pm.add_pass(tribute_passes::wasm::bytes::LowerBytesIntrinsics),
        };
    }

    if stages.contains(&Stage::ExitBoundary) {
        pm.add_pass(tribute_passes::closure_lower::FinalizeClosureStorageLayout)
            .add_pass(trunk_ir::validation::CheckValueIntegrity {
                boundary: "after target evidence lowering",
            })
            // The passes that created conversion casts already emitted their
            // boxing, so the casts left here only retype values for the
            // target type conversion.
            .add_pass(trunk_ir::transforms::GlobalDce {
                config: Default::default(),
            });
        add_canonicalize_passes(&mut pm);
        pm.add_pass(tribute_passes::abi_boundary::CheckBoundaryExit {
            target: config.target_kind(),
        });
    }

    if stages.contains(&Stage::LowerTarget) {
        match config.target_kind() {
            TargetKind::Native => add_native_lowering_passes(&mut pm, config.native),
            TargetKind::Wasm => add_wasm_lowering_passes(&mut pm),
        }
    }

    pm.with_debug_verifier();
    pm
}

/// Forward known struct reads module-wide, then run canonicalization and DCE
/// on every function.
fn add_canonicalize_passes(pm: &mut PassManager) {
    pm.add_pass(tribute_ir::dialect::adt::ForwardStructGets)
        .nest::<func_dialect::Func>()
        .add_pass(trunk_ir::transforms::Canonicalize)
        .add_pass(trunk_ir::transforms::EliminateDeadCode::new());
}

/// Wasm lowering after the boundary exit:
/// 1. Lower the module to wasm dialect operations
/// 2. Convert and reconcile unrealized conversion casts with the Wasm type
///    converter, writing reference upcasts as `wasm.ref_upcast`
/// 3. Assign the module-local indices of the WasmGC operations
///    materialization introduced, and check that the module is ready for
///    emission
fn add_wasm_lowering_passes(pm: &mut PassManager) {
    pm.add_pass(tribute_passes::wasm::lower::LowerToWasm)
        .add_pass(tribute_passes::wasm::lower::LegalizeUnrealizedCasts)
        .add_pass(ReconcileUnrealizedCasts)
        .add_pass(tribute_passes::wasm::lower::FinalizeWasmGcTypes);
}

/// Native lowering after the boundary exit: the native entrypoint, lowering
/// of scf/func/cf/adt/arith to `clif.*`, RTTI and RC insertion and lowering,
/// and cast reconciliation. A dump stops after `native.stop_after`.
fn add_native_lowering_passes(pm: &mut PassManager, native: NativeLowering) {
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

    let NativeLowering {
        sanitize,
        optimizations,
        stop_after,
    } = native;

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
        .add_pass(trunk_ir::transforms::ScfToCf);

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
        return;
    }

    // Eliminate local retain/release pairs while they are still directly
    // observable tribute_rt operations.
    if optimizations.paired_rc_elimination == PairedRcEliminationPolicy::Enabled {
        pm.add_pass(EliminatePairedRc);
    }

    if stop_after == Some(NativePipelineStage::AfterRcOptimization) {
        return;
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
}

/// Run `stages` on the module of an arena session.
fn run_stages(
    ctx: &mut IrContext,
    m: Module,
    config: StageConfig,
    stages: RangeInclusive<Stage>,
) -> PassResult {
    let root =
        core_dialect::Module::from_op(ctx, m.op()).expect("the pipeline runs on a core.module");
    stage_passes(config, stages).run(ctx, root, &mut AnalysisCache::new())
}

/// A frontend result after a run of stages, with the run's result.
struct StagedCompilation {
    context: IrContext,
    module: Module,
    result: PassResult,
}

/// Run the frontend on `source`, then `stages` on its output.
///
/// Returns `None` if parsing fails or the frontend reports an error.
fn run_stages_from_source(
    db: &dyn salsa::Database,
    source: SourceCst,
    config: StageConfig,
    stages: RangeInclusive<Stage>,
) -> Option<StagedCompilation> {
    let frontend = compile_frontend_for_shared_route(db, source)?;
    Some(run_stages_from_frontend(frontend, config, stages))
}

fn run_stages_from_frontend(
    frontend: FrontendCompilation,
    mut config: StageConfig,
    stages: RangeInclusive<Stage>,
) -> StagedCompilation {
    let FrontendCompilation {
        mut context,
        module,
        compiler_intrinsics,
    } = frontend;
    config.compiler_intrinsics = compiler_intrinsics;
    let result = run_stages(&mut context, module, config, stages);
    StagedCompilation {
        context,
        module,
        result,
    }
}

impl StagedCompilation {
    /// The arena session if every stage succeeded.
    fn into_result(self) -> PassResult<(IrContext, Module)> {
        self.result.map(|()| (self.context, self.module))
    }

    /// Report the context's diagnostics and any failure to `db`, and return
    /// the arena session if every stage succeeded.
    fn report(self, db: &dyn salsa::Database) -> Option<(IrContext, Module)> {
        report_context_diagnostics(db, &self.context);
        match self.result {
            Ok(()) => Some((self.context, self.module)),
            Err(error) => {
                report_pass_error(db, &error);
                None
            }
        }
    }
}

// =============================================================================
// Pipeline Entry Points
// =============================================================================

/// Run the frontend and class specialization, and remove the definitions
/// nothing reachable references: the source-logical IR CPS legalization
/// receives.
///
/// Returns `None` if parsing fails or the frontend reports an error.
pub fn compile_specialized_frontend(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    run_stages_from_source(
        db,
        source,
        StageConfig::shared(SourceFunctions::Reachable),
        Stage::Specialize..=Stage::Specialize,
    )
    .map(StagedCompilation::into_result)
    .transpose()
}

/// Run the frontend and CPS legalization on every function (for testing).
///
/// Evidence params are introduced by the physical CPS conversion.
pub fn run_through_cps_lowering(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    run_stages_from_source(
        db,
        source,
        StageConfig::shared(SourceFunctions::All),
        Stage::Specialize..=Stage::LegalizeCps,
    )
    .map(StagedCompilation::into_result)
    .transpose()
}

/// Run the shared middle-end on a frontend result.
///
/// The result is ready for [`run_target_to_boundary_exit`].
pub fn run_shared_middle_end(frontend: FrontendCompilation) -> PassResult<(IrContext, Module)> {
    run_stages_from_frontend(
        frontend,
        StageConfig::shared(SourceFunctions::Reachable),
        Stage::Specialize..=Stage::LowerEffects,
    )
    .into_result()
}

/// Run `target`'s pipeline from the shared middle-end output to the
/// representation/ABI boundary exit.
pub fn run_target_to_boundary_exit(
    ctx: &mut IrContext,
    m: Module,
    target: tribute_passes::abi_boundary::TargetKind,
) -> Result<(), DumpIrError> {
    run_stages(
        ctx,
        m,
        StageConfig::target(target),
        Stage::EnterTarget..=Stage::ExitBoundary,
    )?;
    Ok(())
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
        tribute_passes::abi_boundary::TargetKind::Wasm => lower_and_emit_wasm(ctx, m)
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
    lower_and_emit_native(ctx, m, sanitize, NativeOptimizationOptions::production())
        .map_err(EmitError::Native)
}

/// Lower a module at the boundary exit to Wasm and emit its binary.
fn lower_and_emit_wasm(ctx: &mut IrContext, m: Module) -> WasmCompilationResult<WasmBinary> {
    run_stages(
        ctx,
        m,
        StageConfig::target(tribute_passes::abi_boundary::TargetKind::Wasm),
        Stage::LowerTarget..=Stage::LowerTarget,
    )
    .map_err(|error| CompilationError::ir_validation(error.to_string()))?;
    emit_wasm(ctx, m)
}

/// Lower a module at the boundary exit to native code with `optimizations`
/// and emit its object file.
fn lower_and_emit_native(
    ctx: &mut IrContext,
    m: Module,
    sanitize: bool,
    optimizations: NativeOptimizationOptions,
) -> NativeCompilationResult<Vec<u8>> {
    let config = StageConfig {
        native: NativeLowering {
            sanitize,
            optimizations,
            stop_after: None,
        },
        ..StageConfig::target(tribute_passes::abi_boundary::TargetKind::Native)
    };
    run_stages(ctx, m, config, Stage::LowerTarget..=Stage::LowerTarget).map_err(|error| {
        trunk_ir_cranelift_backend::CompilationError::ir_validation(error.to_string())
    })?;
    emit_native(ctx, m)
}

/// Emit the Wasm binary of a lowered module.
fn emit_wasm(ctx: &mut IrContext, m: Module) -> WasmCompilationResult<WasmBinary> {
    let _span = tracing::info_span!("emit_module_to_wasm").entered();
    trunk_ir_wasm_backend::emit_module_to_wasm(ctx, m)
}

/// Validate and emit the native object file of a lowered module.
fn emit_native(ctx: &mut IrContext, m: Module) -> NativeCompilationResult<Vec<u8>> {
    let _span = tracing::info_span!("emit_module_to_native").entered();
    emit_module_to_native(ctx, m)
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
    let config = StageConfig {
        native: NativeLowering {
            sanitize: false,
            optimizations: options.native,
            stop_after: Some(stage),
        },
        ..StageConfig::target(tribute_passes::abi_boundary::TargetKind::Native)
    };
    let Some(staged) =
        run_stages_from_source(db, source, config, Stage::Specialize..=Stage::LowerTarget)
    else {
        return Ok(String::new());
    };
    dump_staged(db, staged)
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
    let target = if native {
        tribute_passes::abi_boundary::TargetKind::Native
    } else {
        tribute_passes::abi_boundary::TargetKind::Wasm
    };
    let Some(staged) = run_stages_from_source(
        db,
        source,
        StageConfig::target(target),
        Stage::Specialize..=Stage::ExitBoundary,
    ) else {
        return Ok(String::new());
    };
    dump_staged(db, staged)
}

/// Report the context's diagnostics and print the module, or return the
/// run's failure.
fn dump_staged(db: &dyn salsa::Database, staged: StagedCompilation) -> Result<String, DumpIrError> {
    report_context_diagnostics(db, &staged.context);
    staged.result?;
    Ok(trunk_ir::printer::print_module(
        &staged.context,
        staged.module.op(),
    ))
}

#[salsa::tracked(returns(as_deref))]
fn compile_to_wasm_binary_tracked(db: &dyn salsa::Database, source: SourceCst) -> Option<Vec<u8>> {
    let (mut ctx, m) = run_stages_from_source(
        db,
        source,
        StageConfig::target(tribute_passes::abi_boundary::TargetKind::Wasm),
        Stage::Specialize..=Stage::LowerTarget,
    )?
    .report(db)?;
    match emit_wasm(&mut ctx, m) {
        Ok(binary) => Some(binary.bytes),
        Err(e) => {
            report_emission_error(db, format!("WebAssembly compilation failed: {e}"));
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
    let stages = StageConfig {
        native: NativeLowering {
            sanitize: config.sanitize_address(db),
            optimizations: config.optimizations(db).native,
            stop_after: None,
        },
        ..StageConfig::target(tribute_passes::abi_boundary::TargetKind::Native)
    };
    let (mut ctx, m) =
        run_stages_from_source(db, source, stages, Stage::Specialize..=Stage::LowerTarget)?
            .report(db)?;
    match emit_native(&mut ctx, m) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            report_emission_error(db, format!("Native compilation failed: {e}"));
            None
        }
    }
}

/// Pass the diagnostics the stages reported to the context on to Salsa.
fn report_context_diagnostics(db: &dyn salsa::Database, ctx: &IrContext) {
    for diag in ctx.diagnostics().iter() {
        Diagnostic::from_ir(diag.clone(), CompilationPhase::Lowering).accumulate(db);
    }
}

fn report_emission_error(db: &dyn salsa::Database, message: String) {
    Diagnostic::new(
        message,
        Span::new(0, 0),
        DiagnosticSeverity::Error,
        CompilationPhase::Lowering,
    )
    .accumulate(db);
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

/// Run the shared pipeline for diagnostic collection only.
///
/// This is a `#[salsa::tracked]` function so that diagnostics accumulated
/// during compilation can be collected via `compile_ast_tracked::accumulated`.
/// Returns whether the shared pipeline produced a module.
#[salsa::tracked(returns(copy))]
fn compile_ast_tracked(db: &dyn salsa::Database, source: SourceCst) -> bool {
    // Run the shared pipeline; diagnostics are accumulated as side effects
    run_stages_from_source(
        db,
        source,
        StageConfig::shared(SourceFunctions::All),
        Stage::Specialize..=Stage::CheckArity,
    )
    .and_then(|staged| staged.report(db))
    .is_some()
}

/// Compile using the AST-based pipeline.
///
/// Runs the shared pipeline on every function of the program, reachable or
/// not, and returns arena IR. Does NOT run target-specific passes
/// (WASM/native), making it suitable for diagnostic-only compilation
/// ("none" target).
pub fn compile_ast(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    run_stages_from_source(
        db,
        source,
        StageConfig::shared(SourceFunctions::All),
        Stage::Specialize..=Stage::LowerEffects,
    )
    .map(StagedCompilation::into_result)
    .transpose()
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
