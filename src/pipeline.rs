//! Compilation pipeline for Tribute.
//!
//! This module orchestrates the compilation stages with centralized control flow.
//! Each pass is a pure `Module → Module` transformation, and this module handles
//! the sequencing and optional caching of expensive stages.
//!
//! ## Architecture Principles
//!
//! 1. **Pure Transformations**: Each pass is a pure function `(db, Module) -> Module`
//! 2. **Centralized Orchestration**: Pass sequencing is managed here, not in passes
//! 3. **Selective Caching**: Only expensive passes use `#[salsa::tracked]` caching
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
//!     ├─────────────── Evidence & Closure Processing ─┤
//!     ▼ evidence_params (Phase 1)
//! Module (evidence params added to signatures)
//!     │
//!     ▼ closure_lower
//! Module (closure.* lowered)
//!     │
//!     ├─── Shared Pipeline (run_shared_pipeline) ─────┤
//!     ▼ evidence_calls (Phase 2)
//! Module (evidence passed through calls)
//!     │
//!     ▼ resolve_evidence
//! Module (cont.shift uses dynamic tags)
//!     │
//!     ├─► [wasm]   cont_to_trampoline ─► dce ─► resolve_casts ─► compile_to_wasm
//!     └─► [native] cont_to_libmprompt ─► dce ─► resolve_casts ─► compile_to_native
//! ```
//!
//! ## Diagnostics
//!
//! Diagnostics are collected using Salsa accumulators. Each stage can emit
//! diagnostics via `Diagnostic { ... }.accumulate(db)`, which are then
//! collected at the end of compilation.

use crate::SourceCst;
use ropey::Rope;
use salsa::Accumulator;
use std::path::Path;
use tree_sitter::Parser;
use tribute_front::derive_module_name_from_path;
use tribute_front::source_file::parse_with_rope;
use tribute_passes::closure_lower::lower_closures;
use tribute_passes::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_passes::evidence;
use tribute_passes::generate_native_entrypoint;
use tribute_passes::generic_type_converter;
use tribute_passes::lower_cont_to_libmprompt;
use tribute_passes::lower_cont_to_trampoline;
use tribute_passes::lower_evidence_to_native;
use tribute_passes::resolve_evidence::resolve_evidence_dispatch;
use tribute_passes::wasm::lower::lower_to_wasm;
use tribute_passes::wasm::type_converter::wasm_type_converter;
use trunk_ir::Span;
use trunk_ir::arena::bridge::{export_to_salsa, import_salsa_module};
use trunk_ir::conversion::resolve_unrealized_casts;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::ConversionError;
use trunk_ir::transforms::eliminate_dead_functions;
use trunk_ir::{Block, BlockId, DialectOp, IdVec, Region};

// =============================================================================
// Compilation configuration
// =============================================================================

/// Compilation options threaded through the pipeline via Salsa.
#[salsa::input]
pub struct CompilationConfig {
    /// Enable AddressSanitizer instrumentation.
    pub sanitize_address: bool,
}

// AST-based pipeline imports
use tribute_front::ast::ResolvedRef;
use tribute_front::ast::SpanMap;
use tribute_front::ast_to_ir;
use tribute_front::astgen::ParsedAst;
use tribute_front::query as ast_query;
use tribute_front::resolve as ast_resolve;
use tribute_front::resolve::ModuleEnv;
use tribute_front::tdnr as ast_tdnr;
use tribute_front::typeck as ast_typeck;
use tribute_front::typeck::PreludeExports;
use trunk_ir_cranelift_backend::passes::{adt_to_clif, arith_to_clif, cf_to_clif, func_to_clif};
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

/// Parse the prelude source.
///
/// This is the first stage of prelude processing, shared by all prelude-related functions.
/// Returns both the parsed AST and the SourceCst to avoid redundant creation.
fn parse_prelude(db: &dyn salsa::Database) -> Option<(ParsedAst<'_>, crate::SourceCst)> {
    let prelude_source = create_prelude_source(db)?;
    let parsed = ast_query::parsed_ast_with_module_path(
        db,
        prelude_source,
        trunk_ir::Symbol::new("prelude"),
    )?;
    Some((parsed, prelude_source))
}

/// Type alias for resolved AST module.
type ResolvedModule<'db> = tribute_front::ast::Module<ResolvedRef<'db>>;

/// Parse and resolve names in the prelude.
///
/// Returns the resolved AST, span map, and SourceCst, ready for type checking.
fn resolve_prelude(
    db: &dyn salsa::Database,
) -> Option<(ResolvedModule<'_>, SpanMap, crate::SourceCst)> {
    let (parsed, prelude_source) = parse_prelude(db)?;
    let prelude_ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();

    let prelude_env = ast_resolve::build_env(db, &prelude_ast);
    let resolved = ast_resolve::resolve_with_env(db, prelude_ast, prelude_env, span_map.clone());

    Some((resolved, span_map, prelude_source))
}

/// Load and cache the prelude module using the AST pipeline.
///
/// This is a Salsa tracked function, so the prelude is parsed only once
/// and cached for all subsequent compilations.
///
/// Unlike the legacy tirgen-based approach, this uses the AST pipeline:
/// parse → resolve → typecheck → TDNR → ast_to_ir
/// This properly handles case expressions, tuple patterns, and other features
/// that tirgen doesn't support.
#[salsa::tracked]
pub fn prelude_module<'db>(db: &'db dyn salsa::Database) -> Option<Module<'db>> {
    let (resolved, span_map, prelude_source) = resolve_prelude(db)?;

    // Typecheck with independent TypeContext
    let checker = ast_typeck::TypeChecker::new(db, span_map.clone());
    let result = checker.check_module(resolved);

    // TDNR
    let tdnr_ast = ast_tdnr::resolve_tdnr(db, result.module);

    // AST → TrunkIR
    let function_types: std::collections::HashMap<_, _> =
        result.function_types.into_iter().collect();
    let node_types: std::collections::HashMap<_, _> = result.node_types.into_iter().collect();
    let source_uri = prelude_source.uri(db).as_str();
    Some(ast_to_ir::lower_ast_to_ir(
        db,
        tdnr_ast,
        span_map,
        source_uri,
        function_types,
        node_types,
    ))
}

/// Create a SourceCst for the prelude.
fn create_prelude_source(db: &dyn salsa::Database) -> Option<crate::SourceCst> {
    let uri = fluent_uri::Uri::parse_from("prelude:///std/prelude".to_owned())
        .expect("valid prelude URI");
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
#[salsa::tracked]
pub fn prelude_env<'db>(db: &'db dyn salsa::Database) -> Option<ModuleEnv<'db>> {
    let (parsed, _) = parse_prelude(db)?;
    let prelude_ast = parsed.module(db);
    Some(ast_resolve::build_env(db, &prelude_ast))
}

/// Process prelude through AST pipeline and extract type exports.
///
/// This function:
/// 1. Uses `resolve_prelude` for parsing and name resolution (cached)
/// 2. Type checks prelude with independent TypeContext (all UniVars resolved)
/// 3. Extracts PreludeExports (TypeSchemes only, no UniVars)
///
/// Cached by Salsa - computed once and reused.
#[salsa::tracked]
pub fn prelude_exports<'db>(db: &'db dyn salsa::Database) -> Option<PreludeExports<'db>> {
    let (resolved, span_map, _) = resolve_prelude(db)?;

    // Typecheck with independent TypeContext (all UniVars resolved)
    let checker = ast_typeck::TypeChecker::new(db, span_map);
    let prelude_exports = checker.check_module_for_prelude(resolved);

    Some(prelude_exports)
}

/// Merge prelude definitions into a user module.
///
/// This prepends all prelude operations (type definitions, etc.) to the
/// user module's body, making prelude types available for name resolution.
pub fn merge_with_prelude<'db>(
    db: &'db dyn salsa::Database,
    user_module: Module<'db>,
) -> Module<'db> {
    let Some(prelude) = prelude_module(db) else {
        return user_module;
    };

    // Get operations from both modules
    let prelude_body = prelude.body(db);
    let user_body = user_module.body(db);

    // Merge blocks: prepend prelude operations to user operations
    let prelude_blocks = prelude_body.blocks(db);
    let user_blocks = user_body.blocks(db);

    // If both have a single block (common case), merge their operations
    if prelude_blocks.len() == 1 && user_blocks.len() == 1 {
        let prelude_block = &prelude_blocks[0];
        let user_block = &user_blocks[0];

        // Combine operations: prelude first, then user
        let mut combined_ops: IdVec<_> = prelude_block.operations(db).iter().copied().collect();
        combined_ops.extend(user_block.operations(db).iter().copied());

        let merged_block = Block::new(
            db,
            BlockId::fresh(),
            user_block.location(db),
            user_block.args(db).clone(),
            combined_ops,
        );

        let merged_body = Region::new(db, user_body.location(db), IdVec::from(vec![merged_block]));

        return Module::create(
            db,
            user_module.location(db),
            user_module.name(db),
            merged_body,
        );
    }

    // Fallback: emit diagnostic if structure doesn't match
    Diagnostic {
        message: format!(
            "Prelude merge failed: expected single block in both modules, got {} prelude blocks and {} user blocks",
            prelude_blocks.len(),
            user_blocks.len()
        ),
        span: Span::new(0, 0),
        severity: DiagnosticSeverity::Warning,
        phase: CompilationPhase::NameResolution,
    }
    .accumulate(db);

    user_module
}

/// Result of the full compilation pipeline.
pub struct CompilationResult<'db> {
    /// The compiled module with resolved types.
    pub module: Module<'db>,
    /// Diagnostics collected during compilation.
    pub diagnostics: Vec<Diagnostic>,
}

// =============================================================================
// Pipeline Stages (Pure Transformations)
// =============================================================================
//
// Each stage is a #[salsa::tracked] function that takes a Module as input
// and returns a transformed Module. Stages do not call other stages directly;
// orchestration is handled by the compile() function.

/// Lambda Lifting.
///
/// Closure Lowering.
///
/// This pass transforms `func.call_indirect` operations on closures:
/// - Extracts funcref via `closure.func`
/// - Extracts env via `closure.env`
/// - Passes env as first argument to the call
#[salsa::tracked]
pub fn stage_closure_lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    lower_closures(db, module)
}

/// Evidence Parameters (Phase 1).
///
/// Adds evidence parameter as first argument to effectful functions.
/// This must run BEFORE lambda lifting so that lambdas can capture evidence.
///
/// Evidence is a runtime structure for dynamic handler dispatch.
/// Pure functions (with empty effect row) are unchanged.
#[salsa::tracked]
pub fn stage_evidence_params<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    // Early exit: check on Salsa side before paying bridge cost
    let effectful_fns = evidence::collect_effectful_functions(db, &module);
    if effectful_fns.is_empty() {
        return module;
    }

    // Bridge to arena, run arena pass, bridge back
    let (mut ctx, arena_module) = import_salsa_module(db, module.as_operation());
    evidence::arena::arena_add_evidence_params(&mut ctx, arena_module);
    let exported = export_to_salsa(db, &ctx, arena_module);
    Module::from_operation(db, exported).unwrap()
}

/// Evidence Call Transformation (Phase 2).
///
/// Transforms call sites to pass evidence through:
/// - Calls inside effectful functions pass the evidence parameter
/// - Calls inside tribute.handle bodies pass null evidence
///
/// This must run AFTER lambda lifting and closure lowering so that:
/// - Lifted lambdas already have evidence parameters
/// - Closure calls can also receive evidence
#[salsa::tracked]
pub fn stage_evidence_calls<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // Early exit: check on Salsa side before paying bridge cost
    let effectful_fns = evidence::collect_effectful_functions(db, &module);
    let fns_with_evidence = evidence::collect_functions_with_evidence_param(db, &module);
    if effectful_fns.is_empty() && fns_with_evidence.is_empty() {
        return module;
    }

    // Bridge to arena, run arena pass, bridge back
    let (mut ctx, arena_module) = import_salsa_module(db, module.as_operation());
    evidence::arena::arena_transform_evidence_calls(&mut ctx, arena_module);
    let exported = export_to_salsa(db, &ctx, arena_module);
    Module::from_operation(db, exported).unwrap()
}

/// Resolve Evidence-based Dispatch.
///
/// This pass transforms `cont.shift` with placeholder tags into
/// evidence-based dispatch using runtime function calls:
/// - Looks up markers from evidence
/// - Extracts prompt tags from markers
/// - Replaces placeholder tags with dynamically resolved tags
///
/// Must run AFTER evidence_calls (Phase 2) so functions have evidence params.
#[salsa::tracked]
pub fn stage_resolve_evidence<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    resolve_evidence_dispatch(db, module)
}

/// Continuation to Trampoline Lowering.
///
/// This pass transforms continuation operations to trampoline operations:
/// - `cont.shift` → `trampoline.build_state` + `trampoline.build_continuation` + etc.
/// - `cont.resume` → `trampoline.reset_yield_state` + `trampoline.continuation_get` + call
/// - `cont.done` / `cont.suspend` → consumed by handler_dispatch lowering
///
/// This is a backend-agnostic pass that prepares continuation operations for
/// the trampoline (yield-bubbling) implementation strategy.
///
/// Returns an error if any `cont.*` operations (except `cont.drop`) remain after conversion.
#[salsa::tracked]
pub fn stage_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    lower_cont_to_trampoline(db, module)
}

/// Continuation to libmprompt Lowering (native backend).
///
/// This pass transforms continuation operations to libmprompt FFI calls:
/// - `cont.push_prompt` → body outlining + `__tribute_prompt` call
/// - `cont.handler_dispatch` → `scf.loop` with yield state polling
/// - `cont.shift` → `__tribute_yield`
/// - `cont.resume` → `__tribute_resume`
/// - `cont.drop` → `__tribute_resume_drop`
///
/// Returns an error if any `cont.*` operations remain after conversion.
#[salsa::tracked]
pub fn stage_cont_to_libmprompt<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    lower_cont_to_libmprompt(db, module)
}

/// Evidence to Native Lowering.
///
/// Replaces evidence runtime function stubs with extern declarations for
/// the native runtime, rewrites empty evidence creation and extend call sites.
#[salsa::tracked]
pub fn stage_evidence_to_native<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    lower_evidence_to_native(db, module)
}

/// Dead Code Elimination (DCE).
///
/// This pass removes unreachable function definitions from the module.
/// Entry points include:
/// - Functions named "main" or "_start"
/// - Functions referenced by `wasm.export_func` (for wasm target)
#[salsa::tracked]
pub fn stage_dce<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let result = eliminate_dead_functions(db, module);
    if result.removed_count > 0 {
        tracing::debug!(
            "DCE removed {} functions: {:?}",
            result.removed_count,
            result.removed_functions
        );
    }
    result.module
}

/// Resolve unrealized conversion casts.
///
/// This pass eliminates `core.unrealized_conversion_cast` operations that may
/// have been inserted during earlier passes (e.g., handler lowering, trampoline
/// conversion). It uses the generic type converter for target-agnostic type
/// materializations.
///
/// Casts that cannot be resolved with the generic converter will be left for
/// backend-specific converters to handle.
#[salsa::tracked]
pub fn stage_resolve_casts<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let type_converter = generic_type_converter();
    let result = resolve_unrealized_casts(db, module, &type_converter);
    // Always use the partially-resolved module; unresolved casts are left for
    // backend-specific converters to handle.
    result.module
}

/// Compile a TrunkIR module to WebAssembly binary.
///
/// This is a Salsa tracked function that:
/// 1. Lowers the module from func/scf/arith dialects to wasm dialect operations
/// 2. Resolves unrealized conversion casts using Tribute-specific type converter
/// 3. Validates and emits the wasm binary (delegated to trunk-ir-wasm-backend)
#[salsa::tracked]
fn compile_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> WasmCompilationResult<WasmBinary<'db>> {
    // Phase 1 - Lower to wasm dialect (Tribute-specific)
    let lowered = lower_to_wasm(db, module);

    // Phase 2 - Resolve unrealized_conversion_cast operations (Tribute-specific type converter)
    let lowered = {
        let _span = tracing::info_span!("resolve_unrealized_casts").entered();
        let type_converter = wasm_type_converter();
        let result = resolve_unrealized_casts(db, lowered, &type_converter);
        if !result.unresolved.is_empty() {
            return Err(CompilationError::unresolved_casts(
                trunk_ir::conversion::UnresolvedCastError {
                    unresolved: result.unresolved,
                },
            ));
        }
        result.module
    };

    // Phase 3 - Validate and emit (language-agnostic, delegated to trunk-ir-wasm-backend)
    let _span = tracing::info_span!("emit_module_to_wasm").entered();
    trunk_ir_wasm_backend::emit_module_to_wasm(db, lowered)
}

/// Lower to WebAssembly binary.
///
/// This stage compiles the fully-typed, resolved TrunkIR module to WebAssembly binary.
/// Returns None if compilation fails, with error message accumulated.
#[salsa::tracked]
pub fn stage_lower_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Option<WasmBinary<'db>> {
    match compile_to_wasm(db, module) {
        Ok(binary) => Some(binary),
        Err(e) => {
            // Accumulate error as diagnostic for reporting
            Diagnostic {
                message: format!("WebAssembly compilation failed: {}", e),
                span: Span::new(0, 0), // TODO: Extract span from error
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(db);
            None
        }
    }
}

// =============================================================================
// Pipeline Entry Points (SourceCst → Module)
// =============================================================================
//
// These functions take SourceCst and run the pipeline up to a specific stage.
// Useful for testing individual stages or for tools that need intermediate results.

/// Compile for LSP: minimal pipeline preserving source structure.
pub fn compile_for_lsp<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    parse_and_lower_ast(db, source)
}

/// Run pipeline up to evidence params stage (lambdas are now lowered directly in ast_to_ir).
#[salsa::tracked]
pub fn run_lambda_lift<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = parse_and_lower_ast(db, source);
    // Boxing removed - now handled via unrealized_conversion_cast in ast_to_ir
    stage_evidence_params(db, module)
}

/// Run pipeline up to closure lower stage.
#[salsa::tracked]
pub fn run_closure_lower<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = run_lambda_lift(db, source);
    stage_closure_lower(db, module)
}

// =============================================================================
// Full Pipeline (Orchestration)
// =============================================================================

/// Run the shared middle-end pipeline (backend-independent).
///
/// Runs frontend + closure processing, evidence call transformation,
/// and evidence-based dispatch resolution. This produces a module with
/// `cont.*` operations that backend-specific pipelines then lower.
#[salsa::tracked]
pub fn run_shared_pipeline<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    // Frontend + closure processing
    let module = run_closure_lower(db, source);

    // DEBUG: List all operations and nested modules after closure lower
    {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::func;
        let body = module.body(db);
        let mut func_names = Vec::new();
        let mut op_summary = Vec::new();
        for block in body.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let d = op.dialect(db);
                let n = op.name(db);
                op_summary.push(format!("{}.{}", d, n));
                if let Ok(f) = func::Func::from_operation(db, *op) {
                    func_names.push(format!("{}", f.sym_name(db)));
                }
                // Check nested core.module
                if d == "core" && n == "module" {
                    for region in op.regions(db).iter() {
                        for nested_block in region.blocks(db).iter() {
                            for nested_op in nested_block.operations(db).iter() {
                                let nd = nested_op.dialect(db);
                                let nn = nested_op.name(db);
                                if let Ok(f) = func::Func::from_operation(db, *nested_op) {
                                    func_names.push(format!("  nested: {}", f.sym_name(db)));
                                } else {
                                    op_summary.push(format!("  nested: {}.{}", nd, nn));
                                }
                            }
                        }
                    }
                }
            }
        }
        tracing::debug!(
            "After closure_lower: {} top ops, {} funcs",
            op_summary.len(),
            func_names.len()
        );
        tracing::debug!(
            "  Operations: {:?}",
            &op_summary[..op_summary.len().min(40)]
        );
        tracing::debug!("  Functions: {:?}", func_names);
    }

    // Evidence call transformation (Phase 2) - AFTER lambda/closure lowering
    let module = stage_evidence_calls(db, module);

    // Evidence-based dispatch resolution - transforms cont.shift to use dynamic tags
    stage_resolve_evidence(db, module)
}

/// Run the WASM pipeline: shared passes + cont_to_trampoline + DCE + resolve_casts.
///
/// This produces a module ready for WASM backend lowering (trampoline-based
/// continuation implementation).
#[salsa::tracked]
pub fn run_wasm_pipeline<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Result<Module<'db>, ConversionError> {
    let module = run_shared_pipeline(db, source);

    // DEBUG: List all functions before cont_to_trampoline
    {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::func;
        let body = module.body(db);
        let mut func_names = Vec::new();
        for block in body.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if let Ok(f) = func::Func::from_operation(db, *op) {
                    func_names.push(format!("{}", f.sym_name(db)));
                }
            }
        }
        tracing::debug!(
            "Before cont_to_trampoline: {} functions: {:?}",
            func_names.len(),
            func_names
        );
    }

    // Continuation lowering (trampoline/yield-bubbling strategy for WASM)
    let module = stage_cont_to_trampoline(db, module)?;

    // DEBUG: List all functions before DCE
    {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::func;
        let body = module.body(db);
        let mut func_names = Vec::new();
        for block in body.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if let Ok(f) = func::Func::from_operation(db, *op) {
                    func_names.push(format!("{}", f.sym_name(db)));
                }
            }
        }
        tracing::debug!(
            "Before DCE: {} functions: {:?}",
            func_names.len(),
            func_names
        );
    }

    let module = stage_dce(db, module);
    Ok(stage_resolve_casts(db, module))
}

/// Run the native pipeline: shared passes + cont lowering + DCE + resolve_casts.
///
/// Uses `cont_to_libmprompt` for native delimited continuations via libmprompt FFI.
#[salsa::tracked]
pub fn run_native_pipeline<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Result<Module<'db>, ConversionError> {
    let module = run_shared_pipeline(db, source);

    let module = stage_cont_to_libmprompt(db, module)?;
    trunk_ir::validation::debug_assert_value_integrity(db, module, "cont_to_libmprompt");

    let module = stage_evidence_to_native(db, module);
    trunk_ir::validation::debug_assert_value_integrity(db, module, "evidence_to_native");

    let module = stage_dce(db, module);
    Ok(stage_resolve_casts(db, module))
}

/// Run the full middle-end pipeline.
///
/// Delegates to `run_wasm_pipeline`. Prefer using the backend-specific
/// `run_wasm_pipeline` or `run_native_pipeline` directly.
#[salsa::tracked]
pub fn run_full_pipeline<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Result<Module<'db>, ConversionError> {
    run_wasm_pipeline(db, source)
}

/// Compile to WebAssembly binary.
///
/// Runs the WASM pipeline and then lowers to WebAssembly.
#[salsa::tracked]
pub fn compile_to_wasm_binary<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<WasmBinary<'db>> {
    let module = match run_wasm_pipeline(db, source) {
        Ok(m) => m,
        Err(e) => {
            Diagnostic {
                message: format!("Pipeline failed: {}", e),
                span: Span::new(0, 0),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(db);
            return None;
        }
    };

    // Lower to WebAssembly
    stage_lower_to_wasm(db, module)
}

// =============================================================================
// Native Pipeline (Cranelift)
// =============================================================================

/// Compile a TrunkIR module to a native object file.
///
/// This function:
/// 1. Lowers `func.*` operations to `clif.*`
/// 2. Lowers `arith.*` operations to `clif.*`
/// 3. Resolves unrealized conversion casts using native type converter
/// 4. Validates and emits the native object file via Cranelift
fn compile_module_to_native<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    sanitize: bool,
) -> NativeCompilationResult<Vec<u8>> {
    use tribute_passes::native_type_converter;
    use trunk_ir::transforms::lower_scf_to_cf;

    // Phase -1 - Generate native entrypoint (wrap user's main)
    let module = generate_native_entrypoint(db, module, sanitize);

    // Phase 0 - Lower structured control flow to CFG-based control flow
    let module = lower_scf_to_cf(db, module);
    trunk_ir::validation::debug_assert_value_integrity(db, module, "lower_scf_to_cf");

    // Phase 1 - Lower func dialect to clif dialect
    let module = func_to_clif::lower(db, module, native_type_converter())
        .map_err(trunk_ir_cranelift_backend::CompilationError::ir_validation)?;

    trunk_ir::validation::debug_assert_value_integrity(db, module, "func_to_clif");

    // Phase 1.5 - Lower cf dialect to clif dialect
    let module = cf_to_clif::lower(db, module, native_type_converter())
        .map_err(trunk_ir_cranelift_backend::CompilationError::ir_validation)?;
    trunk_ir::validation::debug_assert_value_integrity(db, module, "cf_to_clif");

    // Phase 1.9 - RTTI pass: assign rtti_idx per struct type, generate release functions
    let (module, rtti_map) =
        tribute_passes::native::rtti::generate_rtti(db, module, &native_type_converter());
    trunk_ir::validation::debug_assert_value_integrity(db, module, "generate_rtti");

    // Phase 1.95 - Lower adt.struct_new to clif with RC header (Tribute-specific)
    let module = tribute_passes::native::adt_rc_header::lower(
        db,
        module,
        native_type_converter(),
        &rtti_map.type_to_idx,
    );
    trunk_ir::validation::debug_assert_value_integrity(db, module, "adt_rc_header");

    // Phase 2 - Lower ADT struct access operations to clif dialect (struct_get/struct_set)
    let module = adt_to_clif::lower(db, module, native_type_converter())
        .map_err(trunk_ir_cranelift_backend::CompilationError::ir_validation)?;
    trunk_ir::validation::debug_assert_value_integrity(db, module, "adt_to_clif");

    // Phase 2.5 - Lower arith dialect to clif dialect
    let module = arith_to_clif::lower(db, module, native_type_converter())
        .map_err(trunk_ir_cranelift_backend::CompilationError::ir_validation)?;
    trunk_ir::validation::debug_assert_value_integrity(db, module, "arith_to_clif");

    // Phase 2.7 - Lower tribute_rt boxing/unboxing operations to clif dialect
    let module =
        tribute_passes::native::tribute_rt_to_clif::lower(db, module, native_type_converter());
    trunk_ir::validation::debug_assert_value_integrity(db, module, "tribute_rt_to_clif");

    // Phase 2.8 - Insert reference counting (retain/release) for pointer values
    let module = tribute_passes::native::rc_insertion::insert_rc(db, module);
    trunk_ir::validation::debug_assert_value_integrity(db, module, "rc_insertion");

    // Phase 2.85 - Rewrite continuation ops to use RC-safe wrappers
    let module = tribute_passes::native::cont_rc::rewrite_cont_rc(db, module);
    trunk_ir::validation::debug_assert_value_integrity(db, module, "cont_rc");

    // Phase 3 - Resolve unrealized_conversion_cast operations
    let module = {
        let type_converter = native_type_converter();
        let result = resolve_unrealized_casts(db, module, &type_converter);
        if !result.unresolved.is_empty() {
            let details: Vec<String> = result
                .unresolved
                .iter()
                .map(|c| {
                    format!(
                        "{}.{} -> {}.{}",
                        c.from_type.dialect(db),
                        c.from_type.name(db),
                        c.to_type.dialect(db),
                        c.to_type.name(db),
                    )
                })
                .collect();
            return Err(trunk_ir_cranelift_backend::CompilationError::ir_validation(
                format!(
                    "{} unresolved cast(s) remain after native type conversion: [{}]",
                    result.unresolved.len(),
                    details.join(", "),
                ),
            ));
        }
        result.module
    };
    trunk_ir::validation::debug_assert_value_integrity(db, module, "resolve_unrealized_casts");

    // Phase 3.5 - Lower RC operations (retain/release) to inline clif code
    let module = tribute_passes::native::rc_lowering::lower_rc(db, module);
    trunk_ir::validation::debug_assert_value_integrity(db, module, "lower_rc");

    // Phase 4 - Validate and emit (delegated to trunk-ir-cranelift-backend)
    let _span = tracing::info_span!("emit_module_to_native").entered();
    emit_module_to_native(db, module)
}

/// Compile to native object bytes.
///
/// Runs the native pipeline and then lowers to a native object file via Cranelift.
/// Returns `None` if compilation fails, with diagnostics accumulated.
#[salsa::tracked]
pub fn compile_to_native_binary<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    config: CompilationConfig,
) -> Option<Vec<u8>> {
    let module = match run_native_pipeline(db, source) {
        Ok(m) => m,
        Err(e) => {
            Diagnostic {
                message: format!("Pipeline failed: {}", e),
                span: Span::new(0, 0),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(db);
            return None;
        }
    };

    let sanitize = config.sanitize_address(db);
    match compile_module_to_native(db, module, sanitize) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            Diagnostic {
                message: format!("Native compilation failed: {}", e),
                span: Span::new(0, 0),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Lowering,
            }
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
// This replaces the legacy tirgen-based pipeline that worked directly with IR.

/// Parse and lower source to TrunkIR using the AST pipeline.
///
/// This is the AST-based alternative to `parse_and_lower`.
/// Returns a module with resolved types, ready for further lowering passes.
///
/// Uses the Type Info Injection approach to make prelude types available:
/// 1. Parse user code to AST
/// 2. Merge prelude bindings into user's ModuleEnv
/// 3. Resolve names with merged environment
/// 4. Inject prelude TypeSchemes into user's ModuleTypeEnv
/// 5. Type check with injected types
/// 6. Run TDNR
/// 7. Lower to TrunkIR
/// 8. Merge prelude IR for implementations (still needed for now)
#[salsa::tracked]
pub fn parse_and_lower_ast<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    // Phase 1: Parse user code to AST
    let Some(parsed) = ast_query::parsed_ast(db, source) else {
        let path = trunk_ir::PathId::new(db, source.uri(db).as_str().to_owned());
        let location = trunk_ir::Location::new(path, trunk_ir::Span::new(0, 0));
        let module_name = derive_module_name_from_path(db, path);
        return Module::build(db, location, module_name, |_| {});
    };

    let user_ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();
    tracing::debug!(
        "Phase 1: parsed AST has {} declarations",
        user_ast.decls.len()
    );

    // Phase 2: Build user env and merge prelude bindings
    let mut user_env = ast_resolve::build_env(db, &user_ast);
    if let Some(p_env) = prelude_env(db) {
        user_env.merge(&p_env); // Prelude bindings injected, user definitions take precedence
    }

    // Phase 3: Name resolution with merged environment
    let resolved_ast = ast_resolve::resolve_with_env(db, user_ast, user_env, span_map.clone());

    // Phase 4: Type checking with prelude types injected
    let mut checker = ast_typeck::TypeChecker::new(db, span_map.clone());
    if let Some(p_exports) = prelude_exports(db) {
        checker.inject_prelude(&p_exports); // Prelude TypeSchemes injected (no UniVars)
    }
    let result = checker.check_module(resolved_ast);

    tracing::debug!(
        "Phase 4: after typecheck, {} declarations, {} function_types, {} node_types",
        result.module.decls.len(),
        result.function_types.len(),
        result.node_types.len()
    );

    // Phase 5: TDNR (Type-Directed Name Resolution)
    let tdnr_ast = ast_tdnr::resolve_tdnr(db, result.module);
    tracing::debug!("Phase 5: after TDNR, {} declarations", tdnr_ast.decls.len());

    // Phase 6: AST → TrunkIR
    let function_types: std::collections::HashMap<_, _> =
        result.function_types.into_iter().collect();
    let node_types: std::collections::HashMap<_, _> = result.node_types.into_iter().collect();
    let source_uri = source.uri(db).as_str();
    let user_module = ast_to_ir::lower_ast_to_ir(
        db,
        tdnr_ast,
        span_map,
        source_uri,
        function_types,
        node_types,
    );

    // DEBUG: List user module functions before prelude merge
    {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::func;
        let body = user_module.body(db);
        let mut func_names = Vec::new();
        let mut all_ops = Vec::new();
        for block in body.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let d = op.dialect(db);
                let n = op.name(db);
                all_ops.push(format!("{}.{}", d, n));
                if let Ok(f) = func::Func::from_operation(db, *op) {
                    func_names.push(format!("{}", f.sym_name(db)));
                }
            }
        }
        tracing::debug!(
            "User module before merge: {} ops, {} funcs",
            all_ops.len(),
            func_names.len()
        );
        tracing::debug!("  All ops: {:?}", &all_ops[..all_ops.len().min(20)]);
        tracing::debug!("  Functions: {:?}", func_names);
    }

    // Phase 7: Merge prelude at TrunkIR level
    // Still needed for prelude implementations (function bodies, struct layouts)
    merge_with_prelude(db, user_module)
}

/// Compile using the AST-based pipeline.
///
/// The AST pipeline provides better type safety and separation of concerns.
/// Name resolution, type checking, and TDNR are performed on AST nodes,
/// then lowered to TrunkIR for further transformations.
#[salsa::tracked]
pub fn compile_ast<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Result<Module<'db>, ConversionError> {
    run_full_pipeline(db, source)
}

/// Run compilation and return detailed results including diagnostics.
///
/// Diagnostics are collected using Salsa accumulators from all compilation stages.
pub fn compile_with_diagnostics<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> CompilationResult<'db> {
    // Run the full compilation pipeline
    // Type checking is performed inside the tracked `compile_ast` function,
    // so diagnostics are properly accumulated via Salsa.
    let result = compile_ast(db, source);

    // Collect all accumulated diagnostics from the compilation
    let mut diagnostics: Vec<Diagnostic> = compile_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect();

    let module = match result {
        Ok(module) => module,
        Err(err) => {
            // Add conversion error as diagnostic
            diagnostics.push(Diagnostic {
                message: format!("{}", err),
                span: trunk_ir::Span::new(0, 0),
                severity: tribute_passes::DiagnosticSeverity::Error,
                phase: tribute_passes::CompilationPhase::Lowering,
            });
            // Return a minimal module on error
            let path = trunk_ir::PathId::new(db, source.uri(db).as_str().to_owned());
            let location = trunk_ir::Location::new(path, trunk_ir::Span::new(0, 0));
            let empty_body = trunk_ir::Region::new(db, location, trunk_ir::IdVec::from(Vec::new()));
            Module::create(db, location, trunk_ir::Symbol::new("error"), empty_body)
        }
    };

    CompilationResult {
        module,
        diagnostics,
    }
}

// =============================================================================
// Linking
// =============================================================================

/// Errors that can occur during native binary linking.
#[derive(Debug, derive_more::Display)]
pub enum LinkError {
    #[display("failed to create temporary file: {_0}")]
    TempFile(std::io::Error),
    #[display("failed to invoke linker (cc): {_0}")]
    LinkerNotFound(std::io::Error),
    #[display("linker failed with exit code {_0}")]
    LinkerFailed(i32),
}

impl std::error::Error for LinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LinkError::TempFile(e) | LinkError::LinkerNotFound(e) => Some(e),
            LinkError::LinkerFailed(_) => None,
        }
    }
}

/// Link native object bytes into an executable.
///
/// Writes object bytes to a temp file and invokes the system linker (`cc`),
/// linking against the tribute runtime library (which includes libmprompt).
pub fn link_native_binary(object_bytes: &[u8], output: &Path) -> Result<(), LinkError> {
    let obj_file = tempfile::Builder::new()
        .suffix(".o")
        .tempfile()
        .map_err(LinkError::TempFile)?;
    std::fs::write(obj_file.path(), object_bytes).map_err(LinkError::TempFile)?;

    let mut cmd = std::process::Command::new("cc");
    cmd.arg(obj_file.path());

    // Link tribute-runtime staticlib (bundles both Rust runtime and libmprompt).
    // The library directory is set at build time by build.rs.
    if let Some(static_lib_dir) = option_env!("TRIBUTE_RUNTIME_STATIC_LIB_DIR") {
        cmd.arg("-L").arg(static_lib_dir);
        cmd.arg("-ltribute_runtime");
    }
    if !cfg!(windows) {
        cmd.arg("-lpthread");
    }

    cmd.arg("-o").arg(output);

    let status = cmd.status().map_err(LinkError::LinkerNotFound)?;

    if !status.success() {
        return Err(LinkError::LinkerFailed(status.code().unwrap_or(-1)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;

    #[salsa::tracked]
    fn test_compile(db: &dyn salsa::Database, source: SourceCst) -> Module<'_> {
        compile_ast(db, source).expect("compilation should succeed")
    }

    fn source_from_str(path: &str, text: &str) -> SourceCst {
        salsa::with_attached_database(|db| {
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(text, None).expect("tree");
            SourceCst::from_path(db, path, text.into(), Some(tree))
        })
        .expect("attached db")
    }

    #[salsa_test]
    fn test_full_pipeline(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn compute() -> Int { 42 }\nfn main() { }");

        let module = test_compile(db, source);
        assert_eq!(module.name(db), "test");
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
            d.message.contains("unresolved")
                && d.severity == DiagnosticSeverity::Error
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
            d.severity == DiagnosticSeverity::Error
                && d.phase == CompilationPhase::Parsing
                && d.message.contains("syntax error")
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
    fn test_prelude_option_type(db: &salsa::DatabaseImpl) {
        // Use Option type from prelude
        let source = source_from_str("test.trb", "fn maybe() -> Option(Int) { None }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without "unresolved" errors for Option or None
        let has_option_error = result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("Option") || d.message.contains("None"));
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
            .any(|d| d.message.contains("Result") || d.message.contains("Ok"));
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
            .any(|d| d.message.contains("unresolved") && d.message.contains("y"));
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
            d.message.contains("non-exhaustive") && d.severity == DiagnosticSeverity::Error
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
            .any(|d| d.message.contains("unresolved") && d.message.contains("name"));
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
        // Test that AST pipeline can parse and lower a simple function
        let source = source_from_str("test.trb", "fn compute() -> Int { 42 }\nfn main() { }");

        let module = parse_and_lower_ast(db, source);
        assert_eq!(module.name(db), "test");
    }

    #[salsa_test]
    fn test_ast_pipeline_with_params(db: &salsa::DatabaseImpl) {
        // Test that AST pipeline handles function parameters
        let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

        let module = parse_and_lower_ast(db, source);
        assert_eq!(module.name(db), "test");
    }

    #[salsa_test]
    fn test_ast_pipeline_let_binding(db: &salsa::DatabaseImpl) {
        // Test that AST pipeline handles let bindings
        let source = source_from_str(
            "test.trb",
            r#"
            fn main() -> Int {
                let x = 10;
                let y = 20;
                x + y
            }
            "#,
        );

        let module = parse_and_lower_ast(db, source);
        assert_eq!(module.name(db), "test");
    }

    #[salsa_test]
    fn test_ast_pipeline_struct(db: &salsa::DatabaseImpl) {
        // Test that AST pipeline handles struct definitions
        let source = source_from_str(
            "test.trb",
            r#"
            struct Point {
                x: Int,
                y: Int,
            }

            fn main() -> Int { 0 }
            "#,
        );

        let module = parse_and_lower_ast(db, source);
        assert_eq!(module.name(db), "test");
    }

    // =========================================================================
    // main function return type validation
    // =========================================================================

    #[salsa_test]
    fn test_main_returns_nil_ok(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn main() { }");

        let result = compile_with_diagnostics(db, source);
        let has_main_error = result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("must return Nil"));
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
            d.message.contains("must return Nil")
                && d.severity == DiagnosticSeverity::Error
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
            .any(|d| d.message.contains("must return Nil"));
        assert!(
            !has_main_error,
            "non-main function returning Int should not produce main-specific error, got: {:?}",
            result.diagnostics
        );
    }

    // =========================================================================
    // LinkError & link_native_binary Tests
    // =========================================================================

    #[test]
    fn test_link_error_display_temp_file() {
        let err = LinkError::TempFile(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        ));
        let msg = err.to_string();
        assert!(msg.contains("failed to create temporary file"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_link_error_display_linker_not_found() {
        let err = LinkError::LinkerNotFound(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not found",
        ));
        let msg = err.to_string();
        assert!(msg.contains("failed to invoke linker (cc)"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_link_error_display_linker_failed() {
        let err = LinkError::LinkerFailed(1);
        assert_eq!(err.to_string(), "linker failed with exit code 1");
    }

    #[test]
    fn test_link_error_source() {
        use std::error::Error;

        let io_err = LinkError::TempFile(std::io::Error::other("test"));
        assert!(io_err.source().is_some());

        let linker_err = LinkError::LinkerFailed(1);
        assert!(linker_err.source().is_none());
    }

    #[test]
    fn test_link_native_binary_invalid_object() {
        // Passing garbage bytes should cause the linker to fail
        let output = std::env::temp_dir().join("tribute_test_invalid_link");
        let result = link_native_binary(b"not valid object code", &output);
        assert!(result.is_err());
        match result.unwrap_err() {
            LinkError::LinkerNotFound(_) => {
                eprintln!("warning: system linker (cc) not found, skipping test");
                return;
            }
            LinkError::LinkerFailed(code) => {
                assert_ne!(code, 0, "linker should fail with non-zero exit code");
            }
            other => panic!("expected LinkerFailed, got: {other}"),
        }
        // Clean up in case it somehow succeeded
        let _ = std::fs::remove_file(&output);
    }
}
