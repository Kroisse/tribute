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
//!     ├─── Shared Pipeline (single arena session) ────┤
//!     ▼ evidence_params (Phase 1)
//! Module (evidence params added to signatures)
//!     │
//!     ▼ closure_lower
//! Module (closure.* lowered)
//!     │
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
use tribute_passes::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_passes::evidence;
use tribute_passes::generic_type_converter_arena;
use tribute_passes::lower_cont_to_trampoline;
use trunk_ir::DialectOp;
use trunk_ir::Span;
use trunk_ir::arena::bridge::export_to_salsa;
use trunk_ir::arena::{ArenaModule, IrContext};
use trunk_ir::conversion::resolve_unrealized_casts_arena;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::ConversionError;

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

/// Load and cache the prelude's typed AST using the AST pipeline.
///
/// This is a Salsa tracked function, so the prelude is parsed only once
/// and cached for all subsequent compilations.
///
/// Returns the typed AST (parse → resolve → typecheck → TDNR) without
/// lowering to TrunkIR. The caller is responsible for `ast_to_ir`.
#[salsa::tracked]
fn prelude_module<'db>(db: &'db dyn salsa::Database) -> Option<ast_typeck::TypeCheckOutput<'db>> {
    let (resolved, span_map, _prelude_source) = resolve_prelude(db)?;

    // Typecheck with independent TypeContext
    let checker = ast_typeck::TypeChecker::new(db, span_map.clone());
    let result = checker.check_module(resolved);

    // TDNR
    let tdnr_ast = ast_tdnr::resolve_tdnr(db, result.module);

    Some(ast_typeck::TypeCheckOutput::new(
        db,
        tdnr_ast,
        result.function_types,
        result.node_types,
        span_map,
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
fn prelude_env<'db>(db: &'db dyn salsa::Database) -> Option<ModuleEnv<'db>> {
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
fn prelude_exports<'db>(db: &'db dyn salsa::Database) -> Option<PreludeExports<'db>> {
    let (resolved, span_map, _) = resolve_prelude(db)?;

    // Typecheck with independent TypeContext (all UniVars resolved)
    let checker = ast_typeck::TypeChecker::new(db, span_map);
    let prelude_exports = checker.check_module_for_prelude(resolved);

    Some(prelude_exports)
}

/// Merge prelude decls into user's typed AST and lower to arena IR.
///
/// This performs AST-level prelude merge (prepending prelude decls to user decls),
/// then runs `ast_to_ir` on the merged module.
///
/// Returns `(IrContext, ArenaModule)` — arena IR ready for in-place passes.
fn merge_and_lower_to_ir<'db>(
    db: &'db dyn salsa::Database,
    typed: &ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> (IrContext, ArenaModule) {
    use tribute_front::ast::TypedRef;

    let user_module = typed.module(db);
    let user_fn_types = typed.function_types(db);
    let user_node_types = typed.node_types(db);
    let user_span_map = typed.span_map(db);

    // Merge prelude at AST level
    let (merged_module, merged_fn_types, merged_node_types, merged_span_map) =
        if let Some(prelude) = prelude_module(db) {
            let prelude_module_ast = prelude.module(db);
            let prelude_fn_types = prelude.function_types(db);
            let prelude_node_types = prelude.node_types(db);
            let prelude_span_map = prelude.span_map(db);

            // Prepend prelude decls before user decls
            let mut merged_decls = prelude_module_ast.decls.clone();
            merged_decls.extend(user_module.decls.iter().cloned());

            let merged_ast = tribute_front::ast::Module::<TypedRef<'db>>::new(
                user_module.id,
                user_module.name,
                merged_decls,
            );

            // Merge function_types: prelude first, user overrides
            let mut fn_types: std::collections::HashMap<_, _> =
                prelude_fn_types.iter().cloned().collect();
            fn_types.extend(user_fn_types.iter().cloned());

            // Merge node_types: prelude first, user overrides
            let mut node_types: std::collections::HashMap<_, _> =
                prelude_node_types.iter().cloned().collect();
            node_types.extend(user_node_types.iter().cloned());

            // Merge span maps (user overrides prelude on conflict)
            let merged_span_map = user_span_map.merge(&prelude_span_map);

            (merged_ast, fn_types, node_types, merged_span_map)
        } else {
            let fn_types: std::collections::HashMap<_, _> = user_fn_types.iter().cloned().collect();
            let node_types: std::collections::HashMap<_, _> =
                user_node_types.iter().cloned().collect();
            (
                user_module.clone(),
                fn_types,
                node_types,
                user_span_map.clone(),
            )
        };

    // AST → TrunkIR (arena)
    let source_uri = source.uri(db).as_str();
    let mut ir = IrContext::new();
    let arena_module = ast_to_ir::lower_ast_to_ir(
        db,
        &mut ir,
        merged_module,
        merged_span_map,
        source_uri,
        merged_fn_types,
        merged_node_types,
    );

    (ir, arena_module)
}

/// Run frontend (parse → typecheck → TDNR) and lower to arena IR.
///
/// Returns `None` if parsing fails. Otherwise returns arena IR ready
/// for in-place passes, avoiding unnecessary Salsa↔Arena round-trips.
fn compile_frontend_to_arena(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<(IrContext, ArenaModule)> {
    let typed = parse_and_lower_ast(db, source)?;
    Some(merge_and_lower_to_ir(db, &typed, source))
}

/// Build an empty Salsa `Module` for error/fallback cases.
fn empty_module<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let path = trunk_ir::PathId::new(db, source.uri(db).as_str().to_owned());
    let location = trunk_ir::Location::new(path, trunk_ir::Span::new(0, 0));
    let module_name = derive_module_name_from_path(db, path);
    Module::build(db, location, module_name, |_| {})
}

/// Compile frontend: parse → resolve → typecheck → TDNR → merge prelude → ast_to_ir.
///
/// Returns a Salsa `Module<'db>` for callers that need Salsa-interned IR
/// (LSP, tests). Pipeline functions should prefer `compile_frontend_to_arena`
/// to avoid an extra Salsa↔Arena round-trip.
#[salsa::tracked]
pub fn compile_frontend<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let Some((ir, arena_module)) = compile_frontend_to_arena(db, source) else {
        return empty_module(db, source);
    };

    let exported = export_to_salsa(db, &ir, arena_module);
    Module::from_operation(db, exported).unwrap()
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

/// Compile a TrunkIR module to WebAssembly binary (arena-based).
///
/// Runs all WASM backend passes in a single arena session:
/// 1. Lowers the module from func/scf/arith dialects to wasm dialect operations
/// 2. Resolves unrealized conversion casts using WASM-specific type converter
/// 3. Validates and emits the wasm binary (delegated to trunk-ir-wasm-backend)
fn compile_to_wasm_arena(
    ctx: &mut IrContext,
    arena_module: ArenaModule,
) -> WasmCompilationResult<WasmBinary> {
    let _span = tracing::info_span!("compile_to_wasm").entered();

    // Phase 1 - Lower to wasm dialect (Tribute-specific)
    {
        let _span = tracing::info_span!("lower_to_wasm").entered();
        tribute_passes::wasm::lower::lower_to_wasm_arena(ctx, arena_module);
    }

    // Phase 2 - Resolve unrealized_conversion_cast operations (WASM type converter)
    {
        let _span = tracing::info_span!("resolve_unrealized_casts").entered();
        let tc = tribute_passes::wasm::type_converter::wasm_type_converter(ctx);
        let result = resolve_unrealized_casts_arena(ctx, arena_module, &tc);
        if !result.unresolved.is_empty() {
            let details: Vec<String> = result
                .unresolved
                .iter()
                .map(|c| {
                    let from_td = ctx.types.get(c.from_type);
                    let to_td = ctx.types.get(c.to_type);
                    format!(
                        "{}.{} -> {}.{}",
                        from_td.dialect, from_td.name, to_td.dialect, to_td.name,
                    )
                })
                .collect();
            return Err(CompilationError::unresolved_casts(format!(
                "{} unresolved cast(s) remain after WASM type conversion: [{}]",
                result.unresolved.len(),
                details.join(", "),
            )));
        }
    }

    // Phase 3 - Emit WASM binary
    let _span = tracing::info_span!("emit_module_to_wasm").entered();
    trunk_ir_wasm_backend::emit_module_to_wasm_arena(ctx, arena_module)
}

// =============================================================================
// Pipeline Entry Points (SourceCst → Module)
// =============================================================================
//
// These functions take SourceCst and run the pipeline up to a specific stage.
// Useful for testing individual stages or for tools that need intermediate results.

/// Compile for LSP: minimal pipeline preserving source structure.
pub fn compile_for_lsp<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    compile_frontend(db, source)
}

/// Run pipeline through evidence params (for testing).
///
/// Runs frontend + `add_evidence_params` in a single arena session.
#[salsa::tracked]
pub fn run_through_evidence_params<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Module<'db> {
    let Some((mut ctx, m)) = compile_frontend_to_arena(db, source) else {
        return empty_module(db, source);
    };
    evidence::add_evidence_params(&mut ctx, m);
    let exported = export_to_salsa(db, &ctx, m);
    Module::from_operation(db, exported).unwrap()
}

/// Run pipeline through closure lower (for testing).
///
/// Runs frontend + `add_evidence_params` + `lower_closures`
/// in a single arena session.
#[salsa::tracked]
pub fn run_through_closure_lower<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Module<'db> {
    let Some((mut ctx, m)) = compile_frontend_to_arena(db, source) else {
        return empty_module(db, source);
    };
    evidence::add_evidence_params(&mut ctx, m);
    tribute_passes::closure_lower::lower_closures(&mut ctx, m);
    let exported = export_to_salsa(db, &ctx, m);
    Module::from_operation(db, exported).unwrap()
}

// =============================================================================
// Full Pipeline (Orchestration)
// =============================================================================

/// Run the shared middle-end pipeline (backend-independent) in an arena session.
///
/// Runs all four arena passes (evidence_params, closure_lower, evidence_calls,
/// resolve_evidence) in a **single arena session**, avoiding 4 separate
/// Salsa↔Arena round-trips.
///
/// Returns the arena session so that backend-specific pipelines can continue
/// without a Salsa↔Arena round-trip.
fn run_shared_pipeline_arena(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<(IrContext, ArenaModule)> {
    let (mut ctx, m) = compile_frontend_to_arena(db, source)?;

    evidence::add_evidence_params(&mut ctx, m);
    tribute_passes::closure_lower::lower_closures(&mut ctx, m);
    evidence::transform_evidence_calls(&mut ctx, m);
    tribute_passes::resolve_evidence::resolve_evidence_dispatch(&mut ctx, m);

    Some((ctx, m))
}

/// Run the WASM target pipeline in arena and return IR text for dump-ir.
fn run_wasm_target_pipeline_arena(
    ctx: &mut IrContext,
    m: ArenaModule,
) -> Result<(), ConversionError> {
    lower_cont_to_trampoline(ctx, m).map_err(|illegal_ops| ConversionError {
        illegal_ops: illegal_ops
            .into_iter()
            .map(|op| trunk_ir::rewrite::IllegalOp {
                dialect: op.dialect.to_string(),
                name: op.name.to_string(),
            })
            .collect(),
    })?;

    trunk_ir::arena::transforms::global_dce::eliminate_dead_functions(ctx, m);

    let tc = generic_type_converter_arena(ctx);
    resolve_unrealized_casts_arena(ctx, m, &tc);

    Ok(())
}

/// Run the native target pipeline in arena (shared + native-specific passes).
fn run_native_target_pipeline_arena(ctx: &mut IrContext, m: ArenaModule) {
    tribute_passes::cont_to_libmprompt::lower_cont_to_libmprompt(ctx, m);
    if cfg!(debug_assertions) {
        let result = trunk_ir::arena::validation::validate_value_integrity(ctx, m);
        if !result.is_ok() {
            tracing::warn!(
                "Value integrity errors after cont_to_libmprompt: stale={:?}, use_chain={:?}",
                result.stale_errors,
                result.use_chain_errors
            );
        }
    }

    tribute_passes::native::evidence::lower_evidence_to_native(ctx, m);
    if cfg!(debug_assertions) {
        let result = trunk_ir::arena::validation::validate_value_integrity(ctx, m);
        if !result.is_ok() {
            tracing::warn!(
                "Value integrity errors after evidence_to_native: stale={:?}, use_chain={:?}",
                result.stale_errors,
                result.use_chain_errors
            );
        }
    }

    trunk_ir::arena::transforms::global_dce::eliminate_dead_functions(ctx, m);

    let tc = generic_type_converter_arena(ctx);
    resolve_unrealized_casts_arena(ctx, m, &tc);
}

/// Dump IR text after running the pipeline up to the target-specific passes.
///
/// If `native` is true, runs the native pipeline; otherwise runs the WASM pipeline.
/// Returns the IR text as a string, or an error. Diagnostics are accumulated.
#[salsa::tracked]
pub fn dump_ir(
    db: &dyn salsa::Database,
    source: SourceCst,
    native: bool,
) -> Result<String, ConversionError> {
    let Some((mut ctx, m)) = run_shared_pipeline_arena(db, source) else {
        return Ok(String::new());
    };

    if native {
        run_native_target_pipeline_arena(&mut ctx, m);
    } else {
        run_wasm_target_pipeline_arena(&mut ctx, m)?;
    }

    Ok(trunk_ir::arena::printer::print_module(&ctx, m.op()))
}

/// Compile to WebAssembly binary bytes.
///
/// Runs the full pipeline (frontend → shared passes → WASM lowering → emit)
/// in a single arena session, avoiding Salsa↔Arena round-trips after ast_to_ir.
///
/// Returns the raw WASM bytes on success, or `None` with diagnostics accumulated.
#[salsa::tracked]
pub fn compile_to_wasm_binary(db: &dyn salsa::Database, source: SourceCst) -> Option<Vec<u8>> {
    let (mut ctx, m) = run_shared_pipeline_arena(db, source)?;

    // WASM-specific: cont_to_trampoline + DCE + resolve_casts
    if let Err(e) = lower_cont_to_trampoline(&mut ctx, m).map_err(|illegal_ops| ConversionError {
        illegal_ops: illegal_ops
            .into_iter()
            .map(|op| trunk_ir::rewrite::IllegalOp {
                dialect: op.dialect.to_string(),
                name: op.name.to_string(),
            })
            .collect(),
    }) {
        Diagnostic {
            message: format!("Pipeline failed: {}", e),
            span: Span::new(0, 0),
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::Lowering,
        }
        .accumulate(db);
        return None;
    }

    // Dead code elimination
    trunk_ir::arena::transforms::global_dce::eliminate_dead_functions(&mut ctx, m);

    // Resolve unrealized conversion casts
    let tc = generic_type_converter_arena(&mut ctx);
    resolve_unrealized_casts_arena(&mut ctx, m, &tc);

    // WASM backend lowering + emit
    match compile_to_wasm_arena(&mut ctx, m) {
        Ok(binary) => Some(binary.bytes),
        Err(e) => {
            Diagnostic {
                message: format!("WebAssembly compilation failed: {}", e),
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
// Native Pipeline (Cranelift)
// =============================================================================

/// Compile a TrunkIR module to a native object file (arena-based).
///
/// This function runs all backend passes in a single arena session:
/// 1. Generates native entrypoint
/// 2. Lowers scf/func/cf/adt/arith dialects to `clif.*`
/// 3. Runs RTTI, RC insertion, and RC lowering passes
/// 4. Resolves unrealized conversion casts
/// 5. Validates and emits the native object file via Cranelift
fn compile_module_to_native_arena(
    ctx: &mut IrContext,
    arena_module: ArenaModule,
    sanitize: bool,
) -> NativeCompilationResult<Vec<u8>> {
    let _span = tracing::info_span!("compile_module_to_native").entered();

    // Phase -1 - Generate native entrypoint
    tribute_passes::native::entrypoint::generate_native_entrypoint(ctx, arena_module, sanitize);

    // Phase 0 - Lower structured control flow to CFG-based control flow
    trunk_ir::arena::transforms::scf_to_cf::lower_scf_to_cf(ctx, arena_module);

    // Phase 1 - Lower func dialect to clif dialect
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        func_to_clif::lower(ctx, arena_module, type_converter);
    }

    // Phase 1.5 - Lower cf dialect to clif dialect
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        cf_to_clif::lower(ctx, arena_module, type_converter);
    }

    // Phase 1.9-1.95 - RTTI + ADT RC header
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        let rtti_map =
            tribute_passes::native::rtti::generate_rtti(ctx, arena_module, &type_converter);
        tribute_passes::native::adt_rc_header::lower(
            ctx,
            arena_module,
            type_converter,
            &rtti_map.type_to_idx,
        );
    }

    // Phase 2 - Lower ADT struct access operations to clif dialect
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        adt_to_clif::lower(ctx, arena_module, type_converter);
    }

    // Phase 2.5 - Lower arith dialect to clif dialect
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        arith_to_clif::lower(ctx, arena_module, type_converter);
    }

    // Phase 2.7-2.85 - tribute_rt_to_clif + RC insertion + cont RC
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        tribute_passes::native::tribute_rt_to_clif::lower(ctx, arena_module, type_converter);
        tribute_passes::native::rc_insertion::insert_rc(ctx, arena_module);
        tribute_passes::native::cont_rc::rewrite_cont_rc(ctx, arena_module);
    }

    // Phase 3 - Resolve unrealized_conversion_cast operations
    {
        let (type_converter, _) =
            tribute_passes::native::type_converter::native_type_converter_arena(ctx);
        let result = resolve_unrealized_casts_arena(ctx, arena_module, &type_converter);
        if !result.unresolved.is_empty() {
            let details: Vec<String> = result
                .unresolved
                .iter()
                .map(|c| {
                    let from_td = ctx.types.get(c.from_type);
                    let to_td = ctx.types.get(c.to_type);
                    format!(
                        "{}.{} -> {}.{}",
                        from_td.dialect, from_td.name, to_td.dialect, to_td.name,
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
    }

    // Phase 3.5 - Lower RC operations (retain/release) to inline clif code
    tribute_passes::native::rc_lowering::lower_rc(ctx, arena_module);

    // Phase 4 - Validate and emit
    let _emit_span = tracing::info_span!("emit_module_to_native").entered();
    emit_module_to_native(ctx, arena_module)
}

/// Compile to native object bytes.
///
/// Runs the full pipeline (frontend → shared passes → native lowering → emit)
/// in a single arena session, avoiding Salsa↔Arena round-trips after ast_to_ir.
///
/// Returns `None` if compilation fails, with diagnostics accumulated.
#[salsa::tracked]
pub fn compile_to_native_binary<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    config: CompilationConfig,
) -> Option<Vec<u8>> {
    let (mut ctx, m) = run_shared_pipeline_arena(db, source)?;

    // Native-specific: cont_to_libmprompt + evidence_to_native + DCE + resolve_casts
    tribute_passes::cont_to_libmprompt::lower_cont_to_libmprompt(&mut ctx, m);
    if cfg!(debug_assertions) {
        let result = trunk_ir::arena::validation::validate_value_integrity(&ctx, m);
        if !result.is_ok() {
            tracing::warn!(
                "Value integrity errors after cont_to_libmprompt: stale={:?}, use_chain={:?}",
                result.stale_errors,
                result.use_chain_errors
            );
        }
    }

    tribute_passes::native::evidence::lower_evidence_to_native(&mut ctx, m);
    if cfg!(debug_assertions) {
        let result = trunk_ir::arena::validation::validate_value_integrity(&ctx, m);
        if !result.is_ok() {
            tracing::warn!(
                "Value integrity errors after evidence_to_native: stale={:?}, use_chain={:?}",
                result.stale_errors,
                result.use_chain_errors
            );
        }
    }

    // Dead code elimination
    trunk_ir::arena::transforms::global_dce::eliminate_dead_functions(&mut ctx, m);

    // Resolve unrealized conversion casts
    let tc = generic_type_converter_arena(&mut ctx);
    resolve_unrealized_casts_arena(&mut ctx, m, &tc);

    // Native backend lowering + emit
    let sanitize = config.sanitize_address(db);
    match compile_module_to_native_arena(&mut ctx, m, sanitize) {
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
#[salsa::tracked]
pub fn parse_and_lower_ast<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<ast_typeck::TypeCheckOutput<'db>> {
    // Phase 1: Parse user code to AST
    let parsed = ast_query::parsed_ast(db, source)?;

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

    Some(ast_typeck::TypeCheckOutput::new(
        db,
        tdnr_ast,
        result.function_types,
        result.node_types,
        span_map,
    ))
}

/// Compile using the AST-based pipeline.
///
/// Runs the shared pipeline (frontend + evidence/closure/evidence-calls/
/// resolve-evidence passes) and exports to Salsa Module. Does NOT run
/// target-specific passes (WASM/native), making it suitable for
/// diagnostic-only compilation ("none" target) and LSP.
#[salsa::tracked]
pub fn compile_ast<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let Some((ctx, m)) = run_shared_pipeline_arena(db, source) else {
        return empty_module(db, source);
    };

    let exported = export_to_salsa(db, &ctx, m);
    Module::from_operation(db, exported).unwrap()
}

/// Run compilation and return detailed results including diagnostics.
///
/// Diagnostics are collected using Salsa accumulators from all compilation stages.
pub fn compile_with_diagnostics<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> CompilationResult<'db> {
    let module = compile_ast(db, source);

    // Collect all accumulated diagnostics from the compilation
    let diagnostics: Vec<Diagnostic> = compile_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect();

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
        compile_ast(db, source)
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

        let module = compile_frontend(db, source);
        assert_eq!(module.name(db), "test");
    }

    #[salsa_test]
    fn test_ast_pipeline_with_params(db: &salsa::DatabaseImpl) {
        // Test that AST pipeline handles function parameters
        let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

        let module = compile_frontend(db, source);
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

        let module = compile_frontend(db, source);
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

        let module = compile_frontend(db, source);
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
