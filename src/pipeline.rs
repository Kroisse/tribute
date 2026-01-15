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
//!     ▼ inline_constants
//! Module (const values inlined)
//!     │
//!     ▼ typecheck
//! Typed Module
//!     │
//!     ▼ boxing
//! Module (boxing explicit)
//!     │
//!     ├─────────────── Closure Processing ────────────┤
//!     ▼ lambda_lift
//! Module (lambdas lifted)
//!     │
//!     ▼ closure_lower
//! Module (closure.* lowered)
//!     │
//!     ▼ tdnr
//! Module (UFCS resolved)
//!     │
//!     ├─────────────── Ability Processing ────────────┤
//!     ▼ evidence_insert
//! Module (evidence params added)
//!     │
//!     ▼ handler_lower
//! Module (ability.* → cont.*)
//!     │
//!     ├─────────────── Final Lowering ────────────────┤
//!     ▼ lower_case
//! Module (case → scf.if)
//!     │
//!     ▼ dce
//! Core IR Module
//!     │
//!     └─► [target: wasm] ─► compile_to_wasm ─► WasmBinary
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
use tree_sitter::Parser;
use tribute_front::source_file::parse_with_rope;
use tribute_front::{lower_cst, parse_cst};
use tribute_passes::boxing::insert_boxing;
use tribute_passes::closure_lower::lower_closures;
use tribute_passes::const_inline::inline_module;
use tribute_passes::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_passes::evidence::insert_evidence;
use tribute_passes::handler_lower::lower_handlers;
use tribute_passes::lambda_lift::lift_lambdas;
use tribute_passes::lower_cont_to_trampoline;
use tribute_passes::lower_trampoline_to_adt;
use tribute_passes::lower_tribute_to_cont;
use tribute_passes::lower_tribute_to_scf;
use tribute_passes::resolve::{Resolver, build_env};
use tribute_passes::tdnr::resolve_tdnr;
use tribute_passes::typeck::{TypeChecker, TypeSolver, apply_subst_to_module};
use tribute_wasm_backend::{WasmBinary, compile_to_wasm};
use trunk_ir::Span;
use trunk_ir::dialect::core::Module;
use trunk_ir::transforms::eliminate_dead_functions;
use trunk_ir::{Block, BlockId, IdVec, Region, Symbol};

// =============================================================================
// Standard Library Prelude
// =============================================================================

/// The prelude source code, embedded at compile time.
const PRELUDE_SOURCE: &str = include_str!("../lib/std/prelude.trb");

/// Load and cache the prelude module.
///
/// This is a Salsa tracked function, so the prelude is parsed only once
/// and cached for all subsequent compilations.
#[salsa::tracked]
pub fn prelude_module<'db>(db: &'db dyn salsa::Database) -> Option<Module<'db>> {
    let uri = fluent_uri::Uri::parse_from("prelude:///std/prelude".to_owned())
        .expect("valid prelude URI");
    let text: Rope = PRELUDE_SOURCE.into();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None)?;
    let source = SourceCst::new(db, uri, text, Some(tree));
    let cst = parse_cst(db, source)?;
    Some(lower_cst(db, source, cst))
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

// Re-exports for external use
pub use tribute_passes::resolve::build_env as build_module_env;

/// Result of the full compilation pipeline.
pub struct CompilationResult<'db> {
    /// The compiled module with resolved types.
    pub module: Module<'db>,
    /// The type solver with final substitutions.
    pub solver: TypeSolver<'db>,
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

/// Parse source to CST and lower to initial TrunkIR module.
///
/// This is the entry point that converts source text to TrunkIR.
/// Returns a module with `tribute.*` operations that need further processing.
#[salsa::tracked]
pub fn parse_and_lower<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let Some(cst) = parse_cst(db, source) else {
        let path = trunk_ir::PathId::new(db, source.uri(db).as_str().to_owned());
        let location = trunk_ir::Location::new(path, trunk_ir::Span::new(0, 0));
        return Module::build(db, location, Symbol::new("main"), |_| {});
    };
    emit_parse_errors(db, &cst);
    let user_module = lower_cst(db, source, cst);

    // Merge prelude definitions into the user module
    merge_with_prelude(db, user_module)
}

/// Resolve names in the module.
///
/// This pass resolves:
/// - `tribute.var` → `func.constant` or `adt.struct_new`/`adt.variant_new`
/// - `tribute.call` → `func.call` with resolved callee
/// - `tribute.path` → resolved module paths
///
/// After this pass, all resolvable `tribute.*` operations are transformed.
/// Some may remain for type-directed resolution (UFCS).
#[salsa::tracked]
pub fn stage_resolve<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // Build module environment from declarations (including prelude)
    let env = build_env(db, &module);

    // Resolve names in the module
    let mut resolver = Resolver::new(db, env);
    resolver.resolve_module(&module)
}

fn emit_parse_errors(db: &dyn salsa::Database, cst: &tribute_front::ParsedCst) {
    let root = cst.root_node();
    if !root.has_error() {
        return;
    }

    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.is_error() || node.is_missing() {
            let message = if node.is_missing() {
                // For missing nodes, show what kind was expected
                let kind = node.kind();
                let parent_ctx = node
                    .parent()
                    .map(|p| format!(" in {}", format_node_kind(p.kind())))
                    .unwrap_or_default();
                format!("missing {}{}", format_node_kind(kind), parent_ctx)
            } else {
                // For error nodes, try to provide context
                let parent = node.parent();
                let parent_kind = parent.map(|p| p.kind()).unwrap_or("source_file");

                // Check what unexpected content was found
                let content_preview = if node.end_byte() > node.start_byte() {
                    // We don't have direct access to source here, so just note there's content
                    " (unexpected token)".to_string()
                } else {
                    String::new()
                };

                format!(
                    "syntax error{} while parsing {}",
                    content_preview,
                    format_node_kind(parent_kind)
                )
            };
            Diagnostic {
                message,
                span: Span::new(node.start_byte(), node.end_byte()),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Parsing,
            }
            .accumulate(db);
            continue;
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
}

/// Format a tree-sitter node kind into a human-readable name.
fn format_node_kind(kind: &str) -> String {
    // Convert snake_case to space-separated words
    kind.replace('_', " ")
}

/// Inline constant values.
///
/// This pass inlines constant values at their use sites:
/// - Finds `src.var` operations marked with `resolved_const=true`
/// - Replaces them with `arith.const` operations containing the inlined value
#[salsa::tracked]
pub fn stage_const_inline<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    inline_module(db, &module)
}

/// Infer and check types.
///
/// This pass:
/// - Collects type constraints from the module
/// - Solves constraints via unification
/// - Substitutes inferred types back into the module
/// - Reports type errors
#[salsa::tracked]
pub fn stage_typecheck<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let mut checker = TypeChecker::new(db);
    checker.check_module(&module);

    // Solve constraints
    match checker.solve() {
        Ok(solver) => {
            // Apply substitution to replace type variables with concrete types
            apply_subst_to_module(db, module, solver.type_subst())
        }
        Err(_err) => {
            // TODO: Report type error
            module
        }
    }
}

/// Insert explicit boxing/unboxing operations.
///
/// This pass inserts `tribute_rt.box_*` and `tribute_rt.unbox_*` operations
/// at call sites where polymorphic parameters or results need boxing/unboxing.
/// This makes boxing explicit in the IR, removing the need for emit-time type inference.
#[salsa::tracked]
pub fn stage_boxing<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    insert_boxing(db, module)
}

/// Lambda Lifting.
///
/// This pass transforms lambda expressions into:
/// 1. Lifted top-level functions (with captured variables as parameters)
/// 2. `closure.new` operations at the original lambda locations
#[salsa::tracked]
pub fn stage_lambda_lift<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    lift_lambdas(db, module)
}

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

/// Type-Directed Name Resolution (TDNR).
///
/// This pass resolves UFCS method calls that couldn't be resolved during
/// initial name resolution because they required type information.
///
/// For example:
/// - `list.len()` → `List::len(list)` (based on list's type being `List(a)`)
/// - `x.map(f)` → `Type::map(x, f)` (based on x's inferred type)
#[salsa::tracked]
pub fn stage_tdnr<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    resolve_tdnr(db, module)
}

/// Evidence Insertion.
///
/// This pass transforms effectful functions for ability system support:
/// - Adds evidence parameter as first argument to effectful functions
/// - Passes evidence through call chains
///
/// Evidence is a runtime structure for dynamic handler dispatch.
/// Pure functions (with empty effect row) are unchanged.
#[salsa::tracked]
pub fn stage_evidence<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    insert_evidence(db, module)
}

/// Handler Lowering.
///
/// This pass transforms ability dialect operations to continuation dialect operations:
/// - `ability.prompt` → `cont.push_prompt`
/// - `ability.perform` → `cont.shift` (with evidence lookup)
/// - `ability.resume` → `cont.resume`
/// - `ability.abort` → `cont.drop`
#[salsa::tracked]
pub fn stage_handler_lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    lower_handlers(db, module)
}

/// Continuation to Trampoline Lowering.
///
/// This pass transforms continuation operations to trampoline operations:
/// - `cont.shift` → `trampoline.build_state` + `trampoline.build_continuation` + etc.
/// - `cont.resume` → `trampoline.reset_yield_state` + `trampoline.continuation_get` + call
/// - `cont.get_continuation` → `trampoline.get_yield_continuation`
/// - `cont.get_shift_value` → `trampoline.get_yield_shift_value`
/// - `cont.get_done_value` → `trampoline.step_get`
///
/// This is a backend-agnostic pass that prepares continuation operations for
/// the trampoline (yield-bubbling) implementation strategy.
#[salsa::tracked]
pub fn stage_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    lower_cont_to_trampoline(db, module)
}

/// Trampoline to ADT Lowering.
///
/// This pass transforms trampoline struct operations to ADT operations:
/// - `trampoline.build_state` → `adt.struct_new`
/// - `trampoline.build_continuation` → `adt.struct_new`
/// - `trampoline.build_resume_wrapper` → `adt.struct_new`
/// - `trampoline.step_done` → `adt.struct_new`
/// - `trampoline.step_shift` → `adt.struct_new`
/// - `trampoline.step_get` → `adt.struct_get`
/// - `trampoline.continuation_get` → `adt.struct_get`
/// - `trampoline.resume_wrapper_get` → `adt.struct_get`
///
/// This is a backend-agnostic pass that converts trampoline abstractions
/// to concrete ADT struct operations.
#[salsa::tracked]
pub fn stage_trampoline_to_adt<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    lower_trampoline_to_adt(db, module)
}

/// Lower tribute.handle to cont dialect operations.
///
/// This pass lowers `tribute.handle` expressions to `cont.push_prompt` and
/// `cont.handler_dispatch` operations.
#[salsa::tracked]
pub fn stage_tribute_to_cont<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    lower_tribute_to_cont(db, module)
}

/// Lower tribute.case to scf dialect operations.
///
/// This pass lowers `tribute.case` expressions to `scf.if` operations.
#[salsa::tracked]
pub fn stage_tribute_to_scf<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    lower_tribute_to_scf(db, module)
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
    result.module
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
///
/// This entry point runs only the passes needed for LSP features:
/// - Parse & Lower: CST to TrunkIR
/// - Resolve: Name resolution (completion, go-to-definition)
/// - Typecheck: Type inference (hover)
/// - TDNR: Type-directed name resolution (method calls)
///
/// Excluded passes (preserve source structure):
/// - const_inline: Keep const references for go-to-definition
/// - lambda_lift/closure_lower: Keep closure structure
/// - evidence/handler_lower: Keep ability structure
/// - lower_case: Keep case/pattern structure
/// - dce: Keep all functions
#[salsa::tracked]
pub fn compile_for_lsp<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = parse_and_lower(db, source);
    let module = stage_resolve(db, module);
    let module = stage_typecheck(db, module);
    stage_tdnr(db, module)
}

/// Run pipeline up to resolve stage.
#[salsa::tracked]
pub fn run_resolve<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = parse_and_lower(db, source);
    stage_resolve(db, module)
}

/// Run pipeline up to typecheck stage.
#[salsa::tracked]
pub fn run_typecheck<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = parse_and_lower(db, source);
    let module = stage_resolve(db, module);
    let module = stage_const_inline(db, module);
    stage_typecheck(db, module)
}

/// Run pipeline up to lambda lift stage.
#[salsa::tracked]
pub fn run_lambda_lift<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = parse_and_lower(db, source);
    let module = stage_resolve(db, module);
    let module = stage_const_inline(db, module);
    let module = stage_typecheck(db, module);
    stage_lambda_lift(db, module)
}

/// Run pipeline up to closure lower stage.
#[salsa::tracked]
pub fn run_closure_lower<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = parse_and_lower(db, source);
    let module = stage_resolve(db, module);
    let module = stage_const_inline(db, module);
    let module = stage_typecheck(db, module);
    let module = stage_lambda_lift(db, module);
    stage_closure_lower(db, module)
}

// =============================================================================
// Full Pipeline (Orchestration)
// =============================================================================

/// Run the full compilation pipeline on a source file.
///
/// This is the central orchestration function that sequences all stages.
/// Each stage is a pure transformation that takes a Module and returns a Module.
///
/// Pipeline:
/// 1. Parse & Lower - CST to TrunkIR with prelude merged
/// 2. Resolve - Name resolution
/// 3. Const Inline - Inline constant values
/// 4. Typecheck - Type inference and checking
/// 5. Boxing - Insert explicit boxing/unboxing for polymorphic calls
/// 6. Lambda Lift - Lift lambdas to top-level
/// 7. Closure Lower - Lower closure operations
/// 8. TDNR - Type-directed name resolution
/// 9. Evidence - Insert evidence parameters
/// 10. Handler Lower - Lower ability ops to cont ops
/// 11. Lower Case - Lower case to scf.if
/// 12. DCE - Dead code elimination
/// 13. Final resolve - Report unresolved references
#[salsa::tracked]
pub fn compile<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    // Parse and lower to initial IR
    let module = parse_and_lower(db, source);

    // Frontend passes
    let module = stage_resolve(db, module);
    let module = stage_const_inline(db, module);
    let module = stage_typecheck(db, module);
    let module = stage_boxing(db, module);

    // Closure processing
    let module = stage_lambda_lift(db, module);
    let module = stage_closure_lower(db, module);
    let module = stage_tdnr(db, module);

    // Ability and case lowering
    let module = stage_evidence(db, module);
    let module = stage_tribute_to_cont(db, module); // tribute.handle → cont.push_prompt + cont.handler_dispatch
    let module = stage_tribute_to_scf(db, module); // tribute.case → scf.if
    let module = stage_handler_lower(db, module); // ability.perform → cont.shift, etc.

    // Continuation lowering (backend-agnostic trampoline implementation)
    let module = stage_cont_to_trampoline(db, module); // cont.shift → trampoline ops
    let module = stage_trampoline_to_adt(db, module); // trampoline ops → adt ops

    let module = stage_dce(db, module);

    // Final pass: resolve any remaining unresolved references and emit diagnostics
    let env = build_env(db, &module);
    let mut resolver = Resolver::with_unresolved_reporting(db, env);
    resolver.resolve_module(&module)
}

/// Compile to WebAssembly binary.
///
/// Runs the full pipeline and then lowers to WebAssembly.
///
/// Note: This duplicates the `compile` pipeline stages explicitly rather than
/// calling `compile` because:
/// 1. Each `stage_*` is `#[salsa::tracked]`, so stages are individually cached
///    and reused if `compile` was already called on the same source
/// 2. The wasm path skips the final resolver pass (diagnostic reporting) since
///    we only need the lowered module, not error messages
/// 3. Allows the wasm pipeline to diverge from `compile` in the future if needed
#[salsa::tracked]
pub fn compile_to_wasm_binary<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<WasmBinary<'db>> {
    // Run pipeline up to DCE (stages are cached and reused from compile if already run)
    let module = parse_and_lower(db, source);
    let module = stage_resolve(db, module);
    let module = stage_const_inline(db, module);
    let module = stage_typecheck(db, module);
    let module = stage_boxing(db, module);
    let module = stage_lambda_lift(db, module);
    let module = stage_closure_lower(db, module);
    let module = stage_tdnr(db, module);
    let module = stage_evidence(db, module);
    let module = stage_tribute_to_cont(db, module);
    let module = stage_tribute_to_scf(db, module);
    let module = stage_handler_lower(db, module);

    // Continuation lowering (backend-agnostic trampoline implementation)
    let module = stage_cont_to_trampoline(db, module);
    let module = stage_trampoline_to_adt(db, module);

    let module = stage_dce(db, module);

    // Lower to WebAssembly
    stage_lower_to_wasm(db, module)
}

/// Run compilation and return detailed results including diagnostics.
///
/// Diagnostics are collected using Salsa accumulators from all compilation stages.
pub fn compile_with_diagnostics<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> CompilationResult<'db> {
    // Run the full compilation pipeline (which checks for unresolved references)
    let module = compile(db, source);

    // Re-run type checking to capture the solver for the result
    // (compile already checked types, but we need the solver for diagnostics)
    let parsed_module = parse_and_lower(db, source);
    let resolved_module = stage_resolve(db, parsed_module);
    let mut checker = TypeChecker::new(db);
    checker.check_module(&resolved_module);

    let solver = match checker.solve() {
        Ok(solver) => solver,
        Err(err) => {
            // Emit type error diagnostic via accumulator
            Diagnostic {
                message: format!("Type error: {err}"),
                span: Span::new(0, 0), // TODO: Extract span from error
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(db);
            TypeSolver::new(db)
        }
    };

    // Collect all accumulated diagnostics from the compilation
    let diagnostics = compile::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect();

    CompilationResult {
        module,
        solver,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;

    #[salsa::tracked]
    fn test_compile(db: &dyn salsa::Database, source: SourceCst) -> Module<'_> {
        compile(db, source)
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
        let source = source_from_str("test.trb", "fn main() -> Int { 42 }");

        let module = test_compile(db, source);
        assert_eq!(module.name(db), "main");
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
            fn test(x: Int) -> Int {
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
            fn test(x: Int) -> Int {
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
}
