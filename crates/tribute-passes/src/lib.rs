//! Compiler passes for Tribute.
//!
//! This crate contains the various transformation passes used in the Tribute compiler,
//! including CST to TrunkIR lowering, type inference, name resolution, and more.
//!
//! ## Pipeline
//!
//! The compilation pipeline consists of four stages:
//!
//! 1. **Parsing**: `parse_cst` - Parse source to CST (Tree-sitter)
//! 2. **Lowering**: `lower_cst` - Lower CST to TrunkIR
//! 3. **Name Resolution**: `stage_resolve` - Resolve `src.*` operations
//! 4. **Type Checking**: `stage_typecheck` - Infer and check types
//!
//! All stages are Salsa-tracked for incremental compilation.
//! Use `compile` to run the full pipeline, or `compile_with_diagnostics`
//! for detailed results including error messages.

// === TrunkIR passes ===
pub mod cst_to_tir;
pub mod pipeline;
pub mod resolve;
pub mod rewrite;
pub mod typeck;

// Re-exports
pub use cst_to_tir::{ParsedCst, lower_cst, lower_source_file, parse_cst};
pub use pipeline::{
    CompilationDiagnostic, CompilationResult, DiagnosticSeverity, compile,
    compile_with_diagnostics, stage_resolve, stage_typecheck,
};
pub use resolve::{ModuleEnv, Resolver, resolve_module};
pub use rewrite::{ApplyResult, PatternApplicator, RewriteContext, RewritePattern, RewriteResult};
pub use typeck::{Constraint, EffectRow, TypeChecker, TypeSolver, typecheck_module};
