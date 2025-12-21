//! Compiler passes for Tribute.
//!
//! This crate contains the various transformation passes used in the Tribute compiler,
//! including CST to TrunkIR lowering, type inference, name resolution, and more.
//!
//! ## Pipeline
//!
//! This crate focuses on individual compiler passes and utilities.

// === Diagnostics ===
pub mod diagnostic;

// === TrunkIR passes ===
pub mod case_lowering;
pub mod resolve;
pub mod tdnr;
pub mod typeck;

// Re-exports
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use case_lowering::lower_case_to_scf;
pub use resolve::{ModuleEnv, Resolver, resolve_module};
pub use tdnr::{MethodInfo, MethodRegistry, TdnrResolver, resolve_tdnr};
pub use trunk_ir::rewrite::{
    ApplyResult, PatternApplicator, RewriteContext, RewritePattern, RewriteResult,
};
pub use typeck::{Constraint, EffectRow, TypeChecker, TypeSolver, typecheck_module};
