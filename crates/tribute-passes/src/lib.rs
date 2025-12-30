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
pub mod closure_lower;
pub mod const_inline;
pub mod evidence;
pub mod lambda_lift;
pub mod resolve;
pub mod tdnr;
pub mod typeck;

// Re-exports
pub use case_lowering::lower_case_to_scf;
pub use closure_lower::lower_closures;
pub use const_inline::{ConstInliner, inline_module};
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use evidence::insert_evidence;
pub use lambda_lift::lift_lambdas;
pub use resolve::{ModuleEnv, Resolver, resolve_module};
pub use tdnr::{TdnrResolver, resolve_tdnr};
pub use trunk_ir::rewrite::{
    ApplyResult, PatternApplicator, RewriteContext, RewritePattern, RewriteResult,
};
pub use typeck::{Constraint, EffectRow, TypeChecker, TypeSolver, typecheck_module};
