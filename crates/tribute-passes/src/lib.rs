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
pub mod boxing;
pub mod closure_lower;
pub mod const_inline;
pub mod cont_to_trampoline;
pub mod evidence;
pub mod handler_lower;
pub mod lambda_lift;
pub mod live_vars;
pub mod resolve;
pub mod resolve_type_references;
pub mod tdnr;
pub mod tribute_to_cont;
pub mod tribute_to_scf;
pub mod type_converter;
pub mod typeck;

// Re-exports
pub use boxing::insert_boxing;
pub use closure_lower::lower_closures;
pub use const_inline::{ConstInliner, inline_module};
pub use cont_to_trampoline::lower_cont_to_trampoline;
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use evidence::{
    add_evidence_params, collect_effectful_functions, insert_evidence, is_effectful_type,
    transform_evidence_calls,
};
pub use handler_lower::lower_handlers;
pub use lambda_lift::lift_lambdas;
pub use resolve::{ModuleEnv, Resolver, resolve_module};
pub use tdnr::{TdnrResolver, resolve_tdnr};
pub use tribute_to_cont::lower_tribute_to_cont;
pub use tribute_to_scf::lower_tribute_to_scf;
pub use trunk_ir::rewrite::{
    ApplyResult, PatternApplicator, RewriteContext, RewritePattern, RewriteResult,
};
pub use type_converter::generic_type_converter;
pub use typeck::{Constraint, EffectRow, TypeChecker, TypeSolver, typecheck_module};
