//! Compiler passes for Tribute.
//!
//! This crate contains the various transformation passes used in the Tribute compiler,
//! including boxing, closures, and effect handling.
//!
//! ## Pipeline
//!
//! This crate focuses on individual compiler passes and utilities.
//! Name resolution and TDNR are now handled at the AST level in `tribute-front`.

// === Diagnostics ===
pub mod diagnostic;

// === TrunkIR passes ===
pub mod boxing;
pub mod closure_lower;
pub mod cont_to_yield_bubbling;
pub mod cont_util;
pub mod evidence;
pub mod live_vars;
pub mod lower_closure_lambda;
pub mod native;
pub mod resolve_evidence;
pub mod tail_resumptive;
pub mod tr_dispatch;
pub mod type_converter;
pub mod wasm;

// Re-exports
pub use closure_lower::lower_closures;
pub use cont_to_yield_bubbling::lower_cont_to_yield_bubbling;
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use lower_closure_lambda::lower_closure_lambda;
pub use resolve_evidence::resolve_evidence_dispatch;
pub use trunk_ir::rewrite::{ApplyResult, PatternApplicator, PatternRewriter, RewritePattern};
pub use type_converter::generic_type_converter;
