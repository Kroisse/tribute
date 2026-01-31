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
pub mod cont_to_trampoline;
pub mod evidence;
pub mod handler_lower;
pub mod live_vars;
pub mod resolve_evidence;
pub mod type_converter;
pub mod wasm;

// Re-exports
pub use boxing::insert_boxing;
pub use closure_lower::lower_closures;
pub use cont_to_trampoline::lower_cont_to_trampoline;
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use evidence::{
    add_evidence_params, collect_effectful_functions, insert_evidence, is_effectful_type,
    transform_evidence_calls,
};
pub use handler_lower::lower_handlers;
pub use resolve_evidence::resolve_evidence_dispatch;
pub use trunk_ir::rewrite::{
    ApplyResult, PatternApplicator, RewriteContext, RewritePattern, RewriteResult,
};
pub use type_converter::generic_type_converter;
pub use wasm::type_converter::{closure_adt_type, wasm_type_converter};
