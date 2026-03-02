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
pub mod cont_to_libmprompt;
pub mod cont_to_trampoline;
pub mod cont_util;
pub mod evidence;
pub mod live_vars;
pub mod native;
pub mod resolve_evidence;
pub mod type_converter;
pub mod wasm;

// Re-exports
#[allow(deprecated)]
pub use boxing::insert_boxing; // DEPRECATED: kept for compatibility
pub use closure_lower::lower_closures;
pub use cont_to_libmprompt::lower_cont_to_libmprompt;
pub use cont_to_trampoline::lower_cont_to_trampoline;
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use evidence::{
    collect_effectful_functions, collect_functions_with_evidence_param, is_effectful_type,
};
pub use native::type_converter::native_type_converter;
pub use resolve_evidence::resolve_evidence_dispatch;
pub use trunk_ir::rewrite::{
    ApplyResult, PatternApplicator, PatternRewriter, RewriteContext, RewritePattern,
};
pub use type_converter::generic_type_converter;
// Arena-based type converter is available via wasm::type_converter::wasm_type_converter
// Salsa-based type converter for pipeline: wasm::type_converter::salsa_converter::wasm_type_converter
