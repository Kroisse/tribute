//! Cranelift native backend for TrunkIR.
//!
//! This crate provides language-agnostic lowering passes and emission for converting
//! TrunkIR dialects to native object files via Cranelift.
//!
//! ## Passes
//!
//! - `func_to_clif`: Lowers `func.*` operations to `clif.*`
//! - `arith_to_clif`: Lowers `arith.*` operations to `clif.*`
//!
//! ## Emission
//!
//! - `translate`: Module-level orchestration (validate + emit -> object file)
//! - `function`: clif.* -> Cranelift FunctionBuilder emit
//! - `validation`: Pre-emit validation (all ops must be clif.*)

mod errors;
mod function;
pub mod passes;
mod translate;
mod validation;

pub use errors::{CompilationError, CompilationErrorKind, CompilationResult};
pub use translate::emit_module_to_native;
pub use validation::{ValidationError, validate_clif_ir};
