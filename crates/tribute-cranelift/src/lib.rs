//! Cranelift-based native code compiler for Tribute programming language.
//!
//! This crate is currently a stub. The Cranelift backend is pending migration
//! to the new TrunkIR-based compilation pipeline.
//!
//! ## Status
//!
//! - [ ] TrunkIR → Cranelift IR lowering
//! - [ ] Native code generation
//! - [ ] Object file emission
//!
//! The previous HIR-based backend has been removed as part of the AST/HIR cleanup.

pub mod errors;

pub use errors::{CompilationError, CompilationResult};
