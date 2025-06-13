//! Cranelift-based native code compiler for Tribute
//!
//! This crate provides ahead-of-time (AOT) compilation of Tribute programs
//! to native executables using the Cranelift code generation library.

pub mod compiler;
pub mod codegen;
pub mod runtime;
pub mod types;
pub mod errors;

#[cfg(test)]
mod tests;

pub use compiler::TributeCompiler;
pub use errors::{BoxError, CompilationError, CompilationResult};

use salsa::Database;
use tribute_hir::hir::HirProgram;

/// Compile a Tribute program to an object file
pub fn compile_to_object<'db>(
    db: &'db dyn Database,
    program: HirProgram<'db>,
    target: Option<target_lexicon::Triple>,
) -> CompilationResult<Vec<u8>> {
    let compiler = TributeCompiler::new(target)?;
    compiler.compile_program(db, program)
}