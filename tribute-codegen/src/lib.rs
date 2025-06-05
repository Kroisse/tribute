//! Tribute Programming Language Code Generator
//!
//! This crate provides code generation functionality for the Tribute programming language,
//! transforming HIR (High-level Intermediate Representation) into native binaries via MLIR and LLVM.
//!
//! # Compilation Pipeline
//!
//! The compilation process follows this pipeline:
//! 1. **Parse**: Tribute source → AST (using `tribute-ast` crate)
//! 2. **Lower**: AST → HIR (using `tribute-hir` crate)
//! 3. **Generate**: HIR → MLIR (this crate)
//! 4. **Optimize**: MLIR transformations and optimizations
//! 5. **Codegen**: MLIR → LLVM IR
//! 6. **Link**: LLVM IR → Native binary

pub mod hir_to_mlir;
pub mod codegen;
pub mod error;

pub use error::{Error, Result};

use melior::{
    ir::Location,
    Context,
};
use std::path::Path;
use tribute_ast::{SourceFile, TributeDatabaseImpl};
use tribute_hir::{lower_source_to_hir, HirProgram};

/// Main code generator interface for Tribute programs.
///
/// This struct orchestrates the code generation pipeline from HIR
/// to native binaries.
pub struct TributeCodegen {
    mlir_context: Context,
}

impl TributeCodegen {
    /// Creates a new compiler instance.
    pub fn new() -> Result<Self> {
        let mlir_context = Context::new();
        mlir_context.load_all_available_dialects();
        Ok(Self { mlir_context })
    }

    /// Compiles a Tribute source file to a native binary.
    ///
    /// # Arguments
    /// * `source_path` - Path to the Tribute source file
    /// * `output_path` - Path where the compiled binary should be written
    ///
    /// # Returns
    /// Returns `Ok(())` on successful compilation, or a `CompilerError` on failure.
    pub fn compile_file(&mut self, source_path: &Path, output_path: &Path) -> Result<()> {
        // 1. Parse and lower to HIR
        let source = std::fs::read_to_string(source_path)?;
        let db = TributeDatabaseImpl::default();
        let source_file = SourceFile::new(&db, source_path.to_path_buf(), source);
        
        let hir_program = lower_source_to_hir(&db, source_file)
            .ok_or_else(|| Error::ParseError("Failed to lower to HIR".to_string()))?;

        // 2. Generate code from HIR
        self.compile_hir(&db, hir_program, output_path)
    }

    /// Generates code from HIR for Tribute source code from a string.
    ///
    /// # Arguments
    /// * `source` - Tribute source code as a string
    /// * `output_path` - Path where the compiled binary should be written
    ///
    /// # Returns
    /// Returns `Ok(())` on successful compilation, or a `CompilerError` on failure.
    pub fn compile_string(&mut self, source: &str, output_path: &Path) -> Result<()> {
        // 1. Parse and lower to HIR
        let db = TributeDatabaseImpl::default();
        let source_file = SourceFile::new(&db, "input.trb".into(), source.to_string());
        
        let hir_program = lower_source_to_hir(&db, source_file)
            .ok_or_else(|| Error::ParseError("Failed to lower to HIR".to_string()))?;

        // 2. Generate code from HIR
        self.compile_hir(&db, hir_program, output_path)
    }

    /// Generates code from a HIR program.
    ///
    /// # Arguments
    /// * `db` - Salsa database instance
    /// * `hir_program` - HIR program to compile
    /// * `output_path` - Path where the compiled binary should be written
    pub fn compile_hir(&mut self, db: &dyn salsa::Database, hir_program: HirProgram<'_>, output_path: &Path) -> Result<()> {
        // 1. Generate MLIR representation using Salsa queries
        let mlir_module_data = hir_to_mlir::generate_mlir_module(db, hir_program);
        
        // 2. Convert Salsa-tracked MLIR to actual MLIR module
        let location = Location::unknown(&self.mlir_context);
        let mlir_module = hir_to_mlir::mlir_module_to_melior(db, mlir_module_data, &self.mlir_context, location)?;

        // 3. Generate LLVM IR and compile to binary
        codegen::compile_to_binary(mlir_module, output_path)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_codegen_creation() {
        let codegen = TributeCodegen::new();
        assert!(codegen.is_ok());
    }

    #[test]
    fn test_compile_simple_function() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (main) (print_line "Hello, world!"))"#;
        let output_path = Path::new("/tmp/test_output");

        let result = codegen.compile_string(source, output_path);
        // This will succeed in parsing but fail in code generation (MLIR stub)
        // For now, we just test that it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_compile_arithmetic_function() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (calc x y) (+ x y))"#;
        let output_path = Path::new("/tmp/test_output");

        let result = codegen.compile_string(source, output_path);
        // This will succeed in parsing but fail in code generation (MLIR stub)
        // For now, we just test that it doesn't panic
        let _ = result;
    }
}
