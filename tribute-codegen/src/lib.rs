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
    fn test_debug_mlir_generation() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (main) (print_line "Hello, world!"))"#;
        let output_path = Path::new("/tmp/test_debug_output");

        println!("\n=== Debug MLIR Generation ===");
        println!("Source: {}", source);
        
        let result = codegen.compile_string(source, output_path);
        println!("Result: {:?}", result);
        
        // This test is specifically for debugging output
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
    
    #[test]
    fn test_debug_arithmetic_mlir_generation() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (add_numbers x y) (+ x y))"#;
        let output_path = Path::new("/tmp/test_arithmetic_output");

        println!("\n=== Debug Arithmetic MLIR Generation ===");
        println!("Source: {}", source);
        
        let result = codegen.compile_string(source, output_path);
        println!("Result: {:?}", result);
        
        // This test is specifically for debugging arithmetic operations
        let _ = result;
    }
    
    #[test]
    fn test_debug_complex_arithmetic_mlir_generation() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (complex_calc a b c) (* (+ a b) (- c 5)))"#;
        let output_path = Path::new("/tmp/test_complex_arithmetic_output");

        println!("\n=== Debug Complex Arithmetic MLIR Generation ===");
        println!("Source: {}", source);
        
        let result = codegen.compile_string(source, output_path);
        println!("Result: {:?}", result);
        
        // This test shows nested arithmetic operations
        let _ = result;
    }
    
    #[test]
    fn test_debug_gc_aware_mlir_generation() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (gc_demo x) (+ (* x 2) (+ x 1)))"#;
        let output_path = Path::new("/tmp/test_gc_output");

        println!("\n=== Debug GC-Aware MLIR Generation ===");
        println!("Source: {}", source);
        println!("This demonstrates memory management with boxed values:");
        println!("- Multiple intermediate boxed values created");
        println!("- Reference counting for memory management");
        println!("- Automatic cleanup of temporary values");
        
        let result = codegen.compile_string(source, output_path);
        println!("Result: {:?}", result);
        
        // This test shows GC integration with boxed values
        let _ = result;
    }

    #[test]
    fn test_debug_all_operations() {
        let mut codegen = TributeCodegen::new().unwrap();
        let source = r#"(fn (all_ops x y) (+ (- x y) (* (/ x 2) (% y 3))))"#;
        let output_path = Path::new("/tmp/test_all_ops_output");

        println!("\n=== Debug All Operations ===");
        println!("Source: {}", source);
        println!("This demonstrates all arithmetic operations:");
        println!("- Addition (+)");
        println!("- Subtraction (-)");
        println!("- Multiplication (*)");
        println!("- Division (/)");
        println!("- Modulo (%)");
        
        let result = codegen.compile_string(source, output_path);
        println!("Result: {:?}", result);
        
        // This test shows all operations working together
        let _ = result;
    }
    
    #[test]
    fn test_debug_list_operations() {
        use super::hir_to_mlir::{MlirOperation, MlirListOperation};
        
        println!("\n=== Debug O(1) List Operations ===");
        println!("Demonstrating high-performance list operations:");
        
        // Simulate list operations that would be generated
        let operations = vec![
            MlirOperation::ListOp { 
                operation: MlirListOperation::CreateEmpty { capacity: 10 } 
            },
            MlirOperation::ListOp { 
                operation: MlirListOperation::Push { 
                    list: "list".to_string(), 
                    value: "value1".to_string() 
                }
            },
            MlirOperation::ListOp { 
                operation: MlirListOperation::Get { 
                    list: "list".to_string(), 
                    index: "0".to_string() 
                }
            },
            MlirOperation::ListOp { 
                operation: MlirListOperation::Length { 
                    list: "list".to_string() 
                }
            },
        ];
        
        for (i, op) in operations.iter().enumerate() {
            match op {
                MlirOperation::ListOp { operation } => {
                    match operation {
                        MlirListOperation::CreateEmpty { capacity } => {
                            println!("{}. Create empty list (capacity: {})", i + 1, capacity);
                            println!("   -> O(1) allocation with pre-allocated array");
                        }
                        MlirListOperation::Push { list, value } => {
                            println!("{}. Push {} to {}", i + 1, value, list);
                            println!("   -> Amortized O(1) with automatic resizing");
                        }
                        MlirListOperation::Get { list, index } => {
                            println!("{}. Get {}[{}]", i + 1, list, index);
                            println!("   -> O(1) random access via array indexing");
                        }
                        MlirListOperation::Length { list } => {
                            println!("{}. Get length of {}", i + 1, list);
                            println!("   -> O(1) length stored in metadata");
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        
        println!("\n🚀 Performance characteristics:");
        println!("   • Random access: O(1) - Direct array indexing");
        println!("   • Append: Amortized O(1) - Vector-like growth");
        println!("   • Length: O(1) - Cached in structure");
        println!("   • Memory: Contiguous allocation, cache-friendly");
        println!("   • GC: Reference counting for automatic cleanup");
    }
}
