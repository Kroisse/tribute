//! Salsa integration for MLIR lowering

use crate::{hir_lowering::HirToMLIRLowerer, errors::LoweringError};
use melior::{Context, ir::Module};
use tribute_hir::hir::HirProgram;

/// MLIR module result that can be cached by Salsa
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MLIRModuleResult {
    /// Serialized MLIR module as string for comparison/caching
    pub mlir_text: String,
    /// Whether the lowering was successful
    pub success: bool,
    /// Error message if lowering failed
    pub error: Option<String>,
}

/// Salsa query to lower HIR program to MLIR
#[salsa::tracked]
pub fn lower_program_to_mlir<'db>(db: &'db dyn salsa::Database, program: HirProgram<'db>) -> MLIRModuleResult {
    // Create MLIR context (not cached, created fresh each time)
    let context = Context::new();
    
    // Create lowerer
    let mut lowerer = HirToMLIRLowerer::new(&context);
    
    // Perform lowering
    match lowerer.lower_program(db, program) {
        Ok(module) => {
            // Convert module to string for Salsa caching
            let mlir_text = module.as_operation().to_string();
            MLIRModuleResult {
                mlir_text,
                success: true,
                error: None,
            }
        }
        Err(e) => MLIRModuleResult {
            mlir_text: String::new(),
            success: false,
            error: Some(e.to_string()),
        }
    }
}

/// Helper to get MLIR module from cached result
pub fn mlir_module_from_result(result: &MLIRModuleResult) -> Result<Module<'_>, LoweringError> {
    if !result.success {
        return Err(LoweringError::InvalidMLIR);
    }
    
    // TODO: Parse MLIR text back to Module
    // This requires MLIR parsing capabilities
    todo!("Implement MLIR text -> Module parsing")
}

/// Convenience function for getting MLIR as text (most common use case)
#[salsa::tracked]
pub fn program_to_mlir_text<'db>(db: &'db dyn salsa::Database, program: HirProgram<'db>) -> Result<String, String> {
    let result = lower_program_to_mlir(db, program);
    if result.success {
        Ok(result.mlir_text)
    } else {
        Err(result.error.unwrap_or_else(|| "Unknown MLIR lowering error".to_string()))
    }
}

/// Check if MLIR lowering would succeed (for diagnostics)
#[salsa::tracked] 
pub fn mlir_lowering_diagnostics<'db>(db: &'db dyn salsa::Database, program: HirProgram<'db>) -> Vec<String> {
    let result = lower_program_to_mlir(db, program);
    if result.success {
        vec![]
    } else {
        vec![result.error.unwrap_or_else(|| "Unknown MLIR lowering error".to_string())]
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_salsa_mlir_integration() {
        // This test is temporarily disabled until we have proper HIR construction
        // TODO: Create test HIR programs and verify MLIR lowering works
        
        // let db = create_test_database();
        // let program = create_test_hir_program(&db);
        // let mlir_text = program_to_mlir_text(&db, program).unwrap();
        // assert!(!mlir_text.is_empty());
    }
}