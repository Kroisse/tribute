//! HIR to MLIR lowering for Tribute programs.

pub mod types;
pub mod operations;
pub mod function_gen;
pub mod expression;

use crate::error::Result;
use melior::{
    ir::{BlockLike, Location, Module},
    Context,
};
use tribute_hir::HirProgram;

// Re-export commonly used types and functions
pub use types::{
    MlirModule, MlirFunction, MlirOperation, BoxedType, MlirListOperation, 
    MlirExpressionResult, expected_type_name
};
pub use expression::{generate_mlir_function, generate_mlir_expression};
pub use function_gen::generate_mlir_function_op;

/// Salsa query to generate MLIR module from HIR program.
#[salsa::tracked]
pub fn generate_mlir_module<'db>(db: &'db dyn salsa::Database, hir_program: HirProgram<'db>) -> MlirModule<'db> {
    let mut function_results = Vec::new();
    
    // Generate MLIR for all functions in the HIR program
    let functions = hir_program.functions(db);
    
    for (name, hir_function) in functions {
        let mlir_func = generate_mlir_function(db, hir_function);
        function_results.push((name.clone(), mlir_func));
    }
    
    MlirModule::new(db, function_results)
}

/// Convenience function to convert MlirModule to actual MLIR Module
/// This bridges between Salsa-tracked data and MLIR API
pub fn mlir_module_to_melior<'a>(
    db: &dyn salsa::Database,
    mlir_module: MlirModule<'_>,
    context: &'a Context,
    location: Location<'a>,
) -> Result<Module<'a>> {
    use melior::ir::{Module, BlockLike};

    let module = Module::new(location);
    let functions = mlir_module.functions(db);
    
    println!("Generating MLIR for {} functions", functions.len());
    
    for (name, mlir_func) in functions.iter() {
        let params = mlir_func.params(db);
        let body_ops = mlir_func.body(db);
        
        // Generate MLIR function operation
        let function_op = generate_mlir_function_op(name, &params, &body_ops, context, location);
        
        // Add function to module
        module.body().append_operation(function_op);
        println!("    Successfully generated MLIR function: {}", name);
    }
    
    println!("Successfully generated MLIR module with {} functions", functions.len());
    Ok(module)
}