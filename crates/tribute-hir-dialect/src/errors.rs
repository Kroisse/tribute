//! Error types for MLIR dialect operations

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoweringError {
    #[error("Unsupported AST node: {0}")]
    UnsupportedAst(String),
    
    #[error("Unsupported HIR node: {0}")]
    UnsupportedHir(String),
    
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    
    #[error("Invalid MLIR generated")]
    InvalidMLIR,
    
    #[error("MLIR operation creation failed: {0}")]
    OperationCreationFailed(String),
    
    #[error("Type error: {0}")]
    TypeError(String),
    
    #[error("Symbol table error: {0}")]
    SymbolTableError(String),
    
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("Runtime type error: {0}")]
    RuntimeTypeError(String),
    
    #[error("Function call error: {0}")]
    FunctionCallError(String),
    
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    
    #[error("Division by zero")]
    DivisionByZero,
    
    #[error("MLIR execution error: {0}")]
    MLIRExecutionError(String),
}