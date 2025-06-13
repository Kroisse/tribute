//! Error types for Cranelift compilation

use thiserror::Error;

pub type CompilationResult<T> = Result<T, CompilationError>;

// Helper trait for boxing large errors
pub trait BoxError<T> {
    fn box_err(self) -> CompilationResult<T>;
}

impl<T> BoxError<T> for Result<T, cranelift_module::ModuleError> {
    fn box_err(self) -> CompilationResult<T> {
        self.map_err(|e| CompilationError::ModuleError(Box::new(e)))
    }
}

impl<T> BoxError<T> for Result<T, object::write::Error> {
    fn box_err(self) -> CompilationResult<T> {
        self.map_err(|e| CompilationError::ObjectError(Box::new(e)))
    }
}

#[derive(Error, Debug)]
pub enum CompilationError {
    #[error("Code generation error: {0}")]
    CodegenError(String),
    
    #[error("Module error: {0}")]
    ModuleError(#[from] Box<cranelift_module::ModuleError>),
    
    #[error("Cranelift error: {0}")]
    CraneliftError(String),
    
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
    
    #[error("Type error: {0}")]
    TypeError(String),
    
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    
    #[error("Invalid target: {0}")]
    InvalidTarget(String),
    
    #[error("Object generation failed: {0}")]
    ObjectError(#[from] Box<object::write::Error>),
}