//! Error types for Cranelift compilation

use derive_more::Display;

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

#[derive(Display, Debug)]
pub enum CompilationError {
    #[display("Code generation error: {_0}")]
    CodegenError(String),
    
    #[display("Module error: {_0}")]
    ModuleError(Box<cranelift_module::ModuleError>),
    
    #[display("Cranelift error: {_0}")]
    CraneliftError(String),
    
    #[display("Unsupported feature: {_0}")]
    UnsupportedFeature(String),
    
    #[display("Type error: {_0}")]
    TypeError(String),
    
    #[display("Function not found: {_0}")]
    FunctionNotFound(String),
    
    #[display("Invalid target: {_0}")]
    InvalidTarget(String),
    
    #[display("Object generation failed: {_0}")]
    ObjectError(Box<object::write::Error>),
}

impl From<Box<cranelift_module::ModuleError>> for CompilationError {
    fn from(error: Box<cranelift_module::ModuleError>) -> Self {
        CompilationError::ModuleError(error)
    }
}

impl From<Box<object::write::Error>> for CompilationError {
    fn from(error: Box<object::write::Error>) -> Self {
        CompilationError::ObjectError(error)
    }
}

impl std::error::Error for CompilationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompilationError::ModuleError(e) => Some(e.as_ref()),
            CompilationError::ObjectError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}