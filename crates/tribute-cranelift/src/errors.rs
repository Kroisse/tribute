//! Error types for Cranelift compilation

use derive_more::{Display, From};
use tribute_ast::Identifier;

pub type CompilationResult<T> = Result<T, CompilationError>;

#[derive(Display, Debug, From)]
#[display("{kind}")]
pub struct CompilationError {
    #[from]
    kind: Box<CompilationErrorKind>,
}

impl<E> From<E> for CompilationError
where
    CompilationErrorKind: From<E>,
{
    fn from(error: E) -> Self {
        CompilationError {
            kind: Box::new(CompilationErrorKind::from(error)),
        }
    }
}

#[allow(dead_code)]
impl CompilationError {
    pub(crate) fn unsupported_feature(feature: &'static str) -> Self {
        CompilationErrorKind::UnsupportedFeature(feature).into()
    }

    pub(crate) fn type_error(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::TypeError(msg.to_string()).into()
    }

    pub(crate) fn function_not_found(name: Identifier) -> Self {
        CompilationErrorKind::FunctionNotFound(name).into()
    }

    pub(crate) fn other(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::CraneliftError(msg.to_string()).into()
    }
}

#[derive(Display, Debug)]
pub enum CompilationErrorKind {
    #[display("Code generation error: {_0}")]
    CodegenError(String),

    #[display("Module error: {_0}")]
    ModuleError(cranelift_module::ModuleError),

    #[display("Cranelift error: {_0}")]
    CraneliftError(String),

    #[display("Unsupported feature: {_0}")]
    UnsupportedFeature(&'static str),

    #[display("Type error: {_0}")]
    TypeError(String),

    #[display("Function not found: {_0}")]
    FunctionNotFound(String),

    #[display("Invalid target: {_0}")]
    InvalidTarget(String),

    #[display("Object generation failed: {_0}")]
    ObjectError(object::write::Error),
}

impl From<cranelift_module::ModuleError> for CompilationErrorKind {
    fn from(error: cranelift_module::ModuleError) -> Self {
        CompilationErrorKind::ModuleError(error)
    }
}

impl From<Box<cranelift_module::ModuleError>> for CompilationErrorKind {
    fn from(error: Box<cranelift_module::ModuleError>) -> Self {
        CompilationErrorKind::ModuleError(*error)
    }
}

impl From<object::write::Error> for CompilationErrorKind {
    fn from(error: object::write::Error) -> Self {
        CompilationErrorKind::ObjectError(error)
    }
}

impl From<Box<object::write::Error>> for CompilationErrorKind {
    fn from(error: Box<object::write::Error>) -> Self {
        CompilationErrorKind::ObjectError(*error)
    }
}

impl From<cranelift_codegen::settings::SetError> for CompilationErrorKind {
    fn from(error: cranelift_codegen::settings::SetError) -> Self {
        CompilationErrorKind::CraneliftError(error.to_string())
    }
}

impl From<cranelift_codegen::isa::LookupError> for CompilationErrorKind {
    fn from(error: cranelift_codegen::isa::LookupError) -> Self {
        CompilationErrorKind::InvalidTarget(error.to_string())
    }
}

impl From<cranelift_codegen::CodegenError> for CompilationErrorKind {
    fn from(error: cranelift_codegen::CodegenError) -> Self {
        CompilationErrorKind::CodegenError(error.to_string())
    }
}

impl std::error::Error for CompilationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &*self.kind {
            CompilationErrorKind::ModuleError(e) => Some(e),
            CompilationErrorKind::ObjectError(e) => Some(e),
            _ => None,
        }
    }
}
