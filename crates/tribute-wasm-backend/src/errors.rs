//! Error types for wasm backend emission.

use derive_more::{Display, From};

pub type CompilationResult<T> = Result<T, CompilationError>;

#[derive(Clone, Display, Debug, From, PartialEq, salsa::Update)]
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
    pub fn unsupported_feature(feature: &'static str) -> Self {
        CompilationErrorKind::UnsupportedFeature(feature).into()
    }

    pub fn type_error(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::TypeError(msg.to_string()).into()
    }

    pub fn invalid_module(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::InvalidModule(msg.to_string()).into()
    }

    pub fn function_not_found(name: &str) -> Self {
        CompilationErrorKind::FunctionNotFound(name.to_string()).into()
    }
}

#[derive(Clone, Display, Debug, PartialEq)]
pub enum CompilationErrorKind {
    #[display("Unsupported feature: {_0}")]
    UnsupportedFeature(&'static str),

    #[display("Type error: {_0}")]
    TypeError(String),

    #[display("Invalid module: {_0}")]
    InvalidModule(String),

    #[display("Missing attribute: {_0}")]
    MissingAttribute(&'static str),

    #[display("Invalid attribute: {_0}")]
    InvalidAttribute(&'static str),

    #[display("Function not found: {_0}")]
    FunctionNotFound(String),
}

impl std::error::Error for CompilationError {}
