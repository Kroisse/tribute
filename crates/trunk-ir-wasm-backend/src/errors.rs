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
        CompilationErrorKind::UnsupportedFeature(feature.to_string()).into()
    }

    pub fn unsupported_feature_msg(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::UnsupportedFeature(msg.to_string()).into()
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

    pub fn missing_attribute(attr: &'static str) -> Self {
        CompilationErrorKind::MissingAttribute(attr).into()
    }

    pub fn invalid_operation(op: &'static str) -> Self {
        CompilationErrorKind::InvalidOperation(op).into()
    }

    pub fn invalid_attribute(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::InvalidAttribute(msg.to_string()).into()
    }

    pub fn ir_validation(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::IrValidation(msg.to_string()).into()
    }

    pub fn unresolved_casts(msg: impl std::fmt::Display) -> Self {
        CompilationErrorKind::UnresolvedCasts(msg.to_string()).into()
    }
}

#[derive(Clone, Display, Debug, PartialEq)]
pub enum CompilationErrorKind {
    #[display("Unsupported feature: {_0}")]
    UnsupportedFeature(String),

    #[display("Type error: {_0}")]
    TypeError(String),

    #[display("Invalid module: {_0}")]
    InvalidModule(String),

    #[display("Missing attribute: {_0}")]
    MissingAttribute(&'static str),

    #[display("Invalid attribute: {_0}")]
    InvalidAttribute(String),

    #[display("Function not found: {_0}")]
    FunctionNotFound(String),

    #[display("Invalid operation: {_0}")]
    InvalidOperation(&'static str),

    #[display("IR validation failed: {_0}")]
    IrValidation(String),

    #[display("Unresolved type casts: {_0}")]
    UnresolvedCasts(String),
}

impl std::error::Error for CompilationError {}
