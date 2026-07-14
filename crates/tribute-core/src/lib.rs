//! Tribute compiler utilities.
pub mod callable_abi;
pub mod calling_convention;
pub mod diagnostic;
pub mod fmt;
pub mod target;

pub use callable_abi::CallableAbi;
pub use calling_convention::{
    CALLING_CONVENTION_ATTR, CallingConvention, get_calling_convention, set_calling_convention,
};
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use target::*;
