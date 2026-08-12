//! Tribute compiler utilities.
pub mod callable_abi;
pub mod calling_convention;
pub mod diagnostic;
pub mod fmt;
pub mod target;

pub use callable_abi::CallableAbi;
pub use calling_convention::{
    CALLING_CONVENTION_ATTR, CLOSURE_CALLABLE_TYPE_ATTR, CallingConvention,
    INDIRECT_CALL_SIGNATURE_ATTR, get_calling_convention, get_closure_callable_type,
    get_indirect_call_signature, get_physical_closure_convention, physical_closure_type,
    set_calling_convention, set_closure_callable_type, set_indirect_call_signature,
};
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use target::*;
