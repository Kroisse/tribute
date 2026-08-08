//! Tribute compiler utilities.
pub mod callable_abi;
pub mod calling_convention;
pub mod diagnostic;
pub mod fmt;
pub mod target;

pub use callable_abi::CallableAbi;
pub use calling_convention::{
    CALLING_CONVENTION_ATTR, CPS_PARENT_RESULT_ATTR, CallingConvention, cps_closure_function_type,
    cps_completion_type, cps_dispatch_type, cps_done_type, cps_parent_layout_type,
    cps_parent_ref_type, cps_resume_exact_type, cps_resume_type, get_calling_convention,
    get_physical_closure_convention, physical_closure_type, set_calling_convention,
};
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use target::*;
