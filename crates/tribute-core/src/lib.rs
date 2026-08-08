//! Tribute compiler utilities.
pub mod callable_abi;
pub mod calling_convention;
pub mod diagnostic;
pub mod fmt;
pub mod target;

pub use callable_abi::{CallableAbi, interpose_physical_environment, physical_environment_index};
pub use calling_convention::{
    CALLING_CONVENTION_ATTR, CLOSURE_CALLABLE_TYPE_ATTR, CPS_PARENT_RESULT_ATTR, CallingConvention,
    INDIRECT_CALL_SIGNATURE_ATTR, ROOT_EXPORT_CONVENTION_ATTR, ROOT_SOURCE_RESULT_ATTR,
    cps_closure_function_type, cps_completion_type, cps_dispatch_type, cps_done_type,
    cps_parent_layout_type, cps_parent_ref_type, cps_parent_result_type, cps_resume_exact_type,
    cps_resume_type, get_calling_convention, get_closure_callable_type,
    get_indirect_call_signature, get_physical_closure_convention, get_root_export_convention,
    get_root_source_result, has_canonical_cps_parent_layout, physical_closure_type,
    set_calling_convention, set_closure_callable_type, set_indirect_call_signature,
    set_root_export_convention, set_root_source_result,
};
pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use target::*;
