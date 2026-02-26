//! Arena-based IR transformation passes.
//!
//! These passes operate directly on `IrContext` with in-place mutations,
//! leveraging use-chains for efficient dead code detection and RAUW for
//! value remapping.

pub mod dce;
pub mod global_dce;
pub mod scf_to_cf;

pub use dce::{DceConfig, DceResult, eliminate_dead_code, eliminate_dead_code_with_config};
pub use global_dce::{
    GlobalDceConfig, GlobalDceResult, eliminate_dead_functions,
    eliminate_dead_functions_with_config,
};
pub use scf_to_cf::lower_scf_to_cf;
