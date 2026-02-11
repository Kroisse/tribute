//! IR transformation passes.
//!
//! This module contains compiler optimization passes that transform the IR.
//!
//! ## Pass Order
//!
//! For best results, passes should be run in this order:
//! 1. `global_dce` - Remove unreachable function definitions
//! 2. `dce` - Remove dead operations within functions

pub mod dce;
pub mod global_dce;
pub mod scf_to_cf;

pub use dce::{DceConfig, DceResult, eliminate_dead_code, eliminate_dead_code_with_config};
pub use global_dce::{
    GlobalDceConfig, GlobalDceResult, eliminate_dead_functions,
    eliminate_dead_functions_with_config,
};
pub use scf_to_cf::lower_scf_to_cf;
