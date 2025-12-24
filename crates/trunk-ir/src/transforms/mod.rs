//! IR transformation passes.
//!
//! This module contains compiler optimization passes that transform the IR.

pub mod dce;

pub use dce::{eliminate_dead_code, eliminate_dead_code_with_config, DceConfig, DceResult};
