//! IR transformation passes.
//!
//! This module contains compiler optimization passes that transform the IR.
//!
//! ## Pass Order
//!
//! For best results, passes should be run in this order:
//! 1. `func_dce` - Remove unreachable function definitions
//! 2. `dce` - Remove dead operations within functions

pub mod dce;
pub mod func_dce;

pub use dce::{DceConfig, DceResult, eliminate_dead_code, eliminate_dead_code_with_config};
pub use func_dce::{
    FuncDceConfig, FuncDceResult, eliminate_dead_functions, eliminate_dead_functions_with_config,
};
