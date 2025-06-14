//! Tribute Runtime Library
//!
//! This library provides the runtime support needed for compiled Tribute programs.
//! It implements dynamic values, memory management, and built-in functions.
//! Uses handle-based API for future GC compatibility.

// Use std for simplicity - this is a runtime library
// that will be linked with compiled programs

// Allow unsafe operations for C-compatible runtime functions
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::missing_safety_doc)]

pub mod value;
pub mod memory;
pub mod arithmetic;
pub mod string_ops;
pub mod builtins;

#[cfg(test)]
mod tests;

// Re-export the main API
pub use value::*;
pub use memory::*;
pub use arithmetic::*;
pub use string_ops::*;
pub use builtins::*;

// C-compatible functions are exported directly from each module

// Runtime initialization (called by compiled programs)
#[no_mangle]
pub extern "C" fn tr_runtime_init() {
    // The allocation table is statically initialized, so no setup is needed
    // This function exists for API completeness and future expansion
}

#[no_mangle]
pub extern "C" fn tr_runtime_cleanup() {
    // Clear all allocations from the global allocation table
    value::allocation_table().clear();
}