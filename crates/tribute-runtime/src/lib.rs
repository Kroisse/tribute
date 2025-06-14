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

// Export C-compatible functions
extern "C" {
    // These will be implemented in each module
}

// Runtime initialization (called by compiled programs)
#[no_mangle]
pub extern "C" fn tr_runtime_init() {
    // Initialize any global state if needed
    // For now, this is a no-op
}

#[no_mangle]
pub extern "C" fn tr_runtime_cleanup() {
    // Clean up any global state if needed
    // For now, this is a no-op
}