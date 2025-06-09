//! Tribute Runtime Library
//!
//! This crate provides the native runtime system for compiled Tribute programs.
//! It features a handle-based API for safe memory management and C FFI interfaces
//! for integration with MLIR-generated code.
//!
//! ## Primary API
//!
//! The handle-based API is the recommended approach for all new code:
//! - `tribute_runtime_new()` / `tribute_runtime_destroy()` for runtime management
//! - `tribute_new_*()` for creating values
//! - `tribute_*()` for operations on values
//!
//! This API provides better safety, GC compatibility, and memory management.

#![allow(clippy::missing_safety_doc)]
#![allow(unsafe_op_in_unsafe_fn)]

mod array;
mod boolean;
mod handle;
mod interned_string;
mod list;
mod number;
mod string;
mod value;

pub use array::TributeArray;
pub use interned_string::TributeString;

// Core types (still needed for Handle API)
pub use value::{TributeBoxed, TributeValue};

// Handle-based list API (recommended)
pub use list::{
    tribute_list_get, tribute_list_length, tribute_list_pop,
    tribute_list_push, tribute_list_set, tribute_new_list_empty,
};

// Handle-based API (recommended for new code and GC compatibility)
pub use handle::{
    TRIBUTE_HANDLE_INVALID, TributeHandle, TributeRuntime, HandleTable,
    // Runtime-aware API (primary API)
    tribute_runtime_new, tribute_runtime_destroy,
    tribute_new_number, tribute_new_boolean, tribute_new_nil,
    tribute_new_string, tribute_new_string_from_str, tribute_is_valid, tribute_get_type,
    tribute_unbox_number, tribute_unbox_boolean, 
    tribute_get_string_length, tribute_copy_string_data,
    tribute_add_numbers, tribute_retain, tribute_release,
    tribute_get_ref_count, tribute_get_stats, tribute_clear_all,
};

/// Initialize the Tribute runtime system
/// This function should be called once at program startup
#[unsafe(no_mangle)]
pub extern "C" fn tribute_runtime_init() {
    // Runtime initialization if needed
}

/// Cleanup the Tribute runtime system
/// This function should be called at program shutdown
#[unsafe(no_mangle)]
pub extern "C" fn tribute_runtime_cleanup() {
    // Runtime cleanup if needed
}
