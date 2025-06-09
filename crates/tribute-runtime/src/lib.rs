//! Tribute Runtime Library
//!
//! This crate provides the native runtime system for compiled Tribute programs.
//! It includes memory management, value boxing/unboxing, and C FFI interfaces
//! for integration with MLIR-generated code.

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
pub use boolean::{tribute_box_boolean, tribute_unbox_boolean};
// Legacy pointer-based list API (deprecated)
// pub use list::{tribute_box_list_empty, tribute_box_list_from_array, tribute_list_length, tribute_list_get, tribute_list_set, tribute_list_push, tribute_list_pop};

// Handle-based list API (recommended)
pub use list::{
    tribute_handle_list_get, tribute_handle_list_length, tribute_handle_list_pop,
    tribute_handle_list_push, tribute_handle_list_set, tribute_handle_new_list_empty,
};
pub use number::{
    tribute_add_boxed, tribute_box_number, tribute_div_boxed, tribute_eq_boxed, tribute_gt_boxed,
    tribute_gte_boxed, tribute_lt_boxed, tribute_lte_boxed, tribute_mod_boxed, tribute_mul_boxed,
    tribute_neq_boxed, tribute_sub_boxed, tribute_unbox_number,
};
pub use string::{tribute_box_string, tribute_unbox_string};
pub use value::{
    TributeBoxed, TributeValue, tribute_get_ref_count, tribute_get_type, tribute_release,
    tribute_retain,
};

// Handle-based API (recommended for new code and GC compatibility)
pub use handle::{
    TRIBUTE_HANDLE_INVALID, TributeHandle, TributeRuntime, HandleTable,
    // Runtime-aware API (primary API)
    tribute_runtime_new, tribute_runtime_destroy,
    tribute_handle_new_number, tribute_handle_new_boolean, tribute_handle_new_nil,
    tribute_handle_new_string, tribute_handle_new_string_from_str, tribute_handle_is_valid, tribute_handle_get_type,
    tribute_handle_unbox_number, tribute_handle_unbox_boolean, 
    tribute_handle_get_string_length, tribute_handle_copy_string_data,
    tribute_handle_add_numbers, tribute_handle_retain, tribute_handle_release,
    tribute_handle_get_ref_count, tribute_handle_get_stats, tribute_handle_clear_all,
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
