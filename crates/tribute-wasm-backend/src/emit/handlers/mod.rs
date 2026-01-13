//! Operation handlers for wasm backend.
//!
//! This module contains handlers for different categories of WebAssembly operations.

mod array_handlers;
mod const_handlers;
mod control_flow_handlers;
mod memory_handlers;
mod ref_handlers;
mod struct_handlers;

pub(super) use array_handlers::{
    handle_array_copy, handle_array_get, handle_array_get_s, handle_array_get_u, handle_array_new,
    handle_array_new_default, handle_array_set,
};
pub(super) use const_handlers::{
    handle_f32_const, handle_f64_const, handle_i32_const, handle_i64_const,
};
pub(super) use control_flow_handlers::{
    handle_block, handle_br, handle_br_if, handle_if, handle_loop,
};
pub(super) use memory_handlers::{
    handle_f32_load, handle_f32_store, handle_f64_load, handle_f64_store, handle_i32_load,
    handle_i32_load8_s, handle_i32_load8_u, handle_i32_load16_s, handle_i32_load16_u,
    handle_i32_store, handle_i32_store8, handle_i32_store16, handle_i64_load, handle_i64_load8_s,
    handle_i64_load8_u, handle_i64_load16_s, handle_i64_load16_u, handle_i64_load32_s,
    handle_i64_load32_u, handle_i64_store, handle_i64_store8, handle_i64_store16,
    handle_i64_store32, handle_memory_grow, handle_memory_size,
};
pub(super) use ref_handlers::{handle_ref_cast, handle_ref_func, handle_ref_null, handle_ref_test};
pub(super) use struct_handlers::{handle_struct_get, handle_struct_new, handle_struct_set};
