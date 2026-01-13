//! Operation handlers for wasm backend.
//!
//! This module contains handlers for different categories of WebAssembly operations.

mod const_handlers;
mod memory_handlers;

pub(super) use const_handlers::{
    handle_f32_const, handle_f64_const, handle_i32_const, handle_i64_const,
};
pub(super) use memory_handlers::{
    handle_f32_load, handle_f32_store, handle_f64_load, handle_f64_store, handle_i32_load,
    handle_i32_load8_s, handle_i32_load8_u, handle_i32_load16_s, handle_i32_load16_u,
    handle_i32_store, handle_i32_store8, handle_i32_store16, handle_i64_load, handle_i64_load8_s,
    handle_i64_load8_u, handle_i64_load16_s, handle_i64_load16_u, handle_i64_load32_s,
    handle_i64_load32_u, handle_i64_store, handle_i64_store8, handle_i64_store16,
    handle_i64_store32, handle_memory_grow, handle_memory_size,
};
