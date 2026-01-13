//! Operation handlers for wasm backend.
//!
//! This module contains handlers for different categories of WebAssembly operations.

mod const_handlers;

pub(super) use const_handlers::{
    handle_f32_const, handle_f64_const, handle_i32_const, handle_i64_const,
};
