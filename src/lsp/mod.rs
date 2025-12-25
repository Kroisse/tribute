//! Language Server Protocol implementation for Tribute.
//!
//! This module provides LSP support with features like:
//! - Hover: Display type information at cursor position
//! - Diagnostics: Real-time error and warning reporting

mod pretty;
mod server;
mod tracing_layer;
mod type_index;

pub use server::serve;
