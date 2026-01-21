//! Language Server Protocol implementation for Tribute.
//!
//! This module provides LSP support with features like:
//! - Hover: Display type information at cursor position
//! - Diagnostics: Real-time error and warning reporting
//! - Go to Definition: Navigate to symbol definitions

mod call_index;
mod completion_index;
mod definition_index;
mod pretty;
mod server;
mod type_index;

pub use server::serve;
