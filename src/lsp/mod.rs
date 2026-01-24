//! Language Server Protocol implementation for Tribute.
//!
//! This module provides LSP support with features like:
//! - Hover: Display type information at cursor position
//! - Diagnostics: Real-time error and warning reporting
//! - Go to Definition: Navigate to symbol definitions

mod ast_index;
mod pretty;
mod server;

pub use server::serve;
