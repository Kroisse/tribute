//! Tribute Runtime Library
//!
//! This library provides the runtime support for Tribute programs,
//! including garbage collection, boxing/unboxing, and builtin functions.

pub mod array;
pub mod string;
pub mod list;
pub mod number;
pub mod boolean;
pub mod value;

// Re-export all modules for C FFI
pub use value::*;
pub use number::*;
pub use boolean::*;
pub use string::*;
pub use list::*;

