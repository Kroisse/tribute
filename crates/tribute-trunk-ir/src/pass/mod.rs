//! Compiler passes for TrunkIR.

pub mod ast_to_tir;

pub use ast_to_tir::lower_program;
