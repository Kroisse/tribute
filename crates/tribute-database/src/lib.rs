//! Tribute compiler database
//!
//! This crate provides the Salsa database implementation for the Tribute compiler.

pub mod database;
pub mod target;

pub use database::*;
pub use target::*;
