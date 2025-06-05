//! Tribute AST, parser, and database components
//!
//! This crate provides the core data structures and parsing logic for the Tribute language.

pub mod ast;
pub mod database;
pub mod parser;

pub use ast::*;
pub use database::*;
pub use parser::*;

pub use salsa::Database as Db;
