//! Native lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to native (Cranelift) dialect operations.
//!
//! ## Passes
//!
//! - `type_converter`: Native type converter for IR-level type transformations

pub mod type_converter;
