//! Native lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to native (Cranelift) dialect operations.
//!
//! ## Passes
//!
//! - `type_converter`: Native type converter for IR-level type transformations
//! - `tribute_rt_to_clif`: Lower `tribute_rt.box_*`/`unbox_*` to clif alloc + load/store

pub mod tribute_rt_to_clif;
pub mod type_converter;
