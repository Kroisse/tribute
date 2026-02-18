//! Native lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to native (Cranelift) dialect operations.
//!
//! ## Passes
//!
//! - `entrypoint`: Generate C ABI `main` wrapper for native binaries
//! - `type_converter`: Native type converter for IR-level type transformations
//! - `adt_rc_header`: Lower `adt.struct_new` to clif alloc + RC header init + field stores
//! - `tribute_rt_to_clif`: Lower `tribute_rt.box_*`/`unbox_*` to clif alloc + load/store
//! - `rc_insertion`: Insert `tribute_rt.retain`/`release` for reference counting
//! - `rc_lowering`: Lower `tribute_rt.retain`/`release` to inline `clif.*` ops

pub mod adt_rc_header;
pub mod cont_rc;
pub mod entrypoint;
pub mod evidence;
pub mod rc_insertion;
pub mod rc_lowering;
pub mod rtti;
pub mod tribute_rt_to_clif;
pub mod type_converter;
