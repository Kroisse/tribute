//! TrunkIR to Cranelift lowering passes.
//!
//! These passes lower language-agnostic TrunkIR dialects to clif operations.

pub mod adt_to_clif;
pub mod arith_to_clif;
pub mod cf_to_clif;
pub mod func_to_clif;
