//! TrunkIR dialect modules.
//!
//! Each dialect defines a set of operations with the `<dialect>.<operation>` naming convention.
//! See `new-plans/ir.md` for the full dialect hierarchy.
//!
//! Note: Tribute-specific dialects (ability, case, closure, list, pat, src, ty)
//! have moved to the `tribute-ir` crate.

// === Infrastructure ===
pub mod core;

// === Mid-level (target independent) ===
pub mod adt;
pub mod arith;
pub mod cont;
pub mod func;
pub mod mem;
pub mod scf;

// === Low-level (target specific) ===
pub mod wasm;
