//! TrunkIR dialect modules.
//!
//! Each dialect defines a set of operations with the `<dialect>.<operation>` naming convention.
//! See `new-plans/ir.md` for the full dialect hierarchy.

// === Infrastructure ===
pub mod core;
pub mod ty;

// === High-level (pre-resolution) ===
pub mod ability;
pub mod case;
pub mod closure;
pub mod src;

// === Mid-level (target independent) ===
pub mod adt;
pub mod arith;
pub mod cont;
pub mod func;
pub mod list;
pub mod mem;
pub mod scf;

// === Low-level (target specific) ===
pub mod wasm;
