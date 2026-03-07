//! TrunkIR dialect modules.
//!
//! These modules only contain `register_pure_op!` and `register_isolated_op!` entries
//! for the `inventory`-based operation property registries.
//!
//! Arena dialect definitions with full operation wrappers are in `arena/dialect/`.

pub mod adt;
pub mod arith;
pub mod cont;
pub mod core;
pub mod func;
pub mod mem;
