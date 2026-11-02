//! Typed, value-carrying Tribute source strategies for property-based testing.
//!
//! Each strategy yields a [`Fragment`]: a piece of Tribute source text with
//! the value it evaluates to, and so its type. The strategies emit text
//! directly; there is no term model or printer. Every compound expression is
//! wrapped in `{ … }`, so no precedence handling is needed.
//!
//! Generation is top-down and type-directed: a [`Goal`] names the type to
//! produce, and for `Nat` an upper bound on its value. `+` and `*` choose
//! their right operand within the bound the left one leaves, so every
//! intermediate value stays in the current unsigned 32-bit range
//! (`new-plans/numeric-types.md`) by construction.
//!
//! This crate holds strategies only. Properties live next to the code they
//! test.

mod env;
mod expr;
mod fragment;
mod program;
mod value;

pub use env::{Binding, Env, Goal};
pub use expr::fragment;
pub use fragment::Fragment;
pub use program::{Check, Program, program, program_with};
pub use value::{Ty, Value, ty};
