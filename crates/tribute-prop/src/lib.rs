//! Typed, value-carrying Tribute source strategies for property-based testing.
//!
//! Each strategy yields a [`Fragment`](fragment::Fragment): a piece of Tribute source text with
//! the value it evaluates to, and so its type. The strategies emit text
//! directly; there is no term model or printer. Every compound expression is
//! wrapped in `{ … }`, so no precedence handling is needed.
//!
//! Generation is top-down and type-directed: a [`Goal`](env::Goal) names the type to
//! produce, and for `Nat` an upper bound on its value. `+` and `*` choose
//! their right operand within the bound the left one leaves, so every
//! intermediate value stays in the current unsigned 32-bit range
//! (`design/numeric-types.md`) by construction.
//!
//! This crate holds strategies only. Properties live next to the code they
//! test.

pub mod env;
pub mod expr;
pub mod fragment;
pub mod program;
pub mod value;
