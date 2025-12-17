//! Type checking and inference for TrunkIR.
//!
//! This module implements bidirectional type checking with row-polymorphic
//! effect types, following the design in `new-plans/type-inference.md`.
//!
//! ## Pipeline
//!
//! 1. **Constraint Generation**: Walk TrunkIR ops, generate constraints
//! 2. **Constraint Solving**: Unify types and effect rows
//! 3. **Substitution**: Apply solved types back to IR
//!
//! ## Key Types
//!
//! - [`Constraint`]: Type and row equality constraints
//! - [`EffectRow`]: Row-polymorphic effect types
//! - [`TypeSolver`]: Union-find based constraint solver
//! - [`TypeChecker`]: Bidirectional type checker

mod checker;
mod constraint;
mod effect_row;
mod solver;

pub use checker::{TypeChecker, typecheck_module};
pub use constraint::Constraint;
pub use effect_row::EffectRow;
pub use solver::TypeSolver;
