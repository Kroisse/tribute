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
mod subst;

pub use checker::{
    AbilityOpKey, AbilityOpSignature, FunctionTypeResult, TypeChecker, typecheck_function,
    typecheck_module, typecheck_module_per_function,
};
pub use constraint::Constraint;
pub use effect_row::{AbilityRef, EffectRow, RemoveResult, RowVar};
pub use solver::TypeSolver;
pub use subst::{
    apply_subst_to_module, apply_subst_to_region, has_type_vars, module_has_type_vars,
};
