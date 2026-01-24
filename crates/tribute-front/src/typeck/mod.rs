//! Type checking for the AST.
//!
//! This module transforms `Module<ResolvedRef<'db>>` into `Module<TypedRef<'db>>`
//! by performing bidirectional type checking with row-polymorphic effects.
//!
//! ## Pipeline
//!
//! 1. Collect type definitions and function signatures
//! 2. Generate type constraints by walking the AST
//! 3. Solve constraints via unification
//! 4. Apply the solved substitution to produce typed AST
//!
//! ## Type System Features
//!
//! - Hindley-Milner type inference with let-polymorphism
//! - Bidirectional type checking (infer/check modes)
//! - Row-polymorphic effect tracking
//! - Algebraic data types (structs, enums)
//! - Pattern matching exhaustiveness (future)

mod checker;
mod constraint;
mod context;
pub mod effect_row;
mod solver;

pub use checker::TypeChecker;
pub use constraint::{Constraint, ConstraintSet, TypeVar};
pub use context::TypeContext;
pub use solver::{SolveError, TypeSolver};

use crate::ast::{Module, ResolvedRef, TypedRef};

/// Type check a module.
///
/// This is the main entry point for type checking.
pub fn typecheck_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<ResolvedRef<'db>>,
) -> Module<TypedRef<'db>> {
    let checker = TypeChecker::new(db);
    checker.check_module(module)
}
