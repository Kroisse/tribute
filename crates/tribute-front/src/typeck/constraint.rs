//! Type inference constraints.
//!
//! Constraints are generated during type checking and solved by unification.
//!
//! ```text
//! C ::= τ₁ = τ₂           -- type equality
//!     | ρ₁ = ρ₂           -- effect row equality
//!     | C₁ ∧ C₂           -- conjunction
//! ```

use crate::ast::{EffectRow, NodeId, Type};

/// Source location and semantic role for a generated constraint.
///
/// The solver operates on normalized types and rows, so without this metadata
/// it can only report an error against the enclosing function.  Keeping the
/// originating AST node lets the diagnostic layer point back to the call,
/// lambda, or handler boundary that introduced the failed constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintOrigin {
    pub node_id: NodeId,
    pub kind: ConstraintOriginKind,
}

/// The source construct that introduced a constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintOriginKind {
    Expression,
    Call,
    Lambda,
    HandlerBoundary,
}

/// A type inference constraint.
#[derive(Clone, Debug)]
pub enum Constraint<'db> {
    /// Type equality: τ₁ = τ₂
    TypeEq(Type<'db>, Type<'db>),

    /// Effect row equality: ρ₁ = ρ₂
    RowEq(EffectRow<'db>, EffectRow<'db>),

    /// Type equality with a source origin.
    TypeEqAt(Type<'db>, Type<'db>, ConstraintOrigin),

    /// Effect row equality with a source origin.
    RowEqAt(EffectRow<'db>, EffectRow<'db>, ConstraintOrigin),

    /// Conjunction of constraints.
    And(Vec<Constraint<'db>>),
}

impl<'db> Constraint<'db> {
    /// Create a type equality constraint.
    pub fn type_eq(t1: Type<'db>, t2: Type<'db>) -> Self {
        Self::TypeEq(t1, t2)
    }

    /// Create a row equality constraint.
    pub fn row_eq(r1: EffectRow<'db>, r2: EffectRow<'db>) -> Self {
        Self::RowEq(r1, r2)
    }

    /// Create a source-originated type equality constraint.
    pub fn type_eq_at(t1: Type<'db>, t2: Type<'db>, origin: ConstraintOrigin) -> Self {
        Self::TypeEqAt(t1, t2, origin)
    }

    /// Create a source-originated row equality constraint.
    pub fn row_eq_at(r1: EffectRow<'db>, r2: EffectRow<'db>, origin: ConstraintOrigin) -> Self {
        Self::RowEqAt(r1, r2, origin)
    }

    /// Create a conjunction of constraints.
    pub fn and(constraints: Vec<Constraint<'db>>) -> Self {
        Self::And(constraints)
    }

    /// Flatten a constraint into a list of atomic constraints.
    pub fn flatten(self) -> Vec<Constraint<'db>> {
        match self {
            Self::And(cs) => cs.into_iter().flat_map(|c| c.flatten()).collect(),
            other => vec![other],
        }
    }
}

/// A set of constraints to be solved.
#[derive(Clone, Debug, Default)]
pub struct ConstraintSet<'db> {
    constraints: Vec<Constraint<'db>>,
}

impl<'db> ConstraintSet<'db> {
    /// Create an empty constraint set.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the set.
    pub fn add(&mut self, constraint: Constraint<'db>) {
        self.constraints.push(constraint);
    }

    /// Add a type equality constraint.
    pub fn add_type_eq(&mut self, t1: Type<'db>, t2: Type<'db>) {
        self.add(Constraint::type_eq(t1, t2));
    }

    /// Add a row equality constraint.
    pub fn add_row_eq(&mut self, r1: EffectRow<'db>, r2: EffectRow<'db>) {
        self.add(Constraint::row_eq(r1, r2));
    }

    /// Add a source-originated type equality constraint.
    pub fn add_type_eq_at(&mut self, t1: Type<'db>, t2: Type<'db>, origin: ConstraintOrigin) {
        self.add(Constraint::type_eq_at(t1, t2, origin));
    }

    /// Add a source-originated effect row equality constraint.
    pub fn add_row_eq_at(
        &mut self,
        r1: EffectRow<'db>,
        r2: EffectRow<'db>,
        origin: ConstraintOrigin,
    ) {
        self.add(Constraint::row_eq_at(r1, r2, origin));
    }

    /// Extend with constraints from another set.
    pub fn extend(&mut self, other: ConstraintSet<'db>) {
        self.constraints.extend(other.constraints);
    }

    /// Get all constraints.
    pub fn constraints(&self) -> &[Constraint<'db>] {
        &self.constraints
    }

    /// Take all constraints, consuming the set.
    pub fn into_constraints(self) -> Vec<Constraint<'db>> {
        self.constraints
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get the number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }
}
