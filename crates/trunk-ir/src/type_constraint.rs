//! Static type bounds and projections used by declarative operation schemas.
//!
//! A bound in a typed `#[dialect]` operation definition (`T: IntegerLike`,
//! `S: func::FuncSig`, `P: ResumeToken`) names a Rust type implementing
//! [`TypeConstraint`]. Its [`ConstraintDesc`] tells the generated schema how to
//! test a `TypeRef` against the bound and which projections (`S::Inputs`,
//! `P::Input`) it provides. The `const fn` helpers here run while the schema
//! constant is evaluated, so malformed definitions fail to compile even when the
//! bound comes from another crate.

use crate::{IrContext, TypeRef};

/// Whether a projection yields one type or a type list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionKind {
    One,
    List,
}

/// A named projection provided by a bound.
#[derive(Clone, Copy, Debug)]
pub struct ProjectionDesc {
    pub name: &'static str,
    pub kind: ProjectionKind,
}

/// The value of a projection applied to a concrete type.
#[derive(Clone, Copy, Debug)]
pub enum Projected<'a> {
    One(TypeRef),
    List(&'a [TypeRef]),
}

/// Static description of a bound.
#[derive(Debug)]
pub struct ConstraintDesc {
    /// Diagnostic name: the IR type name for exact bounds (`func.func_sig`),
    /// the Rust name for categories (`IntegerLike`).
    pub name: &'static str,
    /// Exact bounds require one dialect type; a type variable may have at most
    /// one distinct exact bound. Category bounds may be combined freely.
    pub exact: bool,
    /// Projections this bound provides. `Type` is reserved and never listed.
    pub projections: &'static [ProjectionDesc],
    /// Whether a type satisfies the bound, including wrapper invariants.
    pub matches: fn(&IrContext, TypeRef) -> bool,
    /// Projection by index into `projections`. Returns `None` if the type does
    /// not match or the index is out of range.
    pub project: for<'a> fn(&'a IrContext, TypeRef, usize) -> Option<Projected<'a>>,
}

/// Implemented by Rust types usable as bounds in typed `#[dialect]` operations.
pub trait TypeConstraint {
    const DESC: &'static ConstraintDesc;
}

/// Implement [`TypeConstraint`] for a hand-written signature wrapper that
/// exposes `inputs`/`results` as the `Inputs`/`Results` projections.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_func_sig_constraint {
    ($wrapper:ty, $name:literal) => {
        impl $crate::type_constraint::TypeConstraint for $wrapper {
            const DESC: &'static $crate::type_constraint::ConstraintDesc =
                &$crate::type_constraint::ConstraintDesc {
                    name: $name,
                    exact: true,
                    projections: &[
                        $crate::type_constraint::ProjectionDesc {
                            name: "Inputs",
                            kind: $crate::type_constraint::ProjectionKind::List,
                        },
                        $crate::type_constraint::ProjectionDesc {
                            name: "Results",
                            kind: $crate::type_constraint::ProjectionKind::List,
                        },
                    ],
                    matches: |ctx, ty| {
                        <$wrapper as $crate::ops::DialectType>::from_type_ref(ctx, ty).is_some()
                    },
                    project: |ctx, ty, index| {
                        let sig = <$wrapper as $crate::ops::DialectType>::from_type_ref(ctx, ty)?;
                        match index {
                            0 => Some($crate::type_constraint::Projected::List(sig.inputs(ctx))),
                            1 => Some($crate::type_constraint::Projected::List(sig.results(ctx))),
                            _ => None,
                        }
                    },
                };
        }
    };
}

const fn str_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Reject two different exact bounds on one type variable.
pub const fn check_bounds(
    bounds: &'static [&'static ConstraintDesc],
) -> &'static [&'static ConstraintDesc] {
    let mut i = 0;
    while i < bounds.len() {
        if bounds[i].exact {
            let mut j = i + 1;
            while j < bounds.len() {
                if bounds[j].exact && !str_eq(bounds[i].name, bounds[j].name) {
                    panic!("type variable has two different exact bounds");
                }
                j += 1;
            }
        }
        i += 1;
    }
    bounds
}

/// Resolve `<S as B>::name` within one bound, checking its kind.
pub const fn projection_in(bound: &ConstraintDesc, name: &str, kind: ProjectionKind) -> usize {
    let mut i = 0;
    while i < bound.projections.len() {
        let p = &bound.projections[i];
        if str_eq(p.name, name) {
            if !kind_eq(p.kind, kind) {
                match p.kind {
                    ProjectionKind::List => {
                        panic!(
                            "type constraint projection is a type list but is used as a single type"
                        )
                    }
                    ProjectionKind::One => {
                        panic!(
                            "type constraint projection is a single type but is used as a type list"
                        )
                    }
                }
            }
            return i;
        }
        i += 1;
    }
    panic!("type constraint projection is not provided by the selected bound")
}

const fn kind_eq(a: ProjectionKind, b: ProjectionKind) -> bool {
    matches!(
        (a, b),
        (ProjectionKind::One, ProjectionKind::One) | (ProjectionKind::List, ProjectionKind::List)
    )
}

/// Resolve `S::name` across all bounds of a variable: exactly one bound must
/// provide it, with the expected kind. Returns `(bound, projection)` indices.
pub const fn resolve_projection(
    bounds: &[&ConstraintDesc],
    name: &str,
    kind: ProjectionKind,
) -> (usize, usize) {
    let mut found = None;
    let mut i = 0;
    while i < bounds.len() {
        let mut j = 0;
        while j < bounds[i].projections.len() {
            let p = &bounds[i].projections[j];
            if str_eq(p.name, name) {
                if found.is_some() {
                    panic!(
                        "type constraint projection is provided by more than one bound; use `<S as B>::X`"
                    );
                }
                found = Some((i, j, p.kind));
            }
            j += 1;
        }
        i += 1;
    }
    match found {
        Some((i, j, actual)) => {
            if !kind_eq(actual, kind) {
                match actual {
                    ProjectionKind::List => panic!(
                        "type constraint projection is a type list but is used as a single type"
                    ),
                    ProjectionKind::One => panic!(
                        "type constraint projection is a single type but is used as a type list"
                    ),
                }
            }
            (i, j)
        }
        None => {
            panic!("type constraint projection is not provided by any bound of the type variable")
        }
    }
}
