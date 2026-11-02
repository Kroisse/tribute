//! BoundVar substitution utilities.
//!
//! This module provides shared substitution logic for replacing BoundVar types
//! with actual types during type scheme instantiation.

use std::convert::Infallible;

use rustc_hash::FxHashMap as HashMap;

use crate::ast::fold::{TypeFoldable, TypeFolder};
use crate::ast::{AbilityRow, RowVar, Type, TypeKind, TypeScheme};

/// A BoundVar index past the substitution arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundVarOutOfBounds {
    /// The out-of-range BoundVar index.
    pub index: u32,
    /// The number of substitution arguments.
    pub max: usize,
}

/// Replaces scheme `BoundVar(i)` with `args[i]`, without folding into the
/// arguments. An index past `args` is reported; the caller decides whether
/// it is a diagnostic or a broken invariant. `LocalBoundVar` and `UniVar`
/// pass through unchanged.
pub struct BoundVarSubst<'a, 'db> {
    db: &'db dyn salsa::Database,
    args: &'a [Type<'db>],
}

impl<'a, 'db> BoundVarSubst<'a, 'db> {
    pub fn new(db: &'db dyn salsa::Database, args: &'a [Type<'db>]) -> Self {
        Self { db, args }
    }
}

impl<'db> TypeFolder<'db> for BoundVarSubst<'_, 'db> {
    type Error = BoundVarOutOfBounds;

    fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    fn fold_type(&mut self, ty: Type<'db>) -> Result<Type<'db>, BoundVarOutOfBounds> {
        match ty.kind(self.db) {
            TypeKind::BoundVar { index } => {
                self.args
                    .get(*index as usize)
                    .copied()
                    .ok_or(BoundVarOutOfBounds {
                        index: *index,
                        max: self.args.len(),
                    })
            }
            _ => ty.try_super_fold_with(self),
        }
    }
}

/// Substitute BoundVars in a type with the given substitution types.
///
/// Returns [`BoundVarOutOfBounds`] if a BoundVar index exceeds `subst.len()`.
pub fn substitute_bound_vars<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    subst: &[Type<'db>],
) -> Result<Type<'db>, BoundVarOutOfBounds> {
    ty.try_fold_with(&mut BoundVarSubst::new(db, subst))
}

/// Substitute BoundVars within an ability row.
///
/// Returns [`BoundVarOutOfBounds`] if a BoundVar index exceeds `subst.len()`.
pub fn substitute_ability_row<'db>(
    db: &'db dyn salsa::Database,
    row: AbilityRow<'db>,
    subst: &[Type<'db>],
) -> Result<AbilityRow<'db>, BoundVarOutOfBounds> {
    row.try_fold_with(&mut BoundVarSubst::new(db, subst))
}

/// One instantiation mapping is shared by the signature and its retained rows.
pub struct SchemeInstance<'db> {
    pub ty: Type<'db>,
    pub type_args: Vec<Type<'db>>,
    pub row_args: Vec<AbilityRow<'db>>,
    pub row_unions: Vec<crate::ast::RowUnion<'db>>,
    pub row_removals: Vec<crate::ast::RowRemoval<'db>>,
}

/// Instantiates one scheme: its `BoundVar`s become the type arguments and
/// its quantified row tails the row arguments. Substituted type arguments
/// are caller-owned and are not folded, so their rows keep their identities.
struct Instantiate<'a, 'db> {
    types: BoundVarSubst<'a, 'db>,
    rows: HashMap<RowVar, RowVar>,
}

impl<'db> TypeFolder<'db> for Instantiate<'_, 'db> {
    type Error = Infallible;

    fn db(&self) -> &'db dyn salsa::Database {
        self.types.db
    }

    fn fold_type(&mut self, ty: Type<'db>) -> Result<Type<'db>, Infallible> {
        match ty.kind(self.types.db) {
            TypeKind::BoundVar { .. } => {
                Ok(self
                    .types
                    .fold_type(ty)
                    .unwrap_or_else(|BoundVarOutOfBounds { index, max }| {
                        panic!("scheme binder {index} out of {max}")
                    }))
            }
            _ => ty.try_super_fold_with(self),
        }
    }

    fn fold_row(&mut self, row: AbilityRow<'db>) -> Result<AbilityRow<'db>, Infallible> {
        let db = self.types.db;
        let row = row.try_super_fold_with(self)?;
        Ok(match row.rest(db).and_then(|var| self.rows.get(&var)) {
            Some(fresh) => AbilityRow::new(db, row.instances(db).to_vec(), Some(*fresh)),
            None => row,
        })
    }
}

pub fn instantiate_with_arguments<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    type_args: Vec<Type<'db>>,
    row_vars: Vec<RowVar>,
) -> SchemeInstance<'db> {
    assert_eq!(scheme.type_params(db).len(), type_args.len());
    assert_eq!(scheme.effect_params(db).len(), row_vars.len());
    let mut folder = Instantiate {
        types: BoundVarSubst::new(db, &type_args),
        rows: scheme
            .effect_params(db)
            .iter()
            .copied()
            .zip(row_vars.iter().copied())
            .collect(),
    };
    let ty = scheme.body(db).fold_with(&mut folder);
    let row_unions = scheme.row_unions(db).to_vec().fold_with(&mut folder);
    let row_removals = scheme.row_removals(db).to_vec().fold_with(&mut folder);
    SchemeInstance {
        ty,
        type_args,
        row_args: row_vars
            .into_iter()
            .map(|var| AbilityRow::open(db, var))
            .collect(),
        row_unions,
        row_removals,
    }
}

/// Instantiate a TypeScheme for use in the post-solve deferred resolution loop.
///
/// Replaces BoundVars with fresh UniVars from the solver. This is similar to
/// `FunctionInferenceContext::instantiate_scheme` but works with the solver's
/// type variable allocation instead of the context's.
pub fn instantiate_scheme_for_solver<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    solver: &mut super::solver::TypeSolver<'db>,
) -> Type<'db> {
    instantiate_scheme_details_for_solver(db, scheme, solver).ty
}

pub fn instantiate_scheme_details_for_solver<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    solver: &mut super::solver::TypeSolver<'db>,
) -> SchemeInstance<'db> {
    solver.reserve_row_vars_in_type(scheme.body(db));
    for union in scheme.row_unions(db) {
        for row in union.sources.iter().chain(std::iter::once(&union.result)) {
            solver.reserve_row_vars_in_row(*row);
        }
    }
    for removal in scheme.row_removals(db) {
        for row in removal.rows() {
            solver.reserve_row_vars_in_row(row);
        }
    }
    let types = scheme
        .type_params(db)
        .iter()
        .map(|_| solver.fresh_type_var(db))
        .collect();
    let rows = scheme
        .effect_params(db)
        .iter()
        .map(|_| solver.fresh_row_var())
        .collect();
    let instance = instantiate_with_arguments(db, scheme, types, rows);
    solver.add_row_unions(instance.row_unions.clone());
    solver.add_row_removals(instance.row_removals.clone());
    instance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, AbilityInstance, AbilityRow};
    use crate::typeck::TypeSolver;
    use salsa_test_macros::salsa_test;

    /// Helper to create a simple AbilityId with empty module path
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::source(db, trunk_ir::SymbolPath::new([name]))
    }

    // Substitution laws are properties in `laws` below.

    // =========================================================================
    // Instantiation tests
    // =========================================================================

    #[salsa_test]
    fn instantiation_does_not_capture_rows_in_caller_type_arguments(db: &dyn salsa::Database) {
        let shared_id = RowVar { id: 7 };
        let nil = Type::new(db, TypeKind::Nil);
        let bound = Type::new(db, TypeKind::BoundVar { index: 0 });
        let callback = Type::new(
            db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: AbilityRow::open(db, shared_id),
            },
        );
        let row = AbilityRow::new(
            db,
            vec![AbilityInstance {
                ability_id: test_ability_id(db, "Writer"),
                args: vec![bound],
            }],
            Some(shared_id),
        );
        let body = Type::new(
            db,
            TypeKind::Func {
                params: vec![bound],
                result: bound,
                effect: row,
            },
        );
        let scheme = TypeScheme::builder(
            vec![crate::ast::TypeParam::anonymous()],
            vec![shared_id],
            body,
        )
        .row_unions(vec![crate::ast::RowUnion {
            sources: vec![row],
            result: row,
        }])
        .build(db);
        for id in [8, 9] {
            let fresh = RowVar { id };
            let instance = instantiate_with_arguments(db, scheme, vec![callback], vec![fresh]);
            let TypeKind::Func {
                params,
                result,
                effect,
                ..
            } = instance.ty.kind(db)
            else {
                panic!("function")
            };
            assert_eq!(params, &vec![callback]);
            assert_eq!(*result, callback);
            assert_eq!(instance.type_args, vec![callback]);
            for row in std::iter::once(effect)
                .chain(instance.row_unions[0].sources.iter())
                .chain(std::iter::once(&instance.row_unions[0].result))
            {
                assert_eq!(row.rest(db), Some(fresh));
                assert_eq!(row.instances(db)[0].args, vec![callback]);
            }
        }
    }

    #[salsa_test]
    fn bound_var_substitution_reaches_retained_row_constraints(db: &dyn salsa::Database) {
        let tail = RowVar { id: 3 };
        let int = Type::new(db, TypeKind::Int);
        let row = |arg, rest| {
            AbilityRow::new(
                db,
                vec![AbilityInstance {
                    ability_id: test_ability_id(db, "State"),
                    args: vec![arg],
                }],
                rest,
            )
        };
        let scheme = |arg| {
            TypeScheme::builder(
                vec![crate::ast::TypeParam::anonymous()],
                vec![tail],
                Type::new(db, TypeKind::Nil),
            )
            .row_unions(vec![crate::ast::RowUnion {
                sources: vec![row(arg, None), AbilityRow::open(db, tail)],
                result: row(arg, Some(tail)),
            }])
            .row_removals(vec![crate::ast::RowRemoval {
                source: row(arg, Some(tail)),
                removed: row(arg, None),
                result: AbilityRow::open(db, tail),
            }])
            .build(db)
        };
        let bound = Type::new(db, TypeKind::BoundVar { index: 0 });
        assert_eq!(
            scheme(bound).try_fold_with(&mut BoundVarSubst::new(db, &[int])),
            Ok(scheme(int))
        );
        assert_eq!(
            scheme(bound).try_fold_with(&mut BoundVarSubst::new(db, &[])),
            Err(BoundVarOutOfBounds { index: 0, max: 0 })
        );
    }

    #[salsa_test]
    fn solver_instantiation_freshens_quantified_ability_rows(db: &dyn salsa::Database) {
        let quantified_row = RowVar { id: 5000 };
        let function_ty = Type::new(
            db,
            TypeKind::Func {
                params: Vec::new(),
                result: Type::new(db, TypeKind::Nil),
                effect: AbilityRow::new(db, Vec::new(), Some(quantified_row)),
            },
        );
        let scheme = TypeScheme::new(
            db,
            Vec::new(),
            vec![quantified_row],
            Type::new(db, TypeKind::Tuple(vec![function_ty, function_ty])),
        );
        let mut solver = TypeSolver::new(db);

        let first = instantiate_scheme_for_solver(db, scheme, &mut solver);
        let second = instantiate_scheme_for_solver(db, scheme, &mut solver);

        let row_tails = |ty: Type<'_>| {
            let TypeKind::Tuple(elements) = ty.kind(db) else {
                panic!("instantiated scheme should remain a tuple");
            };
            elements
                .iter()
                .map(|element| {
                    let TypeKind::Func { effect, .. } = element.kind(db) else {
                        panic!("tuple element should remain a function");
                    };
                    effect.rest(db).expect("ability row should remain open")
                })
                .collect::<Vec<_>>()
        };
        let first_tails = row_tails(first);
        let second_tails = row_tails(second);

        assert_eq!(first_tails[0], first_tails[1]);
        assert_eq!(second_tails[0], second_tails[1]);
        assert_ne!(first_tails[0], second_tails[0]);
        assert_ne!(first_tails[0], quantified_row);
        assert!(first_tails[0].id > quantified_row.id);
    }
}

/// Laws of bound-variable substitution, checked against a reference model on
/// generated types.
#[cfg(test)]
mod laws {
    use proptest::prelude::*;

    use super::*;
    use crate::typeck::prop::{RowShape, TypeGen, TypeShape, row_shape, type_shape};

    /// Substitution arguments: open types without bound variables.
    const ARGS: TypeGen = TypeGen::GROUND.univars(2).row_vars(2);

    /// A type over `BoundVar(0..n)` with `n` arguments.
    fn scheme_body_and_args(
        cfg: impl Fn(u32) -> TypeGen + Clone + 'static,
    ) -> BoxedStrategy<(TypeShape, Vec<TypeShape>)> {
        (1u32..=3)
            .prop_flat_map(move |n| {
                (
                    type_shape(cfg(n)),
                    proptest::collection::vec(type_shape(ARGS), n as usize),
                )
            })
            .boxed()
    }

    fn body(n: u32) -> TypeGen {
        ARGS.bound_vars(n).continuations(true)
    }

    fn subst<'db>(
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        args: &[Type<'db>],
    ) -> Result<Type<'db>, BoundVarOutOfBounds> {
        ty.try_fold_with(&mut BoundVarSubst::new(db, args))
    }

    fn build_all<'db>(db: &'db dyn salsa::Database, shapes: &[TypeShape]) -> Vec<Type<'db>> {
        shapes.iter().map(|shape| shape.build(db)).collect()
    }

    proptest! {
        /// Substitution replaces `BoundVar(i)` with `args[i]` everywhere,
        /// effect arguments included, and changes nothing else.
        #[test]
        fn substitution_matches_its_model((ty, args) in scheme_body_and_args(body)) {
            let db = salsa::DatabaseImpl::new();
            let expected = ty.substitute_bound(&args).expect("indices in range");
            prop_assert_eq!(
                subst(&db, ty.build(&db), &build_all(&db, &args)),
                Ok(expected.build(&db))
            );
        }

        /// A bound variable is replaced by its argument.
        #[test]
        fn bound_var_is_replaced_by_its_argument(
            args in proptest::collection::vec(type_shape(ARGS), 1..=4),
            pick in any::<proptest::sample::Index>(),
        ) {
            let db = salsa::DatabaseImpl::new();
            let index = pick.index(args.len());
            let bound = Type::new(&db, TypeKind::BoundVar { index: index as u32 });
            prop_assert_eq!(
                subst(&db, bound, &build_all(&db, &args)),
                Ok(args[index].build(&db))
            );
        }

        /// Types without bound variables are returned unchanged, with any
        /// arguments; rows without them keep their interned identity.
        #[test]
        fn types_without_bound_vars_are_unchanged(
            ty in type_shape(ARGS.continuations(true)),
            row in row_shape(ARGS),
            args in proptest::collection::vec(type_shape(ARGS), 0..=2),
        ) {
            let db = salsa::DatabaseImpl::new();
            let args = build_all(&db, &args);
            let ty = ty.build(&db);
            prop_assert_eq!(subst(&db, ty, &args), Ok(ty));
            let row = row.build(&db);
            prop_assert_eq!(substitute_ability_row(&db, row, &args), Ok(row));
        }

        /// Substituting each bound variable by itself is the identity.
        #[test]
        fn identity_arguments_are_the_identity((ty, args) in scheme_body_and_args(body)) {
            let db = salsa::DatabaseImpl::new();
            let identity: Vec<_> = (0..args.len() as u32)
                .map(|index| Type::new(&db, TypeKind::BoundVar { index }))
                .collect();
            let ty = ty.build(&db);
            prop_assert_eq!(subst(&db, ty, &identity), Ok(ty));
        }

        /// Substituting in sequence equals substituting once with the first
        /// arguments substituted by the second.
        #[test]
        fn substitutions_compose(
            (ty, first) in (1u32..=3).prop_flat_map(|n| (
                type_shape(body(n)),
                // Each argument of `first` may mention any BoundVar(0..3) of
                // the second substitution, at any position.
                proptest::collection::vec(type_shape(ARGS.bound_vars(3)), n as usize),
            )),
            second in proptest::collection::vec(type_shape(ARGS), 3),
        ) {
            let db = salsa::DatabaseImpl::new();
            let second = build_all(&db, &second);
            let ty = ty.build(&db);
            let Ok(once) = subst(&db, ty, &build_all(&db, &first)) else {
                panic!("indices in range");
            };
            let composed: Vec<_> = build_all(&db, &first)
                .into_iter()
                .map(|arg| match subst(&db, arg, &second) {
                    Ok(arg) => arg,
                    Err(_) => panic!("indices in range"),
                })
                .collect();
            prop_assert_eq!(
                subst(&db, once, &second),
                subst(&db, ty, &composed)
            );
        }

        /// An index past the arguments is reported with the argument count,
        /// including one inside an ability row.
        #[test]
        fn out_of_range_index_is_reported(
            (ty, args) in (0u32..=2).prop_flat_map(|n| (
                type_shape(ARGS.bound_vars(n + 2).continuations(true)),
                proptest::collection::vec(type_shape(ARGS), n as usize),
            )),
        ) {
            let db = salsa::DatabaseImpl::new();
            let indices = ty.bound_vars();
            let in_range = indices.iter().all(|index| (*index as usize) < args.len());
            let result = subst(&db, ty.build(&db), &build_all(&db, &args));
            match result {
                Ok(_) => prop_assert!(in_range),
                Err(BoundVarOutOfBounds { index, max }) => {
                    prop_assert!(!in_range);
                    prop_assert_eq!(max, args.len());
                    prop_assert!(index as usize >= max && indices.contains(&index));
                }
            }
        }
    }

    /// An out-of-range index in a function type's ability row is reported as
    /// `OutOfBounds` rather than panicking.
    #[test]
    fn out_of_range_bound_var_in_ability_row_is_reported() {
        let db = salsa::DatabaseImpl::new();
        let bound = TypeShape::BoundVar(1);
        let state = crate::typeck::prop::AbilityInstanceShape {
            ability: 1,
            args: vec![bound],
        };
        let func = TypeShape::Func {
            params: vec![],
            result: Box::new(TypeShape::Prim(crate::typeck::prop::Prim::Nil)),
            effect: RowShape::closed(vec![state]),
        };
        let int = Type::new(&db, TypeKind::Int);
        assert_eq!(
            subst(&db, func.build(&db), &[int]),
            Err(BoundVarOutOfBounds { index: 1, max: 1 })
        );
    }
}
