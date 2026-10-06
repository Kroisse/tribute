//! BoundVar substitution utilities.
//!
//! This module provides shared substitution logic for replacing BoundVar types
//! with actual types during type scheme instantiation.

use rustc_hash::FxHashMap as HashMap;

use crate::ast::{Effect, EffectRow, EffectVar, Type, TypeKind, TypeScheme};

/// Result of BoundVar substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubstResult<'db> {
    /// Substitution succeeded.
    Ok(Type<'db>),
    /// BoundVar index was out of bounds.
    OutOfBounds { index: u32, max: usize },
}

/// A BoundVar index past the substitution arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundVarOutOfBounds {
    /// The out-of-range BoundVar index.
    pub index: u32,
    /// The number of substitution arguments.
    pub max: usize,
}

impl<'db> SubstResult<'db> {
    /// Returns the substituted type, or calls the fallback function if out of bounds.
    pub fn unwrap_or_else(self, fallback: impl FnOnce(u32, usize) -> Type<'db>) -> Type<'db> {
        match self {
            SubstResult::Ok(ty) => ty,
            SubstResult::OutOfBounds { index, max } => fallback(index, max),
        }
    }

    /// Returns the substituted type, or returns the given fallback type if out of bounds.
    pub fn unwrap_or(self, fallback: Type<'db>) -> Type<'db> {
        match self {
            SubstResult::Ok(ty) => ty,
            SubstResult::OutOfBounds { .. } => fallback,
        }
    }

    /// Returns true if substitution was successful.
    pub fn is_ok(&self) -> bool {
        matches!(self, SubstResult::Ok(_))
    }
}

/// Substitute BoundVars in a type with the given substitution types.
///
/// Returns `SubstResult::OutOfBounds` if a BoundVar's index exceeds `subst.len()`.
pub fn substitute_bound_vars<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    subst: &[Type<'db>],
) -> SubstResult<'db> {
    match ty.kind(db) {
        TypeKind::BoundVar { index } => {
            if let Some(&ty) = subst.get(*index as usize) {
                SubstResult::Ok(ty)
            } else {
                SubstResult::OutOfBounds {
                    index: *index,
                    max: subst.len(),
                }
            }
        }
        TypeKind::Named { id, name, args } => {
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                match substitute_bound_vars(db, *arg, subst) {
                    SubstResult::Ok(ty) => new_args.push(ty),
                    err @ SubstResult::OutOfBounds { .. } => return err,
                }
            }
            SubstResult::Ok(Type::new(
                db,
                TypeKind::Named {
                    id: *id,
                    name: name.clone(),
                    args: new_args,
                },
            ))
        }
        TypeKind::Func {
            params,
            result,
            effect,
            minimum_convention,
        } => {
            let mut new_params = Vec::with_capacity(params.len());
            for param in params {
                match substitute_bound_vars(db, *param, subst) {
                    SubstResult::Ok(ty) => new_params.push(ty),
                    err @ SubstResult::OutOfBounds { .. } => return err,
                }
            }
            let new_result = match substitute_bound_vars(db, *result, subst) {
                SubstResult::Ok(ty) => ty,
                err @ SubstResult::OutOfBounds { .. } => return err,
            };
            let new_effect = match substitute_effect_row(db, *effect, subst) {
                Ok(row) => row,
                Err(BoundVarOutOfBounds { index, max }) => {
                    return SubstResult::OutOfBounds { index, max };
                }
            };
            SubstResult::Ok(Type::new(
                db,
                TypeKind::Func {
                    params: new_params,
                    result: new_result,
                    effect: new_effect,
                    minimum_convention: *minimum_convention,
                },
            ))
        }
        TypeKind::Tuple(elements) => {
            let mut new_elements = Vec::with_capacity(elements.len());
            for elem in elements {
                match substitute_bound_vars(db, *elem, subst) {
                    SubstResult::Ok(ty) => new_elements.push(ty),
                    err @ SubstResult::OutOfBounds { .. } => return err,
                }
            }
            SubstResult::Ok(Type::new(db, TypeKind::Tuple(new_elements)))
        }
        TypeKind::App { ctor, args } => {
            let new_ctor = match substitute_bound_vars(db, *ctor, subst) {
                SubstResult::Ok(ty) => ty,
                err @ SubstResult::OutOfBounds { .. } => return err,
            };
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                match substitute_bound_vars(db, *arg, subst) {
                    SubstResult::Ok(ty) => new_args.push(ty),
                    err @ SubstResult::OutOfBounds { .. } => return err,
                }
            }
            SubstResult::Ok(Type::new(
                db,
                TypeKind::App {
                    ctor: new_ctor,
                    args: new_args,
                },
            ))
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            let new_arg = match substitute_bound_vars(db, *arg, subst) {
                SubstResult::Ok(ty) => ty,
                err @ SubstResult::OutOfBounds { .. } => return err,
            };
            let new_result = match substitute_bound_vars(db, *result, subst) {
                SubstResult::Ok(ty) => ty,
                err @ SubstResult::OutOfBounds { .. } => return err,
            };
            let new_effect = match substitute_effect_row(db, *effect, subst) {
                Ok(row) => row,
                Err(BoundVarOutOfBounds { index, max }) => {
                    return SubstResult::OutOfBounds { index, max };
                }
            };
            SubstResult::Ok(Type::new(
                db,
                TypeKind::Continuation {
                    arg: new_arg,
                    result: new_result,
                    effect: new_effect,
                },
            ))
        }
        // Primitive types and other type variables are unchanged
        _ => SubstResult::Ok(ty),
    }
}

/// Substitute BoundVars within an effect row.
///
/// Returns [`BoundVarOutOfBounds`] if a BoundVar index exceeds `subst.len()`.
pub fn substitute_effect_row<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    subst: &[Type<'db>],
) -> Result<EffectRow<'db>, BoundVarOutOfBounds> {
    let effects = row.effects(db);
    let mut changed = false;

    let mut new_effects = Vec::with_capacity(effects.len());
    for effect in effects {
        let mut new_args = Vec::with_capacity(effect.args.len());
        for arg in &effect.args {
            match substitute_bound_vars(db, *arg, subst) {
                SubstResult::Ok(ty) => new_args.push(ty),
                SubstResult::OutOfBounds { index, max } => {
                    return Err(BoundVarOutOfBounds { index, max });
                }
            }
        }
        if new_args != effect.args {
            changed = true;
        }
        new_effects.push(Effect {
            ability_id: effect.ability_id,
            args: new_args,
        });
    }

    if changed {
        Ok(EffectRow::new(db, new_effects, row.rest(db)))
    } else {
        Ok(row)
    }
}

/// One instantiation mapping is shared by the signature and its retained rows.
pub struct SchemeInstance<'db> {
    pub ty: Type<'db>,
    pub type_args: Vec<Type<'db>>,
    pub row_args: Vec<EffectRow<'db>>,
    pub row_unions: Vec<crate::ast::RowUnion<'db>>,
    pub row_removals: Vec<crate::ast::RowRemoval<'db>>,
}

pub fn instantiate_with_arguments<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    type_args: Vec<Type<'db>>,
    row_vars: Vec<EffectVar>,
) -> SchemeInstance<'db> {
    assert_eq!(scheme.type_params(db).len(), type_args.len());
    assert_eq!(scheme.effect_params(db).len(), row_vars.len());
    let mut mapping: HashMap<_, _> = scheme
        .effect_params(db)
        .iter()
        .zip(&row_vars)
        .map(|(old, new)| (old.id, *new))
        .collect();
    let mut fresh = || unreachable!("all quantified rows have allocated identities");
    let mut map_type = |ty| {
        // Freshen only scheme-owned rows before inserting caller-owned types.
        let ty =
            freshen_effect_vars_inner(db, ty, scheme.effect_params(db), &mut fresh, &mut mapping);
        substitute_bound_vars(db, ty, &type_args)
            .unwrap_or_else(|index, max| panic!("scheme binder {index} out of {max}"))
    };
    let ty = map_type(scheme.body(db));
    let mut map_row = |row: &mut EffectRow<'db>| {
        let effects: Vec<_> = row
            .effects(db)
            .iter()
            .map(|effect| Effect {
                ability_id: effect.ability_id,
                args: effect.args.iter().copied().map(&mut map_type).collect(),
            })
            .collect();
        let rest = row.rest(db).map(|var| {
            scheme
                .effect_params(db)
                .iter()
                .position(|p| *p == var)
                .map_or(var, |i| row_vars[i])
        });
        *row = EffectRow::new(db, effects, rest);
    };
    let mut row_unions = scheme.row_unions(db).to_vec();
    for union in &mut row_unions {
        union.for_each_row_mut(&mut map_row);
    }
    let mut row_removals = scheme.row_removals(db).to_vec();
    for removal in &mut row_removals {
        removal.for_each_row_mut(&mut map_row);
    }
    SchemeInstance {
        ty,
        type_args,
        row_args: row_vars
            .into_iter()
            .map(|var| EffectRow::open(db, var))
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
    solver.reserve_effect_vars_in_type(scheme.body(db));
    for union in scheme.row_unions(db) {
        for row in union.sources.iter().chain(std::iter::once(&union.result)) {
            solver.reserve_effect_vars_in_row(*row);
        }
    }
    for removal in scheme.row_removals(db) {
        for row in removal.rows() {
            solver.reserve_effect_vars_in_row(row);
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

fn freshen_effect_vars_inner<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    quantified_rows: &[EffectVar],
    fresh_row_var: &mut impl FnMut() -> EffectVar,
    row_vars: &mut HashMap<u64, EffectVar>,
) -> Type<'db> {
    let freshen_row =
        |row: EffectRow<'db>, fresh_row_var: &mut _, row_vars: &mut HashMap<u64, EffectVar>| {
            let effects: Vec<_> = row
                .effects(db)
                .iter()
                .map(|effect| Effect {
                    ability_id: effect.ability_id,
                    args: effect
                        .args
                        .iter()
                        .map(|arg| {
                            freshen_effect_vars_inner(
                                db,
                                *arg,
                                quantified_rows,
                                fresh_row_var,
                                row_vars,
                            )
                        })
                        .collect(),
                })
                .collect();
            let rest = row.rest(db).map(|var| {
                if !quantified_rows.contains(&var) {
                    return var;
                }
                *row_vars.entry(var.id).or_insert_with(&mut *fresh_row_var)
            });
            EffectRow::new(db, effects, rest)
        };

    match ty.kind(db) {
        TypeKind::Named { id, name, args } => Type::new(
            db,
            TypeKind::Named {
                id: *id,
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| {
                        freshen_effect_vars_inner(
                            db,
                            *arg,
                            quantified_rows,
                            fresh_row_var,
                            row_vars,
                        )
                    })
                    .collect(),
            },
        ),
        TypeKind::Func {
            params,
            result,
            effect,
            minimum_convention,
        } => Type::new(
            db,
            TypeKind::Func {
                params: params
                    .iter()
                    .map(|param| {
                        freshen_effect_vars_inner(
                            db,
                            *param,
                            quantified_rows,
                            fresh_row_var,
                            row_vars,
                        )
                    })
                    .collect(),
                result: freshen_effect_vars_inner(
                    db,
                    *result,
                    quantified_rows,
                    fresh_row_var,
                    row_vars,
                ),
                effect: freshen_row(*effect, fresh_row_var, row_vars),
                minimum_convention: *minimum_convention,
            },
        ),
        TypeKind::Tuple(elements) => Type::new(
            db,
            TypeKind::Tuple(
                elements
                    .iter()
                    .map(|element| {
                        freshen_effect_vars_inner(
                            db,
                            *element,
                            quantified_rows,
                            fresh_row_var,
                            row_vars,
                        )
                    })
                    .collect(),
            ),
        ),
        TypeKind::App { ctor, args } => Type::new(
            db,
            TypeKind::App {
                ctor: freshen_effect_vars_inner(
                    db,
                    *ctor,
                    quantified_rows,
                    fresh_row_var,
                    row_vars,
                ),
                args: args
                    .iter()
                    .map(|arg| {
                        freshen_effect_vars_inner(
                            db,
                            *arg,
                            quantified_rows,
                            fresh_row_var,
                            row_vars,
                        )
                    })
                    .collect(),
            },
        ),
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => Type::new(
            db,
            TypeKind::Continuation {
                arg: freshen_effect_vars_inner(db, *arg, quantified_rows, fresh_row_var, row_vars),
                result: freshen_effect_vars_inner(
                    db,
                    *result,
                    quantified_rows,
                    fresh_row_var,
                    row_vars,
                ),
                effect: freshen_row(*effect, fresh_row_var, row_vars),
            },
        ),
        TypeKind::BoundVar { .. }
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::UniVar { .. }
        | TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never
        | TypeKind::Error => ty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, CallingConvention, EffectRow};
    use crate::typeck::TypeSolver;
    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    /// Helper to create a simple AbilityId with empty module path
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::source(db, Symbol::new(name))
    }

    // Substitution laws are properties in `laws` below.

    // =========================================================================
    // Instantiation tests
    // =========================================================================

    #[salsa_test]
    fn instantiation_does_not_capture_rows_in_caller_type_arguments(db: &dyn salsa::Database) {
        let shared_id = EffectVar { id: 7 };
        let nil = Type::new(db, TypeKind::Nil);
        let bound = Type::new(db, TypeKind::BoundVar { index: 0 });
        let callback = Type::new(
            db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::open(db, shared_id),
                minimum_convention: CallingConvention::Direct,
            },
        );
        let row = EffectRow::new(
            db,
            vec![Effect {
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
                minimum_convention: CallingConvention::Direct,
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
            let fresh = EffectVar { id };
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
                assert_eq!(row.effects(db)[0].args, vec![callback]);
            }
        }
    }

    #[salsa_test]
    fn solver_instantiation_freshens_quantified_effect_rows(db: &dyn salsa::Database) {
        let quantified_row = EffectVar { id: 5000 };
        let function_ty = Type::new(
            db,
            TypeKind::Func {
                params: Vec::new(),
                result: Type::new(db, TypeKind::Nil),
                effect: EffectRow::new(db, Vec::new(), Some(quantified_row)),
                minimum_convention: CallingConvention::Direct,
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
                    effect.rest(db).expect("effect row should remain open")
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

    // =========================================================================
    // SubstResult helper method tests
    // =========================================================================

    #[salsa_test]
    fn test_subst_result_unwrap_or_else(db: &dyn salsa::Database) {
        let int_ty = Type::new(db, TypeKind::Int);
        let error_ty = Type::new(db, TypeKind::Error);

        // Ok case
        let ok_result = SubstResult::Ok(int_ty);
        let unwrapped = ok_result.unwrap_or_else(|_, _| error_ty);
        assert_eq!(unwrapped, int_ty);

        // OutOfBounds case
        let err_result = SubstResult::OutOfBounds { index: 5, max: 1 };
        let unwrapped = err_result.unwrap_or_else(|index, max| {
            assert_eq!(index, 5);
            assert_eq!(max, 1);
            error_ty
        });
        assert_eq!(unwrapped, error_ty);
    }

    #[salsa_test]
    fn test_subst_result_unwrap_or(db: &dyn salsa::Database) {
        let int_ty = Type::new(db, TypeKind::Int);
        let error_ty = Type::new(db, TypeKind::Error);

        // Ok case
        let ok_result = SubstResult::Ok(int_ty);
        assert_eq!(ok_result.unwrap_or(error_ty), int_ty);

        // OutOfBounds case
        let err_result: SubstResult<'_> = SubstResult::OutOfBounds { index: 5, max: 1 };
        assert_eq!(err_result.unwrap_or(error_ty), error_ty);
    }

    #[salsa_test]
    fn test_subst_result_is_ok(db: &dyn salsa::Database) {
        let int_ty = Type::new(db, TypeKind::Int);

        let ok_result = SubstResult::Ok(int_ty);
        assert!(ok_result.is_ok());

        let err_result: SubstResult<'_> = SubstResult::OutOfBounds { index: 0, max: 0 };
        assert!(!err_result.is_ok());
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
        ARGS.bound_vars(n).higher_kinded(true).conventions(true)
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
                substitute_bound_vars(&db, ty.build(&db), &build_all(&db, &args)),
                SubstResult::Ok(expected.build(&db))
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
                substitute_bound_vars(&db, bound, &build_all(&db, &args)),
                SubstResult::Ok(args[index].build(&db))
            );
        }

        /// Types without bound variables are returned unchanged, with any
        /// arguments; rows without them keep their interned identity.
        #[test]
        fn types_without_bound_vars_are_unchanged(
            ty in type_shape(ARGS.higher_kinded(true).conventions(true)),
            row in row_shape(ARGS),
            args in proptest::collection::vec(type_shape(ARGS), 0..=2),
        ) {
            let db = salsa::DatabaseImpl::new();
            let args = build_all(&db, &args);
            let ty = ty.build(&db);
            prop_assert_eq!(substitute_bound_vars(&db, ty, &args), SubstResult::Ok(ty));
            let row = row.build(&db);
            prop_assert_eq!(substitute_effect_row(&db, row, &args), Ok(row));
        }

        /// Substituting each bound variable by itself is the identity.
        #[test]
        fn identity_arguments_are_the_identity((ty, args) in scheme_body_and_args(body)) {
            let db = salsa::DatabaseImpl::new();
            let identity: Vec<_> = (0..args.len() as u32)
                .map(|index| Type::new(&db, TypeKind::BoundVar { index }))
                .collect();
            let ty = ty.build(&db);
            prop_assert_eq!(substitute_bound_vars(&db, ty, &identity), SubstResult::Ok(ty));
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
            let SubstResult::Ok(once) = substitute_bound_vars(&db, ty, &build_all(&db, &first)) else {
                panic!("indices in range");
            };
            let composed: Vec<_> = build_all(&db, &first)
                .into_iter()
                .map(|arg| match substitute_bound_vars(&db, arg, &second) {
                    SubstResult::Ok(arg) => arg,
                    SubstResult::OutOfBounds { .. } => panic!("indices in range"),
                })
                .collect();
            prop_assert_eq!(
                substitute_bound_vars(&db, once, &second),
                substitute_bound_vars(&db, ty, &composed)
            );
        }

        /// An index past the arguments is reported with the argument count,
        /// including one inside an effect row.
        #[test]
        fn out_of_range_index_is_reported(
            (ty, args) in (0u32..=2).prop_flat_map(|n| (
                type_shape(ARGS.bound_vars(n + 2).higher_kinded(true)),
                proptest::collection::vec(type_shape(ARGS), n as usize),
            )),
        ) {
            let db = salsa::DatabaseImpl::new();
            let indices = ty.bound_vars();
            let in_range = indices.iter().all(|index| (*index as usize) < args.len());
            let result = substitute_bound_vars(&db, ty.build(&db), &build_all(&db, &args));
            match result {
                SubstResult::Ok(_) => prop_assert!(in_range),
                SubstResult::OutOfBounds { index, max } => {
                    prop_assert!(!in_range);
                    prop_assert_eq!(max, args.len());
                    prop_assert!(index as usize >= max && indices.contains(&index));
                }
            }
        }
    }

    /// An out-of-range index in a function type's effect row is reported as
    /// `OutOfBounds` rather than panicking.
    #[test]
    fn out_of_range_bound_var_in_effect_row_is_reported() {
        let db = salsa::DatabaseImpl::new();
        let bound = TypeShape::BoundVar(1);
        let state = crate::typeck::prop::EffectShape {
            ability: 1,
            args: vec![bound],
        };
        let func = TypeShape::Func {
            params: vec![],
            result: Box::new(TypeShape::Prim(crate::typeck::prop::Prim::Nil)),
            effect: RowShape::closed(vec![state]),
            convention: crate::ast::CallingConvention::Direct,
        };
        let int = Type::new(&db, TypeKind::Int);
        assert_eq!(
            substitute_bound_vars(&db, func.build(&db), &[int]),
            SubstResult::OutOfBounds { index: 1, max: 1 }
        );
    }
}
