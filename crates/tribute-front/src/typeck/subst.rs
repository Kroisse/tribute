//! BoundVar substitution utilities.
//!
//! This module provides shared substitution logic for replacing BoundVar types
//! with actual types during type scheme instantiation.

use crate::ast::{Effect, EffectRow, Type, TypeKind};

/// Result of BoundVar substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubstResult<'db> {
    /// Substitution succeeded.
    Ok(Type<'db>),
    /// BoundVar index was out of bounds.
    OutOfBounds { index: u32, max: usize },
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
        TypeKind::Named { name, args } => {
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
                    name: *name,
                    args: new_args,
                },
            ))
        }
        TypeKind::Func {
            params,
            result,
            effect,
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
            let new_effect = substitute_effect_row(db, *effect, subst);
            SubstResult::Ok(Type::new(
                db,
                TypeKind::Func {
                    params: new_params,
                    result: new_result,
                    effect: new_effect,
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
        // Primitive types and other type variables are unchanged
        _ => SubstResult::Ok(ty),
    }
}

/// Substitute BoundVars within an effect row.
///
/// Panics if a BoundVar index is out of bounds.
pub fn substitute_effect_row<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    subst: &[Type<'db>],
) -> EffectRow<'db> {
    let effects = row.effects(db);
    let mut changed = false;

    let new_effects: Vec<_> = effects
        .iter()
        .map(|effect| {
            let new_args: Vec<_> = effect
                .args
                .iter()
                .map(|a| {
                    substitute_bound_vars(db, *a, subst).unwrap_or_else(|index, max| {
                        panic!(
                            "BoundVar index out of range in effect row: index={}, subst.len()={}",
                            index, max
                        )
                    })
                })
                .collect();
            if new_args != effect.args {
                changed = true;
            }
            Effect {
                name: effect.name,
                args: new_args,
            }
        })
        .collect();

    if changed {
        EffectRow::new(db, new_effects, row.rest(db))
    } else {
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::EffectRow;
    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    // =========================================================================
    // Basic substitution tests
    // =========================================================================

    #[salsa_test]
    fn test_substitute_bound_var_basic(db: &dyn salsa::Database) {
        // BoundVar(0) + [Int] → Int
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_bound_vars(db, bound_var, &subst);
        assert_eq!(result, SubstResult::Ok(int_ty));
    }

    #[salsa_test]
    fn test_substitute_multiple_bound_vars(db: &dyn salsa::Database) {
        // (BoundVar(0), BoundVar(1)) + [Int, Bool] → (Int, Bool)
        let bound0 = Type::new(db, TypeKind::BoundVar { index: 0 });
        let bound1 = Type::new(db, TypeKind::BoundVar { index: 1 });
        let tuple_ty = Type::new(db, TypeKind::Tuple(vec![bound0, bound1]));

        let int_ty = Type::new(db, TypeKind::Int);
        let bool_ty = Type::new(db, TypeKind::Bool);
        let subst = vec![int_ty, bool_ty];

        let result = substitute_bound_vars(db, tuple_ty, &subst);
        let expected = Type::new(db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        assert_eq!(result, SubstResult::Ok(expected));
    }

    #[salsa_test]
    fn test_substitute_in_named_type(db: &dyn salsa::Database) {
        // List(BoundVar(0)) + [Int] → List(Int)
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let list_ty = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![bound_var],
            },
        );

        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_bound_vars(db, list_ty, &subst);
        let expected = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        assert_eq!(result, SubstResult::Ok(expected));
    }

    #[salsa_test]
    fn test_substitute_in_func_type(db: &dyn salsa::Database) {
        // fn(BoundVar(0)) -> BoundVar(0) + [Int] → fn(Int) -> Int
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::pure(db);
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );

        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_bound_vars(db, func_ty, &subst);
        let expected = Type::new(
            db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
            },
        );
        assert_eq!(result, SubstResult::Ok(expected));
    }

    #[salsa_test]
    fn test_substitute_in_tuple_type(db: &dyn salsa::Database) {
        // (BoundVar(0), BoundVar(1)) + [Int, String]
        let bound0 = Type::new(db, TypeKind::BoundVar { index: 0 });
        let bound1 = Type::new(db, TypeKind::BoundVar { index: 1 });
        let tuple_ty = Type::new(db, TypeKind::Tuple(vec![bound0, bound1]));

        let int_ty = Type::new(db, TypeKind::Int);
        let string_ty = Type::new(db, TypeKind::String);
        let subst = vec![int_ty, string_ty];

        let result = substitute_bound_vars(db, tuple_ty, &subst);
        let expected = Type::new(db, TypeKind::Tuple(vec![int_ty, string_ty]));
        assert_eq!(result, SubstResult::Ok(expected));
    }

    #[salsa_test]
    fn test_substitute_out_of_bounds(db: &dyn salsa::Database) {
        // BoundVar(5) + [Int] → OutOfBounds
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 5 });
        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_bound_vars(db, bound_var, &subst);
        assert_eq!(result, SubstResult::OutOfBounds { index: 5, max: 1 });
    }

    #[salsa_test]
    fn test_substitute_out_of_bounds_in_nested(db: &dyn salsa::Database) {
        // List(BoundVar(5)) + [Int] → OutOfBounds
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 5 });
        let list_ty = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![bound_var],
            },
        );

        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_bound_vars(db, list_ty, &subst);
        assert_eq!(result, SubstResult::OutOfBounds { index: 5, max: 1 });
    }

    #[salsa_test]
    fn test_substitute_primitive_unchanged(db: &dyn salsa::Database) {
        // Int + [Bool] → Int (primitives are unchanged)
        let int_ty = Type::new(db, TypeKind::Int);
        let bool_ty = Type::new(db, TypeKind::Bool);
        let subst = vec![bool_ty];

        let result = substitute_bound_vars(db, int_ty, &subst);
        assert_eq!(result, SubstResult::Ok(int_ty));
    }

    #[salsa_test]
    fn test_substitute_empty_subst(db: &dyn salsa::Database) {
        // Primitives with empty subst should work
        let int_ty = Type::new(db, TypeKind::Int);
        let result = substitute_bound_vars(db, int_ty, &[]);
        assert_eq!(result, SubstResult::Ok(int_ty));
    }

    // =========================================================================
    // Effect row substitution tests
    // =========================================================================

    #[salsa_test]
    fn test_substitute_effect_row_with_bound_var(db: &dyn salsa::Database) {
        // {State(BoundVar(0))} + [Int] → {State(Int)}
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::new(
            db,
            vec![Effect {
                name: Symbol::new("State"),
                args: vec![bound_var],
            }],
            None,
        );

        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_effect_row(db, effect, &subst);
        let expected = EffectRow::new(
            db,
            vec![Effect {
                name: Symbol::new("State"),
                args: vec![int_ty],
            }],
            None,
        );
        assert_eq!(result, expected);
    }

    #[salsa_test]
    fn test_substitute_effect_row_no_change(db: &dyn salsa::Database) {
        // {IO} + [Int] → {IO} (no BoundVars in effect)
        let effect = EffectRow::new(
            db,
            vec![Effect {
                name: Symbol::new("IO"),
                args: vec![],
            }],
            None,
        );

        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_effect_row(db, effect, &subst);
        // Should be the same interned value (no change)
        assert_eq!(result, effect);
    }

    #[salsa_test]
    fn test_substitute_pure_effect_row(db: &dyn salsa::Database) {
        let effect = EffectRow::pure(db);
        let int_ty = Type::new(db, TypeKind::Int);
        let subst = vec![int_ty];

        let result = substitute_effect_row(db, effect, &subst);
        assert_eq!(result, effect);
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
