use std::fmt;

use trunk_ir::Symbol;

use super::instance::InstanceKey;
use crate::ast::{
    AbilityOrigin, BuiltinAbility, CallingConvention, EffectRow, EffectVar, Type, TypeDefId,
    TypeKind,
};

/// Generate a mangled symbol for a specialized generic function or type.
///
/// Mangling rules use `$` as the only structural character, with `$0`/`$1`
/// as open/close markers for nested type arguments. Tribute identifiers
/// cannot start with a digit, so every type the compiler constructs rather
/// than a source declaration names starts with a digit tag, and a segment
/// starting with a letter is always a primitive or a source-declared name:
/// `$6` starts a function type, `$7` a tuple, `$8$n` the `n`th bound type
/// variable, and `$5` a compiler-owned nominal type or ability. Before its
/// result, a function type writes its effect row between `$2` and `$1`, or
/// `$3$n` for the `n`th row variable of an open row, and a calling-convention
/// floor above `Direct` after `$4`. Distinct type argument lists get distinct
/// names:
///
/// - `identity + [Int]`              → `identity$Int`
/// - `first + [Int, Text]`           → `first$Int$Text`
/// - `map + [Int, Option(Int)]`      → `map$Int$Option$0$Int$1`
/// - `f + [List(Option(Int))]`       → `f$5$List$0$Option$0$Int$1$1`
/// - `apply + [fn(Int) -> Bool]`     → `apply$6$0$Int$1$Bool`
/// - `swap + [(Int, Bool)]`          → `swap$7$0$Int$Bool$1`
/// - `apply + [fn() ->{Ask} Nat]`    → `apply$6$0$$1$2$Ask$1$Nat`
/// - `apply + [fn() ->{State(Int), e} Nat]`
///   → `apply$6$0$$1$2$State$0$Int$1$3$0$Nat`
pub fn mangle_name(db: &dyn salsa::Database, base: &Symbol, type_args: &[Type<'_>]) -> Symbol {
    let mut buf = String::new();
    let mut row_vars = Vec::new();
    base.with_str(|s| buf.push_str(s));
    for ty in type_args {
        buf.push('$');
        write_type_mangled(db, *ty, &mut row_vars, &mut buf).unwrap();
    }
    Symbol::new(&buf)
}

/// The name of the function instance `key` selects: the type arguments, then
/// `$9` and one letter per class variable. Classes that are all `Cps` are
/// omitted.
pub(crate) fn mangle_instance_name(
    db: &dyn salsa::Database,
    base: &Symbol,
    key: &InstanceKey<'_>,
) -> Symbol {
    let name = mangle_name(db, base, &key.type_args);
    if !key.has_weaker_class() {
        return name;
    }
    class_instance_name(&name, &key.class_args)
}

/// `name` with `$9` and one letter per class.
pub(crate) fn class_instance_name(name: &Symbol, classes: &[CallingConvention]) -> Symbol {
    let mut buf = name.to_string();
    buf.push_str("$9");
    buf.extend(classes.iter().map(|class| match class {
        CallingConvention::Direct => 'D',
        CallingConvention::EvidenceDirect => 'E',
        CallingConvention::Cps => 'C',
    }));
    Symbol::new(&buf)
}

pub fn mangle_type_name(
    db: &dyn salsa::Database,
    id: TypeDefId<'_>,
    name: Symbol,
    type_args: &[Type<'_>],
) -> Symbol {
    mangle_name(db, &nominal_mangle_base(db, id, name), type_args)
}

fn nominal_mangle_base(db: &dyn salsa::Database, id: TypeDefId<'_>, name: Symbol) -> Symbol {
    if id.is_builtin_list(db) {
        Symbol::new("5$List")
    } else {
        let qualified = id.qualified(db);
        if *qualified == name {
            name
        } else {
            qualified.clone()
        }
    }
}

fn write_type_mangled(
    db: &dyn salsa::Database,
    ty: Type<'_>,
    row_vars: &mut Vec<EffectVar>,
    f: &mut impl fmt::Write,
) -> fmt::Result {
    match ty.kind(db) {
        TypeKind::Int => f.write_str("Int"),
        TypeKind::Nat => f.write_str("Nat"),
        TypeKind::Float => f.write_str("Float"),
        TypeKind::Bool => f.write_str("Bool"),
        TypeKind::Bytes => f.write_str("Bytes"),
        TypeKind::Rune => f.write_str("Rune"),
        TypeKind::Nil => f.write_str("Nil"),
        TypeKind::Never => f.write_str("Never"),
        TypeKind::Named { id, name, args } => {
            nominal_mangle_base(db, *id, name.clone()).with_str(|s| f.write_str(s))?;
            if !args.is_empty() {
                f.write_str("$0$")?;
                write_type_mangled_list(db, args, row_vars, f)?;
                f.write_str("$1")?;
            }
            Ok(())
        }
        TypeKind::Func {
            params,
            result,
            effect,
        } => {
            f.write_str("6$0$")?;
            write_type_mangled_list(db, params, row_vars, f)?;
            f.write_str("$1$")?;
            if !effect.is_pure(db) {
                write_effect_row_mangled(db, *effect, row_vars, f)?;
            }
            write_type_mangled(db, *result, row_vars, f)
        }
        TypeKind::Tuple(elems) => {
            f.write_str("7$0$")?;
            write_type_mangled_list(db, elems, row_vars, f)?;
            f.write_str("$1")
        }
        TypeKind::BoundVar { index } => write!(f, "8${index}"),
        // Local binders are source-body metadata. Monomorphization retains the
        // existing uniform representation boundary, where both kinds of
        // quantified variables use their index only.
        TypeKind::LocalBoundVar { index, .. } => write!(f, "8${index}"),
        TypeKind::UniVar { .. } | TypeKind::App { .. } | TypeKind::Continuation { .. } => {
            panic!("mangle_name requires fully-resolved concrete types");
        }
        TypeKind::Error => f.write_str("error"),
    }
}

fn write_type_mangled_list(
    db: &dyn salsa::Database,
    types: &[Type<'_>],
    row_vars: &mut Vec<EffectVar>,
    f: &mut impl fmt::Write,
) -> fmt::Result {
    for (index, ty) in types.iter().enumerate() {
        if index > 0 {
            f.write_char('$')?;
        }
        write_type_mangled(db, *ty, row_vars, f)?;
    }
    Ok(())
}

/// Effects keep their stored order: specializations are keyed by interned
/// type identity, which distinguishes rows listing the same effects in
/// another order. Row variables are numbered by first appearance within the
/// mangled name, so the name does not depend on inference numbering.
fn write_effect_row_mangled(
    db: &dyn salsa::Database,
    row: EffectRow<'_>,
    row_vars: &mut Vec<EffectVar>,
    f: &mut impl fmt::Write,
) -> fmt::Result {
    f.write_str("2")?;
    for effect in row.effects(db) {
        f.write_char('$')?;
        match effect.ability_id.origin(db) {
            AbilityOrigin::Source => {}
            AbilityOrigin::Builtin(BuiltinAbility::Io) => f.write_str("5$")?,
        }
        effect
            .ability_id
            .qualified(db)
            .with_str(|s| f.write_str(s))?;
        if !effect.args.is_empty() {
            f.write_str("$0$")?;
            write_type_mangled_list(db, &effect.args, row_vars, f)?;
            f.write_str("$1")?;
        }
    }
    match row.rest(db) {
        Some(var) => {
            let index = row_vars
                .iter()
                .position(|seen| *seen == var)
                .unwrap_or_else(|| {
                    row_vars.push(var);
                    row_vars.len() - 1
                });
            write!(f, "$3${index}$")
        }
        None => f.write_str("$1$"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    #[test]
    fn test_single_primitive() {
        let db = TestDb::default();
        let base = Symbol::new("identity");
        let int_ty = Type::new(&db, TypeKind::Int);
        let result = mangle_name(&db, &base, &[int_ty]);
        assert_eq!(result.to_string(), "identity$Int");
    }

    #[test]
    fn test_multiple_primitives() {
        let db = TestDb::default();
        let base = Symbol::new("first");
        let int_ty = Type::new(&db, TypeKind::Int);
        let float_ty = Type::new(&db, TypeKind::Float);
        let result = mangle_name(&db, &base, &[int_ty, float_ty]);
        assert_eq!(result.to_string(), "first$Int$Float");
    }

    #[test]
    fn test_named_type_no_args() {
        let db = TestDb::default();
        let base = Symbol::new("wrap");
        let text_ty = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Text")),
                name: Symbol::new("Text"),
                args: vec![],
            },
        );
        let result = mangle_name(&db, &base, &[text_ty]);
        assert_eq!(result.to_string(), "wrap$Text");
    }

    #[test]
    fn test_named_type_with_args() {
        let db = TestDb::default();
        let base = Symbol::new("map");
        let int_ty = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Option")),
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );
        let result = mangle_name(&db, &base, &[int_ty, option_int]);
        assert_eq!(result.to_string(), "map$Int$Option$0$Int$1");
    }

    #[test]
    fn test_nested_named_types() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let int_ty = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Option")),
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );
        let list_option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::builtin_list(&db),
                name: Symbol::new("List"),
                args: vec![option_int],
            },
        );
        let result = mangle_name(&db, &base, &[list_option_int]);
        assert_eq!(result.to_string(), "f$5$List$0$Option$0$Int$1$1");
    }

    #[test]
    fn test_builtin_and_source_list_mangle_distinctly() {
        let db = TestDb::default();
        let base = Symbol::new("identity");
        let name = Symbol::new("List");
        let int_ty = Type::new(&db, TypeKind::Int);
        let builtin = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::builtin_list(&db),
                name: name.clone(),
                args: vec![int_ty],
            },
        );
        let source = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::source(
                    &db,
                    name.clone(),
                    crate::ast::NodeId::from_raw(1),
                ),
                name,
                args: vec![int_ty],
            },
        );

        assert_eq!(
            mangle_name(&db, &base, &[builtin]).to_string(),
            "identity$5$List$0$Int$1"
        );
        assert_eq!(
            mangle_name(&db, &base, &[source]).to_string(),
            "identity$List$0$Int$1"
        );
    }

    #[test]
    fn test_same_spelled_source_types_mangle_distinctly() {
        let db = TestDb::default();
        let base = Symbol::new("identity");
        let name = Symbol::new("Thing");
        let int_ty = Type::new(&db, TypeKind::Int);
        let a_thing = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::source(
                    &db,
                    Symbol::new("A::Thing"),
                    crate::ast::NodeId::from_raw(1),
                ),
                name: name.clone(),
                args: vec![int_ty],
            },
        );
        let b_thing = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::source(
                    &db,
                    Symbol::new("B::Thing"),
                    crate::ast::NodeId::from_raw(2),
                ),
                name,
                args: vec![int_ty],
            },
        );

        assert_eq!(
            mangle_name(&db, &base, &[a_thing]).to_string(),
            "identity$A::Thing$0$Int$1"
        );
        assert_eq!(
            mangle_name(&db, &base, &[b_thing]).to_string(),
            "identity$B::Thing$0$Int$1"
        );
    }

    #[test]
    fn test_named_type_multiple_args() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let int_ty = Type::new(&db, TypeKind::Int);
        let text_ty = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Text")),
                name: Symbol::new("Text"),
                args: vec![],
            },
        );
        let pair = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Pair")),
                name: Symbol::new("Pair"),
                args: vec![int_ty, text_ty],
            },
        );
        let result = mangle_name(&db, &base, &[pair]);
        assert_eq!(result.to_string(), "f$Pair$0$Int$Text$1");
    }

    #[test]
    fn test_function_type() {
        let db = TestDb::default();
        let base = Symbol::new("apply");
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: bool_ty,
                effect: crate::ast::EffectRow::new(&db, vec![], None),
            },
        );
        let result = mangle_name(&db, &base, &[func_ty]);
        assert_eq!(result.to_string(), "apply$6$0$Int$1$Bool");
    }

    fn ability<'db>(db: &'db TestDb, name: &str, args: Vec<Type<'db>>) -> crate::ast::Effect<'db> {
        crate::ast::Effect {
            ability_id: crate::ast::AbilityId::source(db, Symbol::new(name)),
            args,
        }
    }

    fn thunk<'db>(db: &'db TestDb, effect: EffectRow<'db>) -> Type<'db> {
        Type::new(
            db,
            TypeKind::Func {
                params: vec![],
                result: Type::new(db, TypeKind::Nat),
                effect,
            },
        )
    }

    #[test]
    fn test_function_types_mangle_effect_rows_distinctly() {
        let db = TestDb::default();
        let base = Symbol::new("run_state");
        let int_ty = Type::new(&db, TypeKind::Int);
        let mangle = |effects| {
            mangle_name(
                &db,
                &base,
                &[thunk(&db, EffectRow::new(&db, effects, None))],
            )
            .to_string()
        };

        assert_eq!(mangle(vec![]), "run_state$6$0$$1$Nat");
        assert_eq!(
            mangle(vec![ability(&db, "Ask", vec![])]),
            "run_state$6$0$$1$2$Ask$1$Nat"
        );
        assert_eq!(
            mangle(vec![ability(&db, "Tell", vec![])]),
            "run_state$6$0$$1$2$Tell$1$Nat"
        );
        assert_eq!(
            mangle(vec![
                ability(&db, "std::State", vec![int_ty]),
                ability(&db, "Ask", vec![])
            ]),
            "run_state$6$0$$1$2$std::State$0$Int$1$Ask$1$Nat"
        );
        // Rows are distinct interned types in either order.
        assert_ne!(
            mangle(vec![
                ability(&db, "Ask", vec![]),
                ability(&db, "Tell", vec![])
            ]),
            mangle(vec![
                ability(&db, "Tell", vec![]),
                ability(&db, "Ask", vec![])
            ])
        );
    }

    #[test]
    fn test_builtin_and_source_abilities_mangle_distinctly() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let builtin = crate::ast::AbilityId::builtin_io(&db);
        let source = crate::ast::AbilityId::source(&db, builtin.qualified(&db).clone());
        let mangle = |ability_id| {
            let effect = crate::ast::Effect {
                ability_id,
                args: vec![],
            };
            mangle_name(&db, &base, &[thunk(&db, EffectRow::single(&db, effect))]).to_string()
        };

        assert_eq!(mangle(source), "f$6$0$$1$2$std::io::Io$1$Nat");
        assert_eq!(mangle(builtin), "f$6$0$$1$2$5$std::io::Io$1$Nat");
    }

    #[test]
    fn test_effect_row_belongs_to_its_own_function_type() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let ask = || EffectRow::single(&db, ability(&db, "Ask", vec![]));
        let returning = |result, effect| {
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![],
                    result,
                    effect,
                },
            )
        };
        let pure = EffectRow::pure(&db);

        let outer = returning(thunk(&db, pure), ask());
        let inner = returning(thunk(&db, ask()), pure);
        assert_eq!(
            mangle_name(&db, &base, &[outer]).to_string(),
            "f$6$0$$1$2$Ask$1$6$0$$1$Nat"
        );
        assert_eq!(
            mangle_name(&db, &base, &[inner]).to_string(),
            "f$6$0$$1$6$0$$1$2$Ask$1$Nat"
        );
    }

    #[test]
    fn test_open_effect_rows_number_variables_by_first_appearance() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let open = |id| thunk(&db, EffectRow::open(&db, EffectVar { id }));

        assert_eq!(
            mangle_name(&db, &base, &[open(7)]).to_string(),
            "f$6$0$$1$2$3$0$Nat"
        );
        assert_eq!(
            mangle_name(&db, &base, &[open(7)]),
            mangle_name(&db, &base, &[open(42)])
        );
        assert_eq!(
            mangle_name(&db, &base, &[open(7), open(7)]).to_string(),
            "f$6$0$$1$2$3$0$Nat$6$0$$1$2$3$0$Nat"
        );
        assert_eq!(
            mangle_name(&db, &base, &[open(7), open(42)]).to_string(),
            "f$6$0$$1$2$3$0$Nat$6$0$$1$2$3$1$Nat"
        );
    }

    #[test]
    fn test_instance_names_spell_classes_weaker_than_cps() {
        let db = TestDb::default();
        let base = Symbol::new("map");
        let int = Type::new(&db, TypeKind::Int);
        let name = |type_args: &[Type<'_>], class_args: &[CallingConvention]| {
            let key = InstanceKey {
                type_args: type_args.to_vec(),
                class_args: class_args.to_vec(),
            };
            mangle_instance_name(&db, &base, &key).to_string()
        };
        use CallingConvention::{Cps, Direct, EvidenceDirect};

        assert_eq!(name(&[int], &[Direct]), "map$Int$9D");
        assert_eq!(name(&[int], &[EvidenceDirect, Cps]), "map$Int$9EC");
        assert_eq!(name(&[int], &[Cps, Cps]), "map$Int");
        assert_eq!(name(&[], &[Direct]), "map$9D");
    }

    #[test]
    fn test_type_names_mangle_effect_rows_distinctly() {
        let db = TestDb::default();
        let name = Symbol::new("Holder");
        let id = TypeDefId::synthetic(&db, name.clone());
        let mangle = |effect: &str| {
            let row = EffectRow::single(&db, ability(&db, effect, vec![]));
            mangle_type_name(&db, id, name.clone(), &[thunk(&db, row)]).to_string()
        };

        assert_eq!(mangle("Ask"), "Holder$6$0$$1$2$Ask$1$Nat");
        assert_eq!(mangle("Tell"), "Holder$6$0$$1$2$Tell$1$Nat");
    }

    #[test]
    fn test_tuple_type() {
        let db = TestDb::default();
        let base = Symbol::new("swap");
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let tup_ty = Type::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        let result = mangle_name(&db, &base, &[tup_ty]);
        assert_eq!(result.to_string(), "swap$7$0$Int$Bool$1");
    }

    #[test]
    fn test_all_primitives() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let types: Vec<(Type<'_>, &str)> = vec![
            (Type::new(&db, TypeKind::Int), "Int"),
            (Type::new(&db, TypeKind::Nat), "Nat"),
            (Type::new(&db, TypeKind::Float), "Float"),
            (Type::new(&db, TypeKind::Bool), "Bool"),
            (Type::new(&db, TypeKind::Bytes), "Bytes"),
            (Type::new(&db, TypeKind::Rune), "Rune"),
            (Type::new(&db, TypeKind::Nil), "Nil"),
            (Type::new(&db, TypeKind::Never), "Never"),
        ];
        for (ty, expected_suffix) in types {
            let result = mangle_name(&db, &base, &[ty]);
            assert_eq!(result.to_string(), format!("f${expected_suffix}"));
        }
    }

    #[test]
    fn test_empty_type_args() {
        let db = TestDb::default();
        let base = Symbol::new("main");
        let result = mangle_name(&db, &base, &[]);
        assert_eq!(result.to_string(), "main");
    }

    #[test]
    fn test_bound_var() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let bv = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let result = mangle_name(&db, &base, &[bv]);
        assert_eq!(result.to_string(), "f$8$0");
    }

    #[test]
    fn test_error_type() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let err_ty = Type::new(&db, TypeKind::Error);
        let result = mangle_name(&db, &base, &[err_ty]);
        assert_eq!(result.to_string(), "f$error");
    }

    #[test]
    #[should_panic(expected = "mangle_name requires fully-resolved concrete types")]
    fn test_univar_panics() {
        let db = TestDb::default();
        let base = Symbol::new("f");
        let univar_id = crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0);
        let ty = Type::new(&db, TypeKind::UniVar { id: univar_id });
        mangle_name(&db, &base, &[ty]);
    }
}

/// Laws of specialization mangling, checked on generated type arguments.
#[cfg(test)]
mod laws {
    use proptest::prelude::*;
    use rustc_hash::FxHashMap as HashMap;

    use super::*;
    use crate::typeck::prop::{ROW_VAR_BASE, RowShape, TypeGen, TypeShape, type_shape};

    /// Resolved type arguments: no unification variables, `App`, or
    /// `Continuation`, which mangling rejects.
    const ARGS: TypeGen = TypeGen::GROUND.bound_vars(2).row_vars(3).error(true);

    fn args() -> BoxedStrategy<Vec<TypeShape>> {
        proptest::collection::vec(type_shape(ARGS), 0..=3).boxed()
    }

    /// Argument lists with a related second list: the same list, the list
    /// with its row variables renamed bijectively or merged into one, or an
    /// independent list.
    fn arg_pairs() -> BoxedStrategy<(Vec<TypeShape>, Vec<TypeShape>)> {
        args()
            .prop_flat_map(|left| {
                let renamed = |rename: fn(u64) -> u64| {
                    let left = left.clone();
                    Just(
                        left.iter()
                            .map(|ty| ty.rename_row_vars(&rename))
                            .collect::<Vec<_>>(),
                    )
                };
                let right = prop_oneof![
                    Just(left.clone()),
                    renamed(|var| var + 100),
                    renamed(|var| ROW_VAR_BASE + (var - ROW_VAR_BASE + 1) % 3),
                    renamed(|_| ROW_VAR_BASE),
                    args(),
                ];
                (Just(left), right)
            })
            .boxed()
    }

    /// Whether the lists are equal up to a bijective renaming of row
    /// variables across the whole list.
    fn equal_up_to_row_renaming(left: &[TypeShape], right: &[TypeShape]) -> bool {
        #[derive(Default)]
        struct Renaming {
            forward: HashMap<u64, u64>,
            backward: HashMap<u64, u64>,
        }

        impl Renaming {
            fn rows(&mut self, left: &RowShape, right: &RowShape) -> bool {
                left.effects.len() == right.effects.len()
                    && left
                        .effects
                        .iter()
                        .zip(&right.effects)
                        .all(|(l, r)| l.ability == r.ability && self.lists(&l.args, &r.args))
                    && match (left.rest, right.rest) {
                        (None, None) => true,
                        (Some(l), Some(r)) => {
                            *self.forward.entry(l).or_insert(r) == r
                                && *self.backward.entry(r).or_insert(l) == l
                        }
                        _ => false,
                    }
            }

            fn lists(&mut self, left: &[TypeShape], right: &[TypeShape]) -> bool {
                left.len() == right.len() && left.iter().zip(right).all(|(l, r)| self.types(l, r))
            }

            fn types(&mut self, left: &TypeShape, right: &TypeShape) -> bool {
                match (left, right) {
                    (
                        TypeShape::Named {
                            nominal: l,
                            args: la,
                        },
                        TypeShape::Named {
                            nominal: r,
                            args: ra,
                        },
                    ) => l == r && self.lists(la, ra),
                    (
                        TypeShape::Func {
                            params: lp,
                            result: lr,
                            effect: le,
                        },
                        TypeShape::Func {
                            params: rp,
                            result: rr,
                            effect: re,
                        },
                    ) => self.lists(lp, rp) && self.rows(le, re) && self.types(lr, rr),
                    (TypeShape::Tuple(l), TypeShape::Tuple(r)) => self.lists(l, r),
                    (left, right) => left == right,
                }
            }
        }

        Renaming::default().lists(left, right)
    }

    fn mangle(db: &dyn salsa::Database, args: &[TypeShape]) -> String {
        let args: Vec<_> = args.iter().map(|ty| ty.build(db)).collect();
        mangle_name(db, &Symbol::new("f"), &args).to_string()
    }

    proptest! {
        /// Mangling is injective on type arguments up to row-variable
        /// renaming: two lists get the same name exactly when they differ
        /// only by a consistent renaming of row variables.
        #[test]
        fn mangle_is_injective_up_to_row_renaming((left, right) in arg_pairs()) {
            let db = salsa::DatabaseImpl::new();
            prop_assert_eq!(
                mangle(&db, &left) == mangle(&db, &right),
                equal_up_to_row_renaming(&left, &right),
                "{} / {}",
                mangle(&db, &left),
                mangle(&db, &right),
            );
        }

        /// The name depends only on the types, not on the database that
        /// interned them, and extends the base name.
        #[test]
        fn mangle_is_deterministic(args in args()) {
            let (first, second) = (salsa::DatabaseImpl::new(), salsa::DatabaseImpl::new());
            let name = mangle(&first, &args);
            prop_assert_eq!(&name, &mangle(&second, &args));
            if args.is_empty() {
                prop_assert_eq!(name, "f");
            } else {
                prop_assert!(name.starts_with("f$"));
            }
        }
    }

    /// A source type named `Fn` (accepted by the frontend as
    /// `struct Fn(a) { value: a }`) does not mangle like a function type, so
    /// a function type taking `Fn(Int)` and `Bool` and one taking
    /// `fn(Int) -> Bool` get distinct names.
    #[test]
    fn source_type_named_fn_mangles_distinctly() {
        let db = salsa::DatabaseImpl::new();
        let int = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let nat = Type::new(&db, TypeKind::Nat);
        let func = |params, result| {
            Type::new(
                &db,
                TypeKind::Func {
                    params,
                    result,
                    effect: EffectRow::pure(&db),
                },
            )
        };
        let name = Symbol::new("Fn");
        let user_fn = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::source(&db, name.clone(), crate::ast::NodeId::from_raw(1)),
                name,
                args: vec![int],
            },
        );
        let takes_struct = func(vec![user_fn, bool_ty], nat);
        let takes_func = func(vec![func(vec![int], bool_ty)], nat);
        let base = Symbol::new("identity");
        assert_ne!(
            mangle_name(&db, &base, &[takes_struct]),
            mangle_name(&db, &base, &[takes_func])
        );
    }
    /// Source types spelled like the mangled form of a compiler-constructed
    /// type get names distinct from that type.
    #[test]
    fn source_types_spelled_like_compiler_types_mangle_distinctly() {
        let db = salsa::DatabaseImpl::new();
        let int = Type::new(&db, TypeKind::Int);
        let source = |name: &str, node, args| {
            let name = Symbol::new(name);
            Type::new(
                &db,
                TypeKind::Named {
                    id: TypeDefId::source(&db, name.clone(), crate::ast::NodeId::from_raw(node)),
                    name,
                    args,
                },
            )
        };
        let builtin_list = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(&db),
                name: Symbol::new("List"),
                args: vec![int],
            },
        );
        let cases = [
            (
                source("Tup", 1, vec![int]),
                Type::new(&db, TypeKind::Tuple(vec![int])),
            ),
            (
                source("T0", 2, vec![]),
                Type::new(&db, TypeKind::BoundVar { index: 0 }),
            ),
            (source("BuiltinList", 3, vec![int]), builtin_list),
        ];
        let base = Symbol::new("identity");
        for (named, constructed) in cases {
            assert_ne!(
                mangle_name(&db, &base, &[named]),
                mangle_name(&db, &base, &[constructed])
            );
        }
        let type_name =
            |id, args: &[Type<'_>]| mangle_type_name(&db, id, Symbol::new("List"), args);
        let source_list = TypeDefId::source(
            &db,
            Symbol::new("BuiltinList"),
            crate::ast::NodeId::from_raw(3),
        );
        assert_ne!(
            type_name(TypeDefId::builtin_list(&db), &[int]),
            type_name(source_list, &[int])
        );
    }
}
