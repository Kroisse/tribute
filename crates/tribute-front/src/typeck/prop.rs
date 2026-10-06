//! Proptest strategies for the typeck `Type` and `EffectRow` representations.
//!
//! `Type` and `EffectRow` are Salsa-interned, so strategies cannot produce
//! them directly. They produce database-free shapes ([`TypeShape`],
//! [`RowShape`]) that a test interns with `build(db)`. Shapes compare
//! structurally, and interning is structural, so two shapes are equal exactly
//! when the types they build are the same interned value.
//!
//! Nominal types and abilities come from fixed pools whose entries each have
//! one arity, so generated types are well-kinded. The pools include
//! identities that only differ by origin or module (a builtin and a source
//! `List`, `A::Thing` and `B::Thing`, `mod1::State` and `mod2::State`, the
//! builtin and a source `std::io::Io`), so properties exercise the label
//! identity rules.
//!
//! [`TypeGen`] selects which variable kinds and constructors appear. The
//! comparison helpers at the end relate built types modulo effect order,
//! since rows denote sets of effects.

use proptest::prelude::*;
use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, CallingConvention, Effect, EffectRow, EffectVar, NodeId, Type, TypeDefId, TypeKind,
    UniVarId, UniVarSource,
};

/// First row-variable id used by generated rows. Solvers and inference
/// contexts allocate fresh row variables from small counters; generated
/// tails stay clear of them.
pub(crate) const ROW_VAR_BASE: u64 = 1000;

/// Row variables introduced when a generalization opens a row.
const OPENED_ROW_VAR_BASE: u64 = 5000;

/// Unification variables introduced by a linear generalization.
const LINEAR_UNIVAR_BASE: u64 = 1_000_000;

/// Unification variables of effect arguments under
/// `Sharing::Pool { separate_effect_args: true, .. }`.
const EFFECT_ARG_UNIVAR_BASE: u64 = 100;

/// Placeholder ids renumbered by [`renumber_placeholders`].
const PLACEHOLDER: u64 = u64::MAX;

// ============================================================================
// Pools
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum NominalOrigin {
    BuiltinList,
    Synthetic,
    Source(usize),
}

/// A nominal type: origin, qualified identity, display name, arity.
type Nominal = (NominalOrigin, &'static str, &'static str, usize);

const NOMINALS: &[Nominal] = &[
    (NominalOrigin::BuiltinList, "List", "List", 1),
    // Same spelling as the builtin list, different identity.
    (NominalOrigin::Source(1), "List", "List", 1),
    (NominalOrigin::Synthetic, "Option", "Option", 1),
    // Same display name, different declarations.
    (NominalOrigin::Source(2), "A::Thing", "Thing", 1),
    (NominalOrigin::Source(3), "B::Thing", "Thing", 1),
    (NominalOrigin::Synthetic, "Text", "Text", 0),
    (NominalOrigin::Synthetic, "Pair", "Pair", 2),
];

/// An ability: whether it is the compiler-owned `Io`, qualified name, arity.
type Ability = (bool, &'static str, usize);

const ABILITIES: &[Ability] = &[
    (false, "Console", 0),
    // Same name in different modules.
    (false, "mod1::State", 1),
    (false, "mod2::State", 1),
    (false, "Choice", 2),
    // Same path, builtin and source origins.
    (true, "std::io::Io", 0),
    (false, "std::io::Io", 0),
];

pub(crate) fn ability_id<'db>(db: &'db dyn salsa::Database, index: usize) -> AbilityId<'db> {
    let (builtin, name, _) = ABILITIES[index];
    if builtin {
        AbilityId::builtin_io(db)
    } else {
        AbilityId::source(db, Symbol::new(name))
    }
}

fn nominal_id<'db>(db: &'db dyn salsa::Database, index: usize) -> (TypeDefId<'db>, Symbol) {
    let (origin, qualified, name, _) = NOMINALS[index];
    let id = match origin {
        NominalOrigin::BuiltinList => TypeDefId::builtin_list(db),
        NominalOrigin::Synthetic => TypeDefId::synthetic(db, Symbol::new(qualified)),
        NominalOrigin::Source(node) => {
            TypeDefId::source(db, Symbol::new(qualified), NodeId::from_raw(node))
        }
    };
    (id, Symbol::new(name))
}

/// The unification variable a [`TypeShape::UniVar`] id builds.
pub(crate) fn univar<'db>(db: &'db dyn salsa::Database, id: u64) -> Type<'db> {
    let id = UniVarId::new(db, UniVarSource::Anonymous(id), 0);
    Type::new(db, TypeKind::UniVar { id })
}

// ============================================================================
// Shapes
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Prim {
    Int,
    Nat,
    Float,
    Bool,
    Bytes,
    Rune,
    Nil,
    Never,
}

const PRIMS: [Prim; 8] = [
    Prim::Int,
    Prim::Nat,
    Prim::Float,
    Prim::Bool,
    Prim::Bytes,
    Prim::Rune,
    Prim::Nil,
    Prim::Never,
];

/// A database-free description of a [`Type`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TypeShape {
    Prim(Prim),
    Error,
    BoundVar(u32),
    UniVar(u64),
    Named {
        nominal: usize,
        args: Vec<TypeShape>,
    },
    Func {
        params: Vec<TypeShape>,
        result: Box<TypeShape>,
        effect: RowShape,
        convention: CallingConvention,
    },
    Tuple(Vec<TypeShape>),
    App {
        ctor: Box<TypeShape>,
        args: Vec<TypeShape>,
    },
    Continuation {
        arg: Box<TypeShape>,
        result: Box<TypeShape>,
        effect: RowShape,
    },
}

/// A database-free description of an [`Effect`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EffectShape {
    pub ability: usize,
    pub args: Vec<TypeShape>,
}

/// A database-free description of an [`EffectRow`]. Generated rows hold
/// each effect at most once.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RowShape {
    pub effects: Vec<EffectShape>,
    pub rest: Option<u64>,
}

impl TypeShape {
    pub(crate) fn build<'db>(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        let kind = match self {
            TypeShape::Prim(prim) => match prim {
                Prim::Int => TypeKind::Int,
                Prim::Nat => TypeKind::Nat,
                Prim::Float => TypeKind::Float,
                Prim::Bool => TypeKind::Bool,
                Prim::Bytes => TypeKind::Bytes,
                Prim::Rune => TypeKind::Rune,
                Prim::Nil => TypeKind::Nil,
                Prim::Never => TypeKind::Never,
            },
            TypeShape::Error => TypeKind::Error,
            TypeShape::BoundVar(index) => TypeKind::BoundVar { index: *index },
            TypeShape::UniVar(id) => return univar(db, *id),
            TypeShape::Named { nominal, args } => {
                let (id, name) = nominal_id(db, *nominal);
                TypeKind::Named {
                    id,
                    name,
                    args: build_all(db, args),
                }
            }
            TypeShape::Func {
                params,
                result,
                effect,
                convention,
            } => TypeKind::Func {
                params: build_all(db, params),
                result: result.build(db),
                effect: effect.build(db),
                minimum_convention: *convention,
            },
            TypeShape::Tuple(elements) => TypeKind::Tuple(build_all(db, elements)),
            TypeShape::App { ctor, args } => TypeKind::App {
                ctor: ctor.build(db),
                args: build_all(db, args),
            },
            TypeShape::Continuation {
                arg,
                result,
                effect,
            } => TypeKind::Continuation {
                arg: arg.build(db),
                result: result.build(db),
                effect: effect.build(db),
            },
        };
        Type::new(db, kind)
    }

    /// Visit this shape and every shape nested in it, rows included.
    pub(crate) fn visit(&self, f: &mut impl FnMut(&TypeShape)) {
        f(self);
        match self {
            TypeShape::Named { args, .. } | TypeShape::Tuple(args) => {
                args.iter().for_each(|arg| arg.visit(f))
            }
            TypeShape::Func {
                params,
                result,
                effect,
                ..
            } => {
                params.iter().for_each(|param| param.visit(f));
                result.visit(f);
                effect.visit_types(f);
            }
            TypeShape::App { ctor, args } => {
                ctor.visit(f);
                args.iter().for_each(|arg| arg.visit(f));
            }
            TypeShape::Continuation {
                arg,
                result,
                effect,
            } => {
                arg.visit(f);
                result.visit(f);
                effect.visit_types(f);
            }
            TypeShape::Prim(_)
            | TypeShape::Error
            | TypeShape::BoundVar(_)
            | TypeShape::UniVar(_) => {}
        }
    }

    /// Whether any nested shape satisfies `pred`.
    pub(crate) fn any(&self, mut pred: impl FnMut(&TypeShape) -> bool) -> bool {
        let mut found = false;
        self.visit(&mut |shape| found |= pred(shape));
        found
    }

    /// Bound-variable indices in order of appearance.
    pub(crate) fn bound_vars(&self) -> Vec<u32> {
        let mut indices = Vec::new();
        self.visit(&mut |shape| {
            if let TypeShape::BoundVar(index) = shape {
                indices.push(*index);
            }
        });
        indices
    }

    /// Unification-variable ids, each once, in order of first appearance.
    pub(crate) fn univars(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        self.visit(&mut |shape| {
            if let TypeShape::UniVar(id) = shape
                && !ids.contains(id)
            {
                ids.push(*id);
            }
        });
        ids
    }

    /// Number of shapes [`TypeShape::visit`] reaches, this one included.
    pub(crate) fn size(&self) -> usize {
        let mut size = 0;
        self.visit(&mut |_| size += 1);
        size
    }

    /// Replace the `n`th shape in [`TypeShape::visit`] order with `with`.
    pub(crate) fn replace_nth(&self, n: usize, with: &TypeShape) -> TypeShape {
        fn go(shape: &TypeShape, n: &mut Option<usize>, with: &TypeShape) -> TypeShape {
            match n {
                Some(0) => {
                    *n = None;
                    return with.clone();
                }
                Some(k) => *k -= 1,
                None => return shape.clone(),
            }
            let all = |shapes: &[TypeShape], n: &mut Option<usize>| {
                shapes.iter().map(|shape| go(shape, n, with)).collect()
            };
            let row = |row: &RowShape, n: &mut Option<usize>| RowShape {
                effects: row
                    .effects
                    .iter()
                    .map(|effect| EffectShape {
                        ability: effect.ability,
                        args: all(&effect.args, n),
                    })
                    .collect(),
                rest: row.rest,
            };
            match shape {
                TypeShape::Named { nominal, args } => TypeShape::Named {
                    nominal: *nominal,
                    args: all(args, n),
                },
                TypeShape::Func {
                    params,
                    result,
                    effect,
                    convention,
                } => {
                    let params = all(params, n);
                    let result = Box::new(go(result, n, with));
                    TypeShape::Func {
                        params,
                        result,
                        effect: row(effect, n),
                        convention: *convention,
                    }
                }
                TypeShape::Tuple(elements) => TypeShape::Tuple(all(elements, n)),
                TypeShape::App { ctor, args } => {
                    let ctor = Box::new(go(ctor, n, with));
                    TypeShape::App {
                        ctor,
                        args: all(args, n),
                    }
                }
                TypeShape::Continuation {
                    arg,
                    result,
                    effect,
                } => {
                    let arg = Box::new(go(arg, n, with));
                    let result = Box::new(go(result, n, with));
                    TypeShape::Continuation {
                        arg,
                        result,
                        effect: row(effect, n),
                    }
                }
                leaf => leaf.clone(),
            }
        }
        go(self, &mut Some(n), with)
    }

    /// Rebuild with `f` applied to each child, then to the result.
    fn map(&self, f: &mut impl FnMut(TypeShape) -> TypeShape) -> TypeShape {
        let mapped = match self {
            TypeShape::Named { nominal, args } => TypeShape::Named {
                nominal: *nominal,
                args: args.iter().map(|arg| arg.map(f)).collect(),
            },
            TypeShape::Func {
                params,
                result,
                effect,
                convention,
            } => TypeShape::Func {
                params: params.iter().map(|param| param.map(f)).collect(),
                result: Box::new(result.map(f)),
                effect: effect.map_types(f),
                convention: *convention,
            },
            TypeShape::Tuple(elements) => {
                TypeShape::Tuple(elements.iter().map(|element| element.map(f)).collect())
            }
            TypeShape::App { ctor, args } => TypeShape::App {
                ctor: Box::new(ctor.map(f)),
                args: args.iter().map(|arg| arg.map(f)).collect(),
            },
            TypeShape::Continuation {
                arg,
                result,
                effect,
            } => TypeShape::Continuation {
                arg: Box::new(arg.map(f)),
                result: Box::new(result.map(f)),
                effect: effect.map_types(f),
            },
            leaf => leaf.clone(),
        };
        f(mapped)
    }

    /// Reference model of bound-variable substitution: replace `BoundVar(i)`
    /// with `args[i]` everywhere, rows included. `None` if an index is out of
    /// range.
    pub(crate) fn substitute_bound(&self, args: &[TypeShape]) -> Option<TypeShape> {
        let mut in_range = true;
        let result = self.map(&mut |shape| match shape {
            TypeShape::BoundVar(index) => match args.get(index as usize) {
                Some(arg) => arg.clone(),
                None => {
                    in_range = false;
                    shape
                }
            },
            other => other,
        });
        in_range.then_some(result)
    }

    /// Rename every row tail through `rename`.
    pub(crate) fn rename_row_vars(&self, rename: &impl Fn(u64) -> u64) -> TypeShape {
        self.map(&mut |shape| match shape {
            TypeShape::Func {
                params,
                result,
                effect,
                convention,
            } => TypeShape::Func {
                params,
                result,
                effect: effect.rename_tail(rename),
                convention,
            },
            TypeShape::Continuation {
                arg,
                result,
                effect,
            } => TypeShape::Continuation {
                arg,
                result,
                effect: effect.rename_tail(rename),
            },
            other => other,
        })
    }
}

fn build_all<'db>(db: &'db dyn salsa::Database, shapes: &[TypeShape]) -> Vec<Type<'db>> {
    shapes.iter().map(|shape| shape.build(db)).collect()
}

impl EffectShape {
    pub(crate) fn build<'db>(&self, db: &'db dyn salsa::Database) -> Effect<'db> {
        Effect {
            ability_id: ability_id(db, self.ability),
            args: build_all(db, &self.args),
        }
    }
}

impl RowShape {
    pub(crate) fn closed(effects: Vec<EffectShape>) -> Self {
        Self {
            effects,
            rest: None,
        }
    }

    pub(crate) fn build<'db>(&self, db: &'db dyn salsa::Database) -> EffectRow<'db> {
        EffectRow::new(
            db,
            self.effects
                .iter()
                .map(|effect| effect.build(db))
                .collect::<Vec<_>>(),
            self.rest.map(|id| EffectVar { id }),
        )
    }

    fn visit_types(&self, f: &mut impl FnMut(&TypeShape)) {
        for effect in &self.effects {
            effect.args.iter().for_each(|arg| arg.visit(f));
        }
    }

    fn map_types(&self, f: &mut impl FnMut(TypeShape) -> TypeShape) -> RowShape {
        RowShape {
            effects: self
                .effects
                .iter()
                .map(|effect| EffectShape {
                    ability: effect.ability,
                    args: effect.args.iter().map(|arg| arg.map(f)).collect(),
                })
                .collect(),
            rest: self.rest,
        }
    }

    fn rename_tail(self, rename: &impl Fn(u64) -> u64) -> RowShape {
        RowShape {
            rest: self.rest.map(rename),
            ..self
        }
    }

    /// Whether the effects, as a set, equal `other`'s.
    pub(crate) fn same_effect_set(&self, other: &RowShape) -> bool {
        self.effects
            .iter()
            .all(|effect| other.effects.contains(effect))
            && other
                .effects
                .iter()
                .all(|effect| self.effects.contains(effect))
    }
}

// ============================================================================
// Strategies
// ============================================================================

/// Which variable kinds and constructors generated types contain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TypeGen {
    /// Maximum constructor nesting.
    pub depth: u32,
    /// Unification variables `UniVar(0..univars)`; `0` for none.
    pub univars: u64,
    /// Bound variables `BoundVar(0..bound_vars)`; `0` for none.
    pub bound_vars: u32,
    /// Whether bound variables also appear in effect arguments.
    pub bound_vars_in_rows: bool,
    /// Row tails from `ROW_VAR_BASE..ROW_VAR_BASE + row_vars`; `0` closes
    /// every row.
    pub row_vars: u64,
    /// Whether the `Error` type appears.
    pub error: bool,
    /// Whether `App` and `Continuation` appear.
    pub higher_kinded: bool,
    /// Whether function types carry a convention floor above `Direct`.
    pub conventions: bool,
    /// Whether rows hold function types as effect arguments.
    pub row_functions: bool,
    /// Whether a row holds at most one instance of each ability.
    pub unique_abilities: bool,
}

impl TypeGen {
    /// Fully resolved first-order-kinded types: closed rows, no variables,
    /// no `Error`, `Direct` floors.
    pub(crate) const GROUND: Self = Self {
        depth: 3,
        univars: 0,
        bound_vars: 0,
        bound_vars_in_rows: true,
        row_vars: 0,
        error: false,
        higher_kinded: false,
        conventions: false,
        row_functions: true,
        unique_abilities: false,
    };

    pub(crate) const fn univars(self, univars: u64) -> Self {
        Self { univars, ..self }
    }

    pub(crate) const fn bound_vars(self, bound_vars: u32) -> Self {
        Self { bound_vars, ..self }
    }

    pub(crate) const fn bound_vars_in_rows(self, bound_vars_in_rows: bool) -> Self {
        Self {
            bound_vars_in_rows,
            ..self
        }
    }

    pub(crate) const fn row_vars(self, row_vars: u64) -> Self {
        Self { row_vars, ..self }
    }

    pub(crate) const fn error(self, error: bool) -> Self {
        Self { error, ..self }
    }

    pub(crate) const fn higher_kinded(self, higher_kinded: bool) -> Self {
        Self {
            higher_kinded,
            ..self
        }
    }

    pub(crate) const fn conventions(self, conventions: bool) -> Self {
        Self {
            conventions,
            ..self
        }
    }

    pub(crate) const fn row_functions(self, row_functions: bool) -> Self {
        Self {
            row_functions,
            ..self
        }
    }

    pub(crate) const fn unique_abilities(self, unique_abilities: bool) -> Self {
        Self {
            unique_abilities,
            ..self
        }
    }
}

fn leaf(cfg: TypeGen) -> BoxedStrategy<TypeShape> {
    let mut leaves: Vec<(u32, BoxedStrategy<TypeShape>)> = vec![
        (
            6,
            proptest::sample::select(PRIMS.to_vec())
                .prop_map(TypeShape::Prim)
                .boxed(),
        ),
        (
            1,
            Just(TypeShape::Named {
                nominal: 5,
                args: vec![],
            })
            .boxed(),
        ),
    ];
    if cfg.error {
        leaves.push((1, Just(TypeShape::Error).boxed()));
    }
    if cfg.univars > 0 {
        leaves.push((3, (0..cfg.univars).prop_map(TypeShape::UniVar).boxed()));
    }
    if cfg.bound_vars > 0 {
        leaves.push((3, (0..cfg.bound_vars).prop_map(TypeShape::BoundVar).boxed()));
    }
    proptest::strategy::Union::new_weighted(leaves).boxed()
}

fn arity_args(arity: usize, inner: &BoxedStrategy<TypeShape>) -> BoxedStrategy<Vec<TypeShape>> {
    proptest::collection::vec(inner.clone(), arity).boxed()
}

/// Strategy for one effect drawn from the ability pool.
fn effect_with(inner: BoxedStrategy<TypeShape>) -> BoxedStrategy<EffectShape> {
    (0..ABILITIES.len())
        .prop_flat_map(move |ability| {
            arity_args(ABILITIES[ability].2, &inner)
                .prop_map(move |args| EffectShape { ability, args })
        })
        .boxed()
}

/// Remove repeated effects (or repeated abilities, under `unique_abilities`)
/// keeping the first occurrence.
fn dedup_effects(effects: Vec<EffectShape>, unique_abilities: bool) -> Vec<EffectShape> {
    let mut kept: Vec<EffectShape> = Vec::with_capacity(effects.len());
    for effect in effects {
        let repeated = kept.iter().any(|seen| {
            if unique_abilities {
                seen.ability == effect.ability
            } else {
                *seen == effect
            }
        });
        if !repeated {
            kept.push(effect);
        }
    }
    kept
}

/// Strategy for a row whose effect arguments come from `args`.
fn row_with(cfg: TypeGen, args: BoxedStrategy<TypeShape>) -> BoxedStrategy<RowShape> {
    let rest = if cfg.row_vars > 0 {
        proptest::option::of(ROW_VAR_BASE..ROW_VAR_BASE + cfg.row_vars).boxed()
    } else {
        Just(None).boxed()
    };
    (proptest::collection::vec(effect_with(args), 0..=3), rest)
        .prop_map(move |(effects, rest)| RowShape {
            effects: dedup_effects(effects, cfg.unique_abilities),
            rest,
        })
        .boxed()
}

/// Strategy for effect arguments: shallow types without function types
/// unless `row_functions`, and without bound variables unless
/// `bound_vars_in_rows`.
fn row_args(cfg: TypeGen, inner: &BoxedStrategy<TypeShape>) -> BoxedStrategy<TypeShape> {
    let leaf_cfg = TypeGen {
        bound_vars: if cfg.bound_vars_in_rows {
            cfg.bound_vars
        } else {
            0
        },
        ..cfg
    };
    if cfg.row_functions && cfg.bound_vars_in_rows {
        prop_oneof![3 => leaf(leaf_cfg), 1 => inner.clone()].boxed()
    } else {
        leaf(leaf_cfg)
    }
}

/// Strategy for types under `cfg`.
pub(crate) fn type_shape(cfg: TypeGen) -> BoxedStrategy<TypeShape> {
    leaf(cfg)
        .prop_recursive(cfg.depth, 32, 3, move |inner| {
            let row = row_with(cfg, row_args(cfg, &inner));
            let convention = if cfg.conventions {
                proptest::sample::select(vec![
                    CallingConvention::Direct,
                    CallingConvention::EvidenceDirect,
                    CallingConvention::Cps,
                ])
                .boxed()
            } else {
                Just(CallingConvention::Direct).boxed()
            };
            let named = {
                let inner = inner.clone();
                (0..NOMINALS.len()).prop_flat_map(move |nominal| {
                    arity_args(NOMINALS[nominal].3, &inner)
                        .prop_map(move |args| TypeShape::Named { nominal, args })
                })
            };
            let func = (
                proptest::collection::vec(inner.clone(), 0..=2),
                inner.clone(),
                row.clone(),
                convention,
            )
                .prop_map(|(params, result, effect, convention)| TypeShape::Func {
                    params,
                    result: Box::new(result),
                    effect,
                    convention,
                });
            let tuple = proptest::collection::vec(inner.clone(), 2..=3).prop_map(TypeShape::Tuple);
            let mut options: Vec<(u32, BoxedStrategy<TypeShape>)> =
                vec![(3, named.boxed()), (3, func.boxed()), (2, tuple.boxed())];
            if cfg.higher_kinded {
                let app = (
                    inner.clone(),
                    proptest::collection::vec(inner.clone(), 1..=2),
                )
                    .prop_map(|(ctor, args)| TypeShape::App {
                        ctor: Box::new(ctor),
                        args,
                    });
                let continuation =
                    (inner.clone(), inner.clone(), row).prop_map(|(arg, result, effect)| {
                        TypeShape::Continuation {
                            arg: Box::new(arg),
                            result: Box::new(result),
                            effect,
                        }
                    });
                options.push((1, app.boxed()));
                options.push((1, continuation.boxed()));
            }
            proptest::strategy::Union::new_weighted(options)
        })
        .boxed()
}

/// Strategy for rows under `cfg`.
pub(crate) fn row_shape(cfg: TypeGen) -> BoxedStrategy<RowShape> {
    let args = row_args(cfg, &type_shape(TypeGen { depth: 1, ..cfg }));
    row_with(cfg, args)
}

/// Strategy for a single effect under `cfg`.
pub(crate) fn effect_shape(cfg: TypeGen) -> BoxedStrategy<EffectShape> {
    effect_with(row_args(cfg, &type_shape(TypeGen { depth: 1, ..cfg })))
}

/// Strategy for a row together with a permutation of its effects.
pub(crate) fn row_and_shuffle(cfg: TypeGen) -> BoxedStrategy<(RowShape, RowShape)> {
    row_shape(cfg)
        .prop_flat_map(|row| {
            let rest = row.rest;
            let shuffled = Just(row.effects.clone())
                .prop_shuffle()
                .prop_map(move |effects| RowShape { effects, rest });
            (Just(row), shuffled)
        })
        .boxed()
}

/// Strategy for a type under `cfg` that contains `hole` strictly inside a
/// constructor, possibly within an effect argument.
pub(crate) fn context_with_hole(cfg: TypeGen, hole: TypeShape) -> BoxedStrategy<TypeShape> {
    type_shape(cfg)
        .prop_filter("a hole needs a constructor", |shape| shape.size() > 1)
        .prop_flat_map(move |shape| {
            let hole = hole.clone();
            (1..shape.size()).prop_map(move |n| shape.replace_nth(n, &hole))
        })
        .boxed()
}

// ============================================================================
// Generalizations
// ============================================================================

/// How [`generalization`] names the variables it introduces.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Sharing {
    /// Every introduced variable is distinct and fresh, so the result is a
    /// linear pattern that the original type is an instance of.
    Linear,
    /// Introduced unification variables come from `UniVar(0..univars)` and
    /// opened rows from `ROW_VAR_BASE..ROW_VAR_BASE + row_vars`, so
    /// generalizations of one type may share variables inconsistently.
    /// With `separate_effect_args`, variables in effect arguments come from
    /// a disjoint pool, so they never stand for a type outside a row.
    Pool {
        univars: u64,
        row_vars: u64,
        separate_effect_args: bool,
    },
    /// Variables of effect arguments under `separate_effect_args`.
    EffectArgs { univars: u64 },
}

/// Strategy for a type obtained from `shape` by replacing subterms with
/// unification variables and by moving some effects of a row into a row
/// variable (with the remaining effects shuffled).
pub(crate) fn generalization(shape: &TypeShape, sharing: Sharing) -> BoxedStrategy<TypeShape> {
    generalize_inner(shape, sharing)
        .prop_map(move |shape| match sharing {
            Sharing::Linear => renumber_placeholders(shape),
            Sharing::Pool { .. } | Sharing::EffectArgs { .. } => shape,
        })
        .boxed()
}

fn generalize_all(shapes: &[TypeShape], sharing: Sharing) -> BoxedStrategy<Vec<TypeShape>> {
    shapes
        .iter()
        .map(|shape| generalize_inner(shape, sharing))
        .collect::<Vec<_>>()
        .boxed()
}

fn generalize_inner(shape: &TypeShape, sharing: Sharing) -> BoxedStrategy<TypeShape> {
    let replaced = match sharing {
        Sharing::Linear => Just(TypeShape::UniVar(PLACEHOLDER)).boxed(),
        Sharing::Pool { univars, .. } => (0..univars.max(1)).prop_map(TypeShape::UniVar).boxed(),
        Sharing::EffectArgs { univars } => (EFFECT_ARG_UNIVAR_BASE
            ..EFFECT_ARG_UNIVAR_BASE + univars.max(1))
            .prop_map(TypeShape::UniVar)
            .boxed(),
    };
    prop_oneof![1 => replaced, 4 => generalize_children(shape, sharing)].boxed()
}

fn generalize_children(shape: &TypeShape, sharing: Sharing) -> BoxedStrategy<TypeShape> {
    match shape.clone() {
        TypeShape::Named { nominal, args } => generalize_all(&args, sharing)
            .prop_map(move |args| TypeShape::Named { nominal, args })
            .boxed(),
        TypeShape::Func {
            params,
            result,
            effect,
            convention,
        } => (
            generalize_all(&params, sharing),
            generalize_inner(&result, sharing),
            generalize_row(&effect, sharing),
        )
            .prop_map(move |(params, result, effect)| TypeShape::Func {
                params,
                result: Box::new(result),
                effect,
                convention,
            })
            .boxed(),
        TypeShape::Tuple(elements) => generalize_all(&elements, sharing)
            .prop_map(TypeShape::Tuple)
            .boxed(),
        TypeShape::App { ctor, args } => (
            generalize_inner(&ctor, sharing),
            generalize_all(&args, sharing),
        )
            .prop_map(|(ctor, args)| TypeShape::App {
                ctor: Box::new(ctor),
                args,
            })
            .boxed(),
        TypeShape::Continuation {
            arg,
            result,
            effect,
        } => (
            generalize_inner(&arg, sharing),
            generalize_inner(&result, sharing),
            generalize_row(&effect, sharing),
        )
            .prop_map(|(arg, result, effect)| TypeShape::Continuation {
                arg: Box::new(arg),
                result: Box::new(result),
                effect,
            })
            .boxed(),
        leaf => Just(leaf).boxed(),
    }
}

/// Generalize a closed row: generalize effect arguments, then possibly move
/// a subset of the effects into a row variable and shuffle the rest. An
/// open row keeps its tail.
fn generalize_row(row: &RowShape, sharing: Sharing) -> BoxedStrategy<RowShape> {
    let arg_sharing = match sharing {
        Sharing::Pool {
            univars,
            separate_effect_args: true,
            ..
        } => Sharing::EffectArgs { univars },
        other => other,
    };
    let effects = row
        .effects
        .iter()
        .map(|effect| {
            let ability = effect.ability;
            generalize_all(&effect.args, arg_sharing)
                .prop_map(move |args| EffectShape { ability, args })
        })
        .collect::<Vec<_>>();
    let rest = row.rest;
    let count = row.effects.len();
    let tail = match sharing {
        Sharing::Linear => Just(PLACEHOLDER).boxed(),
        Sharing::Pool { row_vars, .. } => (ROW_VAR_BASE..ROW_VAR_BASE + row_vars.max(1)).boxed(),
        Sharing::EffectArgs { .. } => Just(ROW_VAR_BASE).boxed(),
    };
    (
        effects,
        proptest::collection::vec(any::<bool>(), count),
        any::<bool>(),
        tail,
    )
        .prop_flat_map(move |(effects, keep, open, tail)| {
            let opened = rest.is_none() && (open || keep.iter().any(|kept| !kept));
            let kept: Vec<_> = effects
                .into_iter()
                .zip(&keep)
                .filter(|(_, kept)| !opened || **kept)
                .map(|(effect, _)| effect)
                .collect();
            let rest = if opened { Some(tail) } else { rest };
            Just(kept)
                .prop_shuffle()
                .prop_map(move |effects| RowShape { effects, rest })
        })
        .boxed()
}

/// Give each placeholder unification variable and row tail a distinct id.
fn renumber_placeholders(shape: TypeShape) -> TypeShape {
    let mut next_univar = LINEAR_UNIVAR_BASE;
    let mut next_row = OPENED_ROW_VAR_BASE;
    let renumber_row = |row: RowShape, next_row: &mut u64| match row.rest {
        Some(PLACEHOLDER) => {
            *next_row += 1;
            RowShape {
                rest: Some(*next_row),
                ..row
            }
        }
        _ => row,
    };
    shape.map(&mut |shape| match shape {
        TypeShape::UniVar(PLACEHOLDER) => {
            next_univar += 1;
            TypeShape::UniVar(next_univar)
        }
        TypeShape::Func {
            params,
            result,
            effect,
            convention,
        } => TypeShape::Func {
            params,
            result,
            effect: renumber_row(effect, &mut next_row),
            convention,
        },
        TypeShape::Continuation {
            arg,
            result,
            effect,
        } => TypeShape::Continuation {
            arg,
            result,
            effect: renumber_row(effect, &mut next_row),
        },
        other => other,
    })
}

// ============================================================================
// Comparisons of built types
// ============================================================================

/// How [`types_equiv`] relates the effect rows at corresponding positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowRelation {
    /// The same effects as a set and the same tail.
    SetEqual,
    /// [`RowRelation::SetEqual`], or a closed empty row on the left against a
    /// row naming effects on the right (row unification's pure subsumption).
    PureSubsumes,
    /// Rows are not compared.
    Ignore,
}

/// Structural equality of types, comparing rows by `relation`.
pub(crate) fn types_equiv<'db>(
    db: &'db dyn salsa::Database,
    left: Type<'db>,
    right: Type<'db>,
    relation: RowRelation,
) -> bool {
    if left == right {
        return true;
    }
    let all = |left: &[Type<'db>], right: &[Type<'db>]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| types_equiv(db, *left, *right, relation))
    };
    match (left.kind(db), right.kind(db)) {
        (
            TypeKind::Named {
                id: left_id,
                args: left_args,
                ..
            },
            TypeKind::Named {
                id: right_id,
                args: right_args,
                ..
            },
        ) => left_id == right_id && all(left_args, right_args),
        (
            TypeKind::Func {
                params: left_params,
                result: left_result,
                effect: left_effect,
                minimum_convention: left_convention,
            },
            TypeKind::Func {
                params: right_params,
                result: right_result,
                effect: right_effect,
                minimum_convention: right_convention,
            },
        ) => {
            left_convention == right_convention
                && all(left_params, right_params)
                && types_equiv(db, *left_result, *right_result, relation)
                && rows_equiv(db, *left_effect, *right_effect, relation)
        }
        (TypeKind::Tuple(left), TypeKind::Tuple(right)) => all(left, right),
        (
            TypeKind::App {
                ctor: left_ctor,
                args: left_args,
            },
            TypeKind::App {
                ctor: right_ctor,
                args: right_args,
            },
        ) => types_equiv(db, *left_ctor, *right_ctor, relation) && all(left_args, right_args),
        (
            TypeKind::Continuation {
                arg: left_arg,
                result: left_result,
                effect: left_effect,
            },
            TypeKind::Continuation {
                arg: right_arg,
                result: right_result,
                effect: right_effect,
            },
        ) => {
            types_equiv(db, *left_arg, *right_arg, relation)
                && types_equiv(db, *left_result, *right_result, relation)
                && rows_equiv(db, *left_effect, *right_effect, relation)
        }
        _ => false,
    }
}

/// Row comparison used by [`types_equiv`].
pub(crate) fn rows_equiv<'db>(
    db: &'db dyn salsa::Database,
    left: EffectRow<'db>,
    right: EffectRow<'db>,
    relation: RowRelation,
) -> bool {
    match relation {
        RowRelation::Ignore => return true,
        RowRelation::PureSubsumes if left.is_pure(db) && !right.effects(db).is_empty() => {
            return true;
        }
        _ => {}
    }
    let contains = |row: EffectRow<'db>, effect: &Effect<'db>| {
        row.effects(db).iter().any(|candidate| {
            candidate.ability_id == effect.ability_id
                && candidate.args.len() == effect.args.len()
                && candidate
                    .args
                    .iter()
                    .zip(&effect.args)
                    .all(|(a, b)| types_equiv(db, *a, *b, relation))
        })
    };
    left.rest(db) == right.rest(db)
        && left
            .effects(db)
            .iter()
            .all(|effect| contains(right, effect))
        && right
            .effects(db)
            .iter()
            .all(|effect| contains(left, effect))
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        /// Shape equality is interned-type equality.
        #[test]
        fn shapes_build_equal_types_iff_equal(
            left in type_shape(TypeGen::GROUND.univars(2).row_vars(2).higher_kinded(true)),
            right in type_shape(TypeGen::GROUND.univars(2).row_vars(2).higher_kinded(true)),
        ) {
            let db = salsa::DatabaseImpl::new();
            prop_assert_eq!(left == right, left.build(&db) == right.build(&db));
            prop_assert_eq!(left.build(&db), left.clone().build(&db));
        }

        /// Generated rows are sets, and `unique_abilities` rows name each
        /// ability once.
        #[test]
        fn generated_rows_hold_each_effect_once(
            row in row_shape(TypeGen::GROUND.univars(2)),
            unique in row_shape(TypeGen::GROUND.unique_abilities(true)),
        ) {
            for (i, effect) in row.effects.iter().enumerate() {
                prop_assert!(!row.effects[..i].contains(effect));
            }
            for (i, effect) in unique.effects.iter().enumerate() {
                prop_assert!(unique.effects[..i].iter().all(|e| e.ability != effect.ability));
            }
        }

        /// A hole lands strictly inside the context.
        #[test]
        fn context_contains_its_hole(context in context_with_hole(TypeGen::GROUND, TypeShape::UniVar(7))) {
            prop_assert_ne!(&context, &TypeShape::UniVar(7));
            prop_assert_eq!(context.univars(), vec![7]);
        }

        /// A linear generalization has distinct variables and the original as
        /// an instance.
        #[test]
        fn linear_generalization_is_a_pattern(
            (shape, pattern) in type_shape(TypeGen::GROUND)
                .prop_flat_map(|shape| (Just(shape.clone()), generalization(&shape, Sharing::Linear)))
        ) {
            let mut seen = Vec::new();
            pattern.visit(&mut |shape| {
                if let TypeShape::UniVar(id) = shape {
                    seen.push(*id);
                }
            });
            let distinct = pattern.univars();
            prop_assert_eq!(seen.len(), distinct.len());
            prop_assert!(!pattern.any(|shape| *shape == TypeShape::UniVar(PLACEHOLDER)));
            prop_assert!(shape.univars().is_empty());
        }
    }
}
