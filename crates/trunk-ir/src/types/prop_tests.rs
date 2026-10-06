//! Model-based property tests for attribute maps and the interners.
//!
//! Each test is a state machine: a reference model generates the next action
//! from its current state, and the system under test applies the same action.
//! The attribute map test applies random inserts, removals, and extensions to
//! an `AttributeMap` and to a `BTreeMap` keyed by the symbol text, and checks
//! after every step that the map holds, orders, and converts exactly what the
//! model holds. The interner tests intern random data and check that equal
//! data shares one reference, distinct data gets distinct references, and
//! every earlier reference stays valid as more data is interned.

use std::collections::BTreeMap;
use std::hash::{BuildHasher, RandomState};

use proptest::prelude::*;
use proptest::sample::select;
use proptest_state_machine::{ReferenceStateMachine, StateMachineTest, prop_state_machine};

use super::*;

/// Keys drawn mostly from a small pool, so actions often hit existing entries.
fn key() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => prop::sample::select(vec!["", "a", "alpha", "mid", "zeta", "Z", "a.b"])
            .prop_map(str::to_owned),
        1 => any::<String>(),
    ]
}

/// Integers concentrated around the bounds of the typed getters.
fn integer() -> impl Strategy<Value = i128> {
    prop_oneof![
        any::<u8>().prop_map(i128::from),
        any::<i32>().prop_map(i128::from),
        any::<u32>().prop_map(i128::from),
        any::<i64>().prop_map(i128::from),
        any::<u64>().prop_map(i128::from),
        any::<i128>(),
        -2i128..=2,
    ]
}

/// An attribute value; strings are interned into the test's context.
#[derive(Clone, Debug)]
enum ValueSpec {
    Unit,
    Bool(bool),
    Int(i128),
    Str(String),
    Symbol(String),
}

fn value() -> impl Strategy<Value = ValueSpec> {
    prop_oneof![
        Just(ValueSpec::Unit),
        any::<bool>().prop_map(ValueSpec::Bool),
        integer().prop_map(ValueSpec::Int),
        "[a-z]{0,4}".prop_map(ValueSpec::Str),
        "[a-z]{0,4}".prop_map(ValueSpec::Symbol),
    ]
}

fn build_value(ctx: &mut IrContext, spec: &ValueSpec) -> Attribute {
    match spec {
        ValueSpec::Unit => Attribute::Unit,
        ValueSpec::Bool(b) => Attribute::Bool(*b),
        ValueSpec::Int(v) => Attribute::Int(*v),
        ValueSpec::Str(s) => ctx.string_attr(s),
        ValueSpec::Symbol(s) => Attribute::SymbolRef(SymbolPath::from(s.as_str())),
    }
}

#[derive(Clone, Debug)]
enum MapAction {
    Insert(String, ValueSpec),
    Remove(String),
    Extend(Vec<(String, ValueSpec)>),
    Clear,
}

fn map_action() -> impl Strategy<Value = MapAction> {
    prop_oneof![
        6 => (key(), value()).prop_map(|(k, v)| MapAction::Insert(k, v)),
        2 => key().prop_map(MapAction::Remove),
        2 => prop::collection::vec((key(), value()), 0..6).prop_map(MapAction::Extend),
        1 => Just(MapAction::Clear),
    ]
}

/// The model's view of a typed integer getter.
fn expected_integer<T>(
    value: Option<&Attribute>,
    target: &'static str,
    convert: impl FnOnce(i128) -> Result<T, std::num::TryFromIntError>,
) -> Result<Option<T>, IntegerOutOfRange> {
    match value.and_then(Attribute::as_i128) {
        None => Ok(None),
        Some(value) => convert(value)
            .map(Some)
            .map_err(|_| IntegerOutOfRange { value, target }),
    }
}

/// Every lookup of `key` in `attrs` must agree with the model.
fn check_lookup(
    ctx: &IrContext,
    attrs: &AttributeMap,
    model: &BTreeMap<String, Attribute>,
    key: &str,
) {
    let expected = model.get(key);
    let symbol = Symbol::new(key);
    assert_eq!(attrs.get(key), expected);
    assert_eq!(attrs.get(&symbol), expected);
    assert_eq!(attrs.contains_key(key), expected.is_some());
    assert_eq!(attrs.get_bool(key), expected.and_then(Attribute::as_bool));
    assert_eq!(attrs.get_i128(key), expected.and_then(Attribute::as_i128));
    assert_eq!(
        attrs.get_i64(key),
        expected_integer(expected, "i64", i64::try_from)
    );
    assert_eq!(
        attrs.get_i32(key),
        expected_integer(expected, "i32", i32::try_from)
    );
    assert_eq!(
        attrs.get_u64(key),
        expected_integer(expected, "u64", u64::try_from)
    );
    assert_eq!(
        attrs.get_u32(key),
        expected_integer(expected, "u32", u32::try_from)
    );
    assert_eq!(
        attrs.get_u8(key),
        expected_integer(expected, "u8", u8::try_from)
    );
    assert_eq!(
        attrs.get_str(ctx, key),
        expected.and_then(|value| value.as_str(ctx))
    );
    assert_eq!(
        attrs.get_symbol_ref(key),
        expected.and_then(Attribute::as_symbol_ref)
    );
}

/// `attrs` must hold exactly the model's entries, in the model's key order,
/// and equal (with the same hash) a map built from them in reverse order.
fn check_map(attrs: &AttributeMap, model: &BTreeMap<String, Attribute>, hasher: &RandomState) {
    assert_eq!(attrs.len(), model.len());
    assert_eq!(attrs.is_empty(), model.is_empty());
    assert!(
        attrs
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .eq(model.iter().map(|(key, value)| (key.as_str(), value)))
    );
    assert!(
        attrs
            .keys()
            .map(Symbol::as_str)
            .eq(model.keys().map(String::as_str))
    );
    assert!(attrs.values().eq(model.values()));
    assert!(
        attrs
            .iter()
            .rev()
            .map(|(key, _)| key.as_str())
            .eq(model.keys().rev().map(String::as_str))
    );

    let reversed: AttributeMap = model
        .iter()
        .rev()
        .map(|(key, value)| (Symbol::new(key), value.clone()))
        .collect();
    assert_eq!(attrs, &reversed);
    assert_eq!(hasher.hash_one(attrs), hasher.hash_one(&reversed));
}

/// The model's value for `spec`: strings were interned when the system under
/// test built the value.
fn expected_value(ctx: &IrContext, spec: &ValueSpec) -> Attribute {
    match spec {
        ValueSpec::Unit => Attribute::Unit,
        ValueSpec::Bool(b) => Attribute::Bool(*b),
        ValueSpec::Int(v) => Attribute::Int(*v),
        ValueSpec::Str(s) => Attribute::String(ctx.lookup_str(s).expect("interned value")),
        ValueSpec::Symbol(s) => Attribute::SymbolRef(SymbolPath::from(s.as_str())),
    }
}

/// Keys checked after every step besides those an action touches.
#[derive(Clone, Debug)]
struct MapModel {
    probes: Vec<String>,
    entries: BTreeMap<String, ValueSpec>,
}

impl MapModel {
    fn expected(&self, ctx: &IrContext) -> BTreeMap<String, Attribute> {
        self.entries
            .iter()
            .map(|(key, spec)| (key.clone(), expected_value(ctx, spec)))
            .collect()
    }
}

struct MapMachine;

impl ReferenceStateMachine for MapMachine {
    type State = MapModel;
    type Transition = MapAction;

    fn init_state() -> BoxedStrategy<MapModel> {
        prop::collection::vec(key(), 0..4)
            .prop_map(|probes| MapModel {
                probes,
                entries: BTreeMap::new(),
            })
            .boxed()
    }

    fn transitions(_: &MapModel) -> BoxedStrategy<MapAction> {
        map_action().boxed()
    }

    fn apply(mut state: MapModel, action: &MapAction) -> MapModel {
        match action {
            MapAction::Insert(key, spec) => {
                state.entries.insert(key.clone(), spec.clone());
            }
            MapAction::Remove(key) => {
                state.entries.remove(key);
            }
            // Later entries replace earlier ones with the same key.
            MapAction::Extend(entries) => state.entries.extend(entries.iter().cloned()),
            MapAction::Clear => state.entries.clear(),
        }
        state
    }
}

struct MapSut {
    ctx: IrContext,
    hasher: RandomState,
    attrs: AttributeMap,
}

struct MapTest;

impl StateMachineTest for MapTest {
    type SystemUnderTest = MapSut;
    type Reference = MapMachine;

    fn init_test(_: &MapModel) -> MapSut {
        MapSut {
            ctx: IrContext::new(),
            hasher: RandomState::new(),
            attrs: AttributeMap::new(),
        }
    }

    fn apply(mut sut: MapSut, model: &MapModel, action: MapAction) -> MapSut {
        // The previous step checked the map against the model, so its entry
        // for a key is the model's previous entry.
        let mut touched = Vec::new();
        match action {
            MapAction::Insert(key, spec) => {
                let value = build_value(&mut sut.ctx, &spec);
                let previous = sut.attrs.get(key.as_str()).cloned();
                let replaced = sut.attrs.insert(Symbol::new(&key), value);
                assert_eq!(replaced, previous);
                touched.push(key);
            }
            MapAction::Remove(key) => {
                let previous = sut.attrs.get(key.as_str()).cloned();
                assert_eq!(sut.attrs.remove(key.as_str()), previous);
                touched.push(key);
            }
            MapAction::Extend(entries) => {
                let entries: Vec<_> = entries
                    .into_iter()
                    .map(|(key, spec)| (key, build_value(&mut sut.ctx, &spec)))
                    .collect();
                sut.attrs.extend(
                    entries
                        .iter()
                        .map(|(key, value)| (Symbol::new(key), value.clone())),
                );
                touched.extend(entries.into_iter().map(|(key, _)| key));
            }
            MapAction::Clear => sut.attrs.clear(),
        }
        let expected = model.expected(&sut.ctx);
        for key in &touched {
            check_lookup(&sut.ctx, &sut.attrs, &expected, key);
        }
        sut
    }

    fn check_invariants(sut: &MapSut, model: &MapModel) {
        let expected = model.expected(&sut.ctx);
        check_map(&sut.attrs, &expected, &sut.hasher);
        for key in &model.probes {
            check_lookup(&sut.ctx, &sut.attrs, &expected, key);
        }
    }

    fn teardown(sut: MapSut, model: MapModel) {
        let expected = model.expected(&sut.ctx);
        for key in expected.keys() {
            check_lookup(&sut.ctx, &sut.attrs, &expected, key);
        }

        // Collecting the same entries in any order yields an equal map.
        let collected: AttributeMap = expected
            .iter()
            .map(|(key, value)| (Symbol::new(key), value.clone()))
            .collect();
        assert_eq!(&collected, &sut.attrs);
    }
}

prop_state_machine! {
    #[test]
    fn attribute_map_matches_a_btree_map_model(sequential 1..48 => MapTest);
}

// ============================================================================
// Type interner
// ============================================================================

/// Small attribute dictionaries over a few keys and values, so specs collide.
type SmallAttrs = Vec<(&'static str, i8)>;

/// A type to intern: parameters name earlier distinct types by interning
/// order and may carry their own attributes.
#[derive(Clone, Debug)]
struct TypeSpec {
    dialect: &'static str,
    name: &'static str,
    params: Vec<(usize, SmallAttrs)>,
    attrs: SmallAttrs,
}

/// The canonical form of a spec: two specs build equal data exactly when
/// their keys are equal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeKey {
    dialect: &'static str,
    name: &'static str,
    params: Vec<(usize, BTreeMap<&'static str, i8>)>,
    attrs: BTreeMap<&'static str, i8>,
}

impl TypeSpec {
    fn key(&self) -> TypeKey {
        TypeKey {
            dialect: self.dialect,
            name: self.name,
            params: self
                .params
                .iter()
                .map(|(param, attrs)| (*param, attrs.iter().copied().collect()))
                .collect(),
            attrs: self.attrs.iter().copied().collect(),
        }
    }
}

fn small_attrs() -> impl Strategy<Value = SmallAttrs> {
    prop::collection::vec((select(vec!["k", "name"]), 0i8..2), 0..2)
}

/// A spec whose parameters name one of the `known` distinct types.
fn type_spec(known: usize) -> impl Strategy<Value = TypeSpec> {
    let params = if known == 0 {
        Just(Vec::new()).boxed()
    } else {
        prop::collection::vec(
            (
                0..known,
                prop_oneof![3 => Just(Vec::new()), 1 => small_attrs()],
            ),
            0..3,
        )
        .boxed()
    };
    (
        select(vec!["core", "test"]),
        select(vec!["i32", "tuple", "numbered"]),
        params,
        small_attrs(),
    )
        .prop_map(|(dialect, name, params, attrs)| TypeSpec {
            dialect,
            name,
            params,
            attrs,
        })
}

fn small_map(entries: impl IntoIterator<Item = (&'static str, i8)>) -> AttributeMap {
    entries
        .into_iter()
        .map(|(key, value)| (Symbol::new(key), Attribute::Int(value.into())))
        .collect()
}

/// Build the data for `spec`; parameters resolve against `refs`.
fn build_type(spec: &TypeSpec, refs: &[TypeRef]) -> TypeData {
    let mut builder = TypeDataBuilder::new(spec.dialect, spec.name);
    for (param, attrs) in &spec.params {
        builder = builder.param_with_attrs(refs[*param], small_map(attrs.iter().copied()));
    }
    for &(key, value) in &spec.attrs {
        builder = builder.attr(key, Attribute::Int(value.into()));
    }
    builder.build()
}

/// Build a fresh copy of the data for `key`.
fn build_key(key: &TypeKey, refs: &[TypeRef]) -> TypeData {
    let mut builder = TypeDataBuilder::new(key.dialect, key.name);
    for (param, attrs) in &key.params {
        let attrs = small_map(attrs.iter().map(|(&key, &value)| (key, value)));
        builder = builder.param_with_attrs(refs[*param], attrs);
    }
    for (&name, &value) in &key.attrs {
        builder = builder.attr(name, Attribute::Int(value.into()));
    }
    builder.build()
}

#[derive(Clone, Debug)]
enum InternAction {
    /// Intern data built from a spec.
    Intern(TypeSpec),
    /// Intern a fresh copy of data interned before.
    Reintern(usize),
}

/// Each distinct type once, in interning order.
type InternModel = Vec<TypeKey>;

struct TypeInternMachine;

impl ReferenceStateMachine for TypeInternMachine {
    type State = InternModel;
    type Transition = InternAction;

    fn init_state() -> BoxedStrategy<InternModel> {
        Just(Vec::new()).boxed()
    }

    fn transitions(state: &InternModel) -> BoxedStrategy<InternAction> {
        let intern = type_spec(state.len()).prop_map(InternAction::Intern);
        if state.is_empty() {
            intern.boxed()
        } else {
            prop_oneof![
                3 => intern,
                1 => (0..state.len()).prop_map(InternAction::Reintern),
            ]
            .boxed()
        }
    }

    fn preconditions(state: &InternModel, action: &InternAction) -> bool {
        match action {
            InternAction::Intern(spec) => spec.params.iter().all(|&(param, _)| param < state.len()),
            InternAction::Reintern(index) => *index < state.len(),
        }
    }

    fn apply(mut state: InternModel, action: &InternAction) -> InternModel {
        if let InternAction::Intern(spec) = action {
            let key = spec.key();
            if !state.contains(&key) {
                state.push(key);
            }
        }
        state
    }
}

struct TypeInternSut {
    interner: TypeInterner,
    /// The reference of each distinct type, in interning order.
    refs: Vec<TypeRef>,
    /// A copy of each distinct type's data, built from its key.
    known: Vec<TypeData>,
}

struct TypeInternTest;

impl StateMachineTest for TypeInternTest {
    type SystemUnderTest = TypeInternSut;
    type Reference = TypeInternMachine;

    fn init_test(_: &InternModel) -> TypeInternSut {
        TypeInternSut {
            interner: TypeInterner::new(),
            refs: Vec::new(),
            known: Vec::new(),
        }
    }

    fn apply(mut sut: TypeInternSut, model: &InternModel, action: InternAction) -> TypeInternSut {
        let (data, key) = match action {
            InternAction::Intern(spec) => (build_type(&spec, &sut.refs), spec.key()),
            InternAction::Reintern(index) => {
                (build_key(&model[index], &sut.refs), model[index].clone())
            }
        };
        assert_eq!(data.validate_param_attrs(), Ok(()));
        let index = model
            .iter()
            .position(|known| *known == key)
            .expect("interned in the model");
        let previous = sut.refs.get(index).copied();
        assert_eq!(sut.interner.lookup(&data), previous);
        let r = sut.interner.intern(data);
        match previous {
            Some(existing) => assert_eq!(r, existing),
            None => {
                assert!(sut.refs.iter().all(|&other| other != r));
                sut.known.push(build_key(&key, &sut.refs));
                sut.refs.push(r);
            }
        }
        sut
    }

    fn check_invariants(sut: &TypeInternSut, model: &InternModel) {
        assert_eq!(sut.refs.len(), model.len());
        let known = &sut.known;
        for (data, &r) in known.iter().zip(&sut.refs) {
            assert_eq!(sut.interner.get(r), data);
            assert_eq!(sut.interner.lookup(data), Some(r));
        }
        assert_eq!(sut.interner.iter().count(), known.len());
        assert!(sut.interner.iter().eq(sut.refs.iter().copied().zip(known)));
    }
}

prop_state_machine! {
    #[test]
    fn type_interner_dedups_and_keeps_references_stable(
        sequential 1..128 => TypeInternTest
    );
}

// ============================================================================
// Path interner
// ============================================================================

fn path() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => select(vec!["", "file:///a.trb", "file:///b.trb", "a", "A"])
            .prop_map(str::to_owned),
        1 => any::<String>(),
    ]
}

/// A path, and whether to intern it or only look it up.
#[derive(Clone, Debug)]
struct PathAction {
    path: String,
    intern: bool,
}

struct PathInternMachine;

impl ReferenceStateMachine for PathInternMachine {
    /// Each interned path once, in interning order.
    type State = Vec<String>;
    type Transition = PathAction;

    fn init_state() -> BoxedStrategy<Vec<String>> {
        Just(Vec::new()).boxed()
    }

    fn transitions(_: &Vec<String>) -> BoxedStrategy<PathAction> {
        (path(), any::<bool>())
            .prop_map(|(path, intern)| PathAction { path, intern })
            .boxed()
    }

    fn apply(mut state: Vec<String>, action: &PathAction) -> Vec<String> {
        if action.intern && !state.contains(&action.path) {
            state.push(action.path.clone());
        }
        state
    }
}

struct PathInternSut {
    interner: PathInterner,
    /// The reference of each interned path, in interning order.
    refs: Vec<PathRef>,
}

struct PathInternTest;

impl StateMachineTest for PathInternTest {
    type SystemUnderTest = PathInternSut;
    type Reference = PathInternMachine;

    fn init_test(_: &Vec<String>) -> PathInternSut {
        PathInternSut {
            interner: PathInterner::new(),
            refs: Vec::new(),
        }
    }

    fn apply(mut sut: PathInternSut, model: &Vec<String>, action: PathAction) -> PathInternSut {
        let previous = model
            .iter()
            .position(|path| *path == action.path)
            .and_then(|index| sut.refs.get(index).copied());
        assert_eq!(sut.interner.lookup(&action.path), previous);
        if action.intern {
            let r = sut.interner.intern(&action.path);
            match previous {
                Some(existing) => assert_eq!(r, existing),
                None => {
                    assert!(sut.refs.iter().all(|&other| other != r));
                    sut.refs.push(r);
                }
            }
        }
        sut
    }

    fn check_invariants(sut: &PathInternSut, model: &Vec<String>) {
        assert_eq!(sut.refs.len(), model.len());
        for (path, &r) in model.iter().zip(&sut.refs) {
            assert_eq!(sut.interner.get(r), path.as_str());
            assert_eq!(sut.interner.lookup(path), Some(r));
        }
    }
}

prop_state_machine! {
    #[test]
    fn path_interner_dedups_and_keeps_references_stable(
        sequential 1..128 => PathInternTest
    );
}
