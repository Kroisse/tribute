//! Model-based property tests for attribute maps and the interners.
//!
//! The attribute map test applies random inserts, removals, and extensions to
//! an `AttributeMap` and to a `BTreeMap` keyed by the symbol text, and checks
//! after every step that the map holds, orders, and converts exactly what the
//! model holds. The interner tests intern random data and check that equal
//! data shares one reference, distinct data gets distinct references, and
//! every earlier reference stays valid as more data is interned.

use std::collections::BTreeMap;
use std::hash::{BuildHasher, RandomState};

use proptest::prelude::*;
use proptest::sample::Index;
use rustc_hash::FxHashMap as HashMap;

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
) -> Result<(), TestCaseError> {
    let expected = model.get(key);
    let symbol = Symbol::new(key);
    prop_assert_eq!(attrs.get(key), expected);
    prop_assert_eq!(attrs.get(&symbol), expected);
    prop_assert_eq!(attrs.contains_key(key), expected.is_some());
    prop_assert_eq!(attrs.get_bool(key), expected.and_then(Attribute::as_bool));
    prop_assert_eq!(attrs.get_i128(key), expected.and_then(Attribute::as_i128));
    prop_assert_eq!(
        attrs.get_i64(key),
        expected_integer(expected, "i64", i64::try_from)
    );
    prop_assert_eq!(
        attrs.get_i32(key),
        expected_integer(expected, "i32", i32::try_from)
    );
    prop_assert_eq!(
        attrs.get_u64(key),
        expected_integer(expected, "u64", u64::try_from)
    );
    prop_assert_eq!(
        attrs.get_u32(key),
        expected_integer(expected, "u32", u32::try_from)
    );
    prop_assert_eq!(
        attrs.get_u8(key),
        expected_integer(expected, "u8", u8::try_from)
    );
    prop_assert_eq!(
        attrs.get_str(ctx, key),
        expected.and_then(|value| value.as_str(ctx))
    );
    prop_assert_eq!(
        attrs.get_symbol_ref(key),
        expected.and_then(Attribute::as_symbol_ref)
    );
    Ok(())
}

/// `attrs` must hold exactly the model's entries, in the model's key order,
/// and equal (with the same hash) a map built from them in reverse order.
fn check_map(
    attrs: &AttributeMap,
    model: &BTreeMap<String, Attribute>,
    hasher: &RandomState,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(attrs.len(), model.len());
    prop_assert_eq!(attrs.is_empty(), model.is_empty());
    prop_assert!(
        attrs
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .eq(model.iter().map(|(key, value)| (key.as_str(), value)))
    );
    prop_assert!(
        attrs
            .keys()
            .map(Symbol::as_str)
            .eq(model.keys().map(String::as_str))
    );
    prop_assert!(attrs.values().eq(model.values()));
    prop_assert!(
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
    prop_assert_eq!(attrs, &reversed);
    prop_assert_eq!(hasher.hash_one(attrs), hasher.hash_one(&reversed));
    Ok(())
}

proptest! {
    #[test]
    fn attribute_map_matches_a_btree_map_model(
        actions in prop::collection::vec(map_action(), 1..48),
        probes in prop::collection::vec(key(), 0..4),
    ) {
        let mut ctx = IrContext::new();
        let hasher = RandomState::new();
        let mut attrs = AttributeMap::new();
        let mut model = BTreeMap::new();

        for action in actions {
            let mut touched = Vec::new();
            match action {
                MapAction::Insert(key, spec) => {
                    let value = build_value(&mut ctx, &spec);
                    let replaced = attrs.insert(Symbol::new(&key), value.clone());
                    prop_assert_eq!(replaced, model.insert(key.clone(), value));
                    touched.push(key);
                }
                MapAction::Remove(key) => {
                    prop_assert_eq!(attrs.remove(key.as_str()), model.remove(&key));
                    touched.push(key);
                }
                MapAction::Extend(entries) => {
                    // Later entries replace earlier ones with the same key.
                    let entries: Vec<_> = entries
                        .into_iter()
                        .map(|(key, spec)| (key, build_value(&mut ctx, &spec)))
                        .collect();
                    attrs.extend(
                        entries
                            .iter()
                            .map(|(key, value)| (Symbol::new(key), value.clone())),
                    );
                    for (key, value) in entries {
                        model.insert(key.clone(), value);
                        touched.push(key);
                    }
                }
                MapAction::Clear => {
                    attrs.clear();
                    model.clear();
                }
            }

            check_map(&attrs, &model, &hasher)?;
            for key in touched.iter().chain(&probes) {
                check_lookup(&ctx, &attrs, &model, key)?;
            }
        }
        for key in model.keys() {
            check_lookup(&ctx, &attrs, &model, key)?;
        }

        // Collecting the same entries in any order yields an equal map.
        let collected: AttributeMap = model
            .iter()
            .map(|(key, value)| (Symbol::new(key), value.clone()))
            .collect();
        prop_assert_eq!(&collected, &attrs);
    }
}

// ============================================================================
// Type interner
// ============================================================================

/// A type to intern: parameters name earlier types by index and may carry
/// their own attributes.
#[derive(Clone, Debug)]
struct TypeSpec {
    dialect: &'static str,
    name: &'static str,
    params: Vec<(Index, Vec<(&'static str, i8)>)>,
    attrs: Vec<(&'static str, i8)>,
}

/// Small attribute dictionaries over a few keys and values, so specs collide.
fn small_attrs() -> impl Strategy<Value = Vec<(&'static str, i8)>> {
    prop::collection::vec((prop::sample::select(vec!["k", "name"]), 0i8..2), 0..2)
}

fn type_spec() -> impl Strategy<Value = TypeSpec> {
    (
        prop::sample::select(vec!["core", "test"]),
        prop::sample::select(vec!["i32", "tuple", "numbered"]),
        prop::collection::vec(
            (
                any::<Index>(),
                prop_oneof![3 => Just(Vec::new()), 1 => small_attrs()],
            ),
            0..3,
        ),
        small_attrs(),
    )
        .prop_map(|(dialect, name, params, attrs)| TypeSpec {
            dialect,
            name,
            params,
            attrs,
        })
}

fn small_map(entries: &[(&'static str, i8)]) -> AttributeMap {
    entries
        .iter()
        .map(|&(key, value)| (Symbol::new(key), Attribute::Int(value.into())))
        .collect()
}

/// Build the data for `spec`; parameters resolve against `known` and are
/// dropped while nothing has been interned yet.
fn build_type(spec: &TypeSpec, known: &[(TypeData, TypeRef)]) -> TypeData {
    let mut builder = TypeDataBuilder::new(spec.dialect, spec.name);
    if !known.is_empty() {
        for (index, attrs) in &spec.params {
            let (_, ty) = index.get(known);
            builder = builder.param_with_attrs(*ty, small_map(attrs));
        }
    }
    for &(key, value) in &spec.attrs {
        builder = builder.attr(key, Attribute::Int(value.into()));
    }
    builder.build()
}

#[derive(Clone, Debug)]
enum InternAction {
    /// Intern data built from a spec.
    Intern(TypeSpec),
    /// Intern a fresh copy of data interned before.
    Reintern(Index),
}

fn intern_action() -> impl Strategy<Value = InternAction> {
    prop_oneof![
        3 => type_spec().prop_map(InternAction::Intern),
        1 => any::<Index>().prop_map(InternAction::Reintern),
    ]
}

proptest! {
    #[test]
    fn type_interner_dedups_and_keeps_references_stable(
        actions in prop::collection::vec(intern_action(), 1..128),
    ) {
        let mut interner = TypeInterner::new();
        // Each distinct data once, in interning order, with its reference.
        let mut known: Vec<(TypeData, TypeRef)> = Vec::new();
        let mut by_data: HashMap<TypeData, TypeRef> = HashMap::default();

        for action in actions {
            let data = match action {
                InternAction::Intern(spec) => build_type(&spec, &known),
                InternAction::Reintern(index) if !known.is_empty() => index.get(&known).0.clone(),
                InternAction::Reintern(_) => continue,
            };
            prop_assert_eq!(data.validate_param_attrs(), Ok(()));
            let previous = by_data.get(&data).copied();
            prop_assert_eq!(interner.lookup(&data), previous);
            let r = interner.intern(data.clone());
            match previous {
                Some(existing) => prop_assert_eq!(r, existing),
                None => {
                    prop_assert!(known.iter().all(|&(_, other)| other != r));
                    by_data.insert(data.clone(), r);
                    known.push((data, r));
                }
            }

            for (data, r) in &known {
                prop_assert_eq!(interner.get(*r), data);
                prop_assert_eq!(interner.lookup(data), Some(*r));
            }
            prop_assert_eq!(interner.iter().count(), known.len());
            prop_assert!(
                interner
                    .iter()
                    .eq(known.iter().map(|(data, r)| (*r, data)))
            );
        }
    }
}

// ============================================================================
// Path interner
// ============================================================================

fn path() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::sample::select(vec!["", "file:///a.trb", "file:///b.trb", "a", "A"])
            .prop_map(str::to_owned),
        1 => any::<String>(),
    ]
}

proptest! {
    #[test]
    fn path_interner_dedups_and_keeps_references_stable(
        actions in prop::collection::vec((path(), any::<bool>()), 1..128),
    ) {
        let mut interner = PathInterner::new();
        let mut model: HashMap<String, PathRef> = HashMap::default();

        for (path, intern) in actions {
            let previous = model.get(&path).copied();
            prop_assert_eq!(interner.lookup(&path), previous);
            if intern {
                let r = interner.intern(&path);
                match previous {
                    Some(existing) => prop_assert_eq!(r, existing),
                    None => {
                        prop_assert!(model.values().all(|&other| other != r));
                        model.insert(path, r);
                    }
                }
            }

            for (path, r) in &model {
                prop_assert_eq!(interner.get(*r), path.as_str());
                prop_assert_eq!(interner.lookup(path), Some(*r));
            }
        }
    }
}
