//! Properties of the generator itself.

use proptest::prelude::*;
use tribute_prop::env::{Env, Goal};
use tribute_prop::expr::fragment;
use tribute_prop::fragment::Fragment;
use tribute_prop::program::program;
use tribute_prop::value::Value;

/// Strategy for an env of up to four bindings, as enclosing `let`s leave
/// it.
fn env() -> impl Strategy<Value = Env> {
    let value = prop_oneof![
        any::<u32>().prop_map(Value::Nat),
        any::<bool>().prop_map(Value::Bool),
        "[a-z]{1,3}".prop_map(Value::String),
    ];
    prop::collection::vec(value, 0..=4).prop_map(|values| {
        values
            .into_iter()
            .fold(Env::default(), |env, value| env.bind(value).1)
    })
}

/// Strategy for a goal, with a `Nat` bound anywhere in the current range.
fn goal() -> impl Strategy<Value = Goal> {
    prop_oneof![
        prop_oneof![0u32..=1000, any::<u32>()].prop_map(|max| Goal::Nat { max }),
        Just(Goal::Bool),
        Just(Goal::String),
    ]
}

/// An env, a goal, and a fragment generated for them.
fn generated() -> impl Strategy<Value = (Env, Goal, Fragment)> {
    (env(), goal()).prop_flat_map(|(env, goal)| {
        fragment(env.clone(), goal, 3).prop_map(move |fragment| (env.clone(), goal, fragment))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// A fragment has its goal's type, and a `Nat` stays within the goal's
    /// bound. `+` and `*` compute every intermediate value with checked
    /// arithmetic, so a combinator that leaves the range panics here.
    #[test]
    fn values_stay_in_range((_env, goal, fragment) in generated()) {
        prop_assert_eq!(fragment.ty(), goal.ty());
        prop_assert!(goal.accepts(&fragment.value), "{:?} exceeds {:?}", fragment.value, goal);
    }

    /// Every name a fragment refers to is bound in its env, with the value
    /// the env holds.
    #[test]
    fn every_name_is_bound((env, _goal, fragment) in generated()) {
        for name in &fragment.free {
            prop_assert!(env.lookup(name).is_some(), "{} is unbound in {}", name, fragment.src);
        }
    }

    /// A program's checks bind every name they use, so no check refers to
    /// another's bindings, and `main` prints one line per check.
    #[test]
    fn programs_are_closed(program in program()) {
        for check in &program.checks {
            prop_assert!(check.body.free.is_empty(), "{} uses {:?}", check.name, check.body.free);
        }
        prop_assert_eq!(program.expected_stdout().lines().count(), program.checks.len());
    }

    /// A bare reference reads the binding's value.
    #[test]
    fn references_read_their_binding((env, _goal, fragment) in generated()) {
        if let Some(value) = env.lookup(&fragment.src) {
            prop_assert_eq!(value, &fragment.value);
        }
    }
}

#[test]
fn nat_bound_zero_yields_zero() {
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let strategy = fragment(Env::default(), Goal::Nat { max: 0 }, 3);
    for _ in 0..64 {
        let fragment = strategy.new_tree(&mut runner).expect("generates").current();
        assert_eq!(fragment.value, Value::Nat(0), "{}", fragment.src);
    }
}
