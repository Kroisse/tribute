//! Expression strategies, one module per kind of expression.

mod arith;
mod compare;
mod control;
mod string;

use std::collections::BTreeSet;

use proptest::prelude::*;

use crate::env::{Env, Goal};
use crate::fragment::Fragment;
use crate::value::Value;

use self::arith::{add, mul};
use self::compare::{compare_nat, compare_string};
use self::control::{block, case};
use self::string::{concat, nat_to_string};

/// Strategy for a fragment that meets `goal` in `env`, nested at most
/// `depth` levels.
pub fn fragment(env: Env, goal: Goal, depth: u32) -> BoxedStrategy<Fragment> {
    let leaf = leaf(&env, goal);
    if depth == 0 {
        return leaf;
    }
    let depth = depth - 1;
    let compound = match goal {
        Goal::Nat { max } => prop_oneof![
            add(env.clone(), max, depth),
            mul(env.clone(), max, depth),
            block(env.clone(), goal, depth),
            case(env, goal, depth),
        ]
        .boxed(),
        Goal::Bool => prop_oneof![
            compare_nat(env.clone(), depth),
            compare_string(env.clone(), depth),
            block(env.clone(), goal, depth),
            case(env, goal, depth),
        ]
        .boxed(),
        Goal::String => prop_oneof![
            concat(env.clone(), depth),
            nat_to_string(env.clone(), depth),
            block(env.clone(), goal, depth),
            case(env, goal, depth),
        ]
        .boxed(),
    };
    prop_oneof![1 => leaf, 2 => compound].boxed()
}
fn leaf(env: &Env, goal: Goal) -> BoxedStrategy<Fragment> {
    let literal = literal(goal);
    let candidates = env.candidates(goal);
    if candidates.is_empty() {
        return literal;
    }
    let var = prop::sample::select(candidates)
        .prop_map(|binding| Fragment {
            free: BTreeSet::from([binding.name.clone()]),
            src: binding.name,
            value: binding.value,
            yields_case: false,
        })
        .boxed();
    prop_oneof![literal, var].boxed()
}

fn literal(goal: Goal) -> BoxedStrategy<Fragment> {
    match goal {
        Goal::Nat { max } => prop_oneof![0..=max.min(100), 0..=max]
            .prop_map(|n| Fragment::leaf(n.to_string(), Value::Nat(n)))
            .boxed(),
        Goal::Bool => any::<bool>()
            .prop_map(|b| {
                let value = Value::Bool(b);
                Fragment::leaf(value.to_string(), value)
            })
            .boxed(),
        // Nonempty and free of whitespace and escapes, so printed lines
        // compare exactly.
        Goal::String => "[A-Za-z0-9_]{1,6}"
            .prop_map(|s| Fragment::leaf(format!("\"{s}\""), Value::String(s)))
            .boxed(),
    }
}
