use std::fmt::Write;

use proptest::prelude::*;

use super::fragment;
use crate::env::{Env, Goal};
use crate::fragment::Fragment;
use crate::value::ty;

/// `{ let v = rhs` and `body }` on separate lines, with `body` in the env
/// `v` extends. The grammar separates statements by newlines.
pub(super) fn block(env: Env, goal: Goal, depth: u32) -> BoxedStrategy<Fragment> {
    let outer = env.clone();
    ty().prop_flat_map(move |rhs_ty| fragment(outer.clone(), Goal::of(rhs_ty), depth))
        .prop_flat_map(move |rhs| {
            let (name, inner) = env.bind(rhs.value.clone());
            fragment(inner, goal, depth).prop_map(move |body| {
                let src = format!("{{\nlet {name} = {}\n{}\n}}", rhs.src, body.src);
                let mut free = body.free.clone();
                free.remove(&name);
                free.extend(rhs.free.iter().cloned());
                Fragment {
                    src,
                    value: body.value,
                    free,
                    yields_case: body.yields_case,
                }
            })
        })
        .boxed()
}

/// `{ case s { 3 -> a, 7 -> b, _ -> c } }` on distinct `Nat` literal arms
/// and a final wildcard. The arms sometimes include the scrutinee's value,
/// so both a literal arm and the wildcard are taken.
pub(super) fn case(env: Env, goal: Goal, depth: u32) -> BoxedStrategy<Fragment> {
    let arms = (prop::collection::btree_set(0u32..=10, 1..=3), any::<bool>());
    (fragment(env.clone(), Goal::NAT, depth), arms)
        .prop_flat_map(move |(scrutinee, (mut literals, matched))| {
            // TODO(#1797): match any `Nat` once lowering accepts literal
            // patterns of 2^30 and above; it panics on them today.
            if matched && scrutinee.nat() < 1 << 30 {
                literals.insert(scrutinee.nat());
            }
            let literals: Vec<u32> = literals.into_iter().collect();
            let bodies =
                prop::collection::vec(fragment(env.clone(), goal, depth), literals.len() + 1);
            bodies.prop_map(move |bodies| {
                let taken = literals
                    .iter()
                    .position(|&literal| literal == scrutinee.nat())
                    .unwrap_or(literals.len());
                let mut src = format!("{{ case {} {{ ", scrutinee.src);
                for (literal, body) in literals.iter().zip(&bodies) {
                    write!(src, "{literal} -> {}, ", body.src).expect("writing to a String");
                }
                let wildcard = bodies.last().expect("a case has a wildcard arm");
                write!(src, "_ -> {} }} }}", wildcard.src).expect("writing to a String");
                let value = bodies[taken].value.clone();
                Fragment {
                    yields_case: true,
                    ..Fragment::compound(src, value, std::iter::once(&scrutinee).chain(&bodies))
                }
            })
        })
        .boxed()
}
