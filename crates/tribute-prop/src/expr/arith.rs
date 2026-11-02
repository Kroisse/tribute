use proptest::prelude::*;

use super::fragment;
use crate::env::{Env, Goal};
use crate::fragment::Fragment;
use crate::value::Value;

/// `{ l + r }` with `r` at most what `max` leaves after `l`.
pub(super) fn add(env: Env, max: u32, depth: u32) -> BoxedStrategy<Fragment> {
    arithmetic(env, max, depth, "+", |max, l| max - l, u32::checked_add)
}

/// `{ l * r }` with `r` at most what `max` allows after `l`.
pub(super) fn mul(env: Env, max: u32, depth: u32) -> BoxedStrategy<Fragment> {
    arithmetic(
        env,
        max,
        depth,
        "*",
        |max, l| max.checked_div(l).unwrap_or(max),
        u32::checked_mul,
    )
}

/// `{ l op r }`, with `r` at most `rest(max, l)` so that `apply` stays
/// within `max`.
fn arithmetic(
    env: Env,
    max: u32,
    depth: u32,
    op: &'static str,
    rest: fn(u32, u32) -> u32,
    apply: fn(u32, u32) -> Option<u32>,
) -> BoxedStrategy<Fragment> {
    fragment(env.clone(), Goal::Nat { max }, depth)
        .prop_flat_map(move |lhs| {
            let rhs = fragment(
                env.clone(),
                Goal::Nat {
                    max: rest(max, lhs.nat()),
                },
                depth,
            );
            // TODO(#1783): drop this filter once type checking selects `+`
            // and `*` in a `case` arm whose operands are both `case`
            // results; it reports them as ambiguous today.
            let lhs_case = lhs.yields_case;
            rhs.prop_filter("both operands are case results", move |rhs| {
                !(lhs_case && rhs.yields_case)
            })
            .prop_map(move |rhs| {
                let value =
                    apply(lhs.nat(), rhs.nat()).expect("the right operand stays within the bound");
                let src = format!("{{ {} {op} {} }}", lhs.src, rhs.src);
                Fragment::compound(src, Value::Nat(value), [&lhs, &rhs])
            })
        })
        .boxed()
}
