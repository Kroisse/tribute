use proptest::prelude::*;

use super::fragment;
use crate::env::{Env, Goal};
use crate::fragment::Fragment;
use crate::value::Value;

/// `{ l <> r }`.
pub(super) fn concat(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    let side = fragment(env, Goal::String, depth);
    (side.clone(), side)
        .prop_map(|(lhs, rhs)| {
            let (Value::String(l), Value::String(r)) = (&lhs.value, &rhs.value) else {
                panic!("a String goal yields a String");
            };
            let value = Value::String(format!("{l}{r}"));
            let src = format!("{{ {} <> {} }}", lhs.src, rhs.src);
            Fragment::compound(src, value, [&lhs, &rhs])
        })
        .boxed()
}

/// `Nat::to_string`, called by path or as a method.
pub(super) fn nat_to_string(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    (any::<bool>(), fragment(env, Goal::NAT, depth))
        .prop_map(|(method, operand)| {
            let src = if method {
                format!("{{ {{ {} }}.to_string }}", operand.src)
            } else {
                format!("{{ Nat::to_string({}) }}", operand.src)
            };
            let value = Value::String(operand.nat().to_string());
            Fragment::compound(src, value, [&operand])
        })
        .boxed()
}
