use std::cmp::Ordering;

use proptest::prelude::*;

use super::fragment;
use crate::env::{Env, Goal};
use crate::fragment::Fragment;
use crate::value::Value;

#[derive(Clone, Copy, Debug)]
enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Comparison {
    fn symbol(self) -> &'static str {
        match self {
            Comparison::Eq => "==",
            Comparison::Ne => "!=",
            Comparison::Lt => "<",
            Comparison::Le => "<=",
            Comparison::Gt => ">",
            Comparison::Ge => ">=",
        }
    }

    /// Whether the comparison holds of two values of one type.
    fn holds(self, lhs: &Value, rhs: &Value) -> bool {
        let ordering = match (lhs, rhs) {
            (Value::Nat(a), Value::Nat(b)) => a.cmp(b),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => panic!("compared operands share a type: {lhs:?}, {rhs:?}"),
        };
        match self {
            Comparison::Eq => ordering == Ordering::Equal,
            Comparison::Ne => ordering != Ordering::Equal,
            Comparison::Lt => ordering == Ordering::Less,
            Comparison::Le => ordering != Ordering::Greater,
            Comparison::Gt => ordering == Ordering::Greater,
            Comparison::Ge => ordering != Ordering::Less,
        }
    }
}

fn compare(
    env: Env,
    operand: Goal,
    comparisons: &'static [Comparison],
    depth: u32,
) -> BoxedStrategy<Fragment> {
    let side = fragment(env, operand, depth);
    (prop::sample::select(comparisons), side.clone(), side)
        .prop_map(|(comparison, lhs, rhs)| {
            let holds = comparison.holds(&lhs.value, &rhs.value);
            let src = format!("{{ {} {} {} }}", lhs.src, comparison.symbol(), rhs.src);
            Fragment::compound(src, Value::Bool(holds), [&lhs, &rhs])
        })
        .boxed()
}

/// A `Nat` comparison with any of the six operators.
pub(super) fn compare_nat(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    use Comparison::*;
    compare(env, Goal::NAT, &[Eq, Ne, Lt, Le, Gt, Ge], depth)
}

/// A `String` equality test; `String` declares only `==` and `!=`.
pub(super) fn compare_string(env: Env, depth: u32) -> BoxedStrategy<Fragment> {
    compare(env, Goal::String, &[Comparison::Eq, Comparison::Ne], depth)
}
