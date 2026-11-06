//! Generated source text with the value it evaluates to.

use std::collections::BTreeSet;

use crate::value::{Ty, Value};

/// A piece of source text and the value it evaluates to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    pub src: String,
    pub value: Value,
    /// The names the text refers to that it does not bind itself.
    pub free: BTreeSet<String>,
}

impl Fragment {
    pub fn ty(&self) -> Ty {
        self.value.ty()
    }

    pub(crate) fn leaf(src: String, value: Value) -> Self {
        Self {
            src,
            value,
            free: BTreeSet::new(),
        }
    }

    pub(crate) fn compound<'a>(
        src: String,
        value: Value,
        children: impl IntoIterator<Item = &'a Fragment>,
    ) -> Self {
        let free = children
            .into_iter()
            .flat_map(|child| child.free.iter().cloned())
            .collect();
        Self { src, value, free }
    }

    /// The value of a fragment generated for a `Nat` goal.
    pub(crate) fn nat(&self) -> u32 {
        match self.value {
            Value::Nat(n) => n,
            _ => panic!("a Nat goal yields a Nat, got {:?}", self.value),
        }
    }
}
