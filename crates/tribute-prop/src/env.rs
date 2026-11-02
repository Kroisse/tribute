//! The bindings in scope and the goal a strategy generates for.

use crate::value::{Ty, Value};

/// One `let` binding in scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub value: Value,
}

/// The bindings in scope, innermost last.
///
/// [`Env::bind`] names each binding after its depth, so a name is never
/// shadowed and a reference reads the value its binding holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Env {
    bindings: Vec<Binding>,
}

impl Env {
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    pub fn lookup(&self, name: &str) -> Option<&Value> {
        self.bindings
            .iter()
            .rev()
            .find(|binding| binding.name == name)
            .map(|binding| &binding.value)
    }

    /// This env with a fresh name bound to `value`, and that name.
    pub fn bind(&self, value: Value) -> (String, Env) {
        let name = format!("v{}", self.bindings.len());
        let mut env = self.clone();
        env.bindings.push(Binding {
            name: name.clone(),
            value,
        });
        (name, env)
    }

    /// The bindings a reference may use for `goal`.
    pub(crate) fn candidates(&self, goal: Goal) -> Vec<Binding> {
        self.bindings
            .iter()
            .filter(|binding| goal.accepts(&binding.value))
            .cloned()
            .collect()
    }
}

/// What a strategy produces: a type and, for `Nat`, the largest value
/// allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Goal {
    Nat { max: u32 },
    Bool,
    String,
}

impl Goal {
    /// A `Nat` goal over the whole current range.
    pub const NAT: Goal = Goal::Nat { max: u32::MAX };

    pub fn of(ty: Ty) -> Self {
        match ty {
            Ty::Nat => Self::NAT,
            Ty::Bool => Self::Bool,
            Ty::String => Self::String,
        }
    }

    pub fn ty(self) -> Ty {
        match self {
            Goal::Nat { .. } => Ty::Nat,
            Goal::Bool => Ty::Bool,
            Goal::String => Ty::String,
        }
    }

    pub fn accepts(self, value: &Value) -> bool {
        match (self, value) {
            (Goal::Nat { max }, Value::Nat(n)) => *n <= max,
            (Goal::Bool, Value::Bool(_)) | (Goal::String, Value::String(_)) => true,
            _ => false,
        }
    }
}
