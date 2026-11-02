use std::fmt;

use proptest::prelude::*;

/// A type the generator produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Nat,
    Bool,
    String,
}

impl Ty {
    /// The type's name in Tribute source.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Nat => "Nat",
            Ty::Bool => "Bool",
            Ty::String => "String",
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The value a fragment evaluates to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Nat(u32),
    Bool(bool),
    String(String),
}

impl Value {
    pub fn ty(&self) -> Ty {
        match self {
            Value::Nat(_) => Ty::Nat,
            Value::Bool(_) => Ty::Bool,
            Value::String(_) => Ty::String,
        }
    }
}

/// The text a [`Program`](crate::Program) prints for a value: decimal for `Nat`, the
/// constructor name for `Bool`, and the text itself for `String`.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nat(n) => write!(f, "{n}"),
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
            Value::String(s) => f.write_str(s),
        }
    }
}
/// Strategy for a type the generator produces.
pub fn ty() -> impl Strategy<Value = Ty> {
    prop_oneof![Just(Ty::Nat), Just(Ty::Bool), Just(Ty::String)]
}
