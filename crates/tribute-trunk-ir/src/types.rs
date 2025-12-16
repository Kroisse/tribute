//! IR type definitions.

use crate::Symbol;
use serde::{Deserialize, Serialize};

/// IR type representation.
///
/// Note: Uses `Vec` instead of `SmallVec` because the recursive nature
/// (Type containing Vec<Type>) would cause infinite size with inline storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    I {
        bits: u16,
    },
    F {
        bits: u16,
    },
    String,
    Bytes,
    Ptr,
    Never,
    Unit,
    Array(Box<Type>),
    Ref {
        ty: Box<Type>,
        nullable: bool,
    },
    Tuple(Vec<Type>),
    Function {
        params: Vec<Type>,
        results: Vec<Type>,
    },
    Dialect {
        dialect: String,
        name: String,
        params: Vec<Type>,
    },
}

/// IR attribute values.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum Attribute<'db> {
    Unit,
    Bool(bool),
    /// Integer constant stored as raw bits (signless).
    IntBits(u64),
    /// Float constant stored as raw bits.
    FloatBits(u64),
    String(String),
    Bytes(Vec<u8>),
    Type(Type),
    /// Symbol reference path (e.g., ["module", "func_name"])
    SymbolRef(Vec<Symbol<'db>>),
}

impl From<i64> for Attribute<'_> {
    fn from(value: i64) -> Self {
        Attribute::IntBits(u64::from_ne_bytes(value.to_ne_bytes()))
    }
}

impl From<u64> for Attribute<'_> {
    fn from(value: u64) -> Self {
        Attribute::IntBits(value)
    }
}

impl From<bool> for Attribute<'_> {
    fn from(value: bool) -> Self {
        Attribute::IntBits(if value { 1 } else { 0 })
    }
}
