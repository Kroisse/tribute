//! IR type definitions.

use crate::Symbol;
use serde::{Deserialize, Serialize};

/// IR type representation.
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
    Int(i64),
    UInt(u64),
    FloatBits(u64),
    String(String),
    Bytes(Vec<u8>),
    Type(Type),
    /// Symbol reference path (e.g., ["module", "func_name"])
    SymbolRef(Vec<Symbol<'db>>),
}
