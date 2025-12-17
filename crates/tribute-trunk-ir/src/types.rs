//! IR type definitions.

use crate::{Symbol, TrackedVec};

/// IR type representation.
#[salsa::interned(debug)]
pub struct Type<'db> {
    #[returns(ref)]
    pub kind: TypeKind<'db>,
}

impl<'db> Type<'db> {
    pub fn i(db: &'db dyn salsa::Database, bits: u16) -> Self {
        Type::new(db, TypeKind::I { bits })
    }
    pub fn f(db: &'db dyn salsa::Database, bits: u16) -> Self {
        Type::new(db, TypeKind::F { bits })
    }
    pub fn string(db: &'db dyn salsa::Database) -> Self {
        Type::new(db, TypeKind::String)
    }
    pub fn bytes(db: &'db dyn salsa::Database) -> Self {
        Type::new(db, TypeKind::Bytes)
    }
    pub fn ptr(db: &'db dyn salsa::Database) -> Self {
        Type::new(db, TypeKind::Ptr)
    }
    pub fn never(db: &'db dyn salsa::Database) -> Self {
        Type::new(db, TypeKind::Never)
    }
    pub fn unit(db: &'db dyn salsa::Database) -> Self {
        Type::new(db, TypeKind::Unit)
    }
    pub fn array(db: &'db dyn salsa::Database, ty: Type<'db>) -> Self {
        Type::new(db, TypeKind::Array(ty))
    }
    pub fn ref_(db: &'db dyn salsa::Database, ty: Type<'db>, nullable: bool) -> Self {
        Type::new(db, TypeKind::Ref { ty, nullable })
    }
    pub fn tuple(db: &'db dyn salsa::Database, tys: TrackedVec<Type<'db>>) -> Self {
        Type::new(db, TypeKind::Tuple(tys))
    }
    pub fn function(
        db: &'db dyn salsa::Database,
        params: TrackedVec<Type<'db>>,
        results: TrackedVec<Type<'db>>,
    ) -> Self {
        Type::new(db, TypeKind::Function { params, results })
    }
    pub fn dialect(
        db: &'db dyn salsa::Database,
        dialect: String,
        name: String,
        params: TrackedVec<Type<'db>>,
    ) -> Self {
        Type::new(
            db,
            TypeKind::Dialect {
                dialect,
                name,
                params,
            },
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum TypeKind<'db> {
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
    Array(Type<'db>),
    Ref {
        ty: Type<'db>,
        nullable: bool,
    },
    Tuple(TrackedVec<Type<'db>>),
    Function {
        params: TrackedVec<Type<'db>>,
        results: TrackedVec<Type<'db>>,
    },
    Dialect {
        dialect: String,
        name: String,
        params: TrackedVec<Type<'db>>,
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
    Type(Type<'db>),
    /// Symbol reference path (e.g., ["module", "func_name"])
    SymbolRef(TrackedVec<Symbol<'db>>),
    /// List of attributes (for arrays of values like switch cases).
    List(Vec<Attribute<'db>>),
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

impl<'db> From<Vec<Attribute<'db>>> for Attribute<'db> {
    fn from(value: Vec<Attribute<'db>>) -> Self {
        Attribute::List(value)
    }
}
