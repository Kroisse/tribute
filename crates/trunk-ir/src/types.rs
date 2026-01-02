//! IR type definitions.

use std::collections::BTreeMap;

use crate::{IdVec, Location, Symbol, dialect::core};

/// Trait for dialect-specific type wrappers.
///
/// Similar to `DialectOp` for operations, this trait provides a common interface
/// for type wrappers that wrap the generic `Type` with dialect-specific semantics.
pub trait DialectType<'db>: Sized {
    /// Get the underlying `Type`.
    fn as_type(&self) -> Type<'db>;

    /// Try to convert a `Type` to this dialect type wrapper.
    /// Returns `None` if the type doesn't match this dialect type.
    fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self>;
}

/// Attribute map type alias.
pub type Attrs<'db> = BTreeMap<Symbol, Attribute<'db>>;

/// IR type representation.
///
/// All types are dialect-defined with a `dialect.name` naming convention.
#[salsa::interned(debug)]
pub struct Type<'db> {
    pub dialect: Symbol,
    pub name: Symbol,
    #[returns(deref)]
    pub params: IdVec<Type<'db>>,
    #[returns(ref)]
    pub attrs: Attrs<'db>,
}

impl<'db> Type<'db> {
    /// Check if this type matches the given dialect and name.
    pub fn is_dialect(&self, db: &'db dyn salsa::Database, dialect: Symbol, name: Symbol) -> bool {
        self.dialect(db) == dialect && self.name(db) == name
    }

    /// Check if this is a function type (`core.func`).
    pub fn is_function(&self, db: &'db dyn salsa::Database) -> bool {
        self.is_dialect(db, core::DIALECT_NAME(), core::FUNC())
    }

    /// Get function parameter types if this is a function type.
    /// Returns `params[1..]` (skipping the return type at index 0).
    pub fn function_params(&self, db: &'db dyn salsa::Database) -> Option<IdVec<Type<'db>>> {
        if !self.is_function(db) {
            return None;
        }
        let all = self.params(db);
        if all.is_empty() {
            return None;
        }
        Some(all.iter().skip(1).copied().collect())
    }

    /// Get function return type if this is a function type.
    /// Returns `params[0]`.
    pub fn function_result(&self, db: &'db dyn salsa::Database) -> Option<Type<'db>> {
        if !self.is_function(db) {
            return None;
        }
        self.params(db).first().copied()
    }

    /// Get function effect type if this is a function type.
    pub fn function_effect(&self, db: &'db dyn salsa::Database) -> Option<Type<'db>> {
        if !self.is_function(db) {
            return None;
        }
        match self.get_attr(db, core::Func::effect_sym()) {
            Some(Attribute::Type(ty)) => Some(*ty),
            _ => None,
        }
    }

    /// Get an attribute by key.
    pub fn get_attr(&self, db: &'db dyn salsa::Database, key: Symbol) -> Option<&Attribute<'db>> {
        self.attrs(db).get(&key)
    }
}

// Implement Ord for Type using salsa's interned ID.
// This provides a stable ordering for types in collections like BTreeSet.
impl<'db> PartialOrd for Type<'db> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'db> Ord for Type<'db> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use salsa::plumbing::AsId;
        self.as_id().cmp(&other.as_id())
    }
}

/// IR attribute values.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum Attribute<'db> {
    /// Unit/nil value (placeholder for absent or void attributes).
    Unit,
    Bool(bool),
    /// Integer constant stored as raw bits (signless).
    IntBits(u64),
    /// Float constant stored as raw bits.
    FloatBits(u64),
    String(String),
    Bytes(Vec<u8>),
    Type(Type<'db>),
    /// Single interned symbol (e.g., "foo").
    Symbol(Symbol),
    /// List of attributes (for arrays of values like switch cases).
    List(Vec<Attribute<'db>>),
    /// Full source location (file path + span).
    Location(Location<'db>),
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

impl From<Symbol> for Attribute<'_> {
    fn from(value: Symbol) -> Self {
        Attribute::Symbol(value)
    }
}

impl From<String> for Attribute<'_> {
    fn from(value: String) -> Self {
        Attribute::String(value)
    }
}

impl From<&str> for Attribute<'_> {
    fn from(value: &str) -> Self {
        Attribute::String(value.to_string())
    }
}

impl<'db> From<Location<'db>> for Attribute<'db> {
    fn from(value: Location<'db>) -> Self {
        Attribute::Location(value)
    }
}
