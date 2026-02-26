//! Type interning and path interning for arena-based IR.

use std::collections::BTreeMap;

use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;
use std::collections::HashMap;

use super::refs::{PathRef, TypeRef};
use crate::ir::Symbol;
use crate::location::Span;

// ============================================================================
// Location
// ============================================================================

/// Source location in arena IR. Copy-able, no lifetime parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Location {
    pub path: PathRef,
    pub span: Span,
}

impl Location {
    pub const fn new(path: PathRef, span: Span) -> Self {
        Self { path, span }
    }
}

// ============================================================================
// Attribute
// ============================================================================

/// IR attribute values (arena version, no lifetime parameter).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Attribute {
    /// Unit/nil value.
    Unit,
    Bool(bool),
    /// Integer constant stored as raw bits (signless).
    IntBits(u64),
    /// Float constant stored as raw bits.
    FloatBits(u64),
    String(String),
    Bytes(SmallVec<[u8; 16]>),
    Type(TypeRef),
    /// Single interned symbol.
    Symbol(Symbol),
    /// List of attributes.
    List(Vec<Attribute>),
    /// Full source location.
    Location(Location),
}

impl From<i64> for Attribute {
    fn from(value: i64) -> Self {
        Attribute::IntBits(u64::from_ne_bytes(value.to_ne_bytes()))
    }
}

impl From<u64> for Attribute {
    fn from(value: u64) -> Self {
        Attribute::IntBits(value)
    }
}

impl From<bool> for Attribute {
    fn from(value: bool) -> Self {
        Attribute::Bool(value)
    }
}

impl From<Vec<Attribute>> for Attribute {
    fn from(value: Vec<Attribute>) -> Self {
        Attribute::List(value)
    }
}

impl From<Symbol> for Attribute {
    fn from(value: Symbol) -> Self {
        Attribute::Symbol(value)
    }
}

impl From<String> for Attribute {
    fn from(value: String) -> Self {
        Attribute::String(value)
    }
}

impl From<&str> for Attribute {
    fn from(value: &str) -> Self {
        Attribute::String(value.to_string())
    }
}

impl From<Location> for Attribute {
    fn from(value: Location) -> Self {
        Attribute::Location(value)
    }
}

// ============================================================================
// TypeData
// ============================================================================

/// Data for a single interned type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeData {
    pub dialect: Symbol,
    pub name: Symbol,
    pub params: SmallVec<[TypeRef; 4]>,
    pub attrs: BTreeMap<Symbol, Attribute>,
}

/// Builder for constructing `TypeData` with a fluent API.
///
/// Defaults to empty params and empty attrs, matching the most common usage.
pub struct TypeDataBuilder {
    dialect: Symbol,
    name: Symbol,
    params: SmallVec<[TypeRef; 4]>,
    attrs: BTreeMap<Symbol, Attribute>,
}

impl TypeDataBuilder {
    pub fn new(dialect: Symbol, name: Symbol) -> Self {
        Self {
            dialect,
            name,
            params: SmallVec::new(),
            attrs: BTreeMap::new(),
        }
    }

    pub fn param(mut self, ty: TypeRef) -> Self {
        self.params.push(ty);
        self
    }

    pub fn params(mut self, tys: impl IntoIterator<Item = TypeRef>) -> Self {
        self.params.extend(tys);
        self
    }

    pub fn attr(mut self, key: impl Into<Symbol>, val: Attribute) -> Self {
        self.attrs.insert(key.into(), val);
        self
    }

    pub fn build(self) -> TypeData {
        TypeData {
            dialect: self.dialect,
            name: self.name,
            params: self.params,
            attrs: self.attrs,
        }
    }
}

// ============================================================================
// TypeInterner
// ============================================================================

/// Deduplicating type interner. Same `TypeData` always yields the same `TypeRef`.
pub struct TypeInterner {
    types: PrimaryMap<TypeRef, TypeData>,
    dedup: HashMap<TypeData, TypeRef>,
}

impl TypeInterner {
    pub fn new() -> Self {
        Self {
            types: PrimaryMap::new(),
            dedup: HashMap::default(),
        }
    }

    /// Intern a type, returning an existing ref if the data matches.
    pub fn intern(&mut self, data: TypeData) -> TypeRef {
        if let Some(&existing) = self.dedup.get(&data) {
            return existing;
        }
        let r = self.types.push(data.clone());
        self.dedup.insert(data, r);
        r
    }

    /// Look up type data by reference.
    pub fn get(&self, r: TypeRef) -> &TypeData {
        &self.types[r]
    }

    /// Check if this type matches the given dialect and name.
    pub fn is_dialect(&self, r: TypeRef, dialect: Symbol, name: Symbol) -> bool {
        let data = &self.types[r];
        data.dialect == dialect && data.name == name
    }
}

impl Default for TypeInterner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PathInterner
// ============================================================================

/// Deduplicating path (URI string) interner.
pub struct PathInterner {
    paths: PrimaryMap<PathRef, String>,
    dedup: HashMap<String, PathRef>,
}

impl PathInterner {
    pub fn new() -> Self {
        Self {
            paths: PrimaryMap::new(),
            dedup: HashMap::default(),
        }
    }

    /// Intern a path string, returning an existing ref if the string matches.
    pub fn intern(&mut self, path: String) -> PathRef {
        if let Some(&existing) = self.dedup.get(&path) {
            return existing;
        }
        let r = self.paths.push(path.clone());
        self.dedup.insert(path, r);
        r
    }

    /// Look up path string by reference.
    pub fn get(&self, r: PathRef) -> &str {
        &self.paths[r]
    }
}

impl Default for PathInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Symbol;

    #[test]
    fn type_interner_dedup() {
        let mut interner = TypeInterner::new();
        let data = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build();
        let r1 = interner.intern(data.clone());
        let r2 = interner.intern(data);
        assert_eq!(r1, r2, "same TypeData must yield same TypeRef");
    }

    #[test]
    fn type_interner_distinct() {
        let mut interner = TypeInterner::new();
        let i32_data = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build();
        let i64_data = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build();
        let r1 = interner.intern(i32_data);
        let r2 = interner.intern(i64_data);
        assert_ne!(r1, r2, "different TypeData must yield different TypeRef");
    }

    #[test]
    fn type_interner_with_params() {
        let mut interner = TypeInterner::new();
        let i32_ref =
            interner.intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let tuple_data = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("tuple"))
            .param(i32_ref)
            .param(i32_ref)
            .build();
        let r1 = interner.intern(tuple_data.clone());
        let r2 = interner.intern(tuple_data);
        assert_eq!(r1, r2);

        let data = interner.get(r1);
        assert_eq!(data.params.len(), 2);
        assert_eq!(data.params[0], i32_ref);
    }

    #[test]
    fn path_interner_dedup() {
        let mut interner = PathInterner::new();
        let r1 = interner.intern("file:///test.trb".to_owned());
        let r2 = interner.intern("file:///test.trb".to_owned());
        assert_eq!(r1, r2, "same path must yield same PathRef");
    }

    #[test]
    fn path_interner_distinct() {
        let mut interner = PathInterner::new();
        let r1 = interner.intern("file:///a.trb".to_owned());
        let r2 = interner.intern("file:///b.trb".to_owned());
        assert_ne!(r1, r2);
        assert_eq!(interner.get(r1), "file:///a.trb");
        assert_eq!(interner.get(r2), "file:///b.trb");
    }
}
