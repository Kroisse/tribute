//! Type interning and path interning for arena-based IR.

use std::collections::BTreeMap;

use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;
use std::collections::HashMap;

use super::refs::{PathRef, TypeRef};
use crate::location::Span;
use crate::symbol::Symbol;

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
    /// Integer constant (signed).
    Int(i128),
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

impl Attribute {
    /// Extract the inner `Symbol` if this is `Attribute::Symbol`.
    pub fn as_symbol(&self) -> Option<Symbol> {
        match self {
            Attribute::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    /// Extract the inner `TypeRef` if this is `Attribute::Type`.
    pub fn as_type(&self) -> Option<TypeRef> {
        match self {
            Attribute::Type(t) => Some(*t),
            _ => None,
        }
    }

    /// Extract the inner integer if this is `Attribute::Int`.
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Attribute::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract the inner bool if this is `Attribute::Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Attribute::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Extract the inner string slice if this is `Attribute::String`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attribute::String(s) => Some(s),
            _ => None,
        }
    }

    /// Estimate the complexity of this attribute for alias generation heuristics.
    pub fn complexity(&self) -> usize {
        match self {
            Attribute::Unit => 4,
            Attribute::Bool(_) => 5,
            Attribute::Int(v) => {
                // Approximate digit count without allocating
                if *v == 0 {
                    1
                } else {
                    ((*v as f64).abs().log10() as usize) + 1 + usize::from(*v < 0)
                }
            }
            Attribute::FloatBits(_) => 8,
            Attribute::String(s) => s.len() + 2,
            Attribute::Bytes(b) => b.len() * 4 + 7,
            Attribute::Symbol(sym) => sym.with_str(|s| s.len()) + 1,
            Attribute::Type(_) => 10, // rough estimate; actual depends on type
            Attribute::List(list) => {
                list.iter().map(Attribute::complexity).sum::<usize>() + list.len() * 2
            }
            Attribute::Location(_) => 20,
        }
    }
}

impl From<i32> for Attribute {
    fn from(value: i32) -> Self {
        Attribute::Int(value as i128)
    }
}

impl From<u32> for Attribute {
    fn from(value: u32) -> Self {
        Attribute::Int(value as i128)
    }
}

impl From<i64> for Attribute {
    fn from(value: i64) -> Self {
        Attribute::Int(value as i128)
    }
}

impl From<u64> for Attribute {
    fn from(value: u64) -> Self {
        Attribute::Int(value as i128)
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

    /// Iterate over all interned types, yielding `(TypeRef, &TypeData)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (TypeRef, &TypeData)> {
        self.types.iter()
    }

    /// Find a TypeRef by looking up through the dedup map.
    /// Returns `None` if no type with the given data exists.
    pub fn lookup(&self, data: &TypeData) -> Option<TypeRef> {
        self.dedup.get(data).copied()
    }

    /// Estimate the complexity of a type for alias generation heuristics.
    pub fn complexity(&self, ty: TypeRef) -> usize {
        let data = &self.types[ty];
        let mut size = data.dialect.with_str(|s| s.len()) + 1 + data.name.with_str(|s| s.len());
        for &param in &data.params {
            size += self.complexity(param) + 2; // ", " separator
        }
        for (key, val) in &data.attrs {
            size += key.with_str(|s| s.len()) + 3; // "key = "
            size += val.complexity();
        }
        size
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
    use crate::IrContext;
    use crate::Symbol;

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
        let mut ctx = IrContext::new();
        let i32_ref = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let tup = crate::dialect::core::tuple(&mut ctx, [i32_ref, i32_ref]);
        let r1 = tup.as_type_ref();
        // Interning the same tuple again should return the same ref
        let r2 = crate::dialect::core::tuple(&mut ctx, [i32_ref, i32_ref]).as_type_ref();
        assert_eq!(r1, r2);

        let data = ctx.types.get(r1);
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
