//! Type interning and path interning for arena-based IR.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;

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

/// An integer attribute that cannot be represented by the requested type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntegerOutOfRange {
    pub value: i128,
    pub target: &'static str,
}

impl fmt::Display for IntegerOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "integer attribute {} is out of range for {}",
            self.value, self.target
        )
    }
}

impl std::error::Error for IntegerOutOfRange {}

/// Text stored either directly or as an interned symbol attribute.
#[derive(Clone, Copy, Debug)]
pub enum AttributeText<'a> {
    String(&'a str),
    Symbol(Symbol),
}

impl AttributeText<'_> {
    pub fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        match self {
            AttributeText::String(text) => f(text),
            AttributeText::Symbol(symbol) => symbol.with_str(f),
        }
    }
}

impl PartialEq for AttributeText<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.with_str(|text| other.with_str(|other| text == other))
    }
}

impl Eq for AttributeText<'_> {}

impl PartialEq<str> for AttributeText<'_> {
    fn eq(&self, other: &str) -> bool {
        self.with_str(|text| text == other)
    }
}

impl PartialEq<&str> for AttributeText<'_> {
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

impl fmt::Display for AttributeText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with_str(|text| f.write_str(text))
    }
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
    pub fn as_i128(&self) -> Option<i128> {
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

/// A deterministic map of IR attributes with ergonomic symbol and string lookup.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttributeMap(BTreeMap<Symbol, Attribute>);

pub type AttributeIter<'a> = std::collections::btree_map::Iter<'a, Symbol, Attribute>;
pub type AttributeIterMut<'a> = std::collections::btree_map::IterMut<'a, Symbol, Attribute>;
pub type AttributeKeys<'a> = std::collections::btree_map::Keys<'a, Symbol, Attribute>;
pub type AttributeValues<'a> = std::collections::btree_map::Values<'a, Symbol, Attribute>;
pub type AttributeValuesMut<'a> = std::collections::btree_map::ValuesMut<'a, Symbol, Attribute>;
pub type AttributeIntoIter = std::collections::btree_map::IntoIter<Symbol, Attribute>;

/// A key accepted by [`AttributeMap::get`].
pub trait AttributeKey {
    fn lookup_symbol(self) -> Option<Symbol>;
}

impl AttributeKey for Symbol {
    fn lookup_symbol(self) -> Option<Symbol> {
        Some(self)
    }
}

impl AttributeKey for &Symbol {
    fn lookup_symbol(self) -> Option<Symbol> {
        Some(*self)
    }
}

impl AttributeKey for &str {
    fn lookup_symbol(self) -> Option<Symbol> {
        Symbol::lookup(self)
    }
}

impl AttributeMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the attribute associated with a symbol or already-interned string.
    ///
    /// A missing string key is not added to the global symbol interner.
    pub fn get(&self, key: impl AttributeKey) -> Option<&Attribute> {
        let symbol = key.lookup_symbol()?;
        self.0.get(&symbol)
    }

    pub fn get_mut(&mut self, key: impl AttributeKey) -> Option<&mut Attribute> {
        let symbol = key.lookup_symbol()?;
        self.0.get_mut(&symbol)
    }

    pub fn get_bool(&self, key: impl AttributeKey) -> Option<bool> {
        self.get(key).and_then(Attribute::as_bool)
    }

    pub fn get_i128(&self, key: impl AttributeKey) -> Option<i128> {
        self.get(key).and_then(Attribute::as_i128)
    }

    pub fn get_i64(&self, key: impl AttributeKey) -> Result<Option<i64>, IntegerOutOfRange> {
        self.get_integer(key, "i64", i64::try_from)
    }

    pub fn get_i32(&self, key: impl AttributeKey) -> Result<Option<i32>, IntegerOutOfRange> {
        self.get_integer(key, "i32", i32::try_from)
    }

    pub fn get_u64(&self, key: impl AttributeKey) -> Result<Option<u64>, IntegerOutOfRange> {
        self.get_integer(key, "u64", u64::try_from)
    }

    pub fn get_u32(&self, key: impl AttributeKey) -> Result<Option<u32>, IntegerOutOfRange> {
        self.get_integer(key, "u32", u32::try_from)
    }

    pub fn get_u8(&self, key: impl AttributeKey) -> Result<Option<u8>, IntegerOutOfRange> {
        self.get_integer(key, "u8", u8::try_from)
    }

    pub fn get_str(&self, key: impl AttributeKey) -> Option<&str> {
        self.get(key).and_then(Attribute::as_str)
    }

    pub fn get_symbol(&self, key: impl AttributeKey) -> Option<Symbol> {
        self.get(key).and_then(Attribute::as_symbol)
    }

    pub fn get_type(&self, key: impl AttributeKey) -> Option<TypeRef> {
        self.get(key).and_then(Attribute::as_type)
    }

    pub fn get_text(&self, key: impl AttributeKey) -> Option<AttributeText<'_>> {
        match self.get(key)? {
            Attribute::String(text) => Some(AttributeText::String(text)),
            Attribute::Symbol(symbol) => Some(AttributeText::Symbol(*symbol)),
            _ => None,
        }
    }

    fn get_integer<T>(
        &self,
        key: impl AttributeKey,
        target: &'static str,
        convert: impl FnOnce(i128) -> Result<T, std::num::TryFromIntError>,
    ) -> Result<Option<T>, IntegerOutOfRange> {
        let Some(value) = self.get_i128(key) else {
            return Ok(None);
        };
        convert(value)
            .map(Some)
            .map_err(|_| IntegerOutOfRange { value, target })
    }

    pub fn contains_key(&self, key: impl AttributeKey) -> bool {
        let Some(symbol) = key.lookup_symbol() else {
            return false;
        };
        self.0.contains_key(&symbol)
    }

    pub fn insert(&mut self, key: Symbol, value: Attribute) -> Option<Attribute> {
        self.0.insert(key, value)
    }

    pub fn remove(&mut self, key: impl AttributeKey) -> Option<Attribute> {
        let symbol = key.lookup_symbol()?;
        self.0.remove(&symbol)
    }

    pub fn iter(&self) -> AttributeIter<'_> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> AttributeIterMut<'_> {
        self.0.iter_mut()
    }

    pub fn keys(&self) -> AttributeKeys<'_> {
        self.0.keys()
    }

    pub fn values(&self) -> AttributeValues<'_> {
        self.0.values()
    }

    pub fn values_mut(&mut self) -> AttributeValuesMut<'_> {
        self.0.values_mut()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl FromIterator<(Symbol, Attribute)> for AttributeMap {
    fn from_iter<T: IntoIterator<Item = (Symbol, Attribute)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Extend<(Symbol, Attribute)> for AttributeMap {
    fn extend<T: IntoIterator<Item = (Symbol, Attribute)>>(&mut self, iter: T) {
        self.0.extend(iter);
    }
}

impl IntoIterator for AttributeMap {
    type Item = (Symbol, Attribute);
    type IntoIter = AttributeIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a AttributeMap {
    type Item = (&'a Symbol, &'a Attribute);
    type IntoIter = AttributeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut AttributeMap {
    type Item = (&'a Symbol, &'a mut Attribute);
    type IntoIter = AttributeIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
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
    pub attrs: AttributeMap,
}

/// Builder for constructing `TypeData` with a fluent API.
///
/// Defaults to empty params and empty attrs, matching the most common usage.
pub struct TypeDataBuilder {
    dialect: Symbol,
    name: Symbol,
    params: SmallVec<[TypeRef; 4]>,
    attrs: AttributeMap,
}

impl TypeDataBuilder {
    pub fn new(dialect: Symbol, name: Symbol) -> Self {
        Self {
            dialect,
            name,
            params: SmallVec::new(),
            attrs: AttributeMap::new(),
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
    fn attribute_map_accepts_string_and_symbol_keys_without_interning_misses() {
        fn get_by_symbol<'a>(attrs: &'a AttributeMap, key: &Symbol) -> Option<&'a Attribute> {
            attrs.get(key)
        }

        let mut attrs = AttributeMap::new();
        let answer = Symbol::new("answer");
        attrs.insert(answer, Attribute::Int(42));

        assert_eq!(attrs.get("answer"), Some(&Attribute::Int(42)));
        assert_eq!(attrs.get(answer), Some(&Attribute::Int(42)));
        assert_eq!(get_by_symbol(&attrs, &answer), Some(&Attribute::Int(42)));
        assert!(attrs.contains_key("answer"));
        assert_eq!(attrs.keys().copied().collect::<Vec<_>>(), vec![answer]);

        let missing = "__trunk_ir_attribute_map_missing_key__";
        assert_eq!(Symbol::lookup(missing), None);
        assert_eq!(attrs.get(missing), None);
        assert!(!attrs.contains_key(missing));
        assert_eq!(Symbol::lookup(missing), None);

        assert_eq!(attrs.remove(answer), Some(Attribute::Int(42)));
        assert!(attrs.is_empty());
    }

    #[test]
    fn attribute_map_typed_getters_handle_absence_and_integer_range() {
        let mut attrs = AttributeMap::new();
        attrs.insert(Symbol::new("count"), Attribute::Int(i64::MAX as i128));
        attrs.insert(Symbol::new("byte"), Attribute::Int(u8::MAX as i128));
        attrs.insert(Symbol::new("enabled"), Attribute::Bool(true));
        attrs.insert(Symbol::new("name"), Attribute::String("tribute".to_owned()));
        attrs.insert(
            Symbol::new("symbol_name"),
            Attribute::Symbol(Symbol::new("tribute")),
        );

        assert_eq!(attrs.get_i64("count"), Ok(Some(i64::MAX)));
        assert_eq!(attrs.get_i128("count"), Some(i64::MAX as i128));
        assert_eq!(attrs.get_u8("byte"), Ok(Some(u8::MAX)));
        assert_eq!(attrs.get_bool("enabled"), Some(true));
        assert_eq!(attrs.get_str("name"), Some("tribute"));
        assert_eq!(attrs.get_i32("missing"), Ok(None));
        assert_eq!(
            attrs.get_i32("count"),
            Err(IntegerOutOfRange {
                value: i64::MAX as i128,
                target: "i32",
            })
        );
        assert_eq!(
            attrs.get_u8("count"),
            Err(IntegerOutOfRange {
                value: i64::MAX as i128,
                target: "u8",
            })
        );
        assert_eq!(attrs.get_u32("enabled"), Ok(None));

        let string_text = attrs.get_text("name").expect("string text");
        let symbol_text = attrs.get_text("symbol_name").expect("symbol text");
        assert_eq!(string_text, "tribute");
        assert_eq!(symbol_text, "tribute");
        assert_eq!(string_text, symbol_text);
        assert_eq!(attrs.get_text("enabled"), None);
    }

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
