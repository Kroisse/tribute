//! Type interning and path interning for arena-based IR.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash, RandomState};

use cranelift_entity::{EntityRef, PrimaryMap};
use hashbrown::{HashTable, hash_table};
use smallvec::SmallVec;

use super::refs::{PathRef, TypeRef};
use crate::IrContext;
use crate::location::Span;
use crate::symbol::{Symbol, SymbolPath};

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
    /// String uniqued in the owning context's string pool.
    String(StringRef),
    Bytes(SmallVec<[u8; 16]>),
    Type(TypeRef),
    /// Reference to a symbol table definition, by its qualified name.
    SymbolRef(SymbolPath),
    /// List of attributes.
    List(Vec<Attribute>),
    /// Dictionary of attributes keyed by symbol, ordered by key.
    Dict(AttributeMap),
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

impl Attribute {
    /// Extract the referenced path if this is `Attribute::SymbolRef`.
    pub fn as_symbol_ref(&self) -> Option<&SymbolPath> {
        match self {
            Attribute::SymbolRef(s) => Some(s),
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

    /// Extract the inner string handle if this is `Attribute::String`.
    pub fn as_string_ref(&self) -> Option<StringRef> {
        match self {
            Attribute::String(s) => Some(*s),
            _ => None,
        }
    }

    /// Extract the text if this is `Attribute::String`.
    pub fn as_str<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
        match self {
            Attribute::String(s) => Some(ctx.str(*s)),
            _ => None,
        }
    }

    /// Extract the inner list if this is `Attribute::List`.
    pub fn as_list(&self) -> Option<&[Attribute]> {
        match self {
            Attribute::List(items) => Some(items),
            _ => None,
        }
    }

    /// Extract the inner dictionary if this is `Attribute::Dict`.
    pub fn as_dict(&self) -> Option<&AttributeMap> {
        match self {
            Attribute::Dict(dict) => Some(dict),
            _ => None,
        }
    }

    /// Visit every `TypeRef` nested in this attribute, including those inside
    /// lists and dictionaries, in printing order.
    pub fn visit_types(&self, f: &mut impl FnMut(TypeRef)) {
        match self {
            Attribute::Type(ty) => f(*ty),
            Attribute::List(items) => {
                for item in items {
                    item.visit_types(f);
                }
            }
            Attribute::Dict(dict) => {
                for value in dict.values() {
                    value.visit_types(f);
                }
            }
            Attribute::Unit
            | Attribute::Bool(_)
            | Attribute::Int(_)
            | Attribute::FloatBits(_)
            | Attribute::String(_)
            | Attribute::Bytes(_)
            | Attribute::SymbolRef(_)
            | Attribute::Location(_) => {}
        }
    }

    /// Visit every symbol reference nested in this attribute, including those
    /// inside lists and dictionaries, in printing order.
    pub fn visit_symbol_refs(&self, f: &mut impl FnMut(&SymbolPath)) {
        match self {
            Attribute::SymbolRef(path) => f(path),
            Attribute::List(items) => {
                for item in items {
                    item.visit_symbol_refs(f);
                }
            }
            Attribute::Dict(dict) => dict.visit_symbol_refs(f),
            Attribute::Unit
            | Attribute::Bool(_)
            | Attribute::Int(_)
            | Attribute::FloatBits(_)
            | Attribute::String(_)
            | Attribute::Bytes(_)
            | Attribute::Type(_)
            | Attribute::Location(_) => {}
        }
    }

    /// Rebuild this attribute with every nested `TypeRef` replaced by `f`,
    /// stopping at the first error.
    pub fn try_map_types<E>(
        &self,
        f: &mut impl FnMut(TypeRef) -> Result<TypeRef, E>,
    ) -> Result<Attribute, E> {
        Ok(match self {
            Attribute::Type(ty) => Attribute::Type(f(*ty)?),
            Attribute::List(items) => Attribute::List(
                items
                    .iter()
                    .map(|item| item.try_map_types(f))
                    .collect::<Result<_, _>>()?,
            ),
            Attribute::Dict(dict) => Attribute::Dict(AttributeMap(
                dict.iter()
                    .map(|(key, value)| Ok((key, value.try_map_types(f)?)))
                    .collect::<Result<_, _>>()?,
            )),
            Attribute::Unit
            | Attribute::Bool(_)
            | Attribute::Int(_)
            | Attribute::FloatBits(_)
            | Attribute::String(_)
            | Attribute::Bytes(_)
            | Attribute::SymbolRef(_)
            | Attribute::Location(_) => self.clone(),
        })
    }

    /// Rebuild this attribute with every nested `TypeRef` replaced by `f`.
    pub fn map_types(&self, mut f: impl FnMut(TypeRef) -> TypeRef) -> Attribute {
        let Ok(mapped) = self.try_map_types(&mut |ty| Ok::<_, std::convert::Infallible>(f(ty)));
        mapped
    }

    /// Estimate the complexity of this attribute for alias generation heuristics.
    pub fn complexity(&self, strings: &StringPool) -> usize {
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
            Attribute::String(s) => strings.get(*s).len() + 2,
            Attribute::Bytes(b) => b.len() * 4 + 7,
            Attribute::SymbolRef(path) => path.iter().map(|name| name.as_str().len() + 3).sum(),
            Attribute::Type(_) => 10, // rough estimate; actual depends on type
            Attribute::List(list) => {
                list.iter()
                    .map(|item| item.complexity(strings))
                    .sum::<usize>()
                    + list.len() * 2
            }
            Attribute::Dict(dict) => {
                dict.iter()
                    .map(|(key, value)| strings.get(key).len() + 3 + value.complexity(strings))
                    .sum::<usize>()
                    + dict.len() * 2
                    + 2
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

impl From<SymbolPath> for Attribute {
    fn from(value: SymbolPath) -> Self {
        Attribute::SymbolRef(value)
    }
}

impl From<StringRef> for Attribute {
    fn from(value: StringRef) -> Self {
        Attribute::String(value)
    }
}

impl From<AttributeMap> for Attribute {
    fn from(value: AttributeMap) -> Self {
        Attribute::Dict(value)
    }
}

impl From<Location> for Attribute {
    fn from(value: Location) -> Self {
        Attribute::Location(value)
    }
}

/// A deterministic map of IR attributes keyed by name.
///
/// Keys are names, not symbols: they are read and written through the owning
/// context, which resolves their text. Entries are kept in a vector sorted by
/// key text. Maps hold only a few entries, so lookups scan; only inserting a
/// new key orders it. Iteration, equality, and hashing follow key order,
/// independent of the order in which entries were inserted.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct AttributeMap(Vec<(StringRef, Attribute)>);

/// A key accepted by [`AttributeMap::get`]: the key's text, or its pooled
/// handle.
pub trait AttributeKey {
    fn matches(&self, strings: &StringPool, key: StringRef) -> bool;
}

impl AttributeKey for &str {
    fn matches(&self, strings: &StringPool, key: StringRef) -> bool {
        strings.get(key) == *self
    }
}

impl AttributeKey for StringRef {
    fn matches(&self, _strings: &StringPool, key: StringRef) -> bool {
        *self == key
    }
}

impl AttributeMap {
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// A map holding `entries`; later entries replace earlier ones with the
    /// same key.
    pub fn from_entries<K: Into<StringArg>, V: Into<Attribute>>(
        ctx: &mut IrContext,
        entries: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        let mut map = Self::new();
        map.extend(ctx, entries);
        map
    }

    fn position(&self, strings: &StringPool, key: impl AttributeKey) -> Option<usize> {
        self.0
            .iter()
            .position(|&(existing, _)| key.matches(strings, existing))
    }

    /// Return the attribute associated with a key.
    pub fn get(&self, ctx: &impl AsRef<StringPool>, key: impl AttributeKey) -> Option<&Attribute> {
        let index = self.position(ctx.as_ref(), key)?;
        Some(&self.0[index].1)
    }

    pub fn get_mut(
        &mut self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Option<&mut Attribute> {
        let index = self.position(ctx.as_ref(), key)?;
        Some(&mut self.0[index].1)
    }

    pub fn get_bool(&self, ctx: &impl AsRef<StringPool>, key: impl AttributeKey) -> Option<bool> {
        self.get(ctx, key).and_then(Attribute::as_bool)
    }

    pub fn get_i128(&self, ctx: &impl AsRef<StringPool>, key: impl AttributeKey) -> Option<i128> {
        self.get(ctx, key).and_then(Attribute::as_i128)
    }

    pub fn get_i64(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Result<Option<i64>, IntegerOutOfRange> {
        self.get_integer(ctx, key, "i64", i64::try_from)
    }

    pub fn get_i32(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Result<Option<i32>, IntegerOutOfRange> {
        self.get_integer(ctx, key, "i32", i32::try_from)
    }

    pub fn get_u64(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Result<Option<u64>, IntegerOutOfRange> {
        self.get_integer(ctx, key, "u64", u64::try_from)
    }

    pub fn get_u32(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Result<Option<u32>, IntegerOutOfRange> {
        self.get_integer(ctx, key, "u32", u32::try_from)
    }

    pub fn get_u8(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Result<Option<u8>, IntegerOutOfRange> {
        self.get_integer(ctx, key, "u8", u8::try_from)
    }

    pub fn get_string_ref(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Option<StringRef> {
        self.get(ctx, key).and_then(Attribute::as_string_ref)
    }

    /// The text of a string attribute, resolved through the owning context.
    pub fn get_str<'a, C: AsRef<StringPool>>(
        &self,
        ctx: &'a C,
        key: impl AttributeKey,
    ) -> Option<&'a str> {
        let value = self.get_string_ref(ctx, key)?;
        Some(ctx.as_ref().get(value))
    }

    pub fn get_symbol_ref(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Option<&SymbolPath> {
        self.get(ctx, key).and_then(Attribute::as_symbol_ref)
    }

    pub fn get_type(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Option<TypeRef> {
        self.get(ctx, key).and_then(Attribute::as_type)
    }

    fn get_integer<T>(
        &self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
        target: &'static str,
        convert: impl FnOnce(i128) -> Result<T, std::num::TryFromIntError>,
    ) -> Result<Option<T>, IntegerOutOfRange> {
        let Some(value) = self.get_i128(ctx, key) else {
            return Ok(None);
        };
        convert(value)
            .map(Some)
            .map_err(|_| IntegerOutOfRange { value, target })
    }

    pub fn contains_key(&self, ctx: &impl AsRef<StringPool>, key: impl AttributeKey) -> bool {
        self.position(ctx.as_ref(), key).is_some()
    }

    /// Insert or replace an entry, returning the replaced value.
    pub fn insert(
        &mut self,
        ctx: &mut IrContext,
        key: impl Into<StringArg>,
        value: impl Into<Attribute>,
    ) -> Option<Attribute> {
        let key = ctx.intern_string_arg(key.into());
        self.insert_ref(ctx, key, value)
    }

    /// Insert or replace an entry under a key already pooled in `ctx`,
    /// returning the replaced value.
    pub fn insert_ref(
        &mut self,
        ctx: &impl AsRef<StringPool>,
        key: StringRef,
        value: impl Into<Attribute>,
    ) -> Option<Attribute> {
        let strings = ctx.as_ref();
        let value = value.into();
        if let Some(index) = self.0.iter().position(|&(existing, _)| existing == key) {
            return Some(std::mem::replace(&mut self.0[index].1, value));
        }
        // Most maps hold one or two entries. A `Vec`'s first push reserves
        // four, so start a map at exactly one and let later inserts grow it.
        if self.0.capacity() == 0 {
            self.0.reserve_exact(1);
        }
        let text = strings.get(key);
        let index = self
            .0
            .partition_point(|&(existing, _)| strings.get(existing) < text);
        self.0.insert(index, (key, value));
        None
    }

    /// A map holding `entries`, whose keys are already pooled in `ctx`;
    /// later entries replace earlier ones with the same key.
    pub fn from_refs(
        ctx: &impl AsRef<StringPool>,
        entries: impl IntoIterator<Item = (StringRef, Attribute)>,
    ) -> Self {
        let entries = entries.into_iter();
        let mut map = Self::new();
        map.0.reserve_exact(entries.size_hint().0.min(4));
        for (key, value) in entries {
            map.insert_ref(ctx, key, value);
        }
        map
    }

    /// Insert every entry; later entries replace earlier ones with the same
    /// key.
    pub fn extend<K: Into<StringArg>, V: Into<Attribute>>(
        &mut self,
        ctx: &mut IrContext,
        entries: impl IntoIterator<Item = (K, V)>,
    ) {
        let entries = entries.into_iter();
        // Repeated keys replace earlier entries, so the iterator length only
        // bounds the map's size from above. Reserve at most a small map
        // exactly and let inserts grow anything larger.
        if self.0.capacity() == 0 {
            self.0.reserve_exact(entries.size_hint().0.min(4));
        }
        for (key, value) in entries {
            self.insert(ctx, key, value);
        }
    }

    pub fn remove(
        &mut self,
        ctx: &impl AsRef<StringPool>,
        key: impl AttributeKey,
    ) -> Option<Attribute> {
        let index = self.position(ctx.as_ref(), key)?;
        Some(self.0.remove(index).1)
    }

    pub fn iter(&self) -> AttributeIter<'_> {
        AttributeIter(self.0.iter())
    }

    pub fn iter_mut(&mut self) -> AttributeIterMut<'_> {
        AttributeIterMut(self.0.iter_mut())
    }

    pub fn keys(&self) -> AttributeKeys<'_> {
        AttributeKeys(self.0.iter())
    }

    /// Visit every symbol reference in these attributes; see
    /// [`Attribute::visit_symbol_refs`].
    pub fn visit_symbol_refs(&self, f: &mut impl FnMut(&SymbolPath)) {
        for value in self.values() {
            value.visit_symbol_refs(f);
        }
    }

    pub fn values(&self) -> AttributeValues<'_> {
        AttributeValues(self.0.iter())
    }

    pub fn values_mut(&mut self) -> AttributeValuesMut<'_> {
        AttributeValuesMut(self.0.iter_mut())
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

impl fmt::Debug for AttributeMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl AttributeMap {
    /// Format these attributes with keys and string values as text, which
    /// `Debug` cannot resolve without the owning context.
    pub fn display<'a>(&'a self, ctx: &'a impl AsRef<StringPool>) -> AttributeMapDisplay<'a> {
        AttributeMapDisplay {
            map: self,
            strings: ctx.as_ref(),
        }
    }
}

impl AttributeMap {
    /// Read these attributes through `ctx`, so lookups need no context
    /// argument.
    pub fn view<'a>(&'a self, ctx: &'a IrContext) -> AttributeView<'a> {
        AttributeView { map: self, ctx }
    }
}

/// An [`AttributeMap`] bound to its context; see [`AttributeMap::view`] and
/// [`IrContext::op_attrs`].
#[derive(Clone, Copy)]
pub struct AttributeView<'a> {
    map: &'a AttributeMap,
    ctx: &'a IrContext,
}

impl<'a> AttributeView<'a> {
    /// The underlying map.
    pub fn map(self) -> &'a AttributeMap {
        self.map
    }

    pub fn get(self, key: impl AttributeKey) -> Option<&'a Attribute> {
        self.map.get(self.ctx, key)
    }

    pub fn contains_key(self, key: impl AttributeKey) -> bool {
        self.map.contains_key(self.ctx, key)
    }

    pub fn get_bool(self, key: impl AttributeKey) -> Option<bool> {
        self.map.get_bool(self.ctx, key)
    }

    pub fn get_i128(self, key: impl AttributeKey) -> Option<i128> {
        self.map.get_i128(self.ctx, key)
    }

    pub fn get_i64(self, key: impl AttributeKey) -> Result<Option<i64>, IntegerOutOfRange> {
        self.map.get_i64(self.ctx, key)
    }

    pub fn get_i32(self, key: impl AttributeKey) -> Result<Option<i32>, IntegerOutOfRange> {
        self.map.get_i32(self.ctx, key)
    }

    pub fn get_u64(self, key: impl AttributeKey) -> Result<Option<u64>, IntegerOutOfRange> {
        self.map.get_u64(self.ctx, key)
    }

    pub fn get_u32(self, key: impl AttributeKey) -> Result<Option<u32>, IntegerOutOfRange> {
        self.map.get_u32(self.ctx, key)
    }

    pub fn get_u8(self, key: impl AttributeKey) -> Result<Option<u8>, IntegerOutOfRange> {
        self.map.get_u8(self.ctx, key)
    }

    pub fn get_string_ref(self, key: impl AttributeKey) -> Option<StringRef> {
        self.map.get_string_ref(self.ctx, key)
    }

    pub fn get_str(self, key: impl AttributeKey) -> Option<&'a str> {
        self.map.get_str(self.ctx, key)
    }

    pub fn get_symbol_ref(self, key: impl AttributeKey) -> Option<&'a SymbolPath> {
        self.map.get_symbol_ref(self.ctx, key)
    }

    pub fn get_type(self, key: impl AttributeKey) -> Option<TypeRef> {
        self.map.get_type(self.ctx, key)
    }

    /// Each key's text with its value, in key order.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = (&'a str, &'a Attribute)> + 'a {
        self.map
            .iter()
            .map(move |(key, value)| (self.ctx.str(key), value))
    }

    pub fn len(self) -> usize {
        self.map.len()
    }

    pub fn is_empty(self) -> bool {
        self.map.is_empty()
    }
}

/// An [`AttributeMap`] formatted through its context; see
/// [`AttributeMap::display`].
pub struct AttributeMapDisplay<'a> {
    map: &'a AttributeMap,
    strings: &'a StringPool,
}

impl fmt::Debug for AttributeMapDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.map.iter().map(|(key, value)| {
                (
                    self.strings.get(key),
                    AttributeDisplay {
                        attribute: value,
                        strings: self.strings,
                    },
                )
            }))
            .finish()
    }
}

impl fmt::Display for AttributeMapDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

struct AttributeDisplay<'a> {
    attribute: &'a Attribute,
    strings: &'a StringPool,
}

impl fmt::Debug for AttributeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.attribute {
            Attribute::String(value) => fmt::Debug::fmt(self.strings.get(*value), f),
            Attribute::List(items) => f
                .debug_list()
                .entries(items.iter().map(|attribute| AttributeDisplay {
                    attribute,
                    strings: self.strings,
                }))
                .finish(),
            Attribute::Dict(map) => fmt::Debug::fmt(
                &AttributeMapDisplay {
                    map,
                    strings: self.strings,
                },
                f,
            ),
            other => fmt::Debug::fmt(other, f),
        }
    }
}

macro_rules! attribute_iterator {
    ($(#[$meta:meta])* $name:ident<$lt:lifetime>($inner:ty) -> $item:ty = $map:expr) => {
        $(#[$meta])*
        pub struct $name<$lt>($inner);

        impl<$lt> Iterator for $name<$lt> {
            type Item = $item;

            fn next(&mut self) -> Option<Self::Item> {
                self.0.next().map($map)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }

        impl<$lt> DoubleEndedIterator for $name<$lt> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back().map($map)
            }
        }

        impl<$lt> ExactSizeIterator for $name<$lt> {}

        impl<$lt> std::iter::FusedIterator for $name<$lt> {}
    };
}

attribute_iterator!(
    /// Entries of an [`AttributeMap`] in key order.
    #[derive(Clone)]
    AttributeIter<'a>(std::slice::Iter<'a, (StringRef, Attribute)>)
        -> (StringRef, &'a Attribute) = |(key, value)| (*key, value)
);
attribute_iterator!(
    /// Entries of an [`AttributeMap`] in key order, with mutable values.
    AttributeIterMut<'a>(std::slice::IterMut<'a, (StringRef, Attribute)>)
        -> (StringRef, &'a mut Attribute) = |(key, value)| (*key, value)
);
attribute_iterator!(
    /// Keys of an [`AttributeMap`] in order.
    #[derive(Clone)]
    AttributeKeys<'a>(std::slice::Iter<'a, (StringRef, Attribute)>)
        -> StringRef = |(key, _)| *key
);
attribute_iterator!(
    /// Values of an [`AttributeMap`] in key order.
    #[derive(Clone)]
    AttributeValues<'a>(std::slice::Iter<'a, (StringRef, Attribute)>)
        -> &'a Attribute = |(_, value)| value
);
attribute_iterator!(
    /// Mutable values of an [`AttributeMap`] in key order.
    AttributeValuesMut<'a>(std::slice::IterMut<'a, (StringRef, Attribute)>)
        -> &'a mut Attribute = |(_, value)| value
);

/// Owned entries of an [`AttributeMap`] in key order.
pub type AttributeIntoIter = std::vec::IntoIter<(StringRef, Attribute)>;

impl IntoIterator for AttributeMap {
    type Item = (StringRef, Attribute);
    type IntoIter = AttributeIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a AttributeMap {
    type Item = (StringRef, &'a Attribute);
    type IntoIter = AttributeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut AttributeMap {
    type Item = (StringRef, &'a mut Attribute);
    type IntoIter = AttributeIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ============================================================================
// TypeData
// ============================================================================

/// Reserved type attribute holding one dictionary of attributes per type
/// parameter, in `params` order.
///
/// The value is a list of `Attribute::Dict` whose length equals `params.len()`.
/// The key is absent when every dictionary would be empty, so a type without
/// per-parameter attributes keeps a single identity. The textual form never
/// spells this key: it writes each dictionary after its parameter, as in
/// `core.tuple<core.i32, core.ptr {k = @v}>`.
pub const PARAM_ATTRS_ATTR: &str = "param_attrs";

/// Reserved type attribute naming a compiler-owned runtime storage layout.
///
/// The value is a string whose meaning the defining language layer and the
/// implementing target own; TrunkIR does not interpret it.
pub const LAYOUT_ATTR: &str = "layout";

static EMPTY_ATTRIBUTE_MAP: AttributeMap = AttributeMap::new();

/// A malformed [`PARAM_ATTRS_ATTR`] value.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum ParamAttrsError {
    #[display("`{PARAM_ATTRS_ATTR}` must be a list of dictionaries")]
    NotList,
    #[display("`{PARAM_ATTRS_ATTR}` entry {_0} must be a dictionary")]
    NotDict(#[error(not(source))] usize),
    #[display("`{PARAM_ATTRS_ATTR}` has {attrs} entries for {params} type parameters")]
    LengthMismatch { params: usize, attrs: usize },
    #[display("`{PARAM_ATTRS_ATTR}` with only empty dictionaries must be omitted")]
    AllEmpty,
}

/// The canonical [`PARAM_ATTRS_ATTR`] value for `entries`, or `None` when every
/// entry is empty and the key must be omitted.
///
/// Leading empty entries are only counted, so a type whose parameters carry no
/// attributes allocates nothing here.
pub fn param_attrs_attribute(entries: impl IntoIterator<Item = AttributeMap>) -> Option<Attribute> {
    let mut entries = entries.into_iter();
    let mut leading_empty = 0;
    let first = loop {
        let entry = entries.next()?;
        if !entry.is_empty() {
            break entry;
        }
        leading_empty += 1;
    };
    let mut list = Vec::with_capacity(leading_empty + 1 + entries.size_hint().0);
    list.extend(
        std::iter::repeat_with(|| Attribute::Dict(AttributeMap::new())).take(leading_empty),
    );
    list.push(Attribute::Dict(first));
    list.extend(entries.map(Attribute::Dict));
    Some(Attribute::List(list))
}

/// Data for a single interned type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeData {
    pub dialect: Symbol,
    pub name: Symbol,
    pub params: SmallVec<[TypeRef; 4]>,
    pub attrs: AttributeMap,
}

impl TypeData {
    /// The attributes of type parameter `index`; empty when it has none.
    ///
    /// Assumes a well-formed [`PARAM_ATTRS_ATTR`]; see
    /// [`TypeData::validate_param_attrs`].
    pub fn param_attrs(&self, ctx: &impl AsRef<StringPool>, index: usize) -> &AttributeMap {
        self.attrs
            .get(ctx, PARAM_ATTRS_ATTR)
            .and_then(Attribute::as_list)
            .and_then(|entries| entries.get(index))
            .and_then(Attribute::as_dict)
            .unwrap_or(&EMPTY_ATTRIBUTE_MAP)
    }

    /// Each type parameter with its attributes.
    pub fn params_with_attrs<'a>(
        &'a self,
        ctx: &'a impl AsRef<StringPool>,
    ) -> impl Iterator<Item = (TypeRef, &'a AttributeMap)> {
        self.params
            .iter()
            .enumerate()
            .map(move |(index, &ty)| (ty, self.param_attrs(ctx, index)))
    }

    /// Check that [`PARAM_ATTRS_ATTR`], if present, is canonical: one
    /// dictionary per type parameter, not all of them empty.
    pub fn validate_param_attrs(
        &self,
        ctx: &impl AsRef<StringPool>,
    ) -> Result<(), ParamAttrsError> {
        validate_param_attrs(ctx, &self.attrs, self.params.len())
    }
}

/// Check that [`PARAM_ATTRS_ATTR`] in `attrs`, if present, is canonical for a
/// type with `params` type parameters.
pub fn validate_param_attrs(
    ctx: &impl AsRef<StringPool>,
    attrs: &AttributeMap,
    params: usize,
) -> Result<(), ParamAttrsError> {
    let Some(value) = attrs.get(ctx, PARAM_ATTRS_ATTR) else {
        return Ok(());
    };
    let entries = value.as_list().ok_or(ParamAttrsError::NotList)?;
    if entries.len() != params {
        return Err(ParamAttrsError::LengthMismatch {
            params,
            attrs: entries.len(),
        });
    }
    let mut all_empty = true;
    for (index, entry) in entries.iter().enumerate() {
        let dict = entry.as_dict().ok_or(ParamAttrsError::NotDict(index))?;
        all_empty &= dict.is_empty();
    }
    if all_empty {
        return Err(ParamAttrsError::AllEmpty);
    }
    Ok(())
}

/// Bring [`PARAM_ATTRS_ATTR`] in `attrs` to canonical form for a type with
/// `params` type parameters, then validate it.
///
/// A correctly sized list of empty dictionaries, the one non-canonical
/// spelling of "no parameter attributes", is removed; any other malformed
/// value is kept and reported.
pub fn normalize_param_attrs(
    ctx: &impl AsRef<StringPool>,
    attrs: &mut AttributeMap,
    params: usize,
) -> Result<(), ParamAttrsError> {
    let all_empty = attrs
        .get(ctx, PARAM_ATTRS_ATTR)
        .and_then(Attribute::as_list)
        .is_some_and(|entries| {
            entries.len() == params
                && entries
                    .iter()
                    .all(|entry| entry.as_dict().is_some_and(AttributeMap::is_empty))
        });
    if all_empty {
        attrs.remove(ctx, PARAM_ATTRS_ATTR);
    }
    validate_param_attrs(ctx, attrs, params)
}

/// Builder for constructing `TypeData` with a fluent API.
///
/// Defaults to empty params and empty attrs, matching the most common usage.
pub struct TypeDataBuilder {
    dialect: Symbol,
    name: Symbol,
    /// Same inline capacity as [`TypeData::params`], so a type whose
    /// parameters fit inline there does not allocate here either; at 32 bytes
    /// per entry the inline buffer is 128 bytes.
    params: SmallVec<[(TypeRef, AttributeMap); 4]>,
    /// Keys are interned when the type data is built.
    attrs: Vec<(StringArg, Attribute)>,
}

impl TypeDataBuilder {
    pub fn new(dialect: impl Into<Symbol>, name: impl Into<Symbol>) -> Self {
        Self {
            dialect: dialect.into(),
            name: name.into(),
            params: SmallVec::new(),
            attrs: Vec::new(),
        }
    }

    pub fn param(self, ty: TypeRef) -> Self {
        self.param_with_attrs(ty, AttributeMap::new())
    }

    pub fn params(mut self, tys: impl IntoIterator<Item = TypeRef>) -> Self {
        for ty in tys {
            self = self.param(ty);
        }
        self
    }

    /// Add a type parameter carrying its own attributes.
    pub fn param_with_attrs(mut self, ty: TypeRef, attrs: AttributeMap) -> Self {
        self.params.push((ty, attrs));
        self
    }

    pub fn attr(mut self, key: impl Into<StringArg>, val: impl Into<Attribute>) -> Self {
        self.attrs.push((key.into(), val.into()));
        self
    }

    /// Build the type data and intern it.
    pub fn intern(self, ctx: &mut IrContext) -> TypeRef {
        let data = self.build(ctx);
        ctx.intern_type(data)
    }

    /// Build the type data. Parameter attributes given with
    /// [`param_with_attrs`](Self::param_with_attrs) are stored in canonical
    /// [`PARAM_ATTRS_ATTR`] form and replace one set through
    /// [`attr`](Self::attr); an explicit all-empty value is dropped. A
    /// malformed explicit value is kept and rejected when the type is interned
    /// or validated.
    pub fn build(self, ctx: &mut IrContext) -> TypeData {
        let (params, param_attrs): (SmallVec<[TypeRef; 4]>, Vec<_>) =
            self.params.into_iter().unzip();
        let mut attrs = AttributeMap::from_entries(ctx, self.attrs);
        if let Some(value) = param_attrs_attribute(param_attrs) {
            attrs.insert(ctx, PARAM_ATTRS_ATTR, value);
        } else {
            let _ = normalize_param_attrs(ctx, &mut attrs, params.len());
        }
        TypeData {
            dialect: self.dialect,
            name: self.name,
            params,
            attrs,
        }
    }
}

// ============================================================================
// Intern tables
// ============================================================================

/// Values stored once in `values`, deduplicated through an index of their keys.
#[derive(Clone)]
struct InternTable<K: EntityRef, V> {
    values: PrimaryMap<K, V>,
    index: HashTable<K>,
    hasher: RandomState,
}

/// The result of probing an [`InternTable`] once.
pub(crate) enum InternEntry<'a, K: EntityRef, V> {
    Occupied(K),
    Vacant(VacantIntern<'a, K, V>),
}

/// A missing value whose slot has been found; inserting it needs no rehash.
pub(crate) struct VacantIntern<'a, K: EntityRef, V> {
    values: &'a mut PrimaryMap<K, V>,
    entry: hash_table::VacantEntry<'a, K>,
}

impl<K: EntityRef, V> VacantIntern<'_, K, V> {
    /// Store `value`, which must equal the probed key.
    pub(crate) fn insert(self, value: V) -> K {
        let key = self.values.push(value);
        self.entry.insert(key);
        key
    }
}

impl<K: EntityRef, V: Hash + Eq> InternTable<K, V> {
    fn new() -> Self {
        Self {
            values: PrimaryMap::new(),
            index: HashTable::new(),
            hasher: RandomState::new(),
        }
    }

    fn lookup<Q: Hash + Eq + ?Sized>(&self, value: &Q) -> Option<K>
    where
        V: Borrow<Q>,
    {
        let hash = self.hasher.hash_one(value);
        self.index
            .find(hash, |&key| self.values[key].borrow() == value)
            .copied()
    }

    fn entry<Q: Hash + Eq + ?Sized>(&mut self, value: &Q) -> InternEntry<'_, K, V>
    where
        V: Borrow<Q>,
    {
        let hash = self.hasher.hash_one(value);
        let Self {
            values,
            index,
            hasher,
        } = self;
        match index.entry(
            hash,
            |&key| values[key].borrow() == value,
            |&key| hasher.hash_one(&values[key]),
        ) {
            hash_table::Entry::Occupied(entry) => InternEntry::Occupied(*entry.get()),
            hash_table::Entry::Vacant(entry) => InternEntry::Vacant(VacantIntern { values, entry }),
        }
    }

    fn intern(&mut self, value: V) -> K {
        match self.entry(&value) {
            InternEntry::Occupied(key) => key,
            InternEntry::Vacant(entry) => entry.insert(value),
        }
    }
}

// ============================================================================
// TypeInterner
// ============================================================================

/// Deduplicating type interner. Same `TypeData` always yields the same `TypeRef`.
#[derive(Clone)]
pub struct TypeInterner(InternTable<TypeRef, TypeData>);

impl TypeInterner {
    pub fn new() -> Self {
        Self(InternTable::new())
    }

    /// Intern a type, returning an existing ref if the data matches.
    pub fn intern(&mut self, data: TypeData) -> TypeRef {
        self.0.intern(data)
    }

    /// Probe once for `data`, leaving a vacant slot to fill on a miss.
    pub(crate) fn entry(&mut self, data: &TypeData) -> InternEntry<'_, TypeRef, TypeData> {
        self.0.entry(data)
    }

    /// Look up type data by reference.
    pub fn get(&self, r: TypeRef) -> &TypeData {
        &self.0.values[r]
    }

    /// Check if this type matches the given dialect and name.
    ///
    /// `dialect` and `name` are anything a `Symbol` compares with, such as a
    /// `&str` or a `&Symbol`.
    pub fn is_dialect<D, N>(&self, r: TypeRef, dialect: D, name: N) -> bool
    where
        Symbol: PartialEq<D> + PartialEq<N>,
    {
        let data = self.get(r);
        data.dialect == dialect && data.name == name
    }

    /// Iterate over all interned types, yielding `(TypeRef, &TypeData)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (TypeRef, &TypeData)> {
        self.0.values.iter()
    }

    /// Find a TypeRef by looking up through the dedup map.
    /// Returns `None` if no type with the given data exists.
    pub fn lookup(&self, data: &TypeData) -> Option<TypeRef> {
        self.0.lookup(data)
    }

    /// Estimate the complexity of a type for alias generation heuristics.
    pub fn complexity(&self, ty: TypeRef, strings: &StringPool) -> usize {
        let data = self.get(ty);
        let mut size = data.dialect.with_str(|s| s.len()) + 1 + data.name.with_str(|s| s.len());
        for &param in &data.params {
            size += self.complexity(param, strings) + 2; // ", " separator
        }
        for (key, val) in &data.attrs {
            size += strings.get(key).len() + 3; // "key = "
            size += val.complexity(strings);
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
#[derive(Clone)]
pub struct PathInterner(InternTable<PathRef, String>);

impl PathInterner {
    pub fn new() -> Self {
        Self(InternTable::new())
    }

    /// Intern a path string, returning an existing ref if the string matches.
    ///
    /// A path already interned is found without allocating; a new path is
    /// copied into the interner once.
    pub fn intern(&mut self, path: &str) -> PathRef {
        match self.0.entry(path) {
            InternEntry::Occupied(existing) => existing,
            InternEntry::Vacant(entry) => entry.insert(path.to_owned()),
        }
    }

    /// Probe once for `path`, leaving a vacant slot to fill on a miss.
    pub(crate) fn entry(&mut self, path: &str) -> InternEntry<'_, PathRef, String> {
        self.0.entry(path)
    }

    /// Find an existing path without changing the interner.
    pub fn lookup(&self, path: &str) -> Option<PathRef> {
        self.0.lookup(path)
    }

    /// Look up path string by reference.
    pub fn get(&self, r: PathRef) -> &str {
        &self.0.values[r]
    }
}

impl Default for PathInterner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// StringPool
// ============================================================================

/// Handle to a string in an `IrContext`'s string pool.
///
/// Equal handles from one pool denote equal text. Like the context's other
/// handles (`TypeRef`, `OpRef`, ...), a handle is valid only in the context
/// that created it: cloning a context copies the pool, so a handle created
/// before the clone is valid in both copies, and one created afterwards only in
/// the copy that created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringRef(lasso::Spur);

/// A string attribute value given to an operation builder: a pooled handle,
/// or text the builder interns when it creates the operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StringArg {
    Ref(StringRef),
    Text(std::borrow::Cow<'static, str>),
}

impl From<StringRef> for StringArg {
    fn from(value: StringRef) -> Self {
        StringArg::Ref(value)
    }
}

impl From<&'static str> for StringArg {
    fn from(value: &'static str) -> Self {
        StringArg::Text(value.into())
    }
}

impl From<String> for StringArg {
    fn from(value: String) -> Self {
        StringArg::Text(value.into())
    }
}

impl From<std::borrow::Cow<'static, str>> for StringArg {
    fn from(value: std::borrow::Cow<'static, str>) -> Self {
        StringArg::Text(value)
    }
}

/// Deduplicating pool for string attribute values, owned by an `IrContext`.
///
/// Strings are stored in an arena and freed with the pool.
#[derive(Clone)]
pub struct StringPool(lasso::Rodeo<lasso::Spur, rustc_hash::FxBuildHasher>);

impl StringPool {
    pub fn new() -> Self {
        Self(lasso::Rodeo::with_hasher(Default::default()))
    }

    /// Intern `text`, returning the existing handle if it is already pooled.
    pub fn intern(&mut self, text: &str) -> StringRef {
        StringRef(self.0.get_or_intern(text))
    }

    /// Find an existing string without changing the pool.
    pub fn lookup(&self, text: &str) -> Option<StringRef> {
        self.0.get(text).map(StringRef)
    }

    /// The text of a pooled string.
    pub fn get(&self, r: StringRef) -> &str {
        self.0.resolve(&r.0)
    }
}

impl AsRef<StringPool> for StringPool {
    fn as_ref(&self) -> &StringPool {
        self
    }
}

impl Default for StringPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod prop_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrContext;

    #[test]
    fn parameter_attributes_are_canonical_and_part_of_identity() {
        let mut ctx = IrContext::new();
        let i32_ty = TypeDataBuilder::new("core", "i32").intern(&mut ctx);
        let marked = AttributeMap::from_entries(
            &mut ctx,
            [("k", Attribute::SymbolRef(SymbolPath::new(["v"])))],
        );

        let plain = TypeDataBuilder::new("core", "tuple")
            .params([i32_ty, i32_ty])
            .intern(&mut ctx);
        let empty = TypeDataBuilder::new("core", "tuple")
            .param_with_attrs(i32_ty, AttributeMap::new())
            .param(i32_ty)
            .intern(&mut ctx);
        let explicit_empty = TypeDataBuilder::new("core", "tuple")
            .params([i32_ty, i32_ty])
            .attr(
                PARAM_ATTRS_ATTR,
                Attribute::List(vec![
                    Attribute::Dict(AttributeMap::new()),
                    Attribute::Dict(AttributeMap::new()),
                ]),
            )
            .intern(&mut ctx);
        assert_eq!(plain, empty);
        assert_eq!(plain, explicit_empty);
        assert!(
            !ctx.get_type(plain)
                .attrs
                .contains_key(&ctx, PARAM_ATTRS_ATTR)
        );

        let with_attrs = TypeDataBuilder::new("core", "tuple")
            .param(i32_ty)
            .param_with_attrs(i32_ty, marked.clone())
            .intern(&mut ctx);
        assert_ne!(plain, with_attrs);
        let data = ctx.get_type(with_attrs);
        assert!(data.param_attrs(&ctx, 0).is_empty());
        assert_eq!(data.param_attrs(&ctx, 1), &marked);
        itertools::assert_equal(
            data.params_with_attrs(&ctx),
            vec![(i32_ty, &AttributeMap::new()), (i32_ty, &marked)],
        );
        assert_eq!(data.validate_param_attrs(&ctx), Ok(()));
    }

    #[test]
    fn malformed_parameter_attributes_are_reported() {
        let mut ctx = IrContext::new();
        let k_entry = AttributeMap::from_entries(&mut ctx, [("k", Attribute::Unit)]);
        let dict =
            |entries: Option<&AttributeMap>| Attribute::Dict(entries.cloned().unwrap_or_default());
        let key = ctx.intern_str(PARAM_ATTRS_ATTR);
        let attrs = |value| AttributeMap::from_refs(ctx.strings(), [(key, value)]);
        assert_eq!(
            validate_param_attrs(&ctx, &attrs(Attribute::Unit), 1),
            Err(ParamAttrsError::NotList)
        );
        assert_eq!(
            validate_param_attrs(&ctx, &attrs(Attribute::List(vec![dict(Some(&k_entry))])), 2),
            Err(ParamAttrsError::LengthMismatch {
                params: 2,
                attrs: 1
            })
        );
        assert_eq!(
            validate_param_attrs(
                &ctx,
                &attrs(Attribute::List(vec![dict(Some(&k_entry)), Attribute::Unit])),
                2
            ),
            Err(ParamAttrsError::NotDict(1))
        );
        assert_eq!(
            validate_param_attrs(&ctx, &attrs(Attribute::List(vec![dict(None)])), 1),
            Err(ParamAttrsError::AllEmpty)
        );
        assert_eq!(validate_param_attrs(&ctx, &AttributeMap::new(), 3), Ok(()));

        let mut wrong_length = attrs(Attribute::List(vec![dict(None)]));
        assert_eq!(
            normalize_param_attrs(&ctx, &mut wrong_length, 2),
            Err(ParamAttrsError::LengthMismatch {
                params: 2,
                attrs: 1
            })
        );
        assert!(wrong_length.contains_key(&ctx, PARAM_ATTRS_ATTR));
        let mut all_empty = attrs(Attribute::List(vec![dict(None), dict(None)]));
        assert_eq!(normalize_param_attrs(&ctx, &mut all_empty, 2), Ok(()));
        assert!(all_empty.is_empty());
    }

    #[test]
    fn symbol_refs_are_visited_through_lists_and_dicts() {
        let mut ctx = IrContext::new();
        let reference = |name| Attribute::SymbolRef(SymbolPath::new([name]));
        let mut attrs = AttributeMap::new();
        attrs.insert(&mut ctx, "callee", reference("direct"));
        let name = ctx.string_attr("not_a_reference");
        attrs.insert(&mut ctx, "name", name);
        let nested = AttributeMap::from_entries(&mut ctx, [("target", reference("nested"))]);
        attrs.insert(
            &mut ctx,
            "table",
            Attribute::List(vec![reference("first"), Attribute::Dict(nested)]),
        );

        let mut visited = Vec::new();
        attrs.visit_symbol_refs(&mut |symbol| visited.push(symbol.clone()));
        assert_eq!(
            visited,
            ["direct", "first", "nested"]
                .map(|name| SymbolPath::new([name]))
                .to_vec()
        );
    }

    #[test]
    fn nested_types_are_visited_and_mapped_through_lists_and_dicts() {
        let mut ctx = IrContext::new();
        let i32_ty = TypeDataBuilder::new("core", "i32").intern(&mut ctx);
        let i64_ty = TypeDataBuilder::new("core", "i64").intern(&mut ctx);
        let ty_key = ctx.intern_str("ty");
        let tag_key = ctx.intern_str("tag");
        let dict = |ty| {
            Attribute::Dict(AttributeMap::from_refs(
                ctx.strings(),
                [
                    (ty_key, Attribute::Type(ty)),
                    (tag_key, Attribute::SymbolRef(SymbolPath::new(["keep"]))),
                ],
            ))
        };
        let attribute = Attribute::List(vec![Attribute::Type(i32_ty), dict(i32_ty)]);

        let mut visited = Vec::new();
        attribute.visit_types(&mut |ty| visited.push(ty));
        assert_eq!(visited, vec![i32_ty, i32_ty]);

        let mapped = attribute.map_types(|ty| if ty == i32_ty { i64_ty } else { ty });
        assert_eq!(
            mapped,
            Attribute::List(vec![Attribute::Type(i64_ty), dict(i64_ty)])
        );
        assert_eq!(
            attribute.try_map_types(&mut |_| Err::<TypeRef, _>("stop")),
            Err("stop")
        );
    }

    #[test]
    fn attribute_maps_display_their_keys_and_strings_as_text() {
        let mut ctx = IrContext::new();
        let text = ctx.string_attr("hello");
        let nested = AttributeMap::from_entries(&mut ctx, [("inner", text.clone())]);
        let attrs = AttributeMap::from_entries(
            &mut ctx,
            [
                ("name", text),
                ("list", Attribute::List(vec![Attribute::Dict(nested)])),
                ("count", Attribute::Int(3)),
            ],
        );
        assert_eq!(
            attrs.display(&ctx).to_string(),
            r#"{"count": Int(3), "list": [{"inner": "hello"}], "name": "hello"}"#
        );
    }

    #[test]
    fn attribute_map_starts_at_exactly_the_entries_it_holds() {
        let mut ctx = IrContext::new();
        let mut attrs = AttributeMap::new();
        attrs.insert(&mut ctx, "only", Attribute::Unit);
        assert_eq!(attrs.0.capacity(), 1);

        let collected =
            AttributeMap::from_entries(&mut ctx, ["a", "b", "c"].map(|key| (key, Attribute::Unit)));
        assert_eq!(collected.0.capacity(), 3);

        let repeated =
            AttributeMap::from_entries(&mut ctx, (0..100).map(|_| ("same", Attribute::Unit)));
        assert_eq!(repeated.len(), 1);
        assert!(repeated.0.capacity() <= 4);
    }
}
