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
    /// String uniqued in the owning context's string pool.
    String(StringRef),
    Bytes(SmallVec<[u8; 16]>),
    Type(TypeRef),
    /// Reference to a symbol table definition, by its qualified name.
    SymbolRef(Symbol),
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
    /// Extract the inner `Symbol` if this is `Attribute::SymbolRef`.
    pub fn as_symbol_ref(&self) -> Option<Symbol> {
        match self {
            Attribute::SymbolRef(s) => Some(s.clone()),
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
    pub fn visit_symbol_refs(&self, f: &mut impl FnMut(Symbol)) {
        match self {
            Attribute::SymbolRef(symbol) => f(symbol.clone()),
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
            Attribute::Dict(dict) => Attribute::Dict(
                dict.iter()
                    .map(|(key, value)| Ok((key.clone(), value.try_map_types(f)?)))
                    .collect::<Result<_, _>>()?,
            ),
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
            Attribute::SymbolRef(sym) => sym.with_str(|s| s.len()) + 1,
            Attribute::Type(_) => 10, // rough estimate; actual depends on type
            Attribute::List(list) => {
                list.iter()
                    .map(|item| item.complexity(strings))
                    .sum::<usize>()
                    + list.len() * 2
            }
            Attribute::Dict(dict) => {
                dict.iter()
                    .map(|(key, value)| key.with_str(|s| s.len()) + 3 + value.complexity(strings))
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

impl From<Symbol> for Attribute {
    fn from(value: Symbol) -> Self {
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

/// A deterministic map of IR attributes with ergonomic symbol and string lookup.
///
/// Entries are kept in a vector sorted by key. Maps hold only a few entries, so
/// lookups scan with `Symbol` equality, which compares interned ids without
/// resolving strings; only inserting a new key orders it by `Symbol::cmp`.
/// Iteration, equality, and hashing follow key order, independent of the
/// order in which entries were inserted.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct AttributeMap(Vec<(Symbol, Attribute)>);

/// A key accepted by [`AttributeMap::get`].
pub trait AttributeKey {
    fn matches(&self, key: &Symbol) -> bool;
}

impl AttributeKey for Symbol {
    fn matches(&self, key: &Symbol) -> bool {
        self == key
    }
}

impl AttributeKey for &Symbol {
    fn matches(&self, key: &Symbol) -> bool {
        *self == key
    }
}

impl AttributeKey for &str {
    fn matches(&self, key: &Symbol) -> bool {
        key.as_str() == *self
    }
}

impl AttributeMap {
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    fn position(&self, key: impl AttributeKey) -> Option<usize> {
        self.0
            .iter()
            .position(|(existing, _)| key.matches(existing))
    }

    /// Return the attribute associated with a symbol or its text.
    pub fn get(&self, key: impl AttributeKey) -> Option<&Attribute> {
        let index = self.position(key)?;
        Some(&self.0[index].1)
    }

    pub fn get_mut(&mut self, key: impl AttributeKey) -> Option<&mut Attribute> {
        let index = self.position(key)?;
        Some(&mut self.0[index].1)
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

    pub fn get_string_ref(&self, key: impl AttributeKey) -> Option<StringRef> {
        self.get(key).and_then(Attribute::as_string_ref)
    }

    /// The text of a string attribute, resolved through the owning context.
    pub fn get_str<'a>(&self, ctx: &'a IrContext, key: impl AttributeKey) -> Option<&'a str> {
        self.get_string_ref(key).map(|s| ctx.str(s))
    }

    pub fn get_symbol_ref(&self, key: impl AttributeKey) -> Option<Symbol> {
        self.get(key).and_then(Attribute::as_symbol_ref)
    }

    pub fn get_type(&self, key: impl AttributeKey) -> Option<TypeRef> {
        self.get(key).and_then(Attribute::as_type)
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
        self.position(key).is_some()
    }

    /// Insert or replace an entry, returning the replaced value.
    pub fn insert(
        &mut self,
        key: impl Into<Symbol>,
        value: impl Into<Attribute>,
    ) -> Option<Attribute> {
        let key = key.into();
        let value = value.into();
        if let Some(index) = self.position(&key) {
            return Some(std::mem::replace(&mut self.0[index].1, value));
        }
        // Most maps hold one or two entries. A `Vec`'s first push reserves
        // four, so start a map at exactly one and let later inserts grow it.
        if self.0.capacity() == 0 {
            self.0.reserve_exact(1);
        }
        let index = self.0.partition_point(|(existing, _)| *existing < key);
        self.0.insert(index, (key, value));
        None
    }

    pub fn remove(&mut self, key: impl AttributeKey) -> Option<Attribute> {
        let index = self.position(key)?;
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
    pub fn visit_symbol_refs(&self, f: &mut impl FnMut(Symbol)) {
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

/// Later entries replace earlier ones with the same key.
impl FromIterator<(Symbol, Attribute)> for AttributeMap {
    fn from_iter<T: IntoIterator<Item = (Symbol, Attribute)>>(iter: T) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl Extend<(Symbol, Attribute)> for AttributeMap {
    fn extend<T: IntoIterator<Item = (Symbol, Attribute)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        // Repeated keys replace earlier entries, so the iterator length only
        // bounds the map's size from above. Reserve at most a small map
        // exactly and let inserts grow anything larger.
        if self.0.capacity() == 0 {
            self.0.reserve_exact(iter.size_hint().0.min(4));
        }
        for (key, value) in iter {
            self.insert(key, value);
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
    AttributeIter<'a>(std::slice::Iter<'a, (Symbol, Attribute)>)
        -> (&'a Symbol, &'a Attribute) = |(key, value)| (key, value)
);
attribute_iterator!(
    /// Entries of an [`AttributeMap`] in key order, with mutable values.
    AttributeIterMut<'a>(std::slice::IterMut<'a, (Symbol, Attribute)>)
        -> (&'a Symbol, &'a mut Attribute) = |(key, value)| (&*key, value)
);
attribute_iterator!(
    /// Keys of an [`AttributeMap`] in order.
    #[derive(Clone)]
    AttributeKeys<'a>(std::slice::Iter<'a, (Symbol, Attribute)>)
        -> &'a Symbol = |(key, _)| key
);
attribute_iterator!(
    /// Values of an [`AttributeMap`] in key order.
    #[derive(Clone)]
    AttributeValues<'a>(std::slice::Iter<'a, (Symbol, Attribute)>)
        -> &'a Attribute = |(_, value)| value
);
attribute_iterator!(
    /// Mutable values of an [`AttributeMap`] in key order.
    AttributeValuesMut<'a>(std::slice::IterMut<'a, (Symbol, Attribute)>)
        -> &'a mut Attribute = |(_, value)| value
);

/// Owned entries of an [`AttributeMap`] in key order.
pub type AttributeIntoIter = std::vec::IntoIter<(Symbol, Attribute)>;

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
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut AttributeMap {
    type Item = (&'a Symbol, &'a mut Attribute);
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
    pub fn param_attrs(&self, index: usize) -> &AttributeMap {
        self.attrs
            .get(PARAM_ATTRS_ATTR)
            .and_then(Attribute::as_list)
            .and_then(|entries| entries.get(index))
            .and_then(Attribute::as_dict)
            .unwrap_or(&EMPTY_ATTRIBUTE_MAP)
    }

    /// Each type parameter with its attributes.
    pub fn params_with_attrs(&self) -> impl Iterator<Item = (TypeRef, &AttributeMap)> {
        self.params
            .iter()
            .enumerate()
            .map(|(index, &ty)| (ty, self.param_attrs(index)))
    }

    /// Check that [`PARAM_ATTRS_ATTR`], if present, is canonical: one
    /// dictionary per type parameter, not all of them empty.
    pub fn validate_param_attrs(&self) -> Result<(), ParamAttrsError> {
        validate_param_attrs(&self.attrs, self.params.len())
    }
}

/// Check that [`PARAM_ATTRS_ATTR`] in `attrs`, if present, is canonical for a
/// type with `params` type parameters.
pub fn validate_param_attrs(attrs: &AttributeMap, params: usize) -> Result<(), ParamAttrsError> {
    let Some(value) = attrs.get(PARAM_ATTRS_ATTR) else {
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
    attrs: &mut AttributeMap,
    params: usize,
) -> Result<(), ParamAttrsError> {
    let all_empty = attrs
        .get(PARAM_ATTRS_ATTR)
        .and_then(Attribute::as_list)
        .is_some_and(|entries| {
            entries.len() == params
                && entries
                    .iter()
                    .all(|entry| entry.as_dict().is_some_and(AttributeMap::is_empty))
        });
    if all_empty {
        attrs.remove(PARAM_ATTRS_ATTR);
    }
    validate_param_attrs(attrs, params)
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
    attrs: AttributeMap,
}

impl TypeDataBuilder {
    pub fn new(dialect: impl Into<Symbol>, name: impl Into<Symbol>) -> Self {
        Self {
            dialect: dialect.into(),
            name: name.into(),
            params: SmallVec::new(),
            attrs: AttributeMap::new(),
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

    pub fn attr(mut self, key: impl Into<Symbol>, val: impl Into<Attribute>) -> Self {
        self.attrs.insert(key, val);
        self
    }

    /// Build the type data. Parameter attributes given with
    /// [`param_with_attrs`](Self::param_with_attrs) are stored in canonical
    /// [`PARAM_ATTRS_ATTR`] form and replace one set through
    /// [`attr`](Self::attr); an explicit all-empty value is dropped. A
    /// malformed explicit value is kept and rejected when the type is interned
    /// or validated.
    pub fn build(mut self) -> TypeData {
        let (params, param_attrs): (SmallVec<[TypeRef; 4]>, Vec<_>) =
            self.params.into_iter().unzip();
        if let Some(value) = param_attrs_attribute(param_attrs) {
            self.attrs.insert(PARAM_ATTRS_ATTR, value);
        } else {
            let _ = normalize_param_attrs(&mut self.attrs, params.len());
        }
        TypeData {
            dialect: self.dialect,
            name: self.name,
            params,
            attrs: self.attrs,
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
    pub fn is_dialect(&self, r: TypeRef, dialect: Symbol, name: Symbol) -> bool {
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
            size += key.with_str(|s| s.len()) + 3; // "key = "
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
///
/// `Symbol` carries a name from the global interner (such as a function's
/// qualified name) without an intermediate allocation. It is transitional
/// until symbols are owned by the context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StringArg {
    Ref(StringRef),
    Text(std::borrow::Cow<'static, str>),
    Symbol(crate::Symbol),
}

impl From<crate::Symbol> for StringArg {
    fn from(value: crate::Symbol) -> Self {
        StringArg::Symbol(value)
    }
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

impl Default for StringPool {
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
    fn parameter_attributes_are_canonical_and_part_of_identity() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let marked: AttributeMap = [(Symbol::new("k"), Attribute::SymbolRef(Symbol::new("v")))]
            .into_iter()
            .collect();

        let plain = ctx.intern_type(
            TypeDataBuilder::new("core", "tuple")
                .params([i32_ty, i32_ty])
                .build(),
        );
        let empty = ctx.intern_type(
            TypeDataBuilder::new("core", "tuple")
                .param_with_attrs(i32_ty, AttributeMap::new())
                .param(i32_ty)
                .build(),
        );
        let explicit_empty = ctx.intern_type(
            TypeDataBuilder::new("core", "tuple")
                .params([i32_ty, i32_ty])
                .attr(
                    PARAM_ATTRS_ATTR,
                    Attribute::List(vec![
                        Attribute::Dict(AttributeMap::new()),
                        Attribute::Dict(AttributeMap::new()),
                    ]),
                )
                .build(),
        );
        assert_eq!(plain, empty);
        assert_eq!(plain, explicit_empty);
        assert!(!ctx.get_type(plain).attrs.contains_key(PARAM_ATTRS_ATTR));

        let with_attrs = ctx.intern_type(
            TypeDataBuilder::new("core", "tuple")
                .param(i32_ty)
                .param_with_attrs(i32_ty, marked.clone())
                .build(),
        );
        assert_ne!(plain, with_attrs);
        let data = ctx.get_type(with_attrs);
        assert!(data.param_attrs(0).is_empty());
        assert_eq!(data.param_attrs(1), &marked);
        assert_eq!(
            data.params_with_attrs().collect::<Vec<_>>(),
            vec![(i32_ty, &AttributeMap::new()), (i32_ty, &marked)]
        );
        assert_eq!(data.validate_param_attrs(), Ok(()));
    }

    #[test]
    fn malformed_parameter_attributes_are_reported() {
        let dict = |entries: Vec<(&str, Attribute)>| {
            Attribute::Dict(
                entries
                    .into_iter()
                    .map(|(key, value)| (Symbol::from_dynamic(key), value))
                    .collect(),
            )
        };
        let attrs = |value| -> AttributeMap {
            [(Symbol::new(PARAM_ATTRS_ATTR), value)]
                .into_iter()
                .collect()
        };
        assert_eq!(
            validate_param_attrs(&attrs(Attribute::Unit), 1),
            Err(ParamAttrsError::NotList)
        );
        assert_eq!(
            validate_param_attrs(
                &attrs(Attribute::List(vec![dict(vec![("k", Attribute::Unit)])])),
                2
            ),
            Err(ParamAttrsError::LengthMismatch {
                params: 2,
                attrs: 1
            })
        );
        assert_eq!(
            validate_param_attrs(
                &attrs(Attribute::List(vec![
                    dict(vec![("k", Attribute::Unit)]),
                    Attribute::Unit
                ])),
                2
            ),
            Err(ParamAttrsError::NotDict(1))
        );
        assert_eq!(
            validate_param_attrs(&attrs(Attribute::List(vec![dict(vec![])])), 1),
            Err(ParamAttrsError::AllEmpty)
        );
        assert_eq!(validate_param_attrs(&AttributeMap::new(), 3), Ok(()));

        let mut wrong_length = attrs(Attribute::List(vec![dict(vec![])]));
        assert_eq!(
            normalize_param_attrs(&mut wrong_length, 2),
            Err(ParamAttrsError::LengthMismatch {
                params: 2,
                attrs: 1
            })
        );
        assert!(wrong_length.contains_key(PARAM_ATTRS_ATTR));
        let mut all_empty = attrs(Attribute::List(vec![dict(vec![]), dict(vec![])]));
        assert_eq!(normalize_param_attrs(&mut all_empty, 2), Ok(()));
        assert!(all_empty.is_empty());
    }

    #[test]
    fn symbol_refs_are_visited_through_lists_and_dicts() {
        let mut ctx = IrContext::new();
        let reference = |name| Attribute::SymbolRef(Symbol::new(name));
        let mut attrs = AttributeMap::new();
        attrs.insert("callee", reference("direct"));
        attrs.insert("name", ctx.string_attr("not_a_reference"));
        attrs.insert(
            "table",
            Attribute::List(vec![
                reference("first"),
                Attribute::Dict(
                    [(Symbol::new("target"), reference("nested"))]
                        .into_iter()
                        .collect(),
                ),
            ]),
        );

        let mut visited = Vec::new();
        attrs.visit_symbol_refs(&mut |symbol| visited.push(symbol));
        assert_eq!(
            visited,
            ["direct", "first", "nested"].map(Symbol::new).to_vec()
        );
    }

    #[test]
    fn nested_types_are_visited_and_mapped_through_lists_and_dicts() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
        let dict = |ty| {
            Attribute::Dict(
                [
                    (Symbol::new("ty"), Attribute::Type(ty)),
                    (
                        Symbol::new("tag"),
                        Attribute::SymbolRef(Symbol::new("keep")),
                    ),
                ]
                .into_iter()
                .collect(),
            )
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
    fn attribute_map_accepts_string_and_symbol_keys_without_interning_misses() {
        fn get_by_symbol<'a>(attrs: &'a AttributeMap, key: &Symbol) -> Option<&'a Attribute> {
            attrs.get(key)
        }

        let mut attrs = AttributeMap::new();
        let answer = Symbol::new("answer");
        attrs.insert(answer.clone(), Attribute::Int(42));

        assert_eq!(attrs.get("answer"), Some(&Attribute::Int(42)));
        assert_eq!(attrs.get(&answer), Some(&Attribute::Int(42)));
        assert_eq!(get_by_symbol(&attrs, &answer), Some(&Attribute::Int(42)));
        assert!(attrs.contains_key("answer"));
        assert_eq!(
            attrs.keys().cloned().collect::<Vec<_>>(),
            vec![answer.clone()]
        );

        let missing = "__trunk_ir_attribute_map_missing_key__";
        assert_eq!(attrs.get(missing), None);
        assert!(!attrs.contains_key(missing));

        assert_eq!(attrs.remove(answer), Some(Attribute::Int(42)));
        assert!(attrs.is_empty());
    }

    #[test]
    fn attribute_map_is_ordered_by_key_regardless_of_insertion_order() {
        use std::hash::{BuildHasher, RandomState};

        let entries = [
            (Symbol::new("zeta"), Attribute::Int(1)),
            (Symbol::new("alpha"), Attribute::Int(2)),
            (Symbol::new("mid"), Attribute::Int(3)),
        ];
        let forward: AttributeMap = entries.iter().cloned().collect();
        let mut backward = AttributeMap::new();
        for (key, value) in entries.iter().rev().cloned() {
            assert_eq!(backward.insert(key, value), None);
        }

        assert_eq!(forward, backward);
        let hasher = RandomState::new();
        assert_eq!(hasher.hash_one(&forward), hasher.hash_one(&backward));
        let keys = |map: &AttributeMap| map.keys().map(|key| key.to_string()).collect::<Vec<_>>();
        assert_eq!(keys(&forward), ["alpha", "mid", "zeta"]);
        assert_eq!(keys(&backward), ["alpha", "mid", "zeta"]);
        assert_eq!(
            format!("{forward:?}"),
            r#"{Symbol("alpha"): Int(2), Symbol("mid"): Int(3), Symbol("zeta"): Int(1)}"#
        );
    }

    #[test]
    fn attribute_map_starts_at_exactly_the_entries_it_holds() {
        let mut attrs = AttributeMap::new();
        attrs.insert("only", Attribute::Unit);
        assert_eq!(attrs.0.capacity(), 1);

        let collected: AttributeMap = ["a", "b", "c"]
            .into_iter()
            .map(|key| (Symbol::new(key), Attribute::Unit))
            .collect();
        assert_eq!(collected.0.capacity(), 3);

        let repeated: AttributeMap = (0..100)
            .map(|_| (Symbol::new("same"), Attribute::Unit))
            .collect();
        assert_eq!(repeated.len(), 1);
        assert!(repeated.0.capacity() <= 4);
    }

    #[test]
    fn attribute_map_later_entries_replace_earlier_ones() {
        let key = Symbol::new("key");
        let mut attrs = AttributeMap::new();
        assert_eq!(attrs.insert(key.clone(), Attribute::Int(1)), None);
        assert_eq!(
            attrs.insert(key.clone(), Attribute::Int(2)),
            Some(Attribute::Int(1))
        );
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs.get(&key), Some(&Attribute::Int(2)));

        let collected: AttributeMap = [
            (key.clone(), Attribute::Int(1)),
            (Symbol::new("other"), Attribute::Unit),
            (key.clone(), Attribute::Int(3)),
        ]
        .into_iter()
        .collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected.get(&key), Some(&Attribute::Int(3)));

        let mut extended = collected.clone();
        extended.extend([(key.clone(), Attribute::Int(4))]);
        assert_eq!(extended.get(key), Some(&Attribute::Int(4)));
        assert_eq!(extended.len(), 2);
    }

    #[test]
    fn attribute_map_typed_getters_handle_absence_and_integer_range() {
        let mut ctx = IrContext::new();
        let mut attrs = AttributeMap::new();
        attrs.insert("count", Attribute::Int(i64::MAX as i128));
        attrs.insert("byte", Attribute::Int(u8::MAX as i128));
        attrs.insert("enabled", Attribute::Bool(true));
        attrs.insert("name", ctx.string_attr("tribute"));
        attrs.insert("symbol_name", Symbol::new("tribute"));

        assert_eq!(attrs.get_i64("count"), Ok(Some(i64::MAX)));
        assert_eq!(attrs.get_i128("count"), Some(i64::MAX as i128));
        assert_eq!(attrs.get_u8("byte"), Ok(Some(u8::MAX)));
        assert_eq!(attrs.get_bool("enabled"), Some(true));
        assert_eq!(attrs.get_str(&ctx, "name"), Some("tribute"));
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

        assert_eq!(attrs.get_str(&ctx, "name"), Some("tribute"));
        assert_eq!(attrs.get_str(&ctx, "symbol_name"), None);
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
        let i32_ref = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let tup = crate::dialect::core::tuple(&mut ctx, [i32_ref, i32_ref]);
        let r1 = tup.as_type_ref();
        // Interning the same tuple again should return the same ref
        let r2 = crate::dialect::core::tuple(&mut ctx, [i32_ref, i32_ref]).as_type_ref();
        assert_eq!(r1, r2);

        let data = ctx.get_type(r1);
        assert_eq!(data.params.len(), 2);
        assert_eq!(data.params[0], i32_ref);
    }

    #[test]
    fn type_interner_keeps_identities_across_index_growth() {
        let mut interner = TypeInterner::new();
        let data = |index: usize| {
            TypeDataBuilder::new("test", "numbered")
                .attr("index", Attribute::Int(index as i128))
                .build()
        };
        let refs: Vec<_> = (0..1000)
            .map(|index| interner.intern(data(index)))
            .collect();
        for (index, &r) in refs.iter().enumerate() {
            assert_eq!(interner.lookup(&data(index)), Some(r));
            assert_eq!(interner.intern(data(index)), r);
            assert_eq!(interner.get(r), &data(index));
        }
        assert_eq!(interner.iter().count(), refs.len());
        assert_eq!(interner.lookup(&data(1000)), None);
    }

    #[test]
    fn path_interner_looks_up_borrowed_strings() {
        let mut interner = PathInterner::new();
        let r = interner.intern("file:///a.trb");
        assert_eq!(interner.lookup("file:///a.trb"), Some(r));
        assert_eq!(interner.lookup("file:///b.trb"), None);
    }

    #[test]
    fn path_interner_dedup() {
        let mut interner = PathInterner::new();
        let r1 = interner.intern("file:///test.trb");
        let r2 = interner.intern("file:///test.trb");
        assert_eq!(r1, r2, "same path must yield same PathRef");
    }

    #[test]
    fn path_interner_distinct() {
        let mut interner = PathInterner::new();
        let r1 = interner.intern("file:///a.trb");
        let r2 = interner.intern("file:///b.trb");
        assert_ne!(r1, r2);
        assert_eq!(interner.get(r1), "file:///a.trb");
        assert_eq!(interner.get(r2), "file:///b.trb");
    }
}
