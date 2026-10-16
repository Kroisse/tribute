//! Operation and type names.
//!
//! A name is written `dialect.name`, split at its first `.`. The IR stores a
//! [`Name`] handle rather than the text. A registered name points to the
//! static definition `#[dialect]` emits for the declaration, so it is valid in
//! every context and leads to the definition without a lookup. Any other name
//! keeps its dialect and name in the context's [`StringPool`] and is valid
//! only in that context. Text becomes a handle only through
//! [`IrContext::intern_op_name`](crate::IrContext::intern_op_name) or
//! [`IrContext::intern_type_name`](crate::IrContext::intern_type_name), which
//! look for a registered definition first, so one name always has one handle.

use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};

use rustc_hash::FxHashMap as HashMap;

use crate::types::{StringPool, StringRef};

/// The `dialect.name` text of a registered definition.
#[derive(Clone, Copy)]
pub struct QualifiedName {
    text: &'static str,
    dialect_len: usize,
    hash: u64,
}

impl QualifiedName {
    /// Split `text` at its first `.`. Panics, at compile time in a constant,
    /// if either side is empty.
    pub const fn new(text: &'static str) -> Self {
        let bytes = text.as_bytes();
        let mut dialect_len = 0;
        while dialect_len < bytes.len() && bytes[dialect_len] != b'.' {
            dialect_len += 1;
        }
        assert!(
            dialect_len > 0 && dialect_len + 1 < bytes.len(),
            "a qualified name is written `dialect.name`"
        );
        Self {
            text,
            dialect_len,
            hash: fnv1a(bytes),
        }
    }

    /// The whole `dialect.name` text.
    pub fn as_str(&self) -> &'static str {
        self.text
    }

    pub fn dialect(&self) -> &'static str {
        &self.text[..self.dialect_len]
    }

    /// The name without its dialect.
    pub fn strip_dialect(&self) -> &'static str {
        &self.text[self.dialect_len + 1..]
    }
}

impl fmt::Debug for QualifiedName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.text, f)
    }
}

/// A hash of the name's text, so that hashing a registered handle does not
/// depend on where its definition is loaded.
const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        i += 1;
    }
    hash
}

/// Split `dialect.name` text at its first `.`, or `None` if either side is
/// empty.
pub fn split_qualified(text: &str) -> Option<(&str, &str)> {
    let (dialect, name) = text.split_once('.')?;
    (!dialect.is_empty() && !name.is_empty()).then_some((dialect, name))
}

/// A definition that `#[dialect]` registers under a name.
pub trait Definition: Sync + 'static {
    fn qualified_name(&self) -> &QualifiedName;

    /// The registered definition written `dialect.name`.
    fn lookup(name: &str) -> Option<&'static Self>;
}

/// Registered definitions of one kind by their `dialect.name` text.
pub(crate) struct DefinitionTable<D: 'static>(HashMap<&'static str, &'static D>);

impl<D: Definition> DefinitionTable<D> {
    pub(crate) fn collect(kind: &str, defs: impl IntoIterator<Item = &'static D>) -> Self {
        let mut map = HashMap::default();
        for def in defs {
            let name = def.qualified_name();
            if let Some(previous) = map.insert(name.as_str(), def) {
                panic!(
                    "{kind} {} is registered twice ({:p} and {def:p})",
                    name.as_str(),
                    previous,
                );
            }
        }
        Self(map)
    }

    pub(crate) fn get(&self, name: &str) -> Option<&'static D> {
        self.0.get(name).copied()
    }
}

/// An operation or type name: a registered definition or context strings.
pub struct Name<D: 'static>(Repr<D>);

enum Repr<D: 'static> {
    Registered(&'static D),
    Unregistered { dialect: StringRef, name: StringRef },
}

impl<D: 'static> Clone for Name<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: 'static> Copy for Name<D> {}

impl<D: 'static> Clone for Repr<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: 'static> Copy for Repr<D> {}

impl<D: Definition> Name<D> {
    /// The handle of a registered definition.
    pub const fn registered(def: &'static D) -> Self {
        Self(Repr::Registered(def))
    }

    /// The handle of the name written `text`: its registered definition, or
    /// strings interned in `strings`.
    ///
    /// # Panics
    ///
    /// If `text` has no `.` separating a non-empty dialect and name.
    pub(crate) fn intern(strings: &mut StringPool, text: &str) -> Self {
        if let Some(def) = D::lookup(text) {
            return Self::registered(def);
        }
        let (dialect, name) = split_or_panic(text);
        Self(Repr::Unregistered {
            dialect: strings.intern(dialect),
            name: strings.intern(name),
        })
    }

    /// The handle of the name written `text` if it is registered or its
    /// strings are already in `strings`, without adding to the pool.
    pub(crate) fn lookup(strings: &StringPool, text: &str) -> Option<Self> {
        if let Some(def) = D::lookup(text) {
            return Some(Self::registered(def));
        }
        let (dialect, name) = split_or_panic(text);
        Some(Self(Repr::Unregistered {
            dialect: strings.lookup(dialect)?,
            name: strings.lookup(name)?,
        }))
    }

    /// The registered definition, or `None` for an unregistered name.
    pub fn def(self) -> Option<&'static D> {
        match self.0 {
            Repr::Registered(def) => Some(def),
            Repr::Unregistered { .. } => None,
        }
    }

    pub fn is_registered(self) -> bool {
        matches!(self.0, Repr::Registered(_))
    }

    pub fn dialect(self, ctx: &impl AsRef<StringPool>) -> &str {
        match self.0 {
            Repr::Registered(def) => def.qualified_name().dialect(),
            Repr::Unregistered { dialect, .. } => ctx.as_ref().get(dialect),
        }
    }

    /// The name without its dialect.
    pub fn strip_dialect(self, ctx: &impl AsRef<StringPool>) -> &str {
        match self.0 {
            Repr::Registered(def) => def.qualified_name().strip_dialect(),
            Repr::Unregistered { name, .. } => ctx.as_ref().get(name),
        }
    }

    /// The whole `dialect.name` text, borrowed when the name is registered.
    pub fn to_text(self, ctx: &impl AsRef<StringPool>) -> Cow<'static, str> {
        match self.0 {
            Repr::Registered(def) => Cow::Borrowed(def.qualified_name().as_str()),
            Repr::Unregistered { .. } => Cow::Owned(self.display(ctx).to_string()),
        }
    }

    /// Whether this is the name written `text`.
    pub fn is(self, ctx: &impl AsRef<StringPool>, text: &str) -> bool {
        match self.0 {
            Repr::Registered(def) => def.qualified_name().as_str() == text,
            Repr::Unregistered { dialect, name } => split_qualified(text).is_some_and(|(d, n)| {
                let strings = ctx.as_ref();
                strings.get(dialect) == d && strings.get(name) == n
            }),
        }
    }

    /// Display the name as `dialect.name`.
    pub fn display<'a>(self, ctx: &'a impl AsRef<StringPool>) -> impl fmt::Display + 'a {
        let strings = ctx.as_ref();
        DisplayName {
            dialect: self.dialect(strings),
            name: self.strip_dialect(strings),
        }
    }
}

struct DisplayName<'a> {
    dialect: &'a str,
    name: &'a str,
}

impl fmt::Display for DisplayName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.dialect, self.name)
    }
}

impl<D: Definition> PartialEq for Name<D> {
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (Repr::Registered(a), Repr::Registered(b)) => std::ptr::eq(a, b),
            (
                Repr::Unregistered { dialect, name },
                Repr::Unregistered {
                    dialect: other_dialect,
                    name: other_name,
                },
            ) => dialect == other_dialect && name == other_name,
            _ => false,
        }
    }
}

impl<D: Definition> Eq for Name<D> {}

impl<D: Definition> PartialEq<&'static D> for Name<D> {
    fn eq(&self, other: &&'static D) -> bool {
        self.def().is_some_and(|def| std::ptr::eq(def, *other))
    }
}

impl<D: Definition> Hash for Name<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.0 {
            Repr::Registered(def) => state.write_u64(def.qualified_name().hash),
            Repr::Unregistered { dialect, name } => {
                dialect.hash(state);
                name.hash(state);
            }
        }
    }
}

impl<D: Definition> fmt::Debug for Name<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Repr::Registered(def) => fmt::Debug::fmt(def.qualified_name(), f),
            Repr::Unregistered { dialect, name } => f
                .debug_struct("Unregistered")
                .field("dialect", &dialect)
                .field("name", &name)
                .finish(),
        }
    }
}

/// A name given to a builder: a handle, or `dialect.name` text that the
/// builder interns when it builds.
pub enum NameArg<D: 'static> {
    Name(Name<D>),
    Text(Cow<'static, str>),
}

impl<D: Definition> NameArg<D> {
    pub(crate) fn resolve(self, strings: &mut StringPool) -> Name<D> {
        match self {
            NameArg::Name(name) => name,
            NameArg::Text(text) => Name::intern(strings, &text),
        }
    }
}

fn split_or_panic(text: &str) -> (&str, &str) {
    split_qualified(text).unwrap_or_else(|| panic!("`{text}` is not a qualified `dialect.name`"))
}

impl<D: Definition> From<Name<D>> for NameArg<D> {
    fn from(name: Name<D>) -> Self {
        NameArg::Name(name)
    }
}

impl<D: Definition> From<&'static D> for NameArg<D> {
    fn from(def: &'static D) -> Self {
        NameArg::Name(Name::registered(def))
    }
}

impl<D: Definition> From<&'static str> for NameArg<D> {
    fn from(text: &'static str) -> Self {
        NameArg::Text(Cow::Borrowed(text))
    }
}

impl<D: Definition> From<String> for NameArg<D> {
    fn from(text: String) -> Self {
        NameArg::Text(Cow::Owned(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_names_split_at_the_first_dot() {
        const NAME: QualifiedName = QualifiedName::new("llvm.intr.memcpy");
        assert_eq!(NAME.dialect(), "llvm");
        assert_eq!(NAME.strip_dialect(), "intr.memcpy");
        assert_eq!(split_qualified("a.b.c"), Some(("a", "b.c")));
        assert_eq!(split_qualified("a."), None);
        assert_eq!(split_qualified("plain"), None);
    }
}
