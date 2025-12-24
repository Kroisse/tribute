//! Qualified names for identifiers in TrunkIR.
//!
//! A qualified name is a non-empty sequence of symbols representing a fully qualified identifier,
//! such as `std::intrinsics::wasi::fd_write` or `List::map`.

use crate::SymbolVec;
use crate::ir::Symbol;
use smallvec::SmallVec;

/// A fully qualified name consisting of path segments.
///
/// Examples: `std::intrinsics::wasi::preview1::fd_write`, `List::map`
///
/// Used for function callees, type references, and other qualified identifiers.
///
/// This is a non-empty structure: every QualifiedName has at least a name.
/// The parent path can be empty (for simple names like `foo`) or contain multiple segments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QualifiedName {
    /// Parent path segments (can be empty for simple names).
    /// Uses SymbolVec to inline up to 4 parent segments.
    parent: SymbolVec,
    /// The final name segment (guaranteed to exist).
    name: Symbol,
}

impl QualifiedName {
    /// Create a new qualified name with the given parent path and name.
    pub fn new(parent: impl Into<SymbolVec>, name: Symbol) -> Self {
        Self {
            parent: parent.into(),
            name,
        }
    }

    /// Create a qualified name from string segments.
    /// Returns `None` if the iterator is empty.
    pub fn from_strs(segments: impl IntoIterator<Item = &'static str>) -> Option<Self> {
        segments.into_iter().map(Symbol::new).collect()
    }

    /// Create a simple (single-segment) qualified name.
    pub fn simple(name: Symbol) -> Self {
        Self {
            parent: SmallVec::new(),
            name,
        }
    }

    /// Get all segments of this qualified name (parent + name).
    pub fn to_segments(&self) -> SmallVec<[Symbol; 6]> {
        let mut result = SmallVec::with_capacity(self.parent.len() + 1);
        result.extend_from_slice(&self.parent);
        result.push(self.name);
        result
    }

    /// Get the parent path as a slice of symbols.
    pub fn as_parent(&self) -> &[Symbol] {
        &self.parent
    }

    /// Get the parent as a QualifiedName, if it exists.
    /// Returns `None` for simple (single-segment) names.
    pub fn to_parent(&self) -> Option<QualifiedName> {
        QualifiedName::try_from(&self.parent[..]).ok()
    }

    /// Get the last segment (the simple name).
    /// This is guaranteed to exist in a non-empty QualifiedName.
    pub fn name(&self) -> Symbol {
        self.name
    }

    /// Check if this is a simple (single-segment) name.
    pub fn is_simple(&self) -> bool {
        self.parent.is_empty()
    }

    /// Get the path relative to a base path.
    ///
    /// Returns `Some` if this path starts with `base`, containing the remaining segments.
    /// Returns `None` if this path does not start with `base`.
    ///
    /// # Example
    /// ```
    /// use trunk_ir::{QualifiedName, Symbol};
    ///
    /// let full = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
    /// let base = QualifiedName::from_strs(["std", "intrinsics", "wasi"]).unwrap();
    ///
    /// let relative = full.relative(&base).unwrap();
    /// assert!(relative.is_simple());
    /// assert_eq!(relative.name(), "fd_write");
    /// ```
    pub fn relative(&self, base: &QualifiedName) -> Option<QualifiedName> {
        let base_len = base.parent.len() + 1;
        let self_len = self.parent.len() + 1;

        // Must be strictly longer to have remaining segments
        if self_len <= base_len {
            return None;
        }

        // Check if our parent starts with base's parent
        if !self.parent.starts_with(&base.parent) {
            return None;
        }

        // Since self_len > base_len, we know base.parent.len() < self.parent.len()
        // base.name should match self.parent[base.parent.len()]
        if self.parent[base.parent.len()] != base.name {
            return None;
        }

        // Extract remaining segments: parent[base_len..] + name
        Some(QualifiedName::new(&self.parent[base_len..], self.name))
    }

    /// Check if this path starts with the given base path.
    pub fn starts_with(&self, base: &QualifiedName) -> bool {
        // First check if our parent starts with base's parent
        if !self.parent.starts_with(&base.parent) {
            return false;
        }

        // Two cases:
        // 1. base.parent is shorter than our parent: base.name matches self.parent[base.parent.len()]
        // 2. base.parent equals our parent: base.name matches our name
        if base.parent.len() < self.parent.len() {
            self.parent[base.parent.len()] == base.name
        } else if base.parent.len() == self.parent.len() {
            self.name == base.name
        } else {
            // base.parent is longer than our parent → can't start with
            false
        }
    }

    /// Get the number of segments (parent + name).
    ///
    /// This is always at least 1, since QualifiedName is non-empty by design.
    pub fn len(&self) -> usize {
        self.parent.len() + 1
    }

    /// Returns an iterator over all segments (parent + name).
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// Join this qualified name with another, creating a new qualified name.
    ///
    /// # Example
    /// ```
    /// # use trunk_ir::qualified_name::QualifiedName;
    /// # use trunk_ir::ir::Symbol;
    /// let base = QualifiedName::from_strs(["std", "io"]).unwrap();
    /// let suffix = QualifiedName::from_strs(["Reader", "new"]).unwrap();
    /// let full = base.join(&suffix);
    /// assert_eq!(full.to_string(), "std::io::Reader::new");
    /// ```
    pub fn join(&self, other: &QualifiedName) -> QualifiedName {
        QualifiedName::new(
            self.iter()
                .chain(other.parent.iter().copied())
                .collect::<SymbolVec>(),
            other.name,
        )
    }
}

impl std::fmt::Display for QualifiedName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, seg) in self.parent.iter().enumerate() {
            if i > 0 {
                write!(f, "::")?;
            }
            write!(f, "{seg}")?;
        }
        if !self.parent.is_empty() {
            write!(f, "::")?;
        }
        write!(f, "{}", self.name)
    }
}

impl IntoIterator for QualifiedName {
    type Item = Symbol;
    type IntoIter = std::iter::Chain<smallvec::IntoIter<[Symbol; 4]>, std::iter::Once<Symbol>>;

    fn into_iter(self) -> Self::IntoIter {
        self.parent.into_iter().chain(std::iter::once(self.name))
    }
}

impl<'a> IntoIterator for &'a QualifiedName {
    type Item = Symbol;
    type IntoIter =
        std::iter::Chain<std::iter::Copied<std::slice::Iter<'a, Symbol>>, std::iter::Once<Symbol>>;

    fn into_iter(self) -> Self::IntoIter {
        self.parent
            .iter()
            .copied()
            .chain(std::iter::once(self.name))
    }
}

impl std::iter::Extend<Symbol> for QualifiedName {
    fn extend<T: IntoIterator<Item = Symbol>>(&mut self, iter: T) {
        // Move current name to parent
        self.parent.push(self.name);

        // Extend parent with all symbols from iterator
        self.parent.extend(iter);

        // Pop the last element as new name (guaranteed non-empty)
        self.name = self
            .parent
            .pop()
            .expect("extend maintains non-empty invariant");

        // Shrink to fit to avoid wasting memory
        self.parent.shrink_to_fit();
    }
}

impl std::iter::FromIterator<Symbol> for Option<QualifiedName> {
    fn from_iter<T: IntoIterator<Item = Symbol>>(iter: T) -> Self {
        let mut parent = SymbolVec::from_iter(iter);
        let name = parent.pop()?;
        parent.shrink_to_fit();
        Some(QualifiedName::new(parent, name))
    }
}

impl From<Symbol> for QualifiedName {
    fn from(symbol: Symbol) -> Self {
        QualifiedName::simple(symbol)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyError;

impl std::fmt::Display for EmptyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot create QualifiedName from empty collection")
    }
}

impl std::error::Error for EmptyError {}

impl<'a> TryFrom<&'a [Symbol]> for QualifiedName {
    type Error = EmptyError;

    fn try_from(segments: &'a [Symbol]) -> Result<Self, Self::Error> {
        let (name, parent) = segments.split_last().ok_or(EmptyError)?;
        Ok(QualifiedName::new(SymbolVec::from_slice(parent), *name))
    }
}

impl TryFrom<Vec<Symbol>> for QualifiedName {
    type Error = EmptyError;

    fn try_from(mut segments: Vec<Symbol>) -> Result<Self, Self::Error> {
        let name = segments.pop().ok_or(EmptyError)?;
        Ok(QualifiedName::new(SmallVec::from_vec(segments), name))
    }
}

impl From<&'static str> for QualifiedName {
    fn from(text: &'static str) -> Self {
        QualifiedName::simple(Symbol::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualified_name_size() {
        // Ensure QualifiedName stays at 32 bytes
        // This is a regression test to prevent accidental size increases
        assert_eq!(
            std::mem::size_of::<QualifiedName>(),
            32,
            "QualifiedName size changed! Expected 32 bytes (SmallVec<[Symbol; 4]> + Symbol + padding)"
        );
    }

    #[test]
    fn test_simple_name() {
        let name = QualifiedName::simple(Symbol::new("foo"));
        assert!(name.is_simple());
        assert_eq!(name.name(), "foo");
        assert_eq!(name.len(), 1);
        assert_eq!(name.to_string(), "foo");
    }

    #[test]
    fn test_qualified_name() {
        let name = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
        assert!(!name.is_simple());
        assert_eq!(name.name(), "fd_write");
        assert_eq!(name.len(), 4);
        assert_eq!(name.to_string(), "std::intrinsics::wasi::fd_write");
    }

    #[test]
    fn test_relative() {
        let full = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
        let base = QualifiedName::from_strs(["std", "intrinsics", "wasi"]).unwrap();

        let relative = full.relative(&base).unwrap();
        assert!(relative.is_simple());
        assert_eq!(relative.name(), "fd_write");
    }

    #[test]
    fn test_relative_multi_segment() {
        let full = QualifiedName::from_strs(["std", "intrinsics", "wasi", "preview1", "fd_write"])
            .unwrap();
        let base = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();

        let relative = full.relative(&base).unwrap();
        assert_eq!(relative.len(), 3);
        assert_eq!(relative.to_string(), "wasi::preview1::fd_write");
    }

    #[test]
    fn test_relative_no_match() {
        let full = QualifiedName::from_strs(["std", "intrinsics", "wasi"]).unwrap();
        let base = QualifiedName::from_strs(["core", "intrinsics"]).unwrap();

        assert!(full.relative(&base).is_none());
    }

    #[test]
    fn test_relative_same_length() {
        let full = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();
        let base = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();

        // Same length should return None (no remaining segments)
        assert!(full.relative(&base).is_none());
    }

    #[test]
    fn test_starts_with() {
        let name = QualifiedName::from_strs(["std", "intrinsics", "wasi", "fd_write"]).unwrap();
        let prefix = QualifiedName::from_strs(["std", "intrinsics"]).unwrap();
        let other = QualifiedName::from_strs(["core", "intrinsics"]).unwrap();

        assert!(name.starts_with(&prefix));
        assert!(!name.starts_with(&other));
    }

    #[test]
    fn test_from_symbol() {
        let sym = Symbol::new("foo");
        let name: QualifiedName = sym.into();
        assert!(name.is_simple());
        assert_eq!(name.name(), "foo");
    }

    #[test]
    fn test_extend() {
        let mut name = QualifiedName::from_strs(["std", "io"]).unwrap();
        name.extend([Symbol::new("Read"), Symbol::new("read")]);
        assert_eq!(name.to_string(), "std::io::Read::read");

        // Extending with empty iterator should be a no-op
        let mut name2 = QualifiedName::simple(Symbol::new("foo"));
        name2.extend(std::iter::empty());
        assert_eq!(name2.to_string(), "foo");
    }

    #[test]
    fn test_join() {
        let base = QualifiedName::from_strs(["std", "io"]).unwrap();
        let suffix = QualifiedName::from_strs(["Read", "read"]).unwrap();
        let joined = base.join(&suffix);
        assert_eq!(joined.to_string(), "std::io::Read::read");
    }

    #[test]
    fn test_try_from_vec() {
        let symbols = vec![Symbol::new("std"), Symbol::new("io"), Symbol::new("Read")];
        let name = QualifiedName::try_from(symbols).unwrap();
        assert_eq!(name.to_string(), "std::io::Read");
    }

    #[test]
    fn test_try_from_slice() {
        let symbols = [Symbol::new("std"), Symbol::new("io")];
        let name = QualifiedName::try_from(&symbols[..]).unwrap();
        assert_eq!(name.to_string(), "std::io");
    }

    #[test]
    fn test_try_from_empty_vec_fails() {
        let result = QualifiedName::try_from(Vec::<Symbol>::new());
        assert_eq!(result, Err(EmptyError));
    }

    #[test]
    fn test_try_from_empty_slice_fails() {
        let result = QualifiedName::try_from(&[][..]);
        assert_eq!(result, Err(EmptyError));
    }

    #[test]
    fn test_from_iterator_for_option() {
        let symbols = vec![Symbol::new("std"), Symbol::new("io"), Symbol::new("Read")];
        let qn: Option<QualifiedName> = symbols.into_iter().collect();
        assert_eq!(qn.unwrap().to_string(), "std::io::Read");

        // Empty iterator produces None
        let empty: Option<QualifiedName> = std::iter::empty().collect();
        assert!(empty.is_none());
    }

    #[test]
    fn test_from_str() {
        let name: QualifiedName = "bar".into();
        assert!(name.is_simple());
        assert_eq!(name.name(), "bar");
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // Generate arbitrary valid Symbol identifiers
        fn arb_symbol() -> impl Strategy<Value = Symbol> {
            "[a-z][a-z0-9_]{0,15}".prop_map(|s| Symbol::from_dynamic(&s))
        }

        // Generate arbitrary QualifiedName with specified number of segments
        fn arb_qualified_name_with_len(
            len: impl Into<prop::collection::SizeRange>,
        ) -> impl Strategy<Value = QualifiedName> {
            prop::collection::vec(arb_symbol(), len).prop_map(|segments| {
                QualifiedName::try_from(segments).expect("non-empty by construction")
            })
        }

        // Generate arbitrary QualifiedName with 1-8 segments
        fn arb_qualified_name() -> impl Strategy<Value = QualifiedName> {
            arb_qualified_name_with_len(1..=8)
        }

        proptest! {
            #[test]
            fn prop_never_empty(qn in arb_qualified_name()) {
                prop_assert!(qn.len() >= 1);
                prop_assert!(!qn.to_segments().is_empty());
            }

            #[test]
            fn prop_name_equals_last_segment(qn in arb_qualified_name()) {
                let segments = qn.to_segments();
                prop_assert_eq!(qn.name(), *segments.last().unwrap());
            }

            #[test]
            fn prop_segments_roundtrip(qn in arb_qualified_name()) {
                let segments = qn.to_segments();
                let reconstructed = segments.into_iter().collect();
                prop_assert_eq!(Some(qn), reconstructed);
            }

            #[test]
            fn prop_simple_iff_one_segment(qn in arb_qualified_name()) {
                prop_assert_eq!(qn.is_simple(), qn.len() == 1);
            }

            #[test]
            fn prop_parent_length(qn in arb_qualified_name()) {
                prop_assert_eq!(qn.as_parent().len(), qn.len() - 1);
            }

            #[test]
            fn prop_display_contains_colons(qn in arb_qualified_name()) {
                let display = qn.to_string();
                let colon_count = display.matches("::").count();
                prop_assert_eq!(colon_count, qn.len() - 1);
            }

            #[test]
            fn prop_starts_with_reflexive(qn in arb_qualified_name()) {
                prop_assert!(qn.starts_with(&qn));
            }

            #[test]
            fn prop_starts_with_transitive(
                c in arb_qualified_name(),
                b_extra in arb_qualified_name(),
                a_extra in arb_qualified_name(),
            ) {
                // Build a prefix chain: c ⊆ b ⊆ a
                let b = c.join(&b_extra);
                let a = b.join(&a_extra);

                prop_assert!(a.starts_with(&b));
                prop_assert!(b.starts_with(&c));
                prop_assert!(a.starts_with(&c));
            }

            #[test]
            fn prop_relative_some_with_proper_prefix(
                (a, k) in arb_qualified_name_with_len(2..=8)
                    .prop_flat_map(|a| {
                        let a_len = a.len();
                        (Just(a), 1..a_len)
                    })
            ) {
                // b is a proper prefix of a (k < a.len())
                let b = a.iter().take(k).collect::<Option<_>>().unwrap();
                let rel = a.relative(&b);
                prop_assert!(rel.is_some());
                prop_assert_eq!(rel.unwrap().len(), a.len() - b.len());
            }

            #[test]
            fn prop_relative_none_if_equal_or_same_length(
                (a, b) in arb_qualified_name()
                    .prop_flat_map(|a| {
                        let a_len = a.len();
                        let a_clone1 = a.clone();
                        let a_clone2 = a.clone();
                        prop_oneof![
                            // equal
                            1 => Just((a_clone1.clone(), a_clone1)),
                            // same length but not equal
                            9 => arb_qualified_name_with_len(a_len)
                                .prop_map(move |b| (a_clone2.clone(), b))
                        ]
                    })
            ) {
                // Either a == b, or a.len() == b.len() but a != b
                prop_assert_eq!(a.len(), b.len());
                prop_assert!(a.relative(&b).is_none());
            }

            #[test]
            fn prop_relative_implies_starts_with(
                base in arb_qualified_name(),
                extra in arb_qualified_name(),
            ) {
                let full = base.join(&extra);

                let rel = full.relative(&base);
                prop_assert_eq!(rel, Some(extra.clone()));
                prop_assert!(full.starts_with(&base));
            }

            #[test]
            fn prop_relative_length(
                base in arb_qualified_name(),
                extra in arb_qualified_name(),
            ) {
                let full = base.join(&extra);

                let rel = full.relative(&base).unwrap();
                prop_assert_eq!(rel.len(), extra.len());
                prop_assert_eq!(rel.len(), full.len() - base.len());
            }
        }
    }
}
