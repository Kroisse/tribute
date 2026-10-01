//! Interned symbols and block identifiers.
//!
//! These types are Salsa-independent core primitives used throughout the IR.

use std::borrow::Cow;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use lasso::{Spur, ThreadedRodeo};
use smallvec::SmallVec;

// ============================================================================
// Interned Types
// ============================================================================

/// Global string interner for symbols.
///
/// The interner is never cleared, so every interned string lives for the rest
/// of the process. That is what lets `Symbol::as_str` return `&'static str`.
static INTERNER: LazyLock<Interner> = LazyLock::new(|| Interner::with_hasher(Default::default()));

/// Interned names are compiler-generated or come from the program being
/// compiled, so the lookup hash needs no HashDoS resistance. `Symbol::new`
/// runs on every typed operation match, which makes the hasher a hot path.
type Interner = ThreadedRodeo<Spur, rustc_hash::FxBuildHasher>;

/// Interned symbol for efficient comparison of names (functions, variables, fields, etc.)
///
/// Uses lasso for string interning with 4-byte Spur keys.
///
/// Ordering is based on the underlying string content (not interning order),
/// so that key-ordered collections such as `AttributeMap` iterate deterministically.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "salsa", derive(salsa::SalsaValue))]
pub struct Symbol(Spur);

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the string content, not the internal Spur index,
        // so the result is stable regardless of interning order.
        self.as_str().hash(state);
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.0 == other.0 {
            return std::cmp::Ordering::Equal;
        }
        self.as_str().cmp(other.as_str())
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Symbol({:?})", self.as_str())
    }
}

impl Symbol {
    /// Intern a static string and return its symbol. Prefer this over `from_dynamic` when possible.
    pub fn new(text: &'static str) -> Self {
        Symbol(INTERNER.get_or_intern_static(text))
    }

    /// Intern a string and return its symbol. Prefer `new` if the text is static.
    pub fn from_dynamic(text: &str) -> Self {
        Symbol(INTERNER.get_or_intern(text))
    }

    /// Look up an already-interned string without interning it when absent.
    pub fn lookup(text: &str) -> Option<Self> {
        INTERNER.get(text).map(Self)
    }

    /// The symbol's text.
    ///
    /// ```
    /// use trunk_ir::Symbol;
    /// let symbol = Symbol::new("something");
    /// assert_eq!(symbol.as_str(), "something");
    /// ```
    pub fn as_str(self) -> &'static str {
        INTERNER.resolve(&self.0)
    }

    /// Access the symbol's text through a closure.
    ///
    /// Equivalent to `f(self.as_str())`; prefer `as_str` in new code.
    pub fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        f(self.as_str())
    }
}

impl From<&'static str> for Symbol {
    fn from(text: &'static str) -> Self {
        Symbol::new(text)
    }
}

impl From<Cow<'_, str>> for Symbol {
    fn from(text: Cow<'_, str>) -> Self {
        Symbol::from_dynamic(&text)
    }
}

/// Helper macro for declaring multiple symbol helpers at once.
///
/// # Example
/// ```
/// use trunk_ir::symbols;
///
/// symbols! {
///     ATTR_NAME => "name",
///     ATTR_TYPE => "type",
///     #[allow(dead_code)]
///     ATTR_UNUSED => "unused",
/// }
/// ```
#[macro_export]
macro_rules! symbols {
    ($($(#[$attr:meta])* $name:ident => $text:literal),* $(,)?) => {
        $(
            $(#[$attr])*
            #[allow(non_snake_case)]
            #[inline]
            pub fn $name() -> $crate::Symbol {
                $crate::Symbol::new($text)
            }
        )*
    };
}

// Convenient comparison with &str
impl PartialEq<str> for Symbol {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Symbol {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<Symbol> for str {
    fn eq(&self, other: &Symbol) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<Symbol> for &str {
    fn eq(&self, other: &Symbol) -> bool {
        *self == other.as_str()
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_does_not_intern_missing_text() {
        let text = "__trunk_ir_symbol_lookup_missing_text__";
        assert_eq!(Symbol::lookup(text), None);
        assert_eq!(Symbol::lookup(text), None);

        let symbol = Symbol::from_dynamic(text);
        assert_eq!(Symbol::lookup(text), Some(symbol));
    }
}

// ============================================================================
// Block Identity
// ============================================================================

/// Global counter for generating unique block IDs.
static NEXT_BLOCK_ID: AtomicU64 = AtomicU64::new(1);

/// Stable block identifier that survives block recreation.
///
/// Unlike `Block` (which is a Salsa tracked struct with identity tied to creation),
/// `BlockId` is a simple u64 that can be preserved when a block is recreated
/// during IR transformations. This allows block arguments to maintain stable
/// identity across rewrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub u64);

impl BlockId {
    /// Generate a fresh unique block ID.
    pub fn fresh() -> Self {
        BlockId(NEXT_BLOCK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

// ============================================================================
// Small vector type aliases
// ============================================================================

/// Small vector for values tracked by Salsa framework.
pub type IdVec<T> = SmallVec<[T; 2]>;

/// Small vector for symbols.
pub type SymbolVec = SmallVec<[Symbol; 4]>;
