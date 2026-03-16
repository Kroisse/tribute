//! Interned symbols and block identifiers.
//!
//! These types are Salsa-independent core primitives used throughout the IR.

use std::borrow::Cow;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use lasso::{Rodeo, Spur};
use parking_lot::RwLock;
use smallvec::SmallVec;

// ============================================================================
// Interned Types
// ============================================================================

/// Global string interner for symbols.
static INTERNER: LazyLock<RwLock<Rodeo>> = LazyLock::new(|| RwLock::new(Rodeo::default()));

/// Interned symbol for efficient comparison of names (functions, variables, fields, etc.)
///
/// Uses lasso for string interning with 4-byte Spur keys.
///
/// Ordering is based on the underlying string content (not interning order),
/// so that `BTreeMap<Symbol, _>` iteration is deterministic.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "salsa", derive(salsa::Update))]
pub struct Symbol(Spur);

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the string content, not the internal Spur index,
        // so the result is stable regardless of interning order.
        self.with_str(|s| s.hash(state));
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
        let interner = INTERNER.read_recursive();
        let a = interner.resolve(&self.0);
        let b = interner.resolve(&other.0);
        a.cmp(b)
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_str(|s| write!(f, "Symbol({:?})", s))
    }
}

impl Symbol {
    /// Intern a static string and return its symbol. Prefer this over `from_dynamic` when possible.
    pub fn new(text: &'static str) -> Self {
        Self::get_or_else(text, |rodeo| rodeo.get_or_intern_static(text))
    }

    /// Intern a string and return its symbol. Prefer `new` if the text is static.
    pub fn from_dynamic(text: &str) -> Self {
        Self::get_or_else(text, |rodeo| rodeo.get_or_intern(text))
    }

    fn get_or_else(text: &str, f: impl for<'r> FnOnce(&'r mut Rodeo) -> Spur) -> Self {
        let mut lock = INTERNER.upgradable_read();
        Symbol(if let Some(spur) = lock.get(text) {
            spur
        } else {
            lock.with_upgraded(f)
        })
    }

    /// Access the symbol's text with zero-copy.
    ///
    /// Uses `read_recursive()` to allow nested Symbol operations (Display, ==, to_string)
    /// within the closure without risk of deadlock.
    ///
    /// This is useful for optimization: when you need to work with the symbol's text
    /// without allocating a String, use this method. For example:
    ///
    /// ```
    /// use trunk_ir::Symbol;
    /// let symbol = Symbol::new("something");
    /// // Avoid: symbol.to_string() == "something"
    /// // Prefer:
    /// assert!(symbol.with_str(|s| s == "something"));
    /// ```
    pub fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        let interner = INTERNER.read_recursive();
        let text = interner.resolve(&self.0);
        f(text)
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
        self.with_str(|s| s == other)
    }
}

impl PartialEq<&str> for Symbol {
    fn eq(&self, other: &&str) -> bool {
        self.with_str(|s| s == *other)
    }
}

impl PartialEq<Symbol> for str {
    fn eq(&self, other: &Symbol) -> bool {
        other.with_str(|s| s == self)
    }
}

impl PartialEq<Symbol> for &str {
    fn eq(&self, other: &Symbol) -> bool {
        other.with_str(|s| s == *self)
    }
}

// For Display (uses with_str for zero-copy)
impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_str(|s| write!(f, "{}", s))
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
#[cfg_attr(feature = "salsa", derive(salsa::Update))]
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
