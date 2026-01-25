//! Source location types for tracking positions in source files.

use serde::{Deserialize, Serialize};

/// A span of source code, represented as byte offsets.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// A value paired with its source span.
pub type Spanned<T> = (T, Span);

/// Interned URI string for source file identification.
///
/// This is used to efficiently track which file a piece of IR came from.
/// The URI is typically a `file://` URI, but can be any valid URI scheme
/// (e.g., `prelude:///` for built-in sources).
#[salsa::interned(debug)]
pub struct PathId<'db> {
    /// URI string (e.g., "file:///path/to/file.trb")
    #[returns(deref)]
    pub uri: String,
}

/// A location in source code, combining file and span information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Location<'db> {
    pub path: PathId<'db>,
    pub span: Span,
}

impl<'db> Location<'db> {
    pub const fn new(path: PathId<'db>, span: Span) -> Self {
        Self { path, span }
    }
}
