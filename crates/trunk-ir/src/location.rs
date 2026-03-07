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
