//! Source location types for tracking positions in source files.

use serde::{Deserialize, Serialize};

/// A span of source code, represented as byte offsets.
///
/// `source` names the source the offsets are in, as an opaque id its
/// producer assigns. `0` is the source the producer started from, so a
/// producer that reads one source never sets it.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub source: u64,
}

impl Span {
    /// The bytes `start..end` of the source the producer started from.
    pub const fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            source: 0,
        }
    }

    /// The same bytes in the source `source`.
    pub const fn in_source(self, source: u64) -> Self {
        Self { source, ..self }
    }
}

fn is_zero(source: &u64) -> bool {
    *source == 0
}

/// A value paired with its source span.
pub type Spanned<T> = (T, Span);
