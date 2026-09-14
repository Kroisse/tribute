//! Formatting utilities.

/// English plural/verb agreement helpers.
pub trait PluralExt {
    /// Returns `""` for 1, `"s"` otherwise.
    fn plural(&self) -> &'static str;

    /// Returns `singular` for 1, `plural` otherwise.
    fn verb(&self, singular: &'static str, plural: &'static str) -> &'static str;
}

impl PluralExt for usize {
    fn plural(&self) -> &'static str {
        if *self == 1 { "" } else { "s" }
    }

    fn verb(&self, singular: &'static str, plural: &'static str) -> &'static str {
        if *self == 1 { singular } else { plural }
    }
}
