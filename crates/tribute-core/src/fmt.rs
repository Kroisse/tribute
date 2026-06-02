//! Formatting utilities.

use std::fmt;

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

/// Joins items from an iterator with a separator, returning a
/// [`Display`](fmt::Display) value (lazy).
///
/// Thin wrapper over [`itertools::Itertools::format`]. Display the result at
/// most once — the underlying iterator is consumed on first use.
///
/// # Examples
///
/// ```
/// use tribute_core::fmt::joined;
///
/// let items = vec!["State(Int)", "Console"];
/// assert_eq!(format!("{}", joined(", ", &items)), "State(Int), Console");
/// assert_eq!(format!("{}", joined(" | ", &items)), "State(Int) | Console");
///
/// assert_eq!(format!("{}", joined(", ", &[] as &[String])), "");
/// assert_eq!(format!("{}", joined(", ", &["solo"])), "solo");
/// ```
pub fn joined<'a, I>(sep: &'a str, iter: I) -> impl fmt::Display + 'a
where
    I: IntoIterator + 'a,
    I::Item: fmt::Display,
{
    use itertools::Itertools;
    iter.into_iter().format(sep)
}
