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

/// Joins items from an iterator with a separator, returning a [`Display`](fmt::Display) value.
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
    I::IntoIter: Clone,
    I::Item: fmt::Display,
{
    struct Joined<'a, I> {
        sep: &'a str,
        iter: I,
    }

    impl<I> fmt::Display for Joined<'_, I>
    where
        I: Iterator + Clone,
        I::Item: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut first = true;
            for item in self.iter.clone() {
                if !first {
                    f.write_str(self.sep)?;
                }
                first = false;
                write!(f, "{item}")?;
            }
            Ok(())
        }
    }

    Joined {
        sep,
        iter: iter.into_iter(),
    }
}

/// Like [`joined`], but uses a custom formatting closure for each item.
///
/// # Examples
///
/// ```
/// use tribute_core::fmt::joined_by;
///
/// let nums = vec![1, 2, 3];
/// let output = format!("{}", joined_by(", ", &nums, |n, f| write!(f, "#{n}")));
/// assert_eq!(output, "#1, #2, #3");
/// ```
pub fn joined_by<'a, I, F>(sep: &'a str, iter: I, formatter: F) -> impl fmt::Display + 'a
where
    I: IntoIterator + 'a,
    I::IntoIter: Clone,
    F: Fn(I::Item, &mut fmt::Formatter<'_>) -> fmt::Result + 'a,
{
    struct JoinedBy<'a, I, F> {
        sep: &'a str,
        iter: I,
        formatter: F,
    }

    impl<I, F> fmt::Display for JoinedBy<'_, I, F>
    where
        I: Iterator + Clone,
        F: Fn(I::Item, &mut fmt::Formatter<'_>) -> fmt::Result,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut first = true;
            for item in self.iter.clone() {
                if !first {
                    f.write_str(self.sep)?;
                }
                first = false;
                (self.formatter)(item, f)?;
            }
            Ok(())
        }
    }

    JoinedBy {
        sep,
        iter: iter.into_iter(),
        formatter,
    }
}
