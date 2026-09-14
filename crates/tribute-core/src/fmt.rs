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

/// Like [`joined`], but uses a custom formatting closure for each item.
///
/// Clones the iterator for each formatting operation, so the result can be
/// displayed repeatedly. Items are written directly to the formatter.
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

#[cfg(test)]
mod tests {
    use super::{joined, joined_by};
    use std::{cell::Cell, fmt};

    #[test]
    fn joined_consumes_items_only_when_displayed() {
        let visits = Cell::new(0);
        let items = (1..=3).inspect(|_| visits.set(visits.get() + 1));
        let display = joined(" / ", items);
        assert_eq!(visits.get(), 0);
        assert_eq!(format!("[{display}]"), "[1 / 2 / 3]");
        assert_eq!(visits.get(), 3);
    }

    #[test]
    fn joined_by_supports_empty_singleton_and_repeated_display() {
        for (items, expected) in [
            (&[][..], ""),
            (&[1][..], "#1"),
            (&[1, 2, 3][..], "#1, #2, #3"),
        ] {
            let display = joined_by(", ", items, |item, f| write!(f, "#{item}"));
            assert_eq!(format!("{display}"), expected);
            assert_eq!(format!("{display}"), expected);
        }
    }

    #[test]
    fn formatting_failures_are_propagated() {
        struct RejectWrites;

        impl fmt::Write for RejectWrites {
            fn write_str(&mut self, _: &str) -> fmt::Result {
                Err(fmt::Error)
            }
        }

        assert!(fmt::write(&mut RejectWrites, format_args!("{}", joined(", ", [1]))).is_err());
        assert!(
            fmt::write(
                &mut RejectWrites,
                format_args!("{}", joined_by(", ", [1], |item, f| write!(f, "{item}")))
            )
            .is_err()
        );

        let visits = Cell::new(0);
        let display = joined_by(", ", [1, 2], |_, _| {
            visits.set(visits.get() + 1);
            Err(fmt::Error)
        });
        assert!(fmt::write(&mut String::new(), format_args!("{display}")).is_err());
        assert_eq!(visits.get(), 1);
    }
}
