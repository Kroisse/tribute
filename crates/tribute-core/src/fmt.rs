//! Formatting utilities.

use std::fmt;

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
