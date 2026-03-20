//! Formatting utilities.

use std::fmt;

/// Wrapper that displays a slice of items joined by a separator.
///
/// # Examples
///
/// ```
/// use tribute_core::fmt::Sep;
///
/// let items = vec!["State(Int)", "Console"];
/// assert_eq!(format!("{}", Sep(", ", &items)), "State(Int), Console");
/// assert_eq!(format!("{}", Sep(" | ", &items)), "State(Int) | Console");
///
/// assert_eq!(format!("{}", Sep(", ", &[] as &[String])), "");
/// assert_eq!(format!("{}", Sep(", ", &["solo"])), "solo");
/// ```
pub struct Sep<'a, I: fmt::Display>(pub &'a str, pub &'a [I]);

impl<I: fmt::Display> fmt::Display for Sep<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for item in self.1 {
            if !first {
                f.write_str(self.0)?;
            }
            first = false;
            write!(f, "{item}")?;
        }
        Ok(())
    }
}
