//! Formatting utilities.

use std::fmt;

/// Wrapper that displays a slice of items as a comma-separated list.
///
/// # Examples
///
/// ```
/// use tribute_core::fmt::CommaSep;
///
/// let items = vec!["State(Int)", "Console"];
/// let output = format!("{}", CommaSep(&items));
/// assert_eq!(output, "State(Int), Console");
///
/// assert_eq!(format!("{}", CommaSep::<String>(&[])), "");
/// assert_eq!(format!("{}", CommaSep(&["solo"])), "solo");
/// ```
pub struct CommaSep<'a, I: fmt::Display>(pub &'a [I]);

impl<I: fmt::Display> fmt::Display for CommaSep<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for item in self.0 {
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(f, "{item}")?;
        }
        Ok(())
    }
}
