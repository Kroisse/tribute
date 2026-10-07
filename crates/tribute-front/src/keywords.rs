//! Keywords and raw identifiers.
//!
//! Every keyword is strict: it is never an identifier, in any position.
//! `r#name` is a raw identifier that spells a lowercase keyword as a name.
//! See `new-plans/syntax.md`.

use std::borrow::Cow;

use itertools::Itertools;

/// Lowercase keywords. The grammar reserves them, and `r#name` spells them
/// as identifiers.
pub const KEYWORDS: &[&str] = &[
    "fn", "op", "do", "let", "const", "struct", "enum", "ability", "mod", "pub", "use", "extern",
    "case", "handle", "resume", "become", "as",
];

/// Keywords spelled like type identifiers.
pub const LITERAL_KEYWORDS: &[&str] = &["True", "False", "Nil"];

/// Words reserved for future keywords. They are not grammar tokens yet, so
/// the compiler rejects them as identifiers; `r#name` spells them.
pub const RESERVED_WORDS: &[&str] = &["type", "where", "in"];

/// Path keywords. They are reserved and have no raw form.
pub const PATH_KEYWORDS: &[&str] = &["pkg", "super", "self"];

/// Whether `name` must be written as a raw identifier to be used as a name.
pub fn needs_raw(name: &str) -> bool {
    KEYWORDS.contains(&name) || RESERVED_WORDS.contains(&name)
}

/// Whether `name` is a keyword or reserved word, and so cannot be written
/// bare as an identifier.
pub fn is_reserved(name: &str) -> bool {
    needs_raw(name) || LITERAL_KEYWORDS.contains(&name) || PATH_KEYWORDS.contains(&name)
}

/// Whether `name` can be written as a raw identifier `r#name`: a lowercase
/// name other than a path keyword.
pub fn can_be_raw(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !PATH_KEYWORDS.contains(&name)
}

/// The source spelling of the name `name`: raw when it is a keyword.
pub fn source_name(name: &str) -> Cow<'_, str> {
    if needs_raw(name) {
        Cow::Owned(format!("r#{name}"))
    } else {
        Cow::Borrowed(name)
    }
}

/// The name that `text`, an identifier or a `::`-separated path as written,
/// denotes: each raw segment loses its `r#`.
pub fn unraw(text: &str) -> Cow<'_, str> {
    if !text.contains("r#") {
        return Cow::Borrowed(text);
    }
    Cow::Owned(
        text.split("::")
            .map(|segment| {
                let trimmed = segment.trim_start();
                match trimmed.strip_prefix("r#") {
                    Some(name) => Cow::Owned(format!(
                        "{}{name}",
                        &segment[..segment.len() - trimmed.len()]
                    )),
                    None => Cow::Borrowed(segment),
                }
            })
            .join("::"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_segments_are_unwrapped() {
        for (text, expected) in [
            ("foo", "foo"),
            ("r#type", "type"),
            ("r#foo", "foo"),
            ("a::r#op::b", "a::op::b"),
            ("a :: r#op", "a :: op"),
            ("pr#x", "pr#x"),
        ] {
            assert_eq!(unraw(text), expected, "{text}");
        }
    }

    #[test]
    fn keywords_are_spelled_raw() {
        assert_eq!(source_name("type"), "r#type");
        assert_eq!(source_name("op"), "r#op");
        assert_eq!(source_name("foo"), "foo");
        assert_eq!(source_name("become"), "r#become");
        // `if` is an ordinary identifier.
        assert_eq!(source_name("if"), "if");
        assert!(is_reserved("self"));
        assert!(is_reserved("True"));
        assert!(!needs_raw("self"));
        assert!(can_be_raw("type"));
        assert!(can_be_raw("_x1"));
        assert!(!can_be_raw("self"));
        assert!(!can_be_raw("Token"));
        assert!(!can_be_raw("True"));
    }
}
