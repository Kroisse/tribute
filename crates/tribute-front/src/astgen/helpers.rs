//! CST navigation helpers and utility functions for AST lowering.

/// Check if a node is a comment that should be skipped.
pub fn is_comment(kind: &str) -> bool {
    matches!(
        kind,
        "line_comment" | "block_comment" | "line_doc_comment" | "block_doc_comment"
    )
}

/// Truncate token text for display in error messages.
///
/// Returns a `Display` wrapper that lazily truncates to the first line,
/// at most 20 characters, with `...` appended if truncated.
/// No intermediate allocation — writes directly into the formatter.
pub(crate) fn truncate_token_preview(text: &str) -> impl std::fmt::Display + '_ {
    struct TruncatedToken<'a>(&'a str);

    impl std::fmt::Display for TruncatedToken<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if let Some((byte_idx, _)) = self.0.char_indices().nth(20) {
                write!(f, "{}...", &self.0[..byte_idx])
            } else {
                f.write_str(self.0)
            }
        }
    }

    let trimmed = text.trim();
    let first_line = trimmed.lines().next().unwrap_or(trimmed);
    TruncatedToken(first_line)
}

/// An invalid escape sequence found while decoding a literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EscapeError {
    /// Byte range of the escape sequence, relative to the decoded text.
    pub range: std::ops::Range<usize>,
    pub kind: EscapeErrorKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EscapeErrorKind {
    /// A `\u{…}` escape naming a surrogate code point (`D800`–`DFFF`).
    Surrogate(u32),
    /// A `\u{…}` escape naming a value above `10FFFF`.
    OutOfRange(u32),
}

impl EscapeError {
    /// Offset the range by `offset` bytes, mapping a content-relative error
    /// back to the enclosing literal text.
    pub(crate) fn offset_by(self, offset: usize) -> Self {
        Self {
            range: self.range.start + offset..self.range.end + offset,
            kind: self.kind,
        }
    }
}

impl std::fmt::Display for EscapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            EscapeErrorKind::Surrogate(value) => write!(
                f,
                "invalid Unicode escape: U+{value:04X} is a surrogate code point, \
                 not a Unicode scalar value"
            ),
            EscapeErrorKind::OutOfRange(value) => write!(
                f,
                "invalid Unicode escape: U+{value:04X} exceeds the maximum U+10FFFF"
            ),
        }
    }
}

/// Decode the hex digits of a `\u{…}` escape into a Unicode scalar value.
///
/// Returns `Ok(None)` when `hex` is not a hex number; the grammar rejects that
/// shape, so it only occurs for malformed input.
pub(crate) fn decode_unicode_escape(hex: &str) -> Result<Option<char>, EscapeErrorKind> {
    let Ok(value) = u32::from_str_radix(hex, 16) else {
        return Ok(None);
    };
    match char::from_u32(value) {
        Some(ch) => Ok(Some(ch)),
        None if (0xD800..=0xDFFF).contains(&value) => Err(EscapeErrorKind::Surrogate(value)),
        None => Err(EscapeErrorKind::OutOfRange(value)),
    }
}

/// Process escape sequences in a string/bytes literal content.
///
/// Converts backslash escapes (`\n`, `\t`, `\r`, `\0`, `\\`, `\"`, `\xHH`,
/// `\u{H…}`) into their byte values. Unknown escapes are not reachable here
/// because tree-sitter's grammar rejects them at parse time; the grammar also
/// rejects `\u{…}` in bytes literals. A `\u{…}` escape that is not a Unicode
/// scalar value is reported as an [`EscapeError`] relative to `input`.
///
/// Operates on raw bytes to avoid redundant UTF-8 decoding (the input is already
/// validated by tree-sitter).
pub(crate) fn process_escape_sequences(input: &str) -> Result<Vec<u8>, EscapeError> {
    let bytes = input.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] != b'\\' {
            // Scan forward for the next backslash (or end), copy the whole run
            let start = i;
            while i < bytes.len() && bytes[i] != b'\\' {
                i += 1;
            }
            result.extend_from_slice(&bytes[start..i]);
            continue;
        }
        // Backslash — consume it and the escape character
        let escape_start = i;
        i += 1; // skip '\'
        if i >= bytes.len() {
            result.push(b'\\');
            break;
        }
        match bytes[i] {
            b'n' => {
                result.push(b'\n');
                i += 1;
            }
            b'r' => {
                result.push(b'\r');
                i += 1;
            }
            b't' => {
                result.push(b'\t');
                i += 1;
            }
            b'0' => {
                result.push(b'\0');
                i += 1;
            }
            b'\\' => {
                result.push(b'\\');
                i += 1;
            }
            b'"' => {
                result.push(b'"');
                i += 1;
            }
            b'x' if i + 2 < bytes.len() => {
                i += 1; // skip 'x'
                let hex = &bytes[i..i + 2];
                let byte =
                    u8::from_str_radix(std::str::from_utf8(hex).unwrap_or("00"), 16).unwrap_or(0);
                result.push(byte);
                i += 2;
            }
            b'u' if bytes.get(i + 1) == Some(&b'{') => {
                let digits_start = i + 2;
                let Some(len) = bytes[digits_start..].iter().position(|&b| b == b'}') else {
                    // Unterminated escape: keep the rest verbatim
                    result.extend_from_slice(&bytes[escape_start..]);
                    break;
                };
                let digits_end = digits_start + len;
                i = digits_end + 1; // skip '}'
                let hex = &input[digits_start..digits_end];
                match decode_unicode_escape(hex) {
                    Ok(Some(ch)) => {
                        let buf = &mut [0u8; 4];
                        result.extend_from_slice(ch.encode_utf8(buf).as_bytes());
                    }
                    Ok(None) => result.extend_from_slice(&bytes[escape_start..i]),
                    Err(kind) => {
                        return Err(EscapeError {
                            range: escape_start..i,
                            kind,
                        });
                    }
                }
            }
            _ => {
                // Fallback: keep backslash as-is
                result.push(b'\\');
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_escape_scalar_value_boundaries() {
        for (hex, expected) in [
            ("0", '\0'),
            ("D7FF", '\u{D7FF}'),
            ("e000", '\u{E000}'),
            ("10FFFF", '\u{10FFFF}'),
            ("000041", 'A'),
        ] {
            assert_eq!(decode_unicode_escape(hex), Ok(Some(expected)), "{hex}");
        }
        assert_eq!(
            decode_unicode_escape("D800"),
            Err(EscapeErrorKind::Surrogate(0xD800))
        );
        assert_eq!(
            decode_unicode_escape("DFFF"),
            Err(EscapeErrorKind::Surrogate(0xDFFF))
        );
        assert_eq!(
            decode_unicode_escape("110000"),
            Err(EscapeErrorKind::OutOfRange(0x110000))
        );
    }

    #[test]
    fn escape_error_range_covers_the_escape() {
        let content = r"ab\u{D800}c";
        let error = process_escape_sequences(content).unwrap_err();
        assert_eq!(&content[error.range.clone()], r"\u{D800}");
        assert_eq!(error.offset_by(1).range, 3..11);
    }

    #[test]
    fn escapes_decode_to_utf8_bytes() {
        assert_eq!(
            process_escape_sequences(r"\u{E9}\x41\n").unwrap(),
            "éA\n".as_bytes()
        );
    }
}
