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
