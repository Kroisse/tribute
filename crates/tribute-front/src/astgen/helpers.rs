//! CST navigation helpers and utility functions for AST lowering.

/// Check if a node is a comment that should be skipped.
pub fn is_comment(kind: &str) -> bool {
    matches!(
        kind,
        "line_comment" | "block_comment" | "line_doc_comment" | "block_doc_comment"
    )
}
