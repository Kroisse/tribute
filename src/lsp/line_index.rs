//! Line index for converting between LSP positions and byte offsets.
//!
//! LSP uses 0-based (line, character) positions with UTF-16 encoding,
//! while Tribute uses byte offsets. This module provides conversion utilities.

use trunk_ir::Span;

/// Line index for a source file.
///
/// Caches line start positions for efficient position conversion.
pub struct LineIndex {
    /// Byte offset of the start of each line (0-indexed).
    line_starts: Vec<usize>,
    /// Original source text.
    source: String,
}

impl LineIndex {
    /// Create a line index from source text.
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            source: source.to_string(),
        }
    }

    /// Convert LSP (line, character) to byte offset.
    ///
    /// LSP uses 0-based line/column with UTF-16 character offsets.
    pub fn offset(&self, line: u32, character: u32) -> Option<usize> {
        let line_start = *self.line_starts.get(line as usize)?;
        let line_text = self.line_text(line)?;

        // Convert UTF-16 offset to byte offset
        let mut utf16_offset = 0u32;
        for (byte_offset, c) in line_text.char_indices() {
            if utf16_offset >= character {
                return Some(line_start + byte_offset);
            }
            utf16_offset += c.len_utf16() as u32;
        }
        // Character is at or past end of line
        Some(line_start + line_text.len())
    }

    /// Convert byte offset to LSP (line, character).
    pub fn line_col(&self, offset: usize) -> (u32, u32) {
        let offset = offset.min(self.source.len());
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let line_start = self.line_starts[line];
        let text_before = &self.source[line_start..offset];
        let character: u32 = text_before.chars().map(|c| c.len_utf16() as u32).sum();
        (line as u32, character)
    }

    /// Convert a Span to LSP Range.
    pub fn span_to_range(&self, span: Span) -> lsp_types::Range {
        let start = self.line_col(span.start);
        let end = self.line_col(span.end);
        lsp_types::Range {
            start: lsp_types::Position {
                line: start.0,
                character: start.1,
            },
            end: lsp_types::Position {
                line: end.0,
                character: end.1,
            },
        }
    }

    /// Get the text of a specific line.
    fn line_text(&self, line: u32) -> Option<&str> {
        let start = *self.line_starts.get(line as usize)?;
        let end = self
            .line_starts
            .get(line as usize + 1)
            .copied()
            .unwrap_or(self.source.len());
        // Don't include the newline character
        let text = &self.source[start..end];
        Some(text.trim_end_matches('\n'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_line() {
        let index = LineIndex::new("hello");
        assert_eq!(index.offset(0, 0), Some(0));
        assert_eq!(index.offset(0, 5), Some(5));
        assert_eq!(index.line_col(0), (0, 0));
        assert_eq!(index.line_col(5), (0, 5));
    }

    #[test]
    fn test_multiple_lines() {
        let index = LineIndex::new("hello\nworld\n");
        assert_eq!(index.offset(0, 0), Some(0));
        assert_eq!(index.offset(0, 5), Some(5));
        assert_eq!(index.offset(1, 0), Some(6));
        assert_eq!(index.offset(1, 5), Some(11));
        assert_eq!(index.line_col(0), (0, 0));
        assert_eq!(index.line_col(5), (0, 5));
        assert_eq!(index.line_col(6), (1, 0));
        assert_eq!(index.line_col(11), (1, 5));
    }

    #[test]
    fn test_utf16_conversion() {
        // Korean character '가' is 1 char in UTF-8 (3 bytes) but 1 in UTF-16
        let index = LineIndex::new("가나다");
        assert_eq!(index.offset(0, 0), Some(0));
        assert_eq!(index.offset(0, 1), Some(3)); // After '가'
        assert_eq!(index.offset(0, 2), Some(6)); // After '나'
        assert_eq!(index.line_col(0), (0, 0));
        assert_eq!(index.line_col(3), (0, 1));
        assert_eq!(index.line_col(6), (0, 2));
    }

    #[test]
    fn test_emoji() {
        // Emoji '😀' is 4 bytes in UTF-8 but 2 in UTF-16 (surrogate pair)
        let index = LineIndex::new("a😀b");
        assert_eq!(index.offset(0, 0), Some(0)); // 'a'
        assert_eq!(index.offset(0, 1), Some(1)); // Start of emoji
        assert_eq!(index.offset(0, 3), Some(5)); // 'b' (emoji is 2 UTF-16 units)
        assert_eq!(index.line_col(0), (0, 0));
        assert_eq!(index.line_col(1), (0, 1));
        assert_eq!(index.line_col(5), (0, 3));
    }
}
