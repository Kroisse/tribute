use std::ops::Range;
use std::path::Path;

use fluent_uri::Uri;
use ropey::Rope;
use salsa::Setter;
use tree_sitter::{InputEdit, Parser, Point, Tree};
use trunk_ir::Symbol;

#[salsa::input(debug)]
pub struct SourceCst {
    pub uri: Uri<String>,
    pub text: Rope,
    #[returns(as_ref)]
    pub tree: Option<Tree>,
}

impl SourceCst {
    /// Create a SourceCst from a file path (convenience for CLI/tests).
    pub fn from_path(
        db: &dyn salsa::Database,
        path: impl AsRef<Path>,
        text: Rope,
        tree: Option<Tree>,
    ) -> Self {
        let uri = path_to_uri(path.as_ref());
        Self::new(db, uri, text, tree)
    }

    /// Create a SourceCst from a source string, parsing with tree-sitter.
    ///
    /// Convenience for tests that need a fully parsed source.
    pub fn from_source_str(db: &dyn salsa::Database, path: &str, text: &str) -> Self {
        let rope = Rope::from_str(text);
        let tree = new_parser()
            .parse(text, None)
            .expect("Failed to parse source");
        Self::from_path(db, path, rope, Some(tree))
    }
}

/// Convert a filesystem path to a file:// URI.
pub fn path_to_uri(path: &Path) -> Uri<String> {
    let path_str = path.to_string_lossy();
    let uri_string = if path_str.starts_with('/') {
        format!("file://{}", path_str)
    } else {
        // Relative path - just use as-is for testing
        format!("file:///{}", path_str)
    };
    Uri::parse_from(uri_string).expect("valid file URI")
}

/// Parse a text rope with a given parser.
pub fn parse_with_rope(parser: &mut Parser, rope: &Rope, old_tree: Option<&Tree>) -> Option<Tree> {
    let mut callback = |byte: usize, _| chunk_from_byte(rope, byte);
    parser.parse_with_options(&mut callback, old_tree, None)
}

/// A parser for Tribute source.
pub fn new_parser() -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set tree-sitter language");
    parser
}

/// Parse a text rope with a new parser.
pub fn parse_rope(rope: &Rope, old_tree: Option<&Tree>) -> Option<Tree> {
    parse_with_rope(&mut new_parser(), rope, old_tree)
}

/// Replace the bytes `range` of `source`'s text with `text`, and reparse it
/// incrementally from its edited tree.
///
/// `range` must lie within the text on character boundaries.
pub fn apply_edit(
    db: &mut dyn salsa::Database,
    parser: &mut Parser,
    source: SourceCst,
    range: Range<usize>,
    text: &str,
) {
    let mut rope = source.text(db).clone();
    let start_byte = range.start;
    let old_end_byte = range.end;
    let start_position = byte_point(&rope, start_byte);
    let old_end_position = byte_point(&rope, old_end_byte);
    let new_end_position = point_after_text(start_position, text);
    let new_end_byte = start_byte + text.len();

    let start_char = rope.byte_to_char(start_byte);
    let old_end_char = rope.byte_to_char(old_end_byte);
    rope.remove(start_char..old_end_char);
    rope.insert(start_char, text);

    let tree = match source.tree(db).cloned() {
        Some(mut tree) => {
            tree.edit(&InputEdit {
                start_byte,
                old_end_byte,
                new_end_byte,
                start_position,
                old_end_position,
                new_end_position,
            });
            parse_with_rope(parser, &rope, Some(&tree))
        }
        None => parse_with_rope(parser, &rope, None),
    };
    source.set_text(db).to(rope);
    source.set_tree(db).to(tree);
}

fn point_after_text(start: Point, text: &str) -> Point {
    let mut row = start.row;
    let mut column = start.column;
    let mut lines = text.split('\n');
    if let Some(first) = lines.next() {
        column += first.len();
    }
    for line in lines {
        row += 1;
        column = line.len();
    }
    Point { row, column }
}

fn byte_point(rope: &Rope, offset: usize) -> Point {
    let offset = offset.min(rope.len_bytes());
    let row = rope.byte_to_line(offset);
    let column = offset - rope.line_to_byte(row);
    Point { row, column }
}

fn chunk_from_byte(rope: &Rope, byte: usize) -> &[u8] {
    if byte >= rope.len_bytes() {
        return b"";
    }
    let (chunk, chunk_start, _, _) = rope.chunk_at_byte(byte);
    let start = byte - chunk_start;
    &chunk.as_bytes()[start..]
}

/// Derive a module name from a URI string.
///
/// Extracts the file stem (filename without extension) from the URI path.
/// Falls back to "main" if the path cannot be parsed.
pub fn derive_module_name_from_path(uri_str: &str) -> Symbol {
    // Parse URI and extract path component (handle file:// URIs)
    let file_path = Uri::parse(uri_str)
        .ok()
        .and_then(|uri| uri.path().as_str().strip_prefix('/').map(|s| s.to_string()))
        .unwrap_or_else(|| uri_str.to_string());
    Path::new(&file_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(Symbol::new)
        .unwrap_or_else(|| Symbol::new("main"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_to_uri_prefixes_file_scheme() {
        for (path, expected) in [
            (
                "/home/user/project/main.trb",
                "file:///home/user/project/main.trb",
            ),
            ("src/main.trb", "file:///src/main.trb"),
            ("/a/b/c/d/e.trb", "file:///a/b/c/d/e.trb"),
        ] {
            let uri = path_to_uri(Path::new(path));
            assert_eq!(uri.as_str(), expected, "{path}");
        }
    }

    #[test]
    fn derive_module_name_from_path_uses_file_stem() {
        for (path, expected) in [
            ("file:///home/user/project/foo.trb", "foo"),
            ("file:///home/user/project/bar", "bar"),
            ("file:///a/b/c/module.trb", "module"),
            ("/home/user/test.trb", "test"),
            ("", "main"),
            ("file:///project/my.module.trb", "my.module"),
        ] {
            let name = derive_module_name_from_path(path);
            assert_eq!(name, Symbol::new(expected), "{path:?}");
        }
    }

    #[test]
    fn chunk_from_byte_returns_rest_of_chunk() {
        for (text, byte, expected) in [
            ("hello world", 0, &b"hello world"[..]),
            ("hello world", 6, b"world"),
            ("hello", 100, b""),
            ("hello", 5, b""),
            ("", 0, b""),
        ] {
            let rope = Rope::from_str(text);
            let chunk = chunk_from_byte(&rope, byte);
            assert_eq!(chunk, expected, "{text:?} at byte {byte}");
        }
    }

    #[test]
    fn point_after_text_advances_by_rows_and_bytes() {
        for (row, column, text, expected) in [
            (0, 0, "hello", (0, 5)),
            (0, 0, "hello\nworld", (1, 5)),
            (5, 10, "abc", (5, 13)),
            (0, 0, "a\nb\nc", (2, 1)),
        ] {
            let point = point_after_text(Point { row, column }, text);
            assert_eq!((point.row, point.column), expected, "{text:?}");
        }
    }

    #[test]
    fn byte_point_counts_bytes_from_the_line_start() {
        let rope = Rope::from_str("hello\nworld");
        for (offset, expected) in [(0, (0, 0)), (6, (1, 0)), (8, (1, 2)), (100, (1, 5))] {
            let point = byte_point(&rope, offset);
            assert_eq!((point.row, point.column), expected, "at byte {offset}");
        }
    }

    #[test]
    fn parse_rope_parses_and_reparses_with_an_old_tree() {
        let tree = parse_rope(&Rope::from_str("fn foo() { 1 }"), None).expect("a tree");
        assert_eq!(tree.root_node().kind(), "source_file");
        let tree = parse_rope(&Rope::from_str("fn foo() { 2 }"), Some(&tree));
        assert!(tree.is_some(), "reparses with the old tree");
    }
}
