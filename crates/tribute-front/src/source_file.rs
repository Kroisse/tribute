use std::path::Path;

use fluent_uri::Uri;
use ropey::Rope;
use tree_sitter::{Parser, Tree};
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
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set tree-sitter language");
        let tree = parser.parse(text, None).expect("Failed to parse source");
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
}
