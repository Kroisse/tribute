use std::path::Path;

use fluent_uri::Uri;
use ropey::Rope;
use tree_sitter::{Parser, Tree};
use trunk_ir::{PathId, Symbol};

#[salsa::input(debug)]
pub struct SourceCst {
    #[returns(ref)]
    pub uri: Uri<String>,
    #[returns(ref)]
    pub text: Rope,
    #[returns(ref)]
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

/// Derive a module name from a PathId.
///
/// Extracts the file stem (filename without extension) from the URI path.
/// Falls back to "main" if the path cannot be parsed.
pub fn derive_module_name_from_path(db: &dyn salsa::Database, path: PathId<'_>) -> Symbol {
    let uri_str = path.uri(db);
    // Parse URI and extract path component (handle file:// URIs)
    let file_path = Uri::parse(uri_str)
        .ok()
        .and_then(|uri| uri.path().as_str().strip_prefix('/').map(|s| s.to_string()))
        .unwrap_or_else(|| uri_str.to_string());
    Path::new(&file_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(Symbol::from_dynamic)
        .unwrap_or_else(|| Symbol::new("main"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    mod path_to_uri {
        use super::*;

        #[test]
        fn absolute_path() {
            let uri = path_to_uri(Path::new("/home/user/project/main.trb"));
            assert_eq!(uri.as_str(), "file:///home/user/project/main.trb");
        }

        #[test]
        fn relative_path() {
            let uri = path_to_uri(Path::new("src/main.trb"));
            assert_eq!(uri.as_str(), "file:///src/main.trb");
        }

        #[test]
        fn nested_directory() {
            let uri = path_to_uri(Path::new("/a/b/c/d/e.trb"));
            assert_eq!(uri.as_str(), "file:///a/b/c/d/e.trb");
        }
    }

    mod derive_module_name_from_path {
        use super::*;

        #[test]
        fn file_uri_with_extension() {
            let db = test_db();
            let path = PathId::new(&db, "file:///home/user/project/foo.trb".to_owned());
            let name = derive_module_name_from_path(&db, path);
            assert_eq!(name, Symbol::new("foo"));
        }

        #[test]
        fn file_uri_without_extension() {
            let db = test_db();
            let path = PathId::new(&db, "file:///home/user/project/bar".to_owned());
            let name = derive_module_name_from_path(&db, path);
            assert_eq!(name, Symbol::new("bar"));
        }

        #[test]
        fn file_uri_nested_path() {
            let db = test_db();
            let path = PathId::new(&db, "file:///a/b/c/module.trb".to_owned());
            let name = derive_module_name_from_path(&db, path);
            assert_eq!(name, Symbol::new("module"));
        }

        #[test]
        fn plain_path_string() {
            let db = test_db();
            let path = PathId::new(&db, "/home/user/test.trb".to_owned());
            let name = derive_module_name_from_path(&db, path);
            assert_eq!(name, Symbol::new("test"));
        }

        #[test]
        fn fallback_to_main_for_empty() {
            let db = test_db();
            let path = PathId::new(&db, "".to_owned());
            let name = derive_module_name_from_path(&db, path);
            assert_eq!(name, Symbol::new("main"));
        }

        #[test]
        fn module_name_with_dots() {
            let db = test_db();
            let path = PathId::new(&db, "file:///project/my.module.trb".to_owned());
            let name = derive_module_name_from_path(&db, path);
            assert_eq!(name, Symbol::new("my.module"));
        }
    }

    mod chunk_from_byte {
        use super::*;

        #[test]
        fn beginning_of_rope() {
            let rope = Rope::from_str("hello world");
            let chunk = chunk_from_byte(&rope, 0);
            assert!(chunk.starts_with(b"hello"));
        }

        #[test]
        fn middle_of_rope() {
            let rope = Rope::from_str("hello world");
            let chunk = chunk_from_byte(&rope, 6);
            assert!(chunk.starts_with(b"world"));
        }

        #[test]
        fn beyond_rope_length() {
            let rope = Rope::from_str("hello");
            let chunk = chunk_from_byte(&rope, 100);
            assert_eq!(chunk, b"");
        }

        #[test]
        fn exact_end_of_rope() {
            let rope = Rope::from_str("hello");
            let chunk = chunk_from_byte(&rope, 5);
            assert_eq!(chunk, b"");
        }

        #[test]
        fn empty_rope() {
            let rope = Rope::from_str("");
            let chunk = chunk_from_byte(&rope, 0);
            assert_eq!(chunk, b"");
        }
    }
}
