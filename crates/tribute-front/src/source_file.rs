use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

use fluent_uri::Uri;
use ropey::Rope;
use rustc_hash::FxHashMap as HashMap;
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

/// The files a compilation finds by URI, through the database's one
/// [`FileProvider`] (`new-plans/implementation.md` § 패키지 source 트리와
/// 파일 입력). Queries take `&dyn salsa::Database`, so the provider is
/// reached through this singleton input rather than a database trait.
#[salsa::input(singleton, debug)]
pub struct SourceFiles {
    pub provider: Arc<FileProvider>,
}

/// The file at one URI: its source, or `None` while there is no file.
///
/// A query that reads `source` depends on whether the file exists, so
/// creating a file it looked for invalidates it.
#[salsa::input(debug)]
pub struct FileEntry {
    #[returns(ref)]
    pub uri: Uri<String>,
    #[returns(copy)]
    pub source: Option<SourceCst>,
}

/// Makes one [`FileEntry`] per URI on its first request, from what its
/// reader finds there.
///
/// Entries are never removed: a file that disappears keeps its entry with
/// no source, so the queries that read it stay its dependents.
pub struct FileProvider {
    entries: Mutex<HashMap<String, FileEntry>>,
    read: Box<dyn Fn(&Uri<String>) -> Option<Rope> + Send + Sync>,
}

impl FileProvider {
    /// A provider that reads a file with `read` when it is first requested.
    pub fn new(read: impl Fn(&Uri<String>) -> Option<Rope> + Send + Sync + 'static) -> Self {
        Self {
            entries: Mutex::default(),
            read: Box::new(read),
        }
    }

    /// A provider that finds only the files given to [`FileProvider::set`].
    pub fn in_memory() -> Self {
        Self::new(|_| None)
    }

    /// The entry of `uri`, created with what the reader finds there if
    /// this is its first request.
    pub fn entry(&self, db: &dyn salsa::Database, uri: &Uri<String>) -> FileEntry {
        let mut entries = self.entries.lock().expect("file entries lock");
        *entries.entry(uri.as_str().to_owned()).or_insert_with(|| {
            let source = (self.read)(uri).map(|text| new_source(db, uri.clone(), text));
            FileEntry::new(db, uri.clone(), source)
        })
    }

    /// The entry of `uri` if it was requested or set before.
    pub fn existing(&self, uri: &Uri<String>) -> Option<FileEntry> {
        let entries = self.entries.lock().expect("file entries lock");
        entries.get(uri.as_str()).copied()
    }

    /// Make `text` the file at `uri`, or remove the file if `text` is
    /// `None`. A file that exists keeps its source input, with the new text
    /// and a fresh parse.
    pub fn set(
        &self,
        db: &mut dyn salsa::Database,
        uri: &Uri<String>,
        text: Option<Rope>,
    ) -> Option<SourceCst> {
        let Some(entry) = self.existing(uri) else {
            let source = text.map(|text| new_source(db, uri.clone(), text));
            let entry = FileEntry::new(db, uri.clone(), source);
            let mut entries = self.entries.lock().expect("file entries lock");
            entries.insert(uri.as_str().to_owned(), entry);
            return source;
        };
        match (entry.source(db), text) {
            (Some(source), Some(text)) => {
                let tree = parse_rope(&text, None);
                source.set_text(db).to(text);
                source.set_tree(db).to(tree);
                Some(source)
            }
            (None, Some(text)) => {
                let source = new_source(db, uri.clone(), text);
                entry.set_source(db).to(Some(source));
                Some(source)
            }
            (_, None) => {
                entry.set_source(db).to(None);
                None
            }
        }
    }
}

impl std::fmt::Debug for FileProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries = self.entries.lock().map_or(0, |entries| entries.len());
        f.debug_struct("FileProvider")
            .field("entries", &entries)
            .finish_non_exhaustive()
    }
}

/// The source of the file at `uri`, or `None` if there is none or the
/// database has no [`SourceFiles`]. Inside a query, records a dependency on
/// whether the file exists.
pub fn source_at(db: &dyn salsa::Database, uri: &Uri<String>) -> Option<SourceCst> {
    let files = SourceFiles::try_get(db)?;
    files.provider(db).entry(db, uri).source(db)
}

fn new_source(db: &dyn salsa::Database, uri: Uri<String>, text: Rope) -> SourceCst {
    let tree = parse_rope(&text, None);
    SourceCst::new(db, uri, text, tree)
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
