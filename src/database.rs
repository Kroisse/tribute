use std::cell::RefCell;
use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use lsp_types::Uri;
use ropey::Rope;
use tree_sitter::Tree;
use tribute_front::source_file::parse_with_rope;
use tribute_front::{SourceCst, path_to_uri};

thread_local! {
    static PARSER: RefCell<tree_sitter::Parser> = {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        RefCell::new(parser)
    };
}

pub fn parse_with_thread_local(rope: &Rope, old_tree: Option<&Tree>) -> Option<Tree> {
    PARSER.with(|parser| {
        let mut parser = parser.borrow_mut();
        parse_with_rope(&mut parser, rope, old_tree)
    })
}

#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
    documents: Arc<DashMap<String, SourceCst>>,
}

#[salsa::db]
impl salsa::Database for TributeDatabaseImpl {}

impl TributeDatabaseImpl {
    pub fn input(
        &self,
        path: std::path::PathBuf,
    ) -> Result<SourceCst, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.canonicalize()?;
        let uri = path_to_uri(&path);
        let key = uri.as_str().to_owned();
        match self.documents.entry(key) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let file = std::fs::File::open(&path)?;
                let contents = Rope::from_reader(file)?;
                let tree = parse_with_thread_local(&contents, None);
                let source_cst = SourceCst::new(self, uri, contents, tree);
                entry.insert(source_cst);
                Ok(source_cst)
            }
        }
    }

    pub fn open_document(&self, uri: &Uri, text: Rope) {
        let key = uri.as_str().to_owned();
        let tree = parse_with_thread_local(&text, None);
        let source_cst = SourceCst::new(self, (**uri).clone(), text, tree);
        self.documents.insert(key, source_cst);
    }

    pub fn close_document(&self, uri: &Uri) {
        let key = uri.as_str();
        self.documents.remove(key);
    }

    pub fn source_cst(&self, uri: &Uri) -> Option<SourceCst> {
        let key = uri.as_str();
        self.documents.get(key).map(|entry| *entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_with_thread_local() {
        let source = "fn hello() { 42 }";
        let rope = Rope::from_str(source);
        let tree = parse_with_thread_local(&rope, None);
        assert!(tree.is_some(), "Should successfully parse valid source");

        let tree = tree.unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "source_file");
    }

    #[test]
    fn test_parse_with_thread_local_incremental() {
        let source1 = "fn foo() { 1 }";
        let rope1 = Rope::from_str(source1);
        let tree1 = parse_with_thread_local(&rope1, None).unwrap();

        // Incremental parse with old tree
        let source2 = "fn foo() { 2 }";
        let rope2 = Rope::from_str(source2);
        let tree2 = parse_with_thread_local(&rope2, Some(&tree1));
        assert!(tree2.is_some(), "Should successfully parse with old tree");
    }

    #[test]
    fn test_document_lifecycle() {
        let db = TributeDatabaseImpl::default();
        let uri: Uri = "file:///test.trb".parse().unwrap();
        let text = Rope::from_str("fn test() { }");

        // Initially no document
        assert!(db.source_cst(&uri).is_none());

        // Open document
        db.open_document(&uri, text);
        assert!(db.source_cst(&uri).is_some());

        // Close document
        db.close_document(&uri);
        assert!(db.source_cst(&uri).is_none());
    }

    #[test]
    fn test_open_document_overwrites() {
        let db = TributeDatabaseImpl::default();
        let uri: Uri = "file:///test.trb".parse().unwrap();

        // Open with first content
        let text1 = Rope::from_str("fn first() { }");
        db.open_document(&uri, text1);

        let cst1 = db.source_cst(&uri).unwrap();
        assert_eq!(cst1.text(&db).to_string(), "fn first() { }");

        // Open with second content (should overwrite)
        let text2 = Rope::from_str("fn second() { }");
        db.open_document(&uri, text2);

        let cst2 = db.source_cst(&uri).unwrap();
        assert_eq!(cst2.text(&db).to_string(), "fn second() { }");
    }
}
