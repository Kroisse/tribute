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
