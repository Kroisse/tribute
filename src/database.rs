use std::collections::hash_map::Entry;

use lsp_types::Uri;
use parking_lot::RwLock;
use ropey::Rope;
use tribute_front::source_file::parse_rope;
use tribute_front::{SourceCst, path_to_uri};

#[derive(Default)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
    #[allow(
        clippy::disallowed_types,
        reason = "Client-controlled document URI keys require the standard collision-resistant hasher"
    )]
    documents: RwLock<std::collections::HashMap<String, SourceCst>>,
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
        if let Some(entry) = self.documents.read().get(&key) {
            Ok(*entry)
        } else {
            let file = std::fs::File::open(&path)?;
            let contents = Rope::from_reader(file)?;
            let tree = parse_rope(&contents, None);
            let source_cst = SourceCst::new(self, uri, contents, tree);
            let mut lock = self.documents.write();
            match lock.entry(key) {
                Entry::Occupied(entry) => Ok(*entry.get()),
                Entry::Vacant(entry) => {
                    entry.insert(source_cst);
                    Ok(source_cst)
                }
            }
        }
    }

    pub fn open_document(&self, uri: &Uri, text: Rope) {
        let key = uri.as_str().to_owned();
        let tree = parse_rope(&text, None);
        let source_cst = SourceCst::new(self, (**uri).clone(), text, tree);
        self.documents.write().insert(key, source_cst);
    }

    pub fn close_document(&self, uri: &Uri) {
        let key = uri.as_str();
        self.documents.write().remove(key);
    }

    pub fn source_cst(&self, uri: &Uri) -> Option<SourceCst> {
        let key = uri.as_str();
        self.documents.read().get(key).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
