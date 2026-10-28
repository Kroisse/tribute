use std::sync::Arc;

use lsp_types::Uri as LspUri;
use parking_lot::RwLock;
use ropey::Rope;
use tribute_front::source_file::{FileProvider, SourceFiles, parse_rope};
use tribute_front::{SourceCst, Uri, path_to_uri};

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

impl Default for TributeDatabaseImpl {
    fn default() -> Self {
        let db = Self {
            storage: salsa::Storage::default(),
            documents: RwLock::default(),
        };
        SourceFiles::new(&db, Arc::new(FileProvider::new(read_file)));
        db
    }
}

/// The text of the file a `file:` URI names, if it can be read.
fn read_file(uri: &Uri<String>) -> Option<Rope> {
    if uri.scheme().map(|scheme| scheme.as_str()) != Some("file") {
        return None;
    }
    let file = std::fs::File::open(uri.path().as_str()).ok()?;
    Rope::from_reader(file).ok()
}

impl TributeDatabaseImpl {
    /// The source of the file at `path`, read from disk on its first
    /// request. It is the same input a package's source tree finds at that
    /// path.
    pub fn input(
        &self,
        path: std::path::PathBuf,
    ) -> Result<SourceCst, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.canonicalize()?;
        let uri = path_to_uri(&path);
        let provider = SourceFiles::get(self).provider(self).clone();
        provider
            .entry(self, &uri)
            .source(self)
            .ok_or_else(|| format!("cannot read {}", path.display()).into())
    }

    pub fn open_document(&self, uri: &LspUri, text: Rope) {
        let key = uri.as_str().to_owned();
        let tree = parse_rope(&text, None);
        let source_cst = SourceCst::new(self, (**uri).clone(), text, tree);
        self.documents.write().insert(key, source_cst);
    }

    pub fn close_document(&self, uri: &LspUri) {
        let key = uri.as_str();
        self.documents.write().remove(key);
    }

    pub fn source_cst(&self, uri: &LspUri) -> Option<SourceCst> {
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
        let uri: LspUri = "file:///test.trb".parse().unwrap();
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
        let uri: LspUri = "file:///test.trb".parse().unwrap();

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
