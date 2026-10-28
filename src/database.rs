use std::sync::Arc;

use lsp_types::Uri as LspUri;
use parking_lot::RwLock;
use ropey::Rope;
use tribute_front::source_file::{FileProvider, SourceFiles};
use tribute_front::{SourceCst, Uri, path_to_uri, uri_to_path};

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
///
/// The file's name must match the URI's exactly, so that a case-insensitive
/// file system does not find `foo.trb` for `Foo.trb`.
fn read_file(uri: &Uri<String>) -> Option<Rope> {
    let path = uri_to_path(uri)?;
    let name = path.file_name()?;
    let spelled = std::fs::read_dir(path.parent()?)
        .ok()?
        .any(|entry| entry.is_ok_and(|entry| entry.file_name() == name));
    if !spelled {
        return None;
    }
    Rope::from_reader(std::fs::File::open(&path).ok()?).ok()
}

/// The URI the file provider keys `uri`'s file by: a `file:` URI spelled as
/// [`path_to_uri`] spells its path, whatever encoding the client chose.
fn provider_uri(uri: &LspUri) -> Uri<String> {
    uri_to_path(uri).map_or_else(|| (**uri).clone(), |path| path_to_uri(&path))
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

    /// Open the editor buffer `text` for `uri`. The buffer replaces the
    /// file's text in the source input a package's source tree reads at that
    /// path, so the packages that contain the file see its edits.
    pub fn open_document(&mut self, uri: &LspUri, text: Rope) {
        let file = provider_uri(uri);
        let provider = SourceFiles::get(self).provider(self).clone();
        let source_cst = provider
            .set(self, &file, Some(text))
            .expect("a file given text has a source");
        self.documents
            .write()
            .insert(uri.as_str().to_owned(), source_cst);
    }

    /// Close the editor buffer of `uri`, returning its source input to the
    /// file on disk.
    pub fn close_document(&mut self, uri: &LspUri) {
        if self.documents.write().remove(uri.as_str()).is_none() {
            return;
        }
        let file = provider_uri(uri);
        let provider = SourceFiles::get(self).provider(self).clone();
        provider.set(self, &file, read_file(&file));
    }

    /// The URI and source of each open document.
    pub fn open_documents(&self) -> Vec<(String, SourceCst)> {
        self.documents
            .read()
            .iter()
            .map(|(uri, source)| (uri.clone(), *source))
            .collect()
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
        let mut db = TributeDatabaseImpl::default();
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
        let mut db = TributeDatabaseImpl::default();
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
