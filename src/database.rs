use dashmap::{DashMap, Entry};

use tribute_front::{SourceFile, path_to_uri};

#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
    /// Cache of loaded source files, keyed by URI string.
    files: DashMap<String, SourceFile>,
}

#[salsa::db]
impl salsa::Database for TributeDatabaseImpl {}

impl TributeDatabaseImpl {
    pub fn input(
        &self,
        path: std::path::PathBuf,
    ) -> Result<SourceFile, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.canonicalize()?;
        let uri = path_to_uri(&path);
        let uri_str = uri.as_str().to_owned();
        match self.files.entry(uri_str) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let contents = std::fs::read_to_string(&path)?;
                let source_file = SourceFile::new(self, uri, contents);
                Ok(*entry.insert(source_file))
            }
        }
    }
}
