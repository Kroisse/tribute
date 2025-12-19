use std::path::Path;

use dashmap::{DashMap, Entry};
use fluent_uri::Uri;

#[salsa::input(debug)]
pub struct SourceFile {
    #[returns(ref)]
    pub uri: Uri<String>,
    #[returns(deref)]
    pub text: String,
}

impl SourceFile {
    /// Create a SourceFile from a file path (convenience for CLI/tests).
    pub fn from_path(db: &dyn salsa::Database, path: impl AsRef<Path>, text: String) -> Self {
        let uri = path_to_uri(path.as_ref());
        Self::new(db, uri, text)
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

/// Extract the path component from a file:// URI.
pub fn uri_to_path(uri: &Uri<&str>) -> Option<std::path::PathBuf> {
    let scheme = uri.scheme()?;
    if scheme.as_str() != "file" {
        return None;
    }
    let path_str = uri.path().as_str();
    // Handle percent-encoding if needed
    let decoded = percent_decode(path_str);
    Some(std::path::PathBuf::from(decoded))
}

/// Simple percent-decoding for common cases.
fn percent_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[salsa::db]
pub trait Db: salsa::Database {
    fn input(
        &self,
        path: std::path::PathBuf,
    ) -> Result<SourceFile, Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Default, Clone)]
#[salsa::db]
pub struct TributeDatabaseImpl {
    storage: salsa::Storage<Self>,
    /// Cache of loaded source files, keyed by URI string.
    files: DashMap<String, SourceFile>,
}

#[salsa::db]
impl salsa::Database for TributeDatabaseImpl {}

#[salsa::db]
impl Db for TributeDatabaseImpl {
    fn input(
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
