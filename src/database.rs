use std::path::{Path, PathBuf};
use std::sync::Arc;

use lsp_types::Uri as LspUri;
use parking_lot::RwLock;
use ropey::Rope;
use tribute_front::build_inputs::{package_inputs, source_inputs};
use tribute_front::manifest::{MANIFEST_FILE, set_package_root};
use tribute_front::source_file::{FileEntry, FileProvider, SourceFiles};
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

    /// The manifest `tribute.toml` of the package directory `dir`, read
    /// from disk on its first request.
    pub fn manifest(
        &self,
        dir: &std::path::Path,
    ) -> Result<FileEntry, Box<dyn std::error::Error + Send + Sync>> {
        let dir = dir.canonicalize()?;
        let uri = path_to_uri(&dir.join(MANIFEST_FILE));
        let provider = SourceFiles::get(self).provider(self).clone();
        let entry = provider.entry(self, &uri);
        if entry.source(self).is_some() {
            return Ok(entry);
        }
        let misspelled = std::fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .find(|name| name.eq_ignore_ascii_case(MANIFEST_FILE));
        Err(match misspelled {
            Some(name) => format!(
                "cannot find `{MANIFEST_FILE}` in {}; the manifest must be named `{MANIFEST_FILE}`, not `{}`",
                dir.display(),
                name.display()
            ),
            None => format!("cannot find `{MANIFEST_FILE}` in {}", dir.display()),
        }
        .into())
    }

    /// Open the editor buffer `text` for `uri`. The buffer replaces the
    /// file's text in the source input a package's source tree reads at that
    /// path, so the packages that contain the file see its edits. A
    /// document in a package is analyzed as a root with that package's
    /// dependency table (`new-plans/lsp.md` § 문서가 속한 패키지).
    pub fn open_document(&mut self, uri: &LspUri, text: Rope) {
        let file = provider_uri(uri);
        let provider = SourceFiles::get(self).provider(self).clone();
        let source_cst = provider
            .set(self, &file, Some(text))
            .expect("a file given text has a source");
        let manifest = document_manifest(self, &provider, &file);
        set_package_root(self, source_cst, manifest);
        self.documents
            .write()
            .insert(uri.as_str().to_owned(), source_cst);
    }

    /// Close the editor buffer of `uri`, returning its source input to the
    /// file on disk.
    pub fn close_document(&mut self, uri: &LspUri) {
        let Some(source_cst) = self.documents.write().remove(uri.as_str()) else {
            return;
        };
        set_package_root(self, source_cst, None);
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

    /// The files that analyzing the open documents reads, other than the
    /// documents themselves, which the language server watches
    /// (`new-plans/lsp.md` § 입력과 무효화): each document's source tree,
    /// the package it belongs to with its dependencies, and the manifests
    /// looked for to find that package.
    pub fn watched_files(&self) -> Vec<PathBuf> {
        let documents = self.open_documents();
        let provider = SourceFiles::get(self).provider(self).clone();
        let implicit = *crate::frontend_environment(self);
        let mut files = Vec::new();
        for (_, source) in &documents {
            files.extend_from_slice(source_inputs(self, *source));
            let Some(path) = uri_to_path(source.uri(self)) else {
                continue;
            };
            for manifest in manifest_candidates(&path) {
                let entry = provider.entry(self, &path_to_uri(&manifest));
                files.push(manifest);
                if entry.source(self).is_some() {
                    files.extend_from_slice(package_inputs(self, entry, implicit, true));
                    break;
                }
            }
        }
        let open: Vec<PathBuf> = documents
            .iter()
            .filter_map(|(_, source)| uri_to_path(source.uri(self)))
            .collect();
        files.retain(|file| !open.contains(file));
        files.sort();
        files.dedup();
        files
    }

    /// Read the file at `uri` from disk again, after it changed there. An
    /// open document keeps its buffer, and a file no analysis asked for is
    /// left unread. Each open document is then analyzed in the package it
    /// now belongs to, since a manifest may have appeared or disappeared.
    pub fn reload_file(&mut self, uri: &LspUri) {
        if self.source_cst(uri).is_some() {
            return;
        }
        let file = provider_uri(uri);
        let provider = SourceFiles::get(self).provider(self).clone();
        if provider.existing(&file).is_none() {
            return;
        }
        provider.set(self, &file, read_file(&file));
        for (_, source) in self.open_documents() {
            let manifest = document_manifest(self, &provider, source.uri(self));
            set_package_root(self, source, manifest);
        }
    }
}

/// The manifests that may make the document at `path` part of a package,
/// nearest first: `tribute.toml` beside each `src` directory above it.
fn manifest_candidates(path: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    path.ancestors()
        .skip(1)
        .filter(|dir| dir.file_name().is_some_and(|name| name == "src"))
        .filter_map(|src| src.parent())
        .map(|package| package.join(MANIFEST_FILE))
}

/// The manifest of the package the document at `file` belongs to: that of
/// the nearest ancestor directory with a manifest whose `src/` contains the
/// document (`new-plans/lsp.md` § 문서가 속한 패키지).
fn document_manifest(
    db: &dyn salsa::Database,
    provider: &FileProvider,
    file: &Uri<String>,
) -> Option<FileEntry> {
    let path = uri_to_path(file)?;
    manifest_candidates(&path)
        .map(|manifest| provider.entry(db, &path_to_uri(&manifest)))
        .find(|manifest| manifest.source(db).is_some())
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
    /// An open package root file is analyzed against its package's
    /// dependencies until it is closed.
    #[test]
    fn an_open_root_file_reads_its_packages_manifest() {
        let mut db = TributeDatabaseImpl::default();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/packages/app/src/main.trb");
        let uri: LspUri = path_to_uri(&path).as_str().parse().unwrap();
        let text = Rope::from_str(&std::fs::read_to_string(&path).unwrap());
        db.open_document(&uri, text);
        let source = db.source_cst(&uri).unwrap();
        let names = |db: &TributeDatabaseImpl| {
            crate::root_environment(db, source)
                .packages(db)
                .iter()
                .map(|package| package.name(db).name().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&db), ["geometry", "std"]);
        db.close_document(&uri);
        assert_eq!(names(&db), ["std"]);
    }

    /// A file module's document, which no root on disk reads here, is
    /// analyzed with its package's dependency table.
    #[test]
    fn an_open_file_module_reads_its_packages_manifest() {
        let mut db = TributeDatabaseImpl::default();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/packages/app/src/shapes/square.trb");
        let uri: LspUri = path_to_uri(&path).as_str().parse().unwrap();
        db.open_document(&uri, Rope::from_str("pub fn side() -> Nat { 3 }\n"));
        let source = db.source_cst(&uri).unwrap();
        let names = crate::root_environment(&db, source)
            .packages(&db)
            .iter()
            .map(|package| package.name(&db).name().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(names, ["geometry", "std"]);
    }

    /// A document outside any package watches the manifest that would put
    /// it in one; once that manifest appears, it is analyzed in the package
    /// and watches the package's files.
    #[test]
    fn a_new_manifest_puts_an_open_document_in_its_package() {
        let dir = tempfile::tempdir().unwrap();
        let geometry = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/packages/geometry");
        let package = dir.path();
        let main = package.join("src/main.trb");
        let uri: LspUri = path_to_uri(&main).as_str().parse().unwrap();
        let mut db = TributeDatabaseImpl::default();
        db.open_document(&uri, Rope::from_str("fn main() -> Nil { }\n"));
        let source = db.source_cst(&uri).unwrap();
        let manifest = package.join(MANIFEST_FILE);
        assert_eq!(db.watched_files(), [manifest.as_path()]);

        std::fs::write(
            &manifest,
            format!(
                "[package]\nname = \"app\"\n\n[dependencies]\ngeometry = {{ path = {:?} }}\n",
                geometry.display().to_string()
            ),
        )
        .unwrap();
        db.reload_file(&path_to_uri(&manifest).as_str().parse().unwrap());
        let names: Vec<_> = crate::root_environment(&db, source)
            .packages(&db)
            .iter()
            .map(|package| package.name(&db).name().unwrap().to_string())
            .collect();
        assert_eq!(names, ["geometry", "std"]);
        let mut expected = vec![
            geometry.join("src/lib.trb"),
            geometry.join(MANIFEST_FILE),
            package.join("src/lib.trb"),
            manifest,
        ];
        expected.sort();
        assert_eq!(db.watched_files(), expected);
    }
}
