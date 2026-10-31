//! Package manifests and the package graph they declare
//! (`new-plans/modules.md` § 매니페스트 (`tribute.toml`)).
//!
//! Manifests are read through the same per-URI file inputs as source files
//! (`new-plans/implementation.md` § 패키지 source 트리와 파일 입력), so a
//! query that reads a manifest depends on that file alone. Each manifest is
//! parsed by its own query, and each package directory gets its
//! [`Package`] input from a query keyed by its manifest file: editing a
//! manifest recreates only the package it describes and the packages that
//! depend on it.
//!
//! The packages a manifest names are followed by path. Every path in a
//! manifest is relative to the directory that contains it, never to the
//! working directory, and a directory is compared after folding `.` and
//! `..` in its spelling, so the dependencies that name one directory share
//! one package.

use std::path::{Component, Path, PathBuf};

use fluent_uri::Uri;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use salsa::{Accumulator, Setter as _};
use toml::Spanned;
use toml::de::{DeTable, DeValue};
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Span, Symbol};

use crate::analysis::{DependencyTable, Package};
use crate::ast::source_hash;
use crate::keywords::{PATH_KEYWORDS, can_be_raw};
use crate::resolve::PackageName;
use crate::source_file::{FileEntry, SourceCst, file_at, path_to_uri, source_at, uri_to_path};

/// The file name of a package manifest.
pub const MANIFEST_FILE: &str = "tribute.toml";

/// The root file of a library package, relative to its manifest.
pub(crate) const LIBRARY_ROOT: &str = "src/lib.trb";

/// The root file of an executable package, relative to its manifest.
pub(crate) const EXECUTABLE_ROOT: &str = "src/main.trb";

/// A manifest as written, with the errors in it reported.
#[derive(Clone, Debug, PartialEq, Eq, salsa::SalsaValue)]
pub struct Manifest {
    /// The package name, if the manifest gives a valid one.
    pub name: Option<Symbol>,
    /// Where an error about the package itself points: its `[package]`
    /// key, or the start of the file.
    pub span: Span,
    /// The `[dependencies]` entries that are well formed, in key order.
    pub dependencies: Vec<DependencyEntry>,
}

/// One `[dependencies]` entry.
#[derive(Clone, Debug, PartialEq, Eq, salsa::SalsaValue)]
pub struct DependencyEntry {
    /// The name the entry gives the dependency.
    pub name: Symbol,
    /// The entry's key.
    pub name_span: Span,
    /// The entry's `path` value.
    pub path_span: Span,
    /// The package directory the entry's `path` names.
    pub directory: PathBuf,
}

impl DependencyEntry {
    /// The URI of the dependency's manifest.
    pub fn manifest(&self) -> Uri<String> {
        path_to_uri(&self.directory.join(MANIFEST_FILE))
    }
}

/// What a package directory builds.
#[derive(Clone, Debug, PartialEq, Eq, salsa::SalsaValue)]
pub struct LoadedPackage {
    /// The package name, if its manifest gives a valid one.
    pub name: Option<Symbol>,
    /// The library package rooted at `src/lib.trb`.
    pub library: Option<Package>,
    /// `src/main.trb`, the root file of the executable package.
    pub executable: Option<SourceCst>,
}

/// The manifest in `file`, or `None` if there is no such file.
///
/// Reports a TOML syntax error, a missing or invalid package name, and a
/// malformed `[dependencies]` entry, which it leaves out. A manifest that
/// does not parse has no name and no dependencies.
#[salsa::tracked(returns(as_ref))]
pub fn manifest(db: &dyn salsa::Database, file: FileEntry) -> Option<Manifest> {
    let source = file.source(db)?;
    let text = source.text(db).to_string();
    let mut reader = Reader {
        db,
        uri: file.uri(db),
        source: source_hash(file.uri(db).as_str()),
    };
    let start = Span::new(0, 0).in_source(reader.source);
    let document = match DeTable::parse(&text) {
        Ok(document) => document.into_inner(),
        Err(error) => {
            let span = error
                .span()
                .map_or(start, |range| reader.span(range.start..range.end));
            reader.error(span, format!("invalid manifest: {}", error.message()));
            return Some(Manifest {
                name: None,
                span: start,
                dependencies: Vec::new(),
            });
        }
    };
    let (name, span) = match lookup(&document, "package") {
        Some((key, value)) => (reader.package_name(key, value), reader.span_of(key)),
        None => {
            reader.error(start, "the manifest has no `[package]` table".to_owned());
            (None, start)
        }
    };
    let dependencies = match lookup(&document, "dependencies") {
        Some((key, value)) => reader.dependencies(key, value),
        None => Vec::new(),
    };
    Some(Manifest {
        name,
        span,
        dependencies,
    })
}

/// What the package directory of the manifest `file` builds, or `None` if
/// there is no such file. Reports a package with neither root file.
#[salsa::tracked(returns(as_ref))]
pub fn load_package(
    db: &dyn salsa::Database,
    file: FileEntry,
    implicit: DependencyTable,
) -> Option<LoadedPackage> {
    let manifest = manifest(db, file)?;
    let has_library = source_at(db, &beside(file.uri(db), LIBRARY_ROOT)?).is_some();
    let executable = source_at(db, &beside(file.uri(db), EXECUTABLE_ROOT)?);
    if !has_library && executable.is_none() {
        error(
            db,
            manifest.span,
            CompilationPhase::NameResolution,
            format!(
                "the package has neither `{LIBRARY_ROOT}` nor `{EXECUTABLE_ROOT}` beside its manifest"
            ),
        );
    }
    Some(LoadedPackage {
        name: manifest.name.clone(),
        library: *library_package(db, file, implicit),
        executable,
    })
}

/// The library package of the manifest `file`, rooted at its `src/lib.trb`:
/// `None` if the manifest gives no valid name or there is no such file.
///
/// It reads its manifest only through [`package_name`] and
/// [`dependency_table`], so an edit that only moves text in a manifest
/// keeps the package.
#[salsa::tracked]
pub fn library_package(
    db: &dyn salsa::Database,
    file: FileEntry,
    implicit: DependencyTable,
) -> Option<Package> {
    let name = package_name(db, file).clone()?;
    let root = source_at(db, &beside(file.uri(db), LIBRARY_ROOT)?)?;
    Some(Package::new(
        db,
        PackageName::library(name),
        root,
        *dependency_table(db, file, implicit),
    ))
}

/// The dependency table the manifest `file` declares: the library package
/// of each `[dependencies]` entry that loads, then the packages of
/// `implicit`, which every package depends on.
#[salsa::tracked]
pub fn dependency_table(
    db: &dyn salsa::Database,
    file: FileEntry,
    implicit: DependencyTable,
) -> DependencyTable {
    let mut packages: Vec<Package> = loaded_dependencies(db, file)
        .iter()
        .filter_map(|&dependency| *library_package(db, dependency, implicit))
        .collect();
    packages.extend(implicit.packages(db).iter().copied());
    DependencyTable::new(db, packages)
}

/// The package name the manifest `file` gives, if it is valid.
#[salsa::tracked]
fn package_name(db: &dyn salsa::Database, file: FileEntry) -> Option<Symbol> {
    manifest(db, file).as_ref()?.name.clone()
}

/// The manifests of the `[dependencies]` entries of `file` that load.
///
/// Reports at its entry a dependency with no manifest, one whose package
/// name differs from the entry's, one without `src/lib.trb`, and one that
/// depends on this package again; none of them is in the result.
#[salsa::tracked]
fn loaded_dependencies(db: &dyn salsa::Database, file: FileEntry) -> Vec<FileEntry> {
    let Some(manifest) = manifest(db, file) else {
        return Vec::new();
    };
    manifest
        .dependencies
        .iter()
        .filter_map(|entry| dependency(db, file, entry))
        .collect()
}

/// The manifest of the package of each root source a tool analyzes as a
/// package root (`new-plans/implementation.md` § 패키지 source 트리와 파일
/// 입력). The tool that picks a root registers it: the CLI a package
/// directory's `src/main.trb`, and the language server the root of the
/// package an open document belongs to.
#[salsa::input(singleton, debug)]
pub struct PackageRoots {
    #[returns(ref)]
    pub manifests: HashMap<SourceCst, FileEntry>,
}

/// Analyze the root `source` against the dependency table of the manifest
/// `manifest`, or against the implicit table if `manifest` is `None`.
pub fn set_package_root(
    db: &mut dyn salsa::Database,
    source: SourceCst,
    manifest: Option<FileEntry>,
) {
    let Some(roots) = PackageRoots::try_get(db) else {
        if let Some(manifest) = manifest {
            PackageRoots::new(db, [(source, manifest)].into_iter().collect());
        }
        return;
    };
    let mut manifests = roots.manifests(db).clone();
    let changed = match manifest {
        Some(manifest) => manifests.insert(source, manifest) != Some(manifest),
        None => manifests.remove(&source).is_some(),
    };
    if changed {
        roots.set_manifests(db).to(manifests);
    }
}

/// The dependency table the root `source` is analyzed against: that of the
/// manifest [`PackageRoots`] registers for it, or `implicit` for a source
/// compiled without a manifest.
#[salsa::tracked]
pub fn source_dependencies(
    db: &dyn salsa::Database,
    source: SourceCst,
    implicit: DependencyTable,
) -> DependencyTable {
    let manifest =
        PackageRoots::try_get(db).and_then(|roots| roots.manifests(db).get(&source).copied());
    match manifest {
        Some(file) if file.source(db).is_some() => *dependency_table(db, file, implicit),
        _ => implicit,
    }
}

/// The manifest beside the package root file at `uri`, if `uri` is a
/// `src/main.trb` or `src/lib.trb`. Whether the manifest exists is not
/// checked.
pub fn root_file_manifest(uri: &Uri<String>) -> Option<Uri<String>> {
    let path = uri_to_path(uri)?;
    if path.file_name()? != "main.trb" && path.file_name()? != "lib.trb" {
        return None;
    }
    let src = path.parent()?;
    if src.file_name()? != "src" {
        return None;
    }
    Some(path_to_uri(&src.parent()?.join(MANIFEST_FILE)))
}

/// The manifest of the package `entry` of the manifest `from` names, or
/// `None` after reporting why it does not load.
fn dependency(
    db: &dyn salsa::Database,
    from: FileEntry,
    entry: &DependencyEntry,
) -> Option<FileEntry> {
    let directory = entry.directory.display();
    let file = file_at(db, &entry.manifest())?;
    let Some(manifest) = manifest(db, file) else {
        error(
            db,
            entry.path_span,
            CompilationPhase::NameResolution,
            format!(
                "cannot find `{MANIFEST_FILE}` of dependency `{}` in `{directory}`",
                entry.name
            ),
        );
        return None;
    };
    if reaches(db, file, from) {
        error(
            db,
            entry.name_span,
            CompilationPhase::NameResolution,
            format!(
                "dependency `{}` depends on this package again, forming a cycle",
                entry.name
            ),
        );
        return None;
    }
    // A manifest without a valid name has reported why.
    let name = manifest.name.as_ref()?;
    if *name != entry.name {
        error(
            db,
            entry.name_span,
            CompilationPhase::NameResolution,
            format!(
                "dependency `{}` names the package `{name}` in `{directory}`; the entry must be named `{name}`",
                entry.name
            ),
        );
        return None;
    }
    if source_at(db, &beside(file.uri(db), LIBRARY_ROOT)?).is_none() {
        error(
            db,
            entry.path_span,
            CompilationPhase::NameResolution,
            format!("dependency `{name}` has no `{LIBRARY_ROOT}` in `{directory}`"),
        );
        return None;
    }
    Some(file)
}

/// Whether the manifest `from`, or a manifest its dependencies name,
/// transitively, is `target`.
fn reaches(db: &dyn salsa::Database, from: FileEntry, target: FileEntry) -> bool {
    let mut seen = HashSet::default();
    let mut stack = vec![from];
    while let Some(file) = stack.pop() {
        if file == target {
            return true;
        }
        if !seen.insert(file) {
            continue;
        }
        let Some(manifest) = manifest(db, file) else {
            continue;
        };
        stack.extend(
            manifest
                .dependencies
                .iter()
                .filter_map(|entry| file_at(db, &entry.manifest())),
        );
    }
    false
}

/// The URI of `relative` in the directory of the file at `uri`.
pub(crate) fn beside(uri: &Uri<String>, relative: &str) -> Option<Uri<String>> {
    let path = uri_to_path(uri)?;
    Some(path_to_uri(&normalize(&path.parent()?.join(relative))))
}

/// `path` with each `.` dropped and each `..` folded into the component
/// before it, by spelling alone.
fn normalize(path: &Path) -> PathBuf {
    let mut normal = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if normal.file_name().is_some() {
                    normal.pop();
                } else if !normal.has_root() {
                    normal.push(component);
                }
            }
            _ => normal.push(component),
        }
    }
    normal
}

/// The first value of `table` at `key`, with its key.
fn lookup<'t, 'i>(
    table: &'t DeTable<'i>,
    key: &str,
) -> Option<(
    &'t Spanned<std::borrow::Cow<'i, str>>,
    &'t Spanned<DeValue<'i>>,
)> {
    table.iter().find(|(name, _)| name.get_ref() == key)
}

/// Reads one manifest's tables, reporting what is wrong in them.
struct Reader<'a> {
    db: &'a dyn salsa::Database,
    uri: &'a Uri<String>,
    source: u64,
}

impl Reader<'_> {
    /// The package name `[package]` gives, if it is valid.
    fn package_name(
        &mut self,
        key: &Spanned<std::borrow::Cow<'_, str>>,
        value: &Spanned<DeValue<'_>>,
    ) -> Option<Symbol> {
        let DeValue::Table(package) = value.get_ref() else {
            self.error(self.span_of(value), "`package` must be a table".to_owned());
            return None;
        };
        let Some((_, name)) = lookup(package, "name") else {
            self.error(
                self.span_of(key),
                "the `[package]` table has no `name`".to_owned(),
            );
            return None;
        };
        let DeValue::String(text) = name.get_ref() else {
            self.error(self.span_of(name), "`name` must be a string".to_owned());
            return None;
        };
        self.check_name(text, self.span_of(name))
    }

    /// `text` as a package name, or `None` after reporting why it is not one.
    fn check_name(&self, text: &str, span: Span) -> Option<Symbol> {
        let message = if PATH_KEYWORDS.contains(&text) {
            format!("`{text}` is a path keyword and cannot name a package")
        } else if text == "std" {
            "`std` names the standard library, which every package depends on".to_owned()
        } else if !can_be_raw(text) {
            format!("`{text}` is not a package name; it must be a lowercase identifier")
        } else {
            return Some(Symbol::new(text));
        };
        self.error(span, message);
        None
    }

    /// The well-formed entries of `[dependencies]`.
    fn dependencies(
        &mut self,
        key: &Spanned<std::borrow::Cow<'_, str>>,
        value: &Spanned<DeValue<'_>>,
    ) -> Vec<DependencyEntry> {
        let DeValue::Table(table) = value.get_ref() else {
            self.error(
                self.span_of(key),
                "`dependencies` must be a table".to_owned(),
            );
            return Vec::new();
        };
        table
            .iter()
            .filter_map(|(key, value)| self.dependency(key, value))
            .collect()
    }

    fn dependency(
        &mut self,
        key: &Spanned<std::borrow::Cow<'_, str>>,
        value: &Spanned<DeValue<'_>>,
    ) -> Option<DependencyEntry> {
        let name_span = self.span_of(key);
        let name = self.check_name(key.get_ref(), name_span)?;
        let DeValue::Table(entry) = value.get_ref() else {
            self.error(
                self.span_of(value),
                format!("dependency `{name}` must be a table `{{ path = \"…\" }}`"),
            );
            return None;
        };
        for (field, _) in entry.iter().filter(|(field, _)| field.get_ref() != "path") {
            self.error(
                self.span_of(field),
                format!(
                    "unknown key `{}` in dependency `{name}`; a dependency has only `path`",
                    field.get_ref()
                ),
            );
        }
        let Some((_, path)) = lookup(entry, "path") else {
            self.error(name_span, format!("dependency `{name}` has no `path`"));
            return None;
        };
        let DeValue::String(text) = path.get_ref() else {
            self.error(self.span_of(path), "`path` must be a string".to_owned());
            return None;
        };
        let path_span = self.span_of(path);
        let directory = uri_to_path(self.uri)?.parent()?.join(text.as_ref());
        Some(DependencyEntry {
            name,
            name_span,
            path_span,
            directory: normalize(&directory),
        })
    }

    fn span_of<T>(&self, value: &Spanned<T>) -> Span {
        self.span(value.span())
    }

    fn span(&self, range: std::ops::Range<usize>) -> Span {
        Span::new(range.start, range.end).in_source(self.source)
    }

    fn error(&self, span: Span, message: String) {
        error(self.db, span, CompilationPhase::Parsing, message);
    }
}

fn error(db: &dyn salsa::Database, span: Span, phase: CompilationPhase, message: String) {
    Diagnostic::new(message, span, DiagnosticSeverity::Error, phase).accumulate(db);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_folds_dots_by_spelling() {
        for (path, expected) in [
            ("/ws/app/../geometry", "/ws/geometry"),
            ("/ws/app/./lib/../geometry/.", "/ws/app/geometry"),
            ("/ws/app/../../../x", "/x"),
        ] {
            assert_eq!(normalize(Path::new(path)), Path::new(expected), "{path}");
        }
    }

    #[test]
    fn only_a_packages_root_files_have_a_manifest() {
        for (path, expected) in [
            ("/ws/app/src/main.trb", Some("file:///ws/app/tribute.toml")),
            ("/ws/app/src/lib.trb", Some("file:///ws/app/tribute.toml")),
            ("/ws/app/main.trb", None),
            ("/ws/app/src/util/main.trb", None),
        ] {
            let manifest = root_file_manifest(&path_to_uri(Path::new(path)));
            assert_eq!(manifest.as_ref().map(Uri::as_str), expected, "{path}");
        }
    }
}
