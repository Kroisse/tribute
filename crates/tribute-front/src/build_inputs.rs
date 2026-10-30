//! The files a build reads, for listing them and watching them for changes
//! (`new-plans/implementation.md` § 패키지 source 트리와 파일 입력).
//!
//! The lists are made from the paths the source tree and manifest queries
//! asked the file provider about, never from walking a directory. They
//! include paths with no file yet, since creating a file there changes the
//! build.

use std::path::PathBuf;

use rustc_hash::FxHashSet as HashSet;

use crate::SourceCst;
use crate::analysis::{DependencyTable, package_tree};
use crate::manifest::{EXECUTABLE_ROOT, LIBRARY_ROOT, beside, library_package, manifest};
use crate::source_file::{FileEntry, file_at, source_at, uri_to_path};
use crate::source_tree::source_tree_of;

/// The files the build of `source` reads as a root compiled without a
/// manifest: the paths its source tree looked for.
#[salsa::tracked]
pub fn source_inputs(db: &dyn salsa::Database, source: SourceCst) -> Vec<PathBuf> {
    let mut inputs = tree_inputs(db, source);
    finish(&mut inputs);
    inputs
}

/// The files the build of the package directory of the manifest `file`
/// reads: its manifest, its root files, and the paths the source tree of
/// each root file that exists looked for. With `dependencies`, also the
/// manifest of each `[dependencies]` entry, transitively, and the files of
/// each dependency whose manifest exists.
#[salsa::tracked]
pub fn package_inputs(
    db: &dyn salsa::Database,
    file: FileEntry,
    implicit: DependencyTable,
    dependencies: bool,
) -> Vec<PathBuf> {
    let mut inputs = package_files(db, file, implicit, true).clone();
    if dependencies {
        let mut seen = HashSet::default();
        seen.insert(file);
        let mut stack = vec![file];
        while let Some(from) = stack.pop() {
            let Some(manifest) = manifest(db, from) else {
                continue;
            };
            for entry in &manifest.dependencies {
                let uri = entry.manifest();
                inputs.extend(uri_to_path(&uri));
                let Some(dependency) = file_at(db, &uri) else {
                    continue;
                };
                // Manifests only report a cycle, so it is not followed again.
                if dependency.source(db).is_some() && seen.insert(dependency) {
                    inputs.extend_from_slice(package_files(db, dependency, implicit, false));
                    stack.push(dependency);
                }
            }
        }
    }
    finish(&mut inputs);
    inputs
}

/// The files of the package of the manifest `file` alone: the manifest,
/// `src/lib.trb` and the files of the library's source tree, and with
/// `executable`, `src/main.trb` and the files of its source tree.
#[salsa::tracked]
fn package_files(
    db: &dyn salsa::Database,
    file: FileEntry,
    implicit: DependencyTable,
    executable: bool,
) -> Vec<PathBuf> {
    let mut inputs: Vec<PathBuf> = uri_to_path(file.uri(db)).into_iter().collect();
    if let Some(uri) = beside(file.uri(db), LIBRARY_ROOT) {
        inputs.extend(uri_to_path(&uri));
        let library =
            library_package(db, file, implicit).and_then(|package| package_tree(db, package));
        match library {
            Some(tree) => inputs.extend_from_slice(tree.probed()),
            // A manifest without a valid name builds no library package, but
            // its root file is still the package's.
            None => {
                if let Some(root) = source_at(db, &uri) {
                    inputs.extend(tree_inputs(db, root));
                }
            }
        }
    }
    if executable && let Some(uri) = beside(file.uri(db), EXECUTABLE_ROOT) {
        inputs.extend(uri_to_path(&uri));
        if let Some(root) = source_at(db, &uri) {
            inputs.extend(tree_inputs(db, root));
        }
    }
    finish(&mut inputs);
    inputs
}

/// The paths the source tree rooted at `source` looked for, or the path of
/// `source` alone if it does not parse.
fn tree_inputs(db: &dyn salsa::Database, source: SourceCst) -> Vec<PathBuf> {
    match source_tree_of(db, source) {
        Some(tree) => tree.probed().to_vec(),
        None => uri_to_path(source.uri(db)).into_iter().collect(),
    }
}

/// Sort `inputs` and drop duplicates.
fn finish(inputs: &mut Vec<PathBuf>) {
    inputs.sort();
    inputs.dedup();
}
