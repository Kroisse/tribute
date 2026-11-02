//! The packages the language server analyzes a document in
//! (`new-plans/lsp.md` § 문서가 속한 패키지).

use tribute_core::diagnostic::Diagnostic;
use tribute_front::analysis::{self, Package};
use tribute_front::ast::SpanMap;
use tribute_front::ast::source_hash;
use tribute_front::manifest;
use tribute_front::resolve::{NameDisplay, PackageName};
use tribute_front::source_tree::{PackageFile, SourceTree, source_tree_of};
use tribute_front::typeck::TypeCheckOutput;
use trunk_ir::SymbolPath;

use crate::SourceCst;

/// A package whose source tree is analyzed as one: an executable root, or
/// a library package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Supertype)]
pub enum AnalysisUnit {
    /// The executable package rooted at a source, analyzed against the
    /// dependency table [`crate::root_environment`] gives it.
    Executable(SourceCst),
    /// A library package.
    Library(Package),
}

impl AnalysisUnit {
    /// The root file of the package.
    pub fn root(self, db: &dyn salsa::Database) -> SourceCst {
        match self {
            Self::Executable(root) => root,
            Self::Library(package) => *package.root(db),
        }
    }

    /// The name of the package, which its identity paths start with.
    pub fn package_name(self, db: &dyn salsa::Database) -> PackageName {
        match self {
            Self::Executable(_) => PackageName::executable_root(),
            Self::Library(package) => package.name(db).clone(),
        }
    }

    /// The identity path of the module the file `source` declares, or of
    /// the package's root module if its source tree does not read `source`.
    pub fn module_path(self, db: &dyn salsa::Database, source: SourceCst) -> SymbolPath {
        let path = self.file(db, source).map(|file| file.path.clone());
        self.package_name(db)
            .identity_path(path.as_deref().unwrap_or_default())
    }

    /// The package's source tree: its root file and its file modules.
    pub fn tree(self, db: &dyn salsa::Database) -> Option<&SourceTree<'_>> {
        match self {
            Self::Executable(root) => source_tree_of(db, root),
            Self::Library(package) => analysis::package_tree(db, package),
        }
    }

    /// The file `source` of the package's source tree, if the tree reads it.
    pub fn file(self, db: &dyn salsa::Database, source: SourceCst) -> Option<&PackageFile<'_>> {
        self.tree(db)?.file(source_hash(source.uri(db).as_str()))
    }

    /// Whether the package's source tree reads `source`.
    pub fn reads(self, db: &dyn salsa::Database, source: SourceCst) -> bool {
        self.file(db, source).is_some()
    }

    /// The spans of the nodes of the file `source`, numbered as the
    /// package's source tree numbers them.
    pub fn span_map(self, db: &dyn salsa::Database, source: SourceCst) -> Option<SpanMap<'_>> {
        Some(self.file(db, source)?.parsed.span_map(db))
    }

    /// The package's typed AST, even when its analysis reports errors.
    pub fn typed(self, db: &dyn salsa::Database) -> Option<TypeCheckOutput<'_>> {
        match self {
            Self::Executable(root) => crate::parse_and_lower_ast(db, root),
            Self::Library(package) => {
                analysis::checked_package(db, package).map(|checked| checked.checked)
            }
        }
    }

    /// How the package's analysis spells declarations.
    pub fn names(self, db: &dyn salsa::Database) -> NameDisplay {
        match self {
            Self::Executable(root) => crate::diagnostic_names(db, root),
            Self::Library(package) => analysis::package_names(db, package),
        }
    }

    /// The diagnostics of the package's analysis and of its manifests.
    pub fn diagnostics(self, db: &dyn salsa::Database) -> Vec<Diagnostic> {
        match self {
            Self::Executable(root) => crate::compile_with_diagnostics(db, root).diagnostics,
            Self::Library(package) => {
                let implicit = *crate::frontend_environment(db);
                crate::check_library(db, package);
                let manifest = *manifest::owning_manifest(db, *package.root(db));
                let mut diagnostics: Vec<Diagnostic> = manifest
                    .into_iter()
                    .flat_map(|manifest| {
                        manifest::library_package::accumulated::<Diagnostic>(db, manifest, implicit)
                    })
                    .chain(crate::check_library::accumulated::<Diagnostic>(db, package))
                    .cloned()
                    .collect();
                diagnostics.sort_by(tribute_core::diagnostic::compare_diagnostics);
                diagnostics.dedup();
                diagnostics
            }
        }
    }
}
