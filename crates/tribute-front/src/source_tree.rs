//! The source tree of a package: its root file and the files of its file
//! modules, read into one AST (`new-plans/implementation.md` § 패키지 source
//! 트리와 파일 입력, `new-plans/modules.md` § 모듈 선언).
//!
//! Each file is parsed by its own query, so editing one file parses only
//! that file again; [`source_tree`] puts each file's declarations in the
//! body of the `mod` that names it, and keeps each file's parse, where its
//! span map finds a node's span by the source hash the node carries. Name
//! resolution and type checking read the result as one package, as if its
//! modules were written inline.

use fluent_uri::Uri;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Span, Symbol, SymbolPath};

use crate::ast::node_id::source_hash;
use crate::ast::{Module, ModuleItem, SpanMap, UnresolvedName};
use crate::astgen::{self, ParsedAst};
use crate::source_file::{SourceCst, push_uri_path, source_at, uri_to_path};

/// A package's AST with each file module's declarations in its body, and
/// the files it was read from.
///
/// Each file keeps its own parse, spans included; the tree's span map finds
/// a node's file by the source hash its `NodeId` carries, so the files'
/// spans are never copied into one map.
#[derive(Clone, PartialEq, Eq, salsa::SalsaValue)]
pub struct SourceTree<'db> {
    /// The package module with file module bodies filled in, or `None` if
    /// the root declares no file module and its own parse is the package.
    spliced: Option<Module<UnresolvedName>>,
    /// The files of the package, sorted by their module path in it; the
    /// root, at the empty path, is first.
    files: Vec<PackageFile<'db>>,
    /// The index in `files` of the file whose spans carry each source id.
    by_source: HashMap<u64, u32>,
}

/// One file of a package: its module path in the package, its source, and
/// its parse.
#[derive(Clone, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct PackageFile<'db> {
    pub path: SymbolPath,
    pub source: SourceCst,
    pub parsed: ParsedAst<'db>,
}

impl<'db> SourceTree<'db> {
    fn new(
        spliced: Option<Module<UnresolvedName>>,
        mut files: Vec<PackageFile<'db>>,
        db: &'db dyn salsa::Database,
    ) -> Self {
        files.sort_by(|left, right| left.path.cmp(&right.path));
        let mut by_source = HashMap::default();
        for (index, file) in files.iter().enumerate() {
            let index = u32::try_from(index).expect("fewer than 2^32 files");
            by_source.insert(source_hash(file.source.uri(db).as_str()), index);
        }
        Self {
            spliced,
            files,
            by_source,
        }
    }

    /// The package's module.
    pub fn module(&self, db: &'db dyn salsa::Database) -> &Module<UnresolvedName> {
        match &self.spliced {
            Some(module) => module,
            None => self.root().parsed.module(db),
        }
    }

    /// The root file.
    pub fn root(&self) -> &PackageFile<'db> {
        &self.files[0]
    }

    /// The files of the package, in module path order.
    pub fn files(&self) -> &[PackageFile<'db>] {
        &self.files
    }

    /// The files of `module` and the modules under it.
    pub fn files_under(&self, module: &SymbolPath) -> &[PackageFile<'db>] {
        let start = self.files.partition_point(|file| file.path < *module);
        let len = self.files[start..].partition_point(|file| file.path.starts_with(module));
        &self.files[start..start + len]
    }

    /// The file whose spans and nodes carry the source id `source`.
    pub fn file(&self, source: u64) -> Option<&PackageFile<'db>> {
        let index = *self.by_source.get(&source)?;
        self.files.get(index as usize)
    }

    /// The spans of the package's nodes, each found in its own file's
    /// parse.
    pub fn span_map(&'db self, db: &'db dyn salsa::Database) -> SpanMap<'db> {
        SpanMap::tree(db, self)
    }
}

/// The source tree of the package whose root file is `root`, its nodes
/// numbered under `module_path`. `None` if `root` does not parse.
///
/// Reports at the `mod` declaration a module whose file is missing, a
/// module with a body whose file also exists, and a file that the tree
/// reaches twice; such a module gets an empty body.
#[salsa::tracked(returns(as_ref))]
pub fn source_tree<'db>(
    db: &'db dyn salsa::Database,
    root: SourceCst,
    module_path: Option<Symbol>,
) -> Option<SourceTree<'db>> {
    let parsed =
        astgen::lower_source_to_parsed_ast_with_module_path(db, root, module_path.clone())?;
    let root_file = PackageFile {
        path: SymbolPath::root(),
        source: root,
        parsed,
    };
    let module = parsed.module(db);
    if !module
        .decls
        .iter()
        .any(|item| matches!(item, ModuleItem::Module(_)))
    {
        return Some(SourceTree::new(None, vec![root_file], db));
    }
    let mut loader = Loader {
        db,
        module_path,
        seen: vec![root],
        files: vec![root_file],
        changed: false,
    };
    let mut decls = module.decls.clone();
    loader.splice(
        &mut decls,
        &SymbolPath::root(),
        module_dir(root.uri(db).as_str()),
        parsed.span_map(db),
    );
    let spliced = loader.changed.then(|| Module {
        id: module.id,
        name: module.name.clone(),
        decls,
    });
    Some(SourceTree::new(spliced, loader.files, db))
}

/// The source tree rooted at `source`, its nodes numbered under the module
/// name its URI gives, as [`crate::query::parsed_ast`] numbers them.
pub fn source_tree_of(db: &dyn salsa::Database, source: SourceCst) -> Option<&SourceTree<'_>> {
    let module_path = astgen::derive_module_name_from_uri(source.uri(db));
    source_tree(db, source, module_path)
}

struct Loader<'db> {
    db: &'db dyn salsa::Database,
    module_path: Option<Symbol>,
    /// The files the tree reached, to report one reached twice.
    seen: Vec<SourceCst>,
    files: Vec<PackageFile<'db>>,
    /// Whether a module's body was filled in.
    changed: bool,
}

impl<'db> Loader<'db> {
    /// Fill in the body of each file module among `items`, the items of the
    /// module at `path` in the package, declared in a file whose spans are
    /// `spans`, with child module files under `dir`.
    fn splice(
        &mut self,
        items: &mut [ModuleItem<UnresolvedName>],
        path: &SymbolPath,
        dir: &str,
        spans: SpanMap<'_>,
    ) {
        for item in items {
            let ModuleItem::Module(module) = item else {
                continue;
            };
            let sub_path = path.child(module.name.clone());
            let mut sub_dir = dir.to_owned();
            push_uri_path(&mut sub_dir, module.name.as_str());
            let path = format!("{sub_dir}.trb");
            sub_dir.push('/');
            let span = spans.get_or_default(module.id);
            let uri = Uri::parse_from(path).ok();
            let found = uri.as_ref().and_then(|uri| source_at(self.db, uri));
            let shown = uri
                .as_ref()
                .and_then(uri_to_path)
                .map_or_else(String::new, |path| path.display().to_string());
            if let Some(body) = &mut module.body {
                if found.is_some() {
                    self.error(
                        span,
                        format!(
                            "module `{}` has a body, but the file `{shown}` also declares it",
                            module.name
                        ),
                    );
                }
                self.splice(body, &sub_path, &sub_dir, spans);
                continue;
            }
            self.changed = true;
            let Some(source) = found else {
                self.error(
                    span,
                    format!("cannot find the file `{shown}` of module `{}`", module.name),
                );
                module.body = Some(Vec::new());
                continue;
            };
            let uri = source.uri(self.db).as_str();
            if self
                .seen
                .iter()
                .any(|file| *file == source || file.uri(self.db).as_str() == uri)
            {
                self.error(
                    span,
                    format!(
                        "the file `{shown}` of module `{}` is already part of this package",
                        module.name
                    ),
                );
                module.body = Some(Vec::new());
                continue;
            }
            self.seen.push(source);
            let Some(parsed) = astgen::lower_file_module(
                self.db,
                source,
                self.module_path.clone(),
                sub_path.clone(),
            ) else {
                module.body = Some(Vec::new());
                continue;
            };
            let mut body = parsed.module(self.db).decls.clone();
            self.splice(&mut body, &sub_path, &sub_dir, parsed.span_map(self.db));
            self.files.push(PackageFile {
                path: sub_path,
                source,
                parsed,
            });
            module.body = Some(body);
        }
    }

    fn error(&self, span: Span, message: String) {
        Diagnostic::new(
            message,
            span,
            DiagnosticSeverity::Error,
            CompilationPhase::AstGeneration,
        )
        .accumulate(self.db);
    }
}

/// The directory part of a file URI, with its trailing `/`.
fn module_dir(uri: &str) -> &str {
    uri.rfind('/').map_or("", |slash| &uri[..=slash])
}
