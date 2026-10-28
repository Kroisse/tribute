//! The source tree of a package: its root file and the files of its file
//! modules, read into one AST (`new-plans/implementation.md` § 패키지 source
//! 트리와 파일 입력, `new-plans/modules.md` § 모듈 선언).
//!
//! Each file is parsed by its own query, so editing one file parses only
//! that file again; [`source_tree`] puts each file's declarations in the
//! body of the `mod` that names it and merges the span maps. Name
//! resolution and type checking read the result as one package, as if its
//! modules were written inline.

use fluent_uri::Uri;
use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Span, Symbol};

use crate::ast::node_id::source_hash;
use crate::ast::{Module, ModuleItem, SpanMap, UnresolvedName};
use crate::astgen::{self, ParsedAst};
use crate::source_file::{SourceCst, push_uri_path, source_at, uri_to_path};

/// A package's AST with each file module's declarations in its body, and
/// the files they were read from, the root first.
#[derive(Clone, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct SourceTree<'db> {
    pub parsed: ParsedAst<'db>,
    pub files: Vec<SourceCst>,
}

impl SourceTree<'_> {
    /// The file whose spans carry the source id `source`: the root for `0`,
    /// and otherwise the file module whose URI hashes to it.
    pub fn file(&self, db: &dyn salsa::Database, source: u64) -> Option<SourceCst> {
        if source == 0 {
            return self.files.first().copied();
        }
        self.files
            .iter()
            .copied()
            .find(|file| source_hash(file.uri(db).as_str()) == source)
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
    let module = parsed.module(db);
    if !module
        .decls
        .iter()
        .any(|item| matches!(item, ModuleItem::Module(_)))
    {
        return Some(SourceTree {
            parsed,
            files: vec![root],
        });
    }
    let mut loader = Loader {
        db,
        module_path,
        files: vec![root],
        parsed: Vec::new(),
        changed: false,
    };
    let mut decls = module.decls.clone();
    loader.splice(
        &mut decls,
        module_dir(root.uri(db).as_str()),
        parsed.span_map(db),
    );
    if !loader.changed {
        return Some(SourceTree {
            parsed,
            files: loader.files,
        });
    }
    let module = Module {
        id: module.id,
        name: module.name.clone(),
        decls,
    };
    let span_map = SpanMap::merged(
        std::iter::once(parsed.span_map(db))
            .chain(loader.parsed.iter().map(|file| file.span_map(db))),
    );
    Some(SourceTree {
        parsed: ParsedAst::new(db, module, span_map),
        files: loader.files,
    })
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
    files: Vec<SourceCst>,
    parsed: Vec<ParsedAst<'db>>,
    /// Whether a module's body was filled in.
    changed: bool,
}

impl<'db> Loader<'db> {
    /// Fill in the body of each file module among `items`, declared in a
    /// file whose spans are `spans`, with child module files under `dir`.
    fn splice(&mut self, items: &mut [ModuleItem<UnresolvedName>], dir: &str, spans: &SpanMap) {
        for item in items {
            let ModuleItem::Module(module) = item else {
                continue;
            };
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
                self.splice(body, &sub_dir, spans);
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
                .files
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
            self.files.push(source);
            let Some(parsed) = astgen::lower_file_module(self.db, source, self.module_path.clone())
            else {
                module.body = Some(Vec::new());
                continue;
            };
            let mut body = parsed.module(self.db).decls.clone();
            self.splice(&mut body, &sub_dir, parsed.span_map(self.db));
            self.parsed.push(parsed);
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
