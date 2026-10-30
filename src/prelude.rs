//! The prelude source: the root module of package `std`.

use ropey::Rope;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tree_sitter::Parser;
use tribute_front::analysis;
use tribute_front::ast::{SpanMap, SpanMaps};
use tribute_front::ast_to_ir;
use tribute_front::resolve as ast_resolve;
use tribute_front::source_file::parse_with_rope;
use tribute_front::typeck as ast_typeck;

/// The prelude source code, embedded at compile time.
const PRELUDE_SOURCE: &str = include_str!("../lib/std/prelude.trb");

/// The URI the prelude source is registered under.
pub(crate) const PRELUDE_URI: &str = "prelude:///std/prelude";

/// Source appended to the prelude and compiled as part of package `std`.
///
/// Tests use it to exercise package-internal items, such as the
/// compiler-owned runtime under `std::runtime`, that user packages cannot
/// name. Create it before the first query on a database: an absent input
/// records no dependency, so creating it later recompiles nothing. Its
/// declarations are prelude declarations and can shadow the prelude's own.
#[salsa::input(singleton)]
pub struct PreludeExtension {
    pub text: String,
}

/// The prelude source, with any [`PreludeExtension`] appended.
///
/// Tracked so that every query in a revision sees the same source input.
#[salsa::tracked]
pub(crate) fn prelude_source(db: &dyn salsa::Database) -> crate::SourceCst {
    let uri = fluent_uri::Uri::parse_from(PRELUDE_URI.to_owned()).expect("valid prelude URI");
    let text: Rope = match PreludeExtension::try_get(db) {
        Some(extension) => format!("{PRELUDE_SOURCE}\n{}", extension.text(db)).into(),
        None => PRELUDE_SOURCE.into(),
    };
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None);
    crate::SourceCst::new(db, uri, text, tree)
}

/// Package `std`, whose root module is the prelude source and whose
/// dependency table is empty.
#[salsa::tracked]
pub fn std_package(db: &dyn salsa::Database) -> analysis::Package {
    let dependencies = analysis::DependencyTable::new(db, Vec::new());
    analysis::Package::new(
        db,
        ast_resolve::PackageName::std(),
        *prelude_source(db),
        dependencies,
    )
}

/// The dependency table the root crate analyzes user sources against:
/// package `std` from the prelude source.
#[salsa::tracked]
pub fn frontend_environment(db: &dyn salsa::Database) -> analysis::DependencyTable {
    analysis::DependencyTable::new(db, vec![*std_package(db)])
}

/// The dependency table the root crate analyzes the root `source` against:
/// that of the package the CLI or the language server registered it as the
/// root of, with package `std`, or package `std` alone for a source without
/// a manifest ([`tribute_front::manifest::source_dependencies`]).
pub fn root_environment(
    db: &dyn salsa::Database,
    source: crate::SourceCst,
) -> analysis::DependencyTable {
    *tribute_front::manifest::source_dependencies(db, source, *frontend_environment(db))
}

/// The runtime types package `std`'s runtime module declares. The check
/// depends only on `std`'s checked result, as `std`'s specialization does.
///
/// Returns `None` without a prelude, or after reporting why the runtime
/// module does not check.
#[salsa::tracked(returns(as_ref))]
pub(crate) fn prelude_runtime_types(
    db: &dyn salsa::Database,
) -> Option<ast_to_ir::runtime_items::RuntimeTypes> {
    let checked = prelude_module(db)?;
    let function_types: HashMap<_, _> = checked
        .declarations(db)
        .function_types
        .iter()
        .cloned()
        .collect();
    match ast_to_ir::runtime_items::check_runtime(
        db,
        &ast_to_ir::runtime_items::RuntimeInputs {
            module: checked.module(db),
            function_types: &function_types,
        },
        tribute_ir::runtime_items::RUNTIME_MODULE,
        tribute_ir::runtime_items::RUNTIME_ITEMS,
    ) {
        Ok(types) => Some(types),
        Err(errors) => {
            let span_map = SpanMaps::new(prelude_span_map(db));
            let names = analysis::package_names(db, *std_package(db));
            for error in errors {
                error.diagnostic(&span_map, &names).accumulate(db);
            }
            None
        }
    }
}

/// The span map of the prelude's parse, which locates its nodes in a module
/// merged with it.
pub(crate) fn prelude_span_map(db: &dyn salsa::Database) -> Option<SpanMap<'_>> {
    analysis::package_tree(db, *std_package(db)).map(|tree| tree.span_map(db))
}

/// The prelude's typed AST (parse → resolve → typecheck), without
/// lowering to TrunkIR. The caller is responsible for `ast_to_ir`.
pub(crate) fn prelude_module(db: &dyn salsa::Database) -> Option<ast_typeck::TypeCheckOutput<'_>> {
    analysis::checked_package(db, *std_package(db)).map(|std| std.checked)
}

/// The packages a user package depends on: package `std`.
pub fn prelude_dependencies<'db>(db: &'db dyn salsa::Database) -> ast_resolve::Dependencies<'db> {
    analysis::dependencies(db, *frontend_environment(db))
}

/// The prelude's type exports (TypeSchemes only, no UniVars), which user
/// module type checking reads through the dependency table.
#[cfg(test)]
pub(crate) fn prelude_exports(
    db: &dyn salsa::Database,
) -> Option<tribute_front::typeck::PackageExports<'_>> {
    analysis::checked_package(db, *std_package(db)).map(|std| std.exports)
}
