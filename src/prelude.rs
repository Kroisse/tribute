//! The prelude source: the root module of package `std`.

use ropey::Rope;
use tree_sitter::Parser;
use tribute_front::ast::SpanMap;
use tribute_front::astgen::ParsedAst;
use tribute_front::query as ast_query;
use tribute_front::resolve as ast_resolve;
use tribute_front::resolve::PackageEnv;
use tribute_front::source_file::parse_with_rope;
use tribute_front::typeck as ast_typeck;
use tribute_front::typeck::PreludeExports;

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
pub(crate) fn prelude_source(db: &dyn salsa::Database) -> Option<crate::SourceCst> {
    let uri = fluent_uri::Uri::parse_from(PRELUDE_URI.to_owned()).expect("valid prelude URI");
    let text: Rope = match PreludeExtension::try_get(db) {
        Some(extension) => format!("{PRELUDE_SOURCE}\n{}", extension.text(db)).into(),
        None => PRELUDE_SOURCE.into(),
    };
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None)?;
    Some(crate::SourceCst::new(db, uri, text, Some(tree)))
}

/// Parse the prelude source.
///
/// This is the first stage of prelude processing, shared by all prelude-related functions.
/// Returns both the parsed AST and the SourceCst to avoid redundant creation.
#[salsa::tracked(returns(copy))]
fn parse_prelude<'db>(db: &'db dyn salsa::Database) -> Option<(ParsedAst<'db>, crate::SourceCst)> {
    let prelude_source = prelude_source(db)?;
    let parsed = ast_query::parsed_ast_with_module_path(
        db,
        prelude_source,
        trunk_ir::Symbol::new("prelude"),
    )?;
    Some((parsed, prelude_source))
}

/// Parse, resolve, and type check the prelude once.
///
/// Returns both the typed prelude, which user modules import methods from and
/// merge before lowering, and the exports their type checking reads through
/// the dependency table.
/// Salsa caches the result for all subsequent compilations.
#[salsa::tracked(returns(copy))]
fn checked_prelude<'db>(
    db: &'db dyn salsa::Database,
) -> Option<(ast_typeck::TypeCheckOutput<'db>, PreludeExports<'db>)> {
    let (parsed, _) = parse_prelude(db)?;
    let prelude_ast = parsed.module(db);
    let span_map = parsed.span_map(db);
    let prelude_env = ast_resolve::build_library_env(db, prelude_ast);
    let resolved = ast_resolve::resolve_library_with_env(db, prelude_ast, prelude_env, span_map);

    // Typecheck with independent TypeContext (all UniVars resolved)
    let checker = ast_typeck::TypeChecker::new(db, span_map);
    let (result, exports) = checker.check_prelude(&resolved);

    let typed = ast_typeck::TypeCheckOutput::from_check(db, result);
    Some((typed, exports))
}

/// The span map of the prelude's parse, which locates its nodes in a module
/// merged with it.
pub(crate) fn prelude_span_map(db: &dyn salsa::Database) -> Option<&SpanMap> {
    parse_prelude(db).map(|(parsed, _)| parsed.span_map(db))
}

/// The prelude's typed AST (parse → resolve → typecheck), without
/// lowering to TrunkIR. The caller is responsible for `ast_to_ir`.
pub(crate) fn prelude_module(db: &dyn salsa::Database) -> Option<ast_typeck::TypeCheckOutput<'_>> {
    checked_prelude(db).map(|(typed, _)| typed)
}

/// Get prelude's PackageEnv for name resolution.
///
/// This parses the prelude to AST and builds its package environment.
/// Cached by Salsa - computed once and reused.
#[salsa::tracked(returns(as_ref))]
fn prelude_env<'db>(db: &'db dyn salsa::Database) -> Option<PackageEnv<'db>> {
    let (parsed, _) = parse_prelude(db)?;
    Some(ast_resolve::build_library_env(db, parsed.module(db)))
}

/// The packages a user package depends on: package `std`.
pub fn prelude_dependencies<'db>(db: &'db dyn salsa::Database) -> ast_resolve::Dependencies<'db> {
    prelude_env(db)
        .map(ast_resolve::Dependencies::std)
        .unwrap_or_default()
}

/// The prelude's type exports (TypeSchemes only, no UniVars), which user
/// module type checking reads through the dependency table.
pub(crate) fn prelude_exports(db: &dyn salsa::Database) -> Option<PreludeExports<'_>> {
    checked_prelude(db).map(|(_, exports)| exports)
}
