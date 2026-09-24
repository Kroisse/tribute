//! CST to AST lowering.
//!
//! This module converts Tree-sitter CST to the Salsa-tracked AST representation.
//! At this stage, names are unresolved (using `UnresolvedName`).
//!
//! ## Pipeline
//!
//! The lowering produces a `Module<UnresolvedName>` which then flows through:
//! 1. `resolve` - Name resolution → `Module<ResolvedRef>`
//! 2. `typecheck` - Type inference → `Module<TypedRef>`
//! 3. `ast_to_ir` - AST to TrunkIR conversion

mod context;
mod declarations;
mod diagnostics;
mod expressions;
mod helpers;
mod patterns;

use crate::ast::{Module, SpanMap, UnresolvedName};
use crate::query::ParsedCst;
use crate::source_file::SourceCst;
use ropey::Rope;

pub use context::AstLoweringCtx;
pub use declarations::lower_module;
use diagnostics::collect_error_nodes;
pub use expressions::lower_expr;
pub use helpers::is_comment;
pub(super) use helpers::truncate_token_preview;
pub use patterns::lower_pattern;

// =============================================================================
// Entry Points
// =============================================================================

/// Lower a parsed CST to an AST Module (internal, non-Salsa).
///
/// Collects ERROR nodes and lowers the CST into an AST `Module`.
/// Diagnostics are accumulated directly via the context's Salsa database
/// (if present).
fn lower_cst_to_ast_internal(
    ctx: &mut AstLoweringCtx<'_>,
    cst: &ParsedCst,
    module_name: Option<trunk_ir::Symbol>,
) -> Module<UnresolvedName> {
    let root = cst.root_node();

    // Check for ERROR nodes anywhere in the CST
    collect_error_nodes(ctx, root);

    lower_module(ctx, root, module_name)
}

/// Lower a parsed CST to an AST Module.
///
/// This is a convenience entry point for CST → AST conversion that uses
/// `source_hash = 0`. It does not preserve span information and does not
/// distinguish nodes by source origin.
///
/// **Intended for tests and debugging only.** Production code should use
/// [`lower_source_to_parsed_ast`], which computes a proper source hash from
/// the [`SourceCst`] URI and preserves span information.
pub fn lower_cst_to_ast(source: &Rope, cst: &ParsedCst) -> Module<UnresolvedName> {
    let mut ctx = AstLoweringCtx::new(source.clone(), 0);
    lower_cst_to_ast_internal(&mut ctx, cst, None)
}

/// Salsa-tracked parsing result containing both Module and SpanMap.
///
/// This allows both to be computed together and cached efficiently.
#[salsa::tracked]
pub struct ParsedAst<'db> {
    /// The parsed AST module with unresolved names.
    #[returns(clone)]
    pub module: Module<UnresolvedName>,
    /// The span map for looking up source locations.
    #[returns(clone)]
    pub span_map: SpanMap,
}

/// Parse and lower a source file to AST with span information (Salsa-tracked).
///
/// This is the primary entry point for CST → AST conversion.
/// Returns `ParsedAst` containing both the module and span map.
#[salsa::tracked(returns(copy))]
pub fn lower_source_to_parsed_ast<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<ParsedAst<'db>> {
    let module_name = derive_module_name_from_uri(source.uri(db));
    lower_source_to_parsed_ast_with_module_path(db, source, module_name)
}

/// Parse and lower a source file to AST with a specific module path.
///
/// This variant allows specifying a custom module path for the AST nodes,
/// which is useful for parsing library modules (like the prelude) where
/// NodeIds need a different path to avoid collisions with user code.
#[salsa::tracked(returns(copy))]
pub fn lower_source_to_parsed_ast_with_module_path<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    module_path: Option<trunk_ir::Symbol>,
) -> Option<ParsedAst<'db>> {
    use crate::ast::node_id::source_hash;
    use crate::query::parse_cst;

    let cst = parse_cst(db, source)?;
    let text = source.text(db);
    let sh = source_hash(source.uri(db).as_str());
    let mut ctx = AstLoweringCtx::with_db(db, text.clone(), sh);
    let module = lower_cst_to_ast_internal(&mut ctx, &cst, module_path);
    let span_map = ctx.finish().finish();
    Some(ParsedAst::new(db, module, span_map))
}

/// Derive a module name from a source file URI.
///
/// Extracts the file stem (filename without extension) and converts it to a Symbol.
/// Returns `None` if the URI doesn't have a recognizable file path.
fn derive_module_name_from_uri(uri: &fluent_uri::Uri<String>) -> Option<trunk_ir::Symbol> {
    // Get the path component from the URI
    let path_str = uri.path().as_str();

    // Extract the file stem (filename without extension)
    std::path::Path::new(path_str)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(trunk_ir::Symbol::from_dynamic)
}

/// Lower a source file to an AST Module.
///
/// Convenience function that extracts the CST from the source file.
/// Note: This function does not preserve span information.
/// Use `lower_source_to_parsed_ast` for span-preserving lowering.
pub fn lower_source_to_ast(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<UnresolvedName>> {
    lower_source_to_parsed_ast(db, source).map(|parsed| parsed.module(db))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
