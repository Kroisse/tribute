//! Salsa-tracked query functions for incremental compilation.
//!
//! This module provides function-level caching for the compilation pipeline.
//! Each function is processed independently, enabling incremental recompilation
//! when only some functions change.
//!
//! ## Caching Strategy
//!
//! - Module-level queries: Parse the entire module, build environments
//! - Function-level queries: Process individual functions using the environment
//!
//! When a function body changes, only that function needs reprocessing.
//! When a signature changes, dependent functions are invalidated automatically.

use trunk_ir::Symbol;

use crate::ast::{
    Decl, FuncDecl, Module, ResolvedRef, SpanMap, TypeScheme, TypedRef, UnresolvedName,
};
use crate::source_file::SourceCst;
use crate::typeck::TypeCheckOutput;

// =============================================================================
// Module-level queries
// =============================================================================

/// Parse a source file to AST.
///
/// This is the single tracking point for CST → AST conversion.
/// Both `parsed_module` and `span_map` derive from this query.
#[salsa::tracked]
pub fn parsed_ast<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<crate::astgen::ParsedAst<'db>> {
    crate::astgen::lower_source_to_parsed_ast(db, source)
}

/// Parse a source file to an AST module.
///
/// This is the entry point for parsing. The result is cached by Salsa.
/// Use `span_map` to get the corresponding span information.
#[salsa::tracked]
pub fn parsed_module<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<UnresolvedName>> {
    parsed_ast(db, source).map(|parsed| parsed.module(db))
}

/// Get the span map for a parsed source file.
///
/// The SpanMap maps NodeId → Span for looking up source locations.
/// Use together with `parsed_module` - both are derived from the same parse.
#[salsa::tracked]
pub fn span_map<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Option<SpanMap> {
    parsed_ast(db, source).map(|parsed| parsed.span_map(db))
}

/// Get the list of function names in a module.
///
/// This is used to iterate over functions for batch processing.
#[salsa::tracked]
pub fn func_names<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Vec<Symbol> {
    let Some(module) = parsed_module(db, source) else {
        return Vec::new();
    };

    module
        .decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::Function(f) => Some(f.name),
            _ => None,
        })
        .collect()
}

/// Resolve all names in a module.
///
/// This delegates to function-level resolution and aggregates results.
#[salsa::tracked]
pub fn resolved_module<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<ResolvedRef<'db>>> {
    let module = parsed_module(db, source)?;
    let sm = span_map(db, source)?;
    Some(crate::resolve::resolve_module(db, module, sm))
}

/// Base query: run type checking once (cached by Salsa).
///
/// Returns a `TypeCheckOutput` containing both the typed AST module
/// and function type schemes.
#[salsa::tracked]
pub fn type_check_output<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<TypeCheckOutput<'db>> {
    let module = resolved_module(db, source)?;
    Some(crate::typeck::typecheck_module_full(db, module))
}

/// Type check a module.
///
/// Derives the typed module from `type_check_output`.
#[salsa::tracked]
pub fn typed_module<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<TypedRef<'db>>> {
    type_check_output(db, source).map(|o| o.module(db))
}

/// Get function type schemes from type checking.
///
/// Returns the function type schemes collected during type checking,
/// keyed by function name (Symbol).
#[salsa::tracked]
pub fn function_schemes<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Vec<(Symbol, TypeScheme<'db>)>> {
    type_check_output(db, source).map(|o| o.function_types(db).clone())
}

/// Type-directed name resolution (TDNR) on a module.
///
/// This resolves UFCS method calls that couldn't be resolved during
/// initial name resolution because they required type information.
#[salsa::tracked]
pub fn tdnr_module<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<TypedRef<'db>>> {
    let module = typed_module(db, source)?;
    Some(crate::tdnr::resolve_tdnr(db, module))
}

// =============================================================================
// Function-level queries
// =============================================================================

/// Get a parsed function by name.
#[salsa::tracked]
pub fn parsed_func<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    name: Symbol,
) -> Option<FuncDecl<UnresolvedName>> {
    let module = parsed_module(db, source)?;

    module.decls.into_iter().find_map(|decl| match decl {
        Decl::Function(f) if f.name == name => Some(f),
        _ => None,
    })
}

/// Resolve a single function by name.
///
/// The function is resolved in the context of the full module environment.
#[salsa::tracked]
pub fn resolved_func<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    name: Symbol,
) -> Option<FuncDecl<ResolvedRef<'db>>> {
    let module = resolved_module(db, source)?;

    module.decls.into_iter().find_map(|decl| match decl {
        Decl::Function(f) if f.name == name => Some(f),
        _ => None,
    })
}

/// Type check a single function by name.
///
/// The function is type checked in the context of the full module.
#[salsa::tracked]
pub fn typed_func<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    name: Symbol,
) -> Option<FuncDecl<TypedRef<'db>>> {
    let module = typed_module(db, source)?;

    module.decls.into_iter().find_map(|decl| match decl {
        Decl::Function(f) if f.name == name => Some(f),
        _ => None,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_file::path_to_uri;
    use ropey::Rope;
    use tree_sitter::Parser;
    use tribute_core::diagnostic::Diagnostic;

    fn make_source(db: &dyn salsa::Database, text: &str) -> SourceCst {
        let uri = path_to_uri(std::path::Path::new("test.trb"));
        let rope = Rope::from_str(text);

        // Parse the source text to get a tree
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(text, None);

        SourceCst::new(db, uri, rope, tree)
    }

    #[test]
    fn test_parsed_module() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        let module = parsed_module(&db, source);
        assert!(module.is_some());

        let module = module.unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_func_names() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"
            fn foo() { 1 }
            fn bar() { 2 }
            struct Point { x: Int, y: Int }
            fn baz() { 3 }
        "#,
        );

        let names = func_names(&db, source);
        assert_eq!(names.len(), 3);
        assert!(names.iter().any(|n| *n == Symbol::new("foo")));
        assert!(names.iter().any(|n| *n == Symbol::new("bar")));
        assert!(names.iter().any(|n| *n == Symbol::new("baz")));
    }

    #[test]
    fn test_parsed_func() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"
            fn foo() { 1 }
            fn bar() { 2 }
        "#,
        );

        let foo = parsed_func(&db, source, Symbol::new("foo"));
        assert!(foo.is_some());
        assert_eq!(foo.unwrap().name.to_string(), "foo");

        let bar = parsed_func(&db, source, Symbol::new("bar"));
        assert!(bar.is_some());
        assert_eq!(bar.unwrap().name.to_string(), "bar");

        let missing = parsed_func(&db, source, Symbol::new("missing"));
        assert!(missing.is_none());
    }

    #[test]
    fn test_resolved_func() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        let func = resolved_func(&db, source, Symbol::new("main"));
        assert!(func.is_some());
    }

    #[test]
    fn test_typed_func() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        let func = typed_func(&db, source, Symbol::new("main"));
        assert!(func.is_some());
    }

    #[test]
    fn test_parsed_ast_provides_both_module_and_span_map() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        // parsed_ast should provide both module and span_map from same parse
        let ast = parsed_ast(&db, source);
        assert!(ast.is_some());

        let ast = ast.unwrap();
        let module = ast.module(&db);
        let sm = ast.span_map(&db);

        // Verify module has content
        assert_eq!(module.decls.len(), 1);

        // Verify span_map has entries for the module's nodes
        if let crate::ast::Decl::Function(func) = &module.decls[0] {
            assert!(sm.get(func.id).is_some(), "Span map should have entries");
        }
    }

    #[test]
    fn test_parsed_module_and_span_map_derive_from_same_parse() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        // Both should succeed if parsed_ast succeeds
        let module = parsed_module(&db, source);
        let sm = span_map(&db, source);

        assert!(module.is_some());
        assert!(sm.is_some());

        // The module's node IDs should be present in the span map
        let module = module.unwrap();
        let sm = sm.unwrap();

        // The function decl should have a valid span in the map
        if let crate::ast::Decl::Function(func) = &module.decls[0] {
            let func_span = sm.get(func.id);
            assert!(func_span.is_some(), "Function decl should have span in map");
        }
    }

    #[test]
    fn test_unresolved_name_emits_diagnostic() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { undefined_var }");

        // Call resolved_module to trigger name resolution
        let _module = resolved_module(&db, source);

        // Collect diagnostics
        let diagnostics: Vec<_> = resolved_module::accumulated::<Diagnostic>(&db, source);

        // Should have a diagnostic about unresolved name
        assert!(
            !diagnostics.is_empty(),
            "Should emit diagnostic for unresolved name"
        );
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message.contains("unresolved name")),
            "Diagnostic should mention unresolved name: {:?}",
            diagnostics
        );
    }
}
