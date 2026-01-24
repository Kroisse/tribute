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

use crate::ast::{Decl, FuncDecl, Module, ResolvedRef, SpanMap, TypedRef, UnresolvedName};
use crate::source_file::SourceCst;

// =============================================================================
// Module-level queries
// =============================================================================

/// Parse a source file to an AST module.
///
/// This is the entry point for parsing. The result is cached by Salsa.
/// Use `span_map` to get the corresponding span information.
#[salsa::tracked]
pub fn parsed_module<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<UnresolvedName>> {
    crate::astgen::lower_source_to_parsed_ast(db, source).map(|parsed| parsed.module(db))
}

/// Get the span map for a parsed source file.
///
/// The SpanMap maps NodeId → Span for looking up source locations.
/// Use together with `parsed_module` - both are derived from the same parse.
#[salsa::tracked]
pub fn span_map<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Option<SpanMap> {
    crate::astgen::lower_source_to_parsed_ast(db, source).map(|parsed| parsed.span_map(db))
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
    Some(crate::resolve::resolve_module(db, module))
}

/// Type check a module.
///
/// This delegates to function-level type checking and aggregates results.
#[salsa::tracked]
pub fn typed_module<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<TypedRef<'db>>> {
    let module = resolved_module(db, source)?;
    Some(crate::typeck::typecheck_module(db, module))
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
}
