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

use std::hash::{Hash, Hasher};

use tree_sitter::{Node, Tree};
use trunk_ir::Symbol;

use crate::ast::{
    Decl, FuncDecl, Module, ResolvedRef, SpanMap, TypeScheme, TypedRef, UnresolvedName,
};
use crate::source_file::SourceCst;
use crate::typeck::TypeCheckOutput;

// =============================================================================
// ParsedCst (Salsa-cacheable CST wrapper)
// =============================================================================

/// A parsed CST tree, wrapped for Salsa caching.
///
/// Tree-sitter's `Tree` is internally reference-counted (`ts_tree_copy` is O(1)),
/// so cloning is cheap and we can use it directly without additional wrapping.
#[derive(Clone, Debug)]
pub struct ParsedCst(Tree);

impl ParsedCst {
    /// Create a new ParsedCst from a tree-sitter Tree.
    pub fn new(tree: Tree) -> Self {
        Self(tree)
    }

    /// Get a reference to the underlying tree.
    pub fn tree(&self) -> &Tree {
        &self.0
    }

    /// Get the root node of the CST.
    pub fn root_node(&self) -> Node<'_> {
        self.0.root_node()
    }
}

impl PartialEq for ParsedCst {
    fn eq(&self, other: &Self) -> bool {
        // Trees are equal if they have the same root node id AND byte range.
        // The byte range helps distinguish different parses that happen to
        // have the same node id but different source lengths.
        let self_root = self.0.root_node();
        let other_root = other.0.root_node();
        self_root.id() == other_root.id() && self_root.byte_range() == other_root.byte_range()
    }
}

impl Eq for ParsedCst {}

impl Hash for ParsedCst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by root node id and byte range to reduce collision risk
        // across different parse sessions.
        let root = self.0.root_node();
        root.id().hash(state);
        root.byte_range().hash(state);
    }
}

// =============================================================================
// CST Parsing
// =============================================================================

/// Wrap a pre-parsed CST stored in the database.
#[salsa::tracked]
pub fn parse_cst(db: &dyn salsa::Database, source: SourceCst) -> Option<ParsedCst> {
    let tree = source.tree(db).clone()?;
    Some(ParsedCst::new(tree))
}

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

/// Parse a source file to AST with a specific module path.
///
/// This variant is used for parsing the prelude or other library modules
/// where NodeIds need a different module path to avoid collisions.
#[salsa::tracked]
pub fn parsed_ast_with_module_path<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    module_path: Symbol,
) -> Option<crate::astgen::ParsedAst<'db>> {
    crate::astgen::lower_source_to_parsed_ast_with_module_path(db, source, Some(module_path))
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
    let sm = span_map(db, source)?;
    Some(crate::typeck::typecheck_module_full(db, module, sm))
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
    fn test_function_schemes_returns_entries() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { 42 }");

        let schemes = function_schemes(&db, source);
        assert!(schemes.is_some(), "function_schemes should return Some");

        let schemes = schemes.unwrap();
        assert!(
            schemes.iter().any(|(name, _)| *name == Symbol::new("foo")),
            "function_schemes should contain 'foo', got: {:?}",
            schemes
                .iter()
                .map(|(n, _)| n.to_string())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_type_check_output_has_both_fields() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        let output = type_check_output(&db, source);
        assert!(output.is_some(), "type_check_output should return Some");

        let output = output.unwrap();
        // Module should have declarations
        let module = output.module(&db);
        assert!(!module.decls.is_empty(), "Module should have declarations");

        // function_types should be a valid Vec (possibly empty for simple cases)
        let _ft = output.function_types(&db);
    }

    /// Recursively check if a type contains any unresolved UniVar.
    fn contains_univar(db: &dyn salsa::Database, ty: crate::ast::Type) -> bool {
        match ty.kind(db) {
            crate::ast::TypeKind::UniVar { .. } => true,
            crate::ast::TypeKind::Func {
                params,
                result,
                effect: _,
            } => params.iter().any(|p| contains_univar(db, *p)) || contains_univar(db, *result),
            crate::ast::TypeKind::Named { args, .. } => {
                args.iter().any(|a| contains_univar(db, *a))
            }
            crate::ast::TypeKind::Tuple(elems) => elems.iter().any(|e| contains_univar(db, *e)),
            crate::ast::TypeKind::App { ctor, args } => {
                contains_univar(db, *ctor) || args.iter().any(|a| contains_univar(db, *a))
            }
            _ => false,
        }
    }

    #[test]
    fn test_function_schemes_no_univars() {
        let db = salsa::DatabaseImpl::default();
        // Use annotated parameters to ensure type inference resolves fully
        let source = make_source(
            &db,
            r#"
            fn add(x: Int, y: Int) -> Int { x + y }
            fn greet() -> String { "hello" }
        "#,
        );

        let schemes = function_schemes(&db, source);
        assert!(schemes.is_some(), "function_schemes should return Some");

        let schemes = schemes.unwrap();
        for (name, scheme) in &schemes {
            let body = scheme.body(&db);
            assert!(
                !contains_univar(&db, body),
                "Function '{}' scheme body should not contain UniVar, got: {:?}",
                name,
                body.kind(&db),
            );
        }
    }

    #[test]
    fn test_function_schemes_fully_annotated_no_univars() {
        let db = salsa::DatabaseImpl::default();
        // Fully annotated function (params + return) should never have UniVars
        let source = make_source(&db, "fn add(x: Int, y: Int) -> Int { x + y }");

        let schemes = function_schemes(&db, source);
        assert!(schemes.is_some(), "function_schemes should return Some");

        let schemes = schemes.unwrap();
        let add = schemes
            .iter()
            .find(|(name, _)| *name == Symbol::new("add"))
            .expect("should have 'add' scheme");

        let body = add.1.body(&db);
        assert!(
            !contains_univar(&db, body),
            "Fully annotated function 'add' scheme body should not contain UniVar, got: {:?}",
            body.kind(&db),
        );
    }

    #[test]
    fn test_function_schemes_annotated_preserved() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn inc(x: Int) -> Int { x + 1 }");

        let schemes = function_schemes(&db, source);
        assert!(schemes.is_some(), "function_schemes should return Some");

        let schemes = schemes.unwrap();
        let inc_scheme = schemes
            .iter()
            .find(|(name, _)| *name == Symbol::new("inc"))
            .expect("should have 'inc' scheme");

        let body = inc_scheme.1.body(&db);
        // The body should be a function type Int -> Int
        match body.kind(&db) {
            crate::ast::TypeKind::Func { params, result, .. } => {
                assert_eq!(params.len(), 1, "inc should have 1 parameter");
                assert!(
                    matches!(params[0].kind(&db), crate::ast::TypeKind::Int),
                    "Parameter should be Int, got: {:?}",
                    params[0].kind(&db),
                );
                assert!(
                    matches!(result.kind(&db), crate::ast::TypeKind::Int),
                    "Return type should be Int, got: {:?}",
                    result.kind(&db),
                );
            }
            other => panic!("Expected Func type, got: {:?}", other),
        }
    }

    #[test]
    fn test_polymorphic_function_scheme() {
        // fn apply(f: fn(a) -> b, x: a) -> b { f(x) }
        //
        // Lowercase names in type annotations are type variables.
        // The scheme should quantify over them as type_params,
        // and the body should reference them via BoundVar.
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn apply(f: fn(a) -> b, x: a) -> b { f(x) }");

        let schemes = function_schemes(&db, source);
        assert!(schemes.is_some(), "function_schemes should return Some");

        let schemes = schemes.unwrap();
        let apply = schemes
            .iter()
            .find(|(name, _)| *name == Symbol::new("apply"))
            .expect("should have 'apply' scheme");

        // type_params should contain a and b (in declaration order)
        let type_params = apply.1.type_params(&db);
        assert_eq!(
            type_params.len(),
            2,
            "Expected 2 type params (a, b), got: {:?}",
            type_params,
        );

        let body = apply.1.body(&db);
        let crate::ast::TypeKind::Func { params, result, .. } = body.kind(&db) else {
            panic!(
                "Expected Func type for apply scheme body, got: {:?}",
                body.kind(&db)
            );
        };
        assert_eq!(params.len(), 2, "apply should have 2 parameters");

        // Second param `x: a` → BoundVar(0)
        assert!(
            matches!(
                params[1].kind(&db),
                crate::ast::TypeKind::BoundVar { index: 0 }
            ),
            "Second param should be BoundVar(0) for 'a', got: {:?}",
            params[1].kind(&db),
        );

        // Return type `b` → BoundVar(1)
        assert!(
            matches!(
                result.kind(&db),
                crate::ast::TypeKind::BoundVar { index: 1 }
            ),
            "Return type should be BoundVar(1) for 'b', got: {:?}",
            result.kind(&db),
        );

        // No UniVars should remain
        assert!(
            !contains_univar(&db, body),
            "apply scheme body should not contain UniVar, got: {:?}",
            body.kind(&db),
        );
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

    #[test]
    fn test_let_binding_with_tuple_pattern() {
        // Test that let-binding with tuple pattern correctly infers types
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"
fn test() -> Int {
    let #(a, b) = #(1, 2);
    a + b
}
"#,
        );

        // Type checking should succeed
        let module = typed_module(&db, source);
        assert!(module.is_some(), "Type checking should succeed");
    }

    #[test]
    fn test_let_binding_simple_pattern() {
        // Test that let-binding with simple bind pattern works
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"
fn test() -> Int {
    let x = 42;
    x
}
"#,
        );

        // Type checking should succeed
        let module = typed_module(&db, source);
        assert!(module.is_some(), "Type checking should succeed");
    }
}
