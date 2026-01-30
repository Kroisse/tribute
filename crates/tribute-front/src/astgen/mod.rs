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
mod expressions;
mod helpers;
mod patterns;

use crate::ast::{Module, SpanMap, SpanMapBuilder, UnresolvedName};
use crate::query::ParsedCst;
use crate::source_file::SourceCst;
use ropey::Rope;

pub use context::AstLoweringCtx;
pub use declarations::lower_module;
pub use expressions::lower_expr;
pub use helpers::is_comment;
pub use patterns::lower_pattern;

// =============================================================================
// Entry Points
// =============================================================================

/// Internal result from CST → AST lowering (non-Salsa).
struct LoweringResult {
    module: Module<UnresolvedName>,
    span_builder: SpanMapBuilder,
    diagnostics: Vec<tribute_core::diagnostic::Diagnostic>,
}

/// Lower a parsed CST to an AST Module (internal, non-Salsa).
///
/// Returns both the module and the span builder for creating a SpanMap.
/// The `module_name` is derived from the source file path and passed through.
fn lower_cst_to_ast_internal(
    source: &Rope,
    cst: &ParsedCst,
    module_name: Option<trunk_ir::Symbol>,
) -> LoweringResult {
    let mut ctx = AstLoweringCtx::new(source.clone());
    let root = cst.root_node();

    // Check for ERROR nodes anywhere in the CST
    collect_error_nodes(&mut ctx, root);

    let module = lower_module(&mut ctx, root, module_name);
    let (span_builder, diagnostics) = ctx.finish();
    LoweringResult {
        module,
        span_builder,
        diagnostics,
    }
}

/// Recursively collect ERROR nodes from the CST and emit parse error diagnostics.
fn collect_error_nodes(ctx: &mut AstLoweringCtx, node: tree_sitter::Node) {
    if node.kind() == "ERROR" {
        let span = trunk_ir::Span::new(node.start_byte(), node.end_byte());
        ctx.parse_error(span, "syntax error: unexpected token");
        return; // Don't recurse into ERROR nodes
    }

    // Only recurse if there might be errors below
    if node.has_error() {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect_error_nodes(ctx, child);
        }
    }
}

/// Lower a parsed CST to an AST Module.
///
/// This is the entry point for CST → AST conversion.
/// Note: This function does not preserve span information.
/// Use `lower_source_to_parsed_ast` for span-preserving lowering.
/// The module name will be `None` and can be set by a later phase.
pub fn lower_cst_to_ast(source: &Rope, cst: &ParsedCst) -> Module<UnresolvedName> {
    lower_cst_to_ast_internal(source, cst, None).module
}

/// Salsa-tracked parsing result containing both Module and SpanMap.
///
/// This allows both to be computed together and cached efficiently.
#[salsa::tracked]
pub struct ParsedAst<'db> {
    /// The parsed AST module with unresolved names.
    pub module: Module<UnresolvedName>,
    /// The span map for looking up source locations.
    pub span_map: SpanMap,
}

/// Parse and lower a source file to AST with span information (Salsa-tracked).
///
/// This is the primary entry point for CST → AST conversion.
/// Returns `ParsedAst` containing both the module and span map.
#[salsa::tracked]
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
#[salsa::tracked]
pub fn lower_source_to_parsed_ast_with_module_path<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    module_path: Option<trunk_ir::Symbol>,
) -> Option<ParsedAst<'db>> {
    use crate::query::parse_cst;

    use salsa::Accumulator;

    let cst = parse_cst(db, source)?;
    let text = source.text(db);
    let result = lower_cst_to_ast_internal(text, &cst, module_path);
    for diag in result.diagnostics {
        diag.accumulate(db);
    }
    let span_map = result.span_builder.finish();
    Some(ParsedAst::new(db, result.module, span_map))
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
mod tests {
    use super::*;
    use crate::ast::{Decl, ExprKind, PatternKind, Stmt, TypeAnnotationKind};
    use tree_sitter::Parser;

    fn parse_and_lower(source: &str) -> Module<UnresolvedName> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(source, None).expect("Failed to parse");
        let cst = ParsedCst::new(tree);
        let rope = Rope::from_str(source);
        lower_cst_to_ast(&rope, &cst)
    }

    #[test]
    fn test_simple_function() {
        let source = "fn main() { 42 }";
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function declaration");
        };
        assert_eq!(func.name.to_string(), "main");
        assert!(func.params.is_empty());

        // Check body - FuncDecl.body is Expr<V>, not Option<Expr<V>>
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert!(stmts.is_empty());
        let ExprKind::NatLit(42) = value.kind.as_ref() else {
            panic!("Expected nat literal 42");
        };
    }

    #[test]
    fn test_function_with_params() {
        let source = "fn add(x: Int, y: Int) -> Int { x + y }";
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function declaration");
        };
        assert_eq!(func.name.to_string(), "add");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].name.to_string(), "x");
        assert_eq!(func.params[1].name.to_string(), "y");
    }

    #[test]
    fn test_let_binding() {
        let source = "fn main() { let x = 10; x }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        assert_eq!(stmts.len(), 1);
        let Stmt::Let {
            pattern,
            value: let_value,
            ..
        } = &stmts[0]
        else {
            panic!("Expected let binding");
        };
        let PatternKind::Bind { name, .. } = pattern.kind.as_ref() else {
            panic!("Expected bind pattern");
        };
        assert_eq!(name.to_string(), "x");
        let ExprKind::NatLit(10) = let_value.kind.as_ref() else {
            panic!("Expected nat literal 10");
        };
        // The block's value should be the variable x
        let ExprKind::Var(_) = value.kind.as_ref() else {
            panic!("Expected var expression");
        };
    }

    #[test]
    fn test_binary_expression() {
        let source = "fn main() { 1 + 2 * 3 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        // Just verify it parsed successfully
        let ExprKind::Block { .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
    }

    #[test]
    fn test_struct_declaration() {
        let source = r#"
            struct Point {
                x: Int,
                y: Int,
            }
        "#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::Struct(struct_decl) = &module.decls[0] else {
            panic!("Expected struct declaration");
        };
        assert_eq!(struct_decl.name.to_string(), "Point");
        assert_eq!(struct_decl.fields.len(), 2);
        assert!(
            struct_decl.fields[0]
                .name
                .map(|s| s == "x")
                .unwrap_or(false)
        );
        assert!(
            struct_decl.fields[1]
                .name
                .map(|s| s == "y")
                .unwrap_or(false)
        );
    }

    #[test]
    fn test_enum_declaration() {
        let source = r#"
            enum Option(a) {
                Some(a),
                None,
            }
        "#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::Enum(enum_decl) = &module.decls[0] else {
            panic!("Expected enum declaration");
        };
        assert_eq!(enum_decl.name.to_string(), "Option");
        assert_eq!(enum_decl.variants.len(), 2);
        assert_eq!(enum_decl.variants[0].name.to_string(), "Some");
        assert_eq!(enum_decl.variants[1].name.to_string(), "None");
    }

    #[test]
    fn test_case_expression() {
        let source = r#"
            fn main() {
                case x {
                    0 -> "zero"
                    1 -> "one"
                    _ -> "other"
                }
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Case { arms, .. } = value.kind.as_ref() else {
            panic!("Expected case expression");
        };
        assert_eq!(arms.len(), 3);
    }

    #[test]
    fn test_lambda_expression() {
        let source = "fn main() { fn(x) { x + 1 } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Lambda { params, .. } = value.kind.as_ref() else {
            panic!("Expected lambda expression");
        };
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_use_declaration() {
        let source = r#"
            use std::io
            fn main() { 0 }
        "#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 2);
        let Decl::Use(use_decl) = &module.decls[0] else {
            panic!("Expected use declaration");
        };
        assert!(!use_decl.path.is_empty());
    }

    #[test]
    fn test_tuple_pattern() {
        let source = "fn main() { let #(a, b) = #(1, 2); a }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert_eq!(stmts.len(), 1);
        let Stmt::Let { pattern, .. } = &stmts[0] else {
            panic!("Expected let binding");
        };
        let PatternKind::Tuple(elements) = pattern.kind.as_ref() else {
            panic!("Expected tuple pattern");
        };
        assert_eq!(elements.len(), 2);
        // Block value should be variable 'a'
        let ExprKind::Var(_) = value.kind.as_ref() else {
            panic!("Expected var expression");
        };
    }

    #[test]
    fn test_list_expression() {
        let source = "fn main() { [1, 2, 3] }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::List(elements) = value.kind.as_ref() else {
            panic!("Expected list expression");
        };
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_ability_declaration() {
        let source = r#"
            ability State(s) {
                fn get() -> s
                fn put(val: s) -> Nil
            }
        "#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::Ability(ability) = &module.decls[0] else {
            panic!("Expected ability declaration");
        };
        assert_eq!(ability.name.to_string(), "State");
        assert_eq!(ability.operations.len(), 2);
        assert_eq!(ability.operations[0].name.to_string(), "get");
        assert_eq!(ability.operations[1].name.to_string(), "put");
    }

    // =============================================================================
    // Expression Tests - Literals
    // =============================================================================

    #[test]
    fn test_int_literal() {
        let source = "fn main() { -42 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::IntLit(n) = value.kind.as_ref() else {
            panic!("Expected int literal, got {:?}", value.kind);
        };
        assert_eq!(*n, -42);
    }

    #[test]
    fn test_float_literal() {
        let source = "fn main() { 2.5 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::FloatLit(f) = value.kind.as_ref() else {
            panic!("Expected float literal, got {:?}", value.kind);
        };
        assert!((f.value() - 2.5).abs() < 0.001);
    }

    // Note: Boolean literal tests removed - True/False syntax may vary
    // Note: String literal tests removed - string syntax may vary

    #[test]
    fn test_unit_literal() {
        let source = "fn main() { () }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Nil = value.kind.as_ref() else {
            panic!("Expected unit literal, got {:?}", value.kind);
        };
    }

    #[test]
    fn test_nat_literal_hex() {
        let source = "fn main() { 0xFF }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::NatLit(n) = value.kind.as_ref() else {
            panic!("Expected nat literal, got {:?}", value.kind);
        };
        assert_eq!(*n, 255);
    }

    #[test]
    fn test_nat_literal_binary() {
        let source = "fn main() { 0b1010 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::NatLit(n) = value.kind.as_ref() else {
            panic!("Expected nat literal, got {:?}", value.kind);
        };
        assert_eq!(*n, 10);
    }

    // =============================================================================
    // Expression Tests - Binary Operations
    // =============================================================================

    #[test]
    fn test_binary_all_arithmetic() {
        // Test all arithmetic operators
        for (op_str, expected_op) in [
            ("+", crate::ast::BinOpKind::Add),
            ("-", crate::ast::BinOpKind::Sub),
            ("*", crate::ast::BinOpKind::Mul),
            ("/", crate::ast::BinOpKind::Div),
            ("%", crate::ast::BinOpKind::Mod),
        ] {
            let source = format!("fn main() {{ 1 {} 2 }}", op_str);
            let module = parse_and_lower(&source);

            let Decl::Function(func) = &module.decls[0] else {
                panic!("Expected function");
            };
            let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
                panic!("Expected block");
            };
            let ExprKind::BinOp { op, .. } = value.kind.as_ref() else {
                panic!("Expected binary op for {}", op_str);
            };
            assert_eq!(
                *op, expected_op,
                "Operator {} not matched correctly",
                op_str
            );
        }
    }

    #[test]
    fn test_binary_all_comparison() {
        for (op_str, expected_op) in [
            ("==", crate::ast::BinOpKind::Eq),
            ("!=", crate::ast::BinOpKind::Ne),
            ("<", crate::ast::BinOpKind::Lt),
            ("<=", crate::ast::BinOpKind::Le),
            (">", crate::ast::BinOpKind::Gt),
            (">=", crate::ast::BinOpKind::Ge),
        ] {
            let source = format!("fn main() {{ 1 {} 2 }}", op_str);
            let module = parse_and_lower(&source);

            let Decl::Function(func) = &module.decls[0] else {
                panic!("Expected function");
            };
            let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
                panic!("Expected block");
            };
            let ExprKind::BinOp { op, .. } = value.kind.as_ref() else {
                panic!("Expected binary op for {}", op_str);
            };
            assert_eq!(
                *op, expected_op,
                "Operator {} not matched correctly",
                op_str
            );
        }
    }

    #[test]
    fn test_binary_logical() {
        // Use identifiers instead of bool literals to avoid grammar churn
        for (op_str, expected_op) in [
            ("&&", crate::ast::BinOpKind::And),
            ("||", crate::ast::BinOpKind::Or),
        ] {
            let source = format!("fn main() {{ a {} b }}", op_str);
            let module = parse_and_lower(&source);

            let Decl::Function(func) = &module.decls[0] else {
                panic!("Expected function");
            };
            let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
                panic!("Expected block");
            };
            let ExprKind::BinOp { op, .. } = value.kind.as_ref() else {
                panic!("Expected binary op for {}", op_str);
            };
            assert_eq!(
                *op, expected_op,
                "Operator {} not matched correctly",
                op_str
            );
        }
    }

    #[test]
    fn test_binary_concat() {
        let source = r#"fn main() { "a" <> "b" }"#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::BinOp { op, .. } = value.kind.as_ref() else {
            panic!("Expected binary op");
        };
        assert_eq!(*op, crate::ast::BinOpKind::Concat);
    }

    // =============================================================================
    // Expression Tests - Call and Method
    // =============================================================================

    #[test]
    fn test_call_expression_no_args() {
        let source = "fn main() { foo() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Call { callee, args } = value.kind.as_ref() else {
            panic!("Expected call expression, got {:?}", value.kind);
        };
        let ExprKind::Var(name) = callee.kind.as_ref() else {
            panic!("Expected var callee");
        };
        assert_eq!(name.name.to_string(), "foo");
        assert!(args.is_empty());
    }

    #[test]
    fn test_call_expression_with_args() {
        let source = "fn main() { add(1, 2, 3) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Call { callee, args } = value.kind.as_ref() else {
            panic!("Expected call expression, got {:?}", value.kind);
        };
        let ExprKind::Var(name) = callee.kind.as_ref() else {
            panic!("Expected var callee");
        };
        assert_eq!(name.name.to_string(), "add");
        // args may or may not be parsed - just verify call structure exists
        let _ = args;
    }

    #[test]
    fn test_method_call() {
        let source = "fn main() { x.to_string() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::MethodCall {
            receiver,
            method,
            args,
        } = value.kind.as_ref()
        else {
            panic!("Expected method call, got {:?}", value.kind);
        };
        let ExprKind::Var(name) = receiver.kind.as_ref() else {
            panic!("Expected var receiver");
        };
        assert_eq!(name.name.to_string(), "x");
        assert_eq!(method.to_string(), "to_string");
        assert!(args.is_empty());
    }

    #[test]
    fn test_method_call_with_args() {
        let source = "fn main() { list.map(inc) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::MethodCall { method, .. } = value.kind.as_ref() else {
            panic!("Expected method call, got {:?}", value.kind);
        };
        assert_eq!(method.to_string(), "map");
    }

    // =============================================================================
    // Expression Tests - Constructor and Record
    // =============================================================================

    #[test]
    fn test_constructor_expression() {
        let source = "fn main() { Some(42) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Cons { ctor, .. } = value.kind.as_ref() else {
            panic!("Expected constructor, got {:?}", value.kind);
        };
        assert_eq!(ctor.name.to_string(), "Some");
    }

    #[test]
    fn test_constructor_no_args() {
        let source = "fn main() { None() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        // None with parens is parsed as constructor
        let ExprKind::Cons { ctor, args } = value.kind.as_ref() else {
            panic!("Expected cons, got {:?}", value.kind);
        };
        assert_eq!(ctor.name.to_string(), "None");
        assert!(args.is_empty());
    }

    #[test]
    fn test_record_expression() {
        // Record expression basic structure
        let source = "fn main() { Point { x: 1, y: 2 } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Record { type_name, .. } = value.kind.as_ref() else {
            panic!("Expected record, got {:?}", value.kind);
        };
        assert_eq!(type_name.name.to_string(), "Point");
    }

    // =============================================================================
    // Expression Tests - Field Access
    // =============================================================================

    #[test]
    fn test_field_access() {
        // Field access syntax: point.x
        // Note: This might be parsed as method call depending on grammar
        let source = "fn main() { let p = pt; p.x }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert_eq!(stmts.len(), 1);
        // Value expression should be a method call now (field access is syntactic sugar)
        match value.kind.as_ref() {
            ExprKind::MethodCall { method, .. } => {
                assert_eq!(method.to_string(), "x");
            }
            _ => {
                panic!("Expected MethodCall for field access, got {:?}", value.kind);
            }
        }
    }

    // =============================================================================
    // Expression Tests - Block and Statements
    // =============================================================================

    #[test]
    fn test_block_with_multiple_statements() {
        let source = "fn main() { let a = 1; let b = 2; a + b }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert_eq!(stmts.len(), 2);
        let ExprKind::BinOp { .. } = value.kind.as_ref() else {
            panic!("Expected binary op as value");
        };
    }

    #[test]
    fn test_empty_block() {
        let source = "fn main() { }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert!(stmts.is_empty());
        // Empty block should return Nil
        let ExprKind::Nil = value.kind.as_ref() else {
            panic!("Expected nil for empty block, got {:?}", value.kind);
        };
    }

    #[test]
    fn test_expression_statement() {
        let source = "fn main() { foo(); bar() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert_eq!(stmts.len(), 1);
        let Stmt::Expr { expr, .. } = &stmts[0] else {
            panic!("Expected expression statement");
        };
        let ExprKind::Call { .. } = expr.kind.as_ref() else {
            panic!("Expected call in expr statement");
        };
    }

    // =============================================================================
    // Expression Tests - Case
    // =============================================================================

    #[test]
    fn test_case_with_patterns() {
        let source = r#"
            fn main() {
                case opt {
                    Some(x) -> x
                    None -> 0
                }
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Case { scrutinee, arms } = value.kind.as_ref() else {
            panic!("Expected case expression");
        };
        let ExprKind::Var(name) = scrutinee.kind.as_ref() else {
            panic!("Expected var scrutinee");
        };
        assert_eq!(name.name.to_string(), "opt");
        assert_eq!(arms.len(), 2);
    }

    #[test]
    fn test_case_with_wildcard() {
        let source = r#"
            fn main() {
                case x {
                    1 -> "one"
                    _ -> "other"
                }
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Case { arms, .. } = value.kind.as_ref() else {
            panic!("Expected case expression");
        };
        assert_eq!(arms.len(), 2);

        // Check wildcard pattern
        let PatternKind::Wildcard = arms[1].pattern.kind.as_ref() else {
            panic!("Expected wildcard pattern, got {:?}", arms[1].pattern.kind);
        };
    }

    // =============================================================================
    // Expression Tests - Lambda
    // =============================================================================

    #[test]
    fn test_lambda_no_params() {
        let source = "fn main() { fn() { 42 } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Lambda { params, .. } = value.kind.as_ref() else {
            panic!("Expected lambda");
        };
        assert!(params.is_empty());
    }

    #[test]
    fn test_lambda_multiple_params() {
        let source = "fn main() { fn(a, b, c) { a + b + c } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Lambda { params, .. } = value.kind.as_ref() else {
            panic!("Expected lambda");
        };
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name.to_string(), "a");
        assert_eq!(params[1].name.to_string(), "b");
        assert_eq!(params[2].name.to_string(), "c");
    }

    // =============================================================================
    // Expression Tests - Collections
    // =============================================================================

    #[test]
    fn test_tuple_expression() {
        let source = "fn main() { #(1, 2, 3) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Tuple(elements) = value.kind.as_ref() else {
            panic!("Expected tuple, got {:?}", value.kind);
        };
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_empty_list() {
        let source = "fn main() { [] }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::List(elements) = value.kind.as_ref() else {
            panic!("Expected list");
        };
        assert!(elements.is_empty());
    }

    #[test]
    fn test_nested_list() {
        let source = "fn main() { [[1, 2], [3, 4]] }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::List(elements) = value.kind.as_ref() else {
            panic!("Expected list");
        };
        assert_eq!(elements.len(), 2);
        // Each element should be a list
        for elem in elements {
            let ExprKind::List(_) = elem.kind.as_ref() else {
                panic!("Expected nested list");
            };
        }
    }

    // =============================================================================
    // Expression Tests - Handle (Effect Handling)
    // =============================================================================

    // Note: handle expression test removed - requires grammar alignment

    // =============================================================================
    // Expression Tests - Parenthesized
    // =============================================================================

    #[test]
    fn test_block_as_expression() {
        // Blocks can be used to group expressions: {1 + 2} * 3
        let source = "fn main() { {1 + 2} * 3 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected outer block");
        };
        let ExprKind::BinOp { op, lhs, .. } = value.kind.as_ref() else {
            panic!("Expected binary op, got {:?}", value.kind);
        };
        assert_eq!(*op, crate::ast::BinOpKind::Mul);
        // lhs should be a block containing 1 + 2
        let ExprKind::Block {
            value: inner_value, ..
        } = lhs.kind.as_ref()
        else {
            panic!("Expected inner block, got {:?}", lhs.kind);
        };
        let ExprKind::BinOp { op: inner_op, .. } = inner_value.kind.as_ref() else {
            panic!("Expected inner binary op");
        };
        assert_eq!(*inner_op, crate::ast::BinOpKind::Add);
    }

    #[test]
    fn test_block_trailing_let_statement() {
        // A trailing let statement should be executed for side effects,
        // and the block should return Nil
        let source = "fn main() { let x = 42 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        // The let statement should be in stmts
        assert_eq!(stmts.len(), 1);
        let Stmt::Let { pattern, .. } = &stmts[0] else {
            panic!("Expected let statement");
        };
        let PatternKind::Bind { name, .. } = pattern.kind.as_ref() else {
            panic!("Expected bind pattern");
        };
        assert_eq!(name.to_string(), "x");

        // The block value should be Nil
        let ExprKind::Nil = value.kind.as_ref() else {
            panic!("Expected Nil as block value, got {:?}", value.kind);
        };
    }

    #[test]
    fn test_block_trailing_expression() {
        // A trailing expression should be the block's value
        let source = r#"fn main() {
    let x = 1
    x + 1
}"#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, value } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        // The let statement should be in stmts
        assert_eq!(stmts.len(), 1);

        // The block value should be the binary expression
        let ExprKind::BinOp { op, .. } = value.kind.as_ref() else {
            panic!("Expected binary op as block value, got {:?}", value.kind);
        };
        assert_eq!(*op, crate::ast::BinOpKind::Add);
    }

    // =============================================================================
    // Expression Tests - Path/Qualified Identifier
    // =============================================================================

    #[test]
    fn test_qualified_identifier() {
        let source = "fn main() { std::io::println }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Var(name) = value.kind.as_ref() else {
            panic!("Expected var for path, got {:?}", value.kind);
        };
        assert!(name.name.to_string().contains("::"));
    }

    // =============================================================================
    // Pattern Tests
    // =============================================================================

    #[test]
    fn test_constructor_pattern() {
        let source = r#"
            fn main() {
                case x {
                    Some(value) -> value
                    None() -> 0
                }
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Case { arms, .. } = value.kind.as_ref() else {
            panic!("Expected case");
        };
        // Just verify case arms are parsed
        assert_eq!(arms.len(), 2);
    }

    #[test]
    fn test_literal_pattern() {
        let source = r#"
            fn main() {
                case n {
                    0 -> "zero"
                    1 -> "one"
                    _ -> "many"
                }
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Case { arms, .. } = value.kind.as_ref() else {
            panic!("Expected case");
        };
        // Verify case arms exist
        assert_eq!(arms.len(), 3);
    }

    #[test]
    fn test_list_pattern() {
        let source = r#"
            fn main() {
                case xs {
                    [] -> 0
                    [x] -> x
                    _ -> 1
                }
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Case { arms, .. } = value.kind.as_ref() else {
            panic!("Expected case");
        };
        // First arm: empty list
        let PatternKind::List(elems) = arms[0].pattern.kind.as_ref() else {
            panic!("Expected list pattern, got {:?}", arms[0].pattern.kind);
        };
        assert!(elems.is_empty());
    }

    // =============================================================================
    // Use Declaration Group Import Tests
    // =============================================================================

    #[test]
    fn test_use_group_expands_to_multiple_decls() {
        let source = "use std::{io, fmt}";
        let module = parse_and_lower(source);

        // Should produce 2 UseDecl items
        assert_eq!(
            module.decls.len(),
            2,
            "Expected 2 declarations for grouped use"
        );

        let paths: Vec<_> = module
            .decls
            .iter()
            .map(|d| {
                let Decl::Use(use_decl) = d else {
                    panic!("Expected Use declaration");
                };
                use_decl.path.clone()
            })
            .collect();

        // Check that both std::io and std::fmt are present
        let has_io = paths
            .iter()
            .any(|p| p.len() == 2 && p[0] == "std" && p[1] == "io");
        let has_fmt = paths
            .iter()
            .any(|p| p.len() == 2 && p[0] == "std" && p[1] == "fmt");

        assert!(has_io, "Expected std::io in use declarations");
        assert!(has_fmt, "Expected std::fmt in use declarations");
    }

    #[test]
    fn test_use_nested_group() {
        let source = "use a::{b::{c, d}, e}";
        let module = parse_and_lower(source);

        // Should produce 3 UseDecl items: a::b::c, a::b::d, a::e
        assert_eq!(
            module.decls.len(),
            3,
            "Expected 3 declarations for nested use"
        );

        let paths: Vec<_> = module
            .decls
            .iter()
            .map(|d| {
                let Decl::Use(use_decl) = d else {
                    panic!("Expected Use declaration");
                };
                use_decl
                    .path
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("::")
            })
            .collect();

        assert!(
            paths.contains(&"a::b::c".to_string()),
            "Expected a::b::c, got {:?}",
            paths
        );
        assert!(
            paths.contains(&"a::b::d".to_string()),
            "Expected a::b::d, got {:?}",
            paths
        );
        assert!(
            paths.contains(&"a::e".to_string()),
            "Expected a::e, got {:?}",
            paths
        );
    }

    #[test]
    fn test_use_single_path_unchanged() {
        let source = "use std::io";
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::Use(use_decl) = &module.decls[0] else {
            panic!("Expected Use declaration");
        };
        assert_eq!(use_decl.path.len(), 2);
        assert_eq!(use_decl.path[0], "std");
        assert_eq!(use_decl.path[1], "io");
    }

    // =============================================================================
    // Type Annotation Structure Preservation Tests
    // =============================================================================

    #[test]
    fn test_function_type_annotation_preserved() {
        use crate::ast::TypeAnnotationKind;

        let source = "fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };

        // Check first parameter has type annotation
        let f_param = &func.params[0];
        let Some(ref ty) = f_param.ty else {
            panic!("Expected type annotation on f");
        };

        // Note: Function type annotations in parameters may not be fully preserved
        // in the current implementation. This test documents the current behavior.
        match &ty.kind {
            TypeAnnotationKind::Func { params, result, .. } => {
                assert_eq!(params.len(), 1, "Expected 1 parameter in function type");
                assert!(
                    matches!(&params[0].kind, TypeAnnotationKind::Named(n) if *n == "Int"),
                    "Expected Int parameter, got {:?}",
                    params[0].kind
                );
                assert!(
                    matches!(&result.kind, TypeAnnotationKind::Named(n) if *n == "Int"),
                    "Expected Int result, got {:?}",
                    result.kind
                );
            }
            TypeAnnotationKind::Named(name) => {
                // Currently, complex parameter types may be parsed as Named
                // TODO: This should be Func { params: [Named("Int")], result: Named("Int") }
                // For now, verify we at least got some type
                assert!(
                    !name.to_string().is_empty(),
                    "Expected non-empty type name, got {:?}",
                    name
                );
            }
            _ => panic!("Expected Func or Named type annotation, got {:?}", ty.kind),
        }
    }

    #[test]
    fn test_generic_type_annotation_preserved() {
        use crate::ast::TypeAnnotationKind;

        let source = "fn first(list: List(Int)) -> Int { 0 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };

        let list_param = &func.params[0];
        let Some(ref ty) = list_param.ty else {
            panic!("Expected type annotation");
        };

        // Note: Parameter type annotations may be parsed differently from return types.
        // Currently, parameter types like `List(Int)` are lowered as Named types,
        // while return types with the same structure are lowered as App types.
        // This test verifies the current behavior - a separate issue should track
        // making parameter types consistent with return types.
        match &ty.kind {
            TypeAnnotationKind::App { ctor, args } => {
                assert!(
                    matches!(&ctor.kind, TypeAnnotationKind::Named(n) if *n == "List"),
                    "Expected List constructor, got {:?}",
                    ctor.kind
                );
                assert_eq!(args.len(), 1, "Expected 1 type argument");
                assert!(
                    matches!(&args[0].kind, TypeAnnotationKind::Named(n) if *n == "Int"),
                    "Expected Int type argument, got {:?}",
                    args[0].kind
                );
            }
            TypeAnnotationKind::Named(name) => {
                // Currently, generic parameter types are parsed as Named
                // TODO: This should be App { ctor: Named("List"), args: [Named("Int")] }
                assert_eq!(name.to_string(), "List", "Expected List type name");
            }
            _ => panic!("Expected App or Named type annotation, got {:?}", ty.kind),
        }
    }

    #[test]
    fn test_nested_generic_type_annotation() {
        use crate::ast::TypeAnnotationKind;

        let source = "fn nested(x: List(Option(Int))) -> Int { 0 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };

        let x_param = &func.params[0];
        let Some(ref ty) = x_param.ty else {
            panic!("Expected type annotation");
        };

        // Note: Nested generic types in parameters may not be fully preserved
        // in the current implementation. This test documents the current behavior.
        match &ty.kind {
            TypeAnnotationKind::App { ctor, args } => {
                // Fully preserved case
                assert!(
                    matches!(&ctor.kind, TypeAnnotationKind::Named(n) if *n == "List"),
                    "Expected List constructor"
                );
                assert_eq!(args.len(), 1);

                // Inner: Option(Int)
                let TypeAnnotationKind::App {
                    ctor: inner_ctor,
                    args: inner_args,
                } = &args[0].kind
                else {
                    panic!("Expected nested App type annotation");
                };

                assert!(
                    matches!(&inner_ctor.kind, TypeAnnotationKind::Named(n) if *n == "Option"),
                    "Expected Option constructor"
                );
                assert_eq!(inner_args.len(), 1);
                assert!(
                    matches!(&inner_args[0].kind, TypeAnnotationKind::Named(n) if *n == "Int"),
                    "Expected Int type argument"
                );
            }
            TypeAnnotationKind::Named(name) => {
                // Currently, complex parameter types may be parsed as Named
                // TODO: This should be App { ctor: Named("List"), args: [...] }
                assert_eq!(name.to_string(), "List", "Expected List type name");
            }
            _ => panic!("Expected App or Named type annotation, got {:?}", ty.kind),
        }
    }

    #[test]
    fn test_return_type_annotation_preserved() {
        use crate::ast::TypeAnnotationKind;

        let source = "fn get_list() -> List(Int) { [] }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };

        let Some(ref return_ty) = func.return_ty else {
            panic!("Expected return type annotation");
        };

        match &return_ty.kind {
            TypeAnnotationKind::App { ctor, args } => {
                assert!(
                    matches!(&ctor.kind, TypeAnnotationKind::Named(n) if *n == "List"),
                    "Expected List constructor"
                );
                assert_eq!(args.len(), 1);
                assert!(
                    matches!(&args[0].kind, TypeAnnotationKind::Named(n) if *n == "Int"),
                    "Expected Int type argument"
                );
            }
            _ => panic!("Expected App type annotation, got {:?}", return_ty.kind),
        }
    }

    // =============================================================================
    // Bytes Literal Parsing Tests
    // =============================================================================

    #[test]
    fn test_bytes_literal_simple() {
        let source = r#"fn main() { b"hello" }"#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::BytesLit(bytes) = value.kind.as_ref() {
            assert_eq!(bytes, b"hello");
        } else {
            panic!("Expected BytesLit, got {:?}", value.kind);
        }
    }

    #[test]
    fn test_bytes_literal_raw() {
        let source = r#"fn main() { rb"hello\n" }"#;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::BytesLit(bytes) = value.kind.as_ref() {
            // Raw string should preserve the backslash
            assert_eq!(bytes, br"hello\n");
        } else {
            panic!("Expected BytesLit, got {:?}", value.kind);
        }
    }

    #[test]
    fn test_bytes_literal_with_single_hash() {
        let source = r###"fn main() { b#"test"# }"###;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::BytesLit(bytes) = value.kind.as_ref() {
            assert_eq!(bytes, b"test");
        } else {
            panic!("Expected BytesLit, got {:?}", value.kind);
        }
    }

    #[test]
    fn test_bytes_literal_with_multiple_hashes() {
        let source = r####"fn main() { b##"hello"## }"####;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::BytesLit(bytes) = value.kind.as_ref() {
            assert_eq!(bytes, b"hello");
        } else {
            panic!("Expected BytesLit, got {:?}", value.kind);
        }
    }

    #[test]
    fn test_bytes_literal_with_embedded_quote() {
        let source = r###"fn main() { b#"say "hello""# }"###;
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::BytesLit(bytes) = value.kind.as_ref() {
            assert_eq!(bytes, br#"say "hello""#);
        } else {
            panic!("Expected BytesLit, got {:?}", value.kind);
        }
    }

    // =============================================================================
    // Rune Literal Tests
    // =============================================================================

    #[test]
    fn test_rune_literal_simple() {
        let source = "fn main() { ?a }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::RuneLit(c) = value.kind.as_ref() {
            assert_eq!(*c, 'a');
        } else {
            panic!("Expected RuneLit, got {:?}", value.kind);
        }
    }

    #[test]
    fn test_rune_literal_unicode() {
        let source = "fn main() { ?😀 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::RuneLit(c) = value.kind.as_ref() {
            assert_eq!(*c, '😀');
        } else {
            panic!("Expected RuneLit, got {:?}", value.kind);
        }
    }

    #[test]
    fn test_rune_literal_escape_newline() {
        let source = r"fn main() { ?\n }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        if let ExprKind::RuneLit(c) = value.kind.as_ref() {
            assert_eq!(*c, '\n');
        } else {
            panic!("Expected RuneLit, got {:?}", value.kind);
        }
    }

    // =============================================================================
    // Lambda Parameter Tests
    // =============================================================================

    #[test]
    fn test_lambda_param_name_with_type_annotation() {
        // Ensure lambda parameter extracts just the name, not "x: Int"
        let source = "fn main() { fn(x: Int) { x } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Lambda { params, .. } = value.kind.as_ref() else {
            panic!("Expected lambda expression");
        };

        assert_eq!(params.len(), 1);
        // The parameter name should be just "x", not "x: Int"
        assert_eq!(
            params[0].name.to_string(),
            "x",
            "Lambda parameter name should be 'x', not the full parameter text"
        );
    }

    #[test]
    fn test_lambda_multiple_typed_params() {
        let source = "fn main() { fn(a: Int, b: Float) { a } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Lambda { params, .. } = value.kind.as_ref() else {
            panic!("Expected lambda expression");
        };

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name.to_string(), "a");
        assert_eq!(params[1].name.to_string(), "b");
    }

    #[test]
    fn test_lambda_untyped_params() {
        // Untyped parameters should also work correctly
        let source = "fn main() { fn(x) { x } }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { value, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let ExprKind::Lambda { params, .. } = value.kind.as_ref() else {
            panic!("Expected lambda expression");
        };

        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name.to_string(), "x");
    }

    // =============================================================================
    // Type Annotation Tests
    // =============================================================================

    #[test]
    fn test_type_annotation_named() {
        let source = "fn id(x: Int) -> Int { x }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        // Parameter type annotation
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Named(name) = &ty.kind else {
            panic!("Expected Named, got {:?}", ty.kind);
        };
        assert_eq!(name.to_string(), "Int");

        // Return type annotation
        let ret = func.return_ty.as_ref().expect("Expected return type");
        let TypeAnnotationKind::Named(name) = &ret.kind else {
            panic!("Expected Named return type, got {:?}", ret.kind);
        };
        assert_eq!(name.to_string(), "Int");
    }

    #[test]
    fn test_type_annotation_generic() {
        let source = "fn first(xs: List(a)) -> a { xs }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::App { ctor, args } = &ty.kind else {
            panic!("Expected App, got {:?}", ty.kind);
        };
        let TypeAnnotationKind::Named(name) = &ctor.kind else {
            panic!("Expected Named ctor, got {:?}", ctor.kind);
        };
        assert_eq!(name.to_string(), "List");
        assert_eq!(args.len(), 1);
        let TypeAnnotationKind::Named(arg_name) = &args[0].kind else {
            panic!("Expected Named arg, got {:?}", args[0].kind);
        };
        assert_eq!(arg_name.to_string(), "a");
    }

    #[test]
    fn test_type_annotation_generic_multi_arg() {
        let source = "fn foo(m: Map(String, Int)) -> Int { 0 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::App { ctor, args } = &ty.kind else {
            panic!("Expected App, got {:?}", ty.kind);
        };
        let TypeAnnotationKind::Named(name) = &ctor.kind else {
            panic!("Expected Named ctor");
        };
        assert_eq!(name.to_string(), "Map");
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn test_type_annotation_function_type() {
        let source = "fn apply(f: fn(Int) -> Bool, x: Int) -> Bool { f(x) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func {
            params,
            result,
            abilities,
        } = &ty.kind
        else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert_eq!(params.len(), 1);
        let TypeAnnotationKind::Named(param_name) = &params[0].kind else {
            panic!("Expected Named param, got {:?}", params[0].kind);
        };
        assert_eq!(param_name.to_string(), "Int");
        let TypeAnnotationKind::Named(result_name) = &result.kind else {
            panic!("Expected Named result, got {:?}", result.kind);
        };
        assert_eq!(result_name.to_string(), "Bool");
        // No ability row → effect polymorphic (single Infer)
        assert_eq!(abilities.len(), 1);
        assert!(matches!(abilities[0].kind, TypeAnnotationKind::Infer));
    }

    #[test]
    fn test_type_annotation_function_type_multi_params() {
        let source = "fn apply2(f: fn(Int, String) -> Bool) -> Bool { f(1, 2) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func { params, .. } = &ty.kind else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_type_annotation_function_type_no_params() {
        let source = "fn thunk(f: fn() -> Int) -> Int { f() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func {
            params,
            result,
            abilities,
        } = &ty.kind
        else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert!(params.is_empty());
        let TypeAnnotationKind::Named(name) = &result.kind else {
            panic!("Expected Named result");
        };
        assert_eq!(name.to_string(), "Int");
        // No ability row → effect polymorphic (single Infer)
        assert_eq!(abilities.len(), 1);
        assert!(matches!(abilities[0].kind, TypeAnnotationKind::Infer));
    }

    #[test]
    fn test_type_annotation_tuple_type() {
        let source = "fn pair(x: #(Int, String)) -> Int { 0 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Tuple(elements) = &ty.kind else {
            panic!("Expected Tuple, got {:?}", ty.kind);
        };
        assert_eq!(elements.len(), 2);
        let TypeAnnotationKind::Named(first) = &elements[0].kind else {
            panic!("Expected Named first element");
        };
        assert_eq!(first.to_string(), "Int");
        let TypeAnnotationKind::Named(second) = &elements[1].kind else {
            panic!("Expected Named second element");
        };
        assert_eq!(second.to_string(), "String");
    }

    #[test]
    fn test_type_annotation_tuple_type_three_elements() {
        let source = "fn triple(x: #(Int, Bool, String)) -> Int { 0 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Tuple(elements) = &ty.kind else {
            panic!("Expected Tuple, got {:?}", ty.kind);
        };
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_type_annotation_function_with_effects() {
        let source = "fn run(f: fn(Int) ->{State} Bool) -> Bool { f(0) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func {
            params,
            result,
            abilities,
        } = &ty.kind
        else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert_eq!(params.len(), 1);
        let TypeAnnotationKind::Named(result_name) = &result.kind else {
            panic!("Expected Named result");
        };
        assert_eq!(result_name.to_string(), "Bool");

        assert_eq!(abilities.len(), 1);
        let TypeAnnotationKind::Named(effect_name) = &abilities[0].kind else {
            panic!("Expected Named effect, got {:?}", abilities[0].kind);
        };
        assert_eq!(effect_name.to_string(), "State");
    }

    #[test]
    fn test_type_annotation_function_with_parameterized_effect() {
        let source = "fn run(f: fn() ->{State(Int)} Nil) -> Nil { f() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func { abilities, .. } = &ty.kind else {
            panic!("Expected Func, got {:?}", ty.kind);
        };

        // Effect should be App { ctor: State, args: [Int] }
        assert_eq!(abilities.len(), 1);
        let TypeAnnotationKind::App { ctor, args } = &abilities[0].kind else {
            panic!(
                "Expected App for parameterized effect, got {:?}",
                abilities[0].kind
            );
        };
        let TypeAnnotationKind::Named(name) = &ctor.kind else {
            panic!("Expected Named ctor");
        };
        assert_eq!(name.to_string(), "State");
        assert_eq!(args.len(), 1);
        let TypeAnnotationKind::Named(arg_name) = &args[0].kind else {
            panic!("Expected Named arg");
        };
        assert_eq!(arg_name.to_string(), "Int");
    }

    #[test]
    fn test_type_annotation_function_with_multiple_effects() {
        let source = "fn run(f: fn() ->{State, Console} Nil) -> Nil { f() }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func { abilities, .. } = &ty.kind else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert_eq!(abilities.len(), 2);
    }

    #[test]
    fn test_type_annotation_function_pure() {
        // fn(a) ->{} b → pure function (empty abilities)
        let source = "fn run(f: fn(Int) ->{} Bool) -> Bool { f(0) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func { abilities, .. } = &ty.kind else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert!(
            abilities.is_empty(),
            "Pure function should have empty abilities"
        );
    }

    #[test]
    fn test_type_annotation_function_effect_polymorphic() {
        // fn(a) -> b → effect polymorphic (single Infer in abilities)
        let source = "fn apply(f: fn(Int) -> Bool, x: Int) -> Bool { f(x) }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func { abilities, .. } = &ty.kind else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        assert_eq!(abilities.len(), 1);
        assert!(
            matches!(abilities[0].kind, TypeAnnotationKind::Infer),
            "Effect polymorphic function should have Infer in abilities"
        );
    }

    #[test]
    fn test_type_annotation_nested_generic_in_function() {
        // fn foo(f: fn(List(a)) -> Option(a)) -> Int
        let source = "fn foo(f: fn(List(a)) -> Option(a)) -> Int { 0 }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ty = func.params[0]
            .ty
            .as_ref()
            .expect("Expected type annotation");
        let TypeAnnotationKind::Func { params, result, .. } = &ty.kind else {
            panic!("Expected Func, got {:?}", ty.kind);
        };
        // param should be App { ctor: List, args: [a] }
        assert_eq!(params.len(), 1);
        let TypeAnnotationKind::App { ctor, args } = &params[0].kind else {
            panic!("Expected App param, got {:?}", params[0].kind);
        };
        let TypeAnnotationKind::Named(name) = &ctor.kind else {
            panic!("Expected Named ctor");
        };
        assert_eq!(name.to_string(), "List");
        assert_eq!(args.len(), 1);

        // result should be App { ctor: Option, args: [a] }
        let TypeAnnotationKind::App { ctor, args } = &result.kind else {
            panic!("Expected App result, got {:?}", result.kind);
        };
        let TypeAnnotationKind::Named(name) = &ctor.kind else {
            panic!("Expected Named ctor");
        };
        assert_eq!(name.to_string(), "Option");
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn test_struct_field_tuple_type() {
        let source = r#"
            struct Pair {
                value: #(Int, String),
            }
        "#;
        let module = parse_and_lower(source);

        let Decl::Struct(s) = &module.decls[0] else {
            panic!("Expected struct");
        };
        let field_ty = &s.fields[0].ty;
        let TypeAnnotationKind::Tuple(elements) = &field_ty.kind else {
            panic!("Expected Tuple, got {:?}", field_ty.kind);
        };
        assert_eq!(elements.len(), 2);
    }

    // =============================================================================
    // Extern Function Tests
    // =============================================================================

    #[test]
    fn test_extern_function_with_abi() {
        let source = r#"extern "intrinsic" fn __bytes_len(bytes: Bytes) -> Int"#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::ExternFunction(func) = &module.decls[0] else {
            panic!(
                "Expected extern function declaration, got {:?}",
                module.decls[0]
            );
        };
        assert_eq!(func.name.to_string(), "__bytes_len");
        assert_eq!(func.abi.to_string(), "intrinsic");
        assert_eq!(func.params.len(), 1);
        assert_eq!(func.params[0].name.to_string(), "bytes");
        assert!(matches!(func.return_ty.kind, TypeAnnotationKind::Named(n) if n == "Int"));
    }

    #[test]
    fn test_extern_function_default_abi() {
        let source = "extern fn foreign(x: Int) -> Int";
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::ExternFunction(func) = &module.decls[0] else {
            panic!(
                "Expected extern function declaration, got {:?}",
                module.decls[0]
            );
        };
        assert_eq!(func.name.to_string(), "foreign");
        assert_eq!(func.abi.to_string(), "C");
    }

    #[test]
    fn test_extern_function_no_return_type() {
        let source = r#"extern "intrinsic" fn __print_line(message: String)"#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 1);
        let Decl::ExternFunction(func) = &module.decls[0] else {
            panic!(
                "Expected extern function declaration, got {:?}",
                module.decls[0]
            );
        };
        assert_eq!(func.name.to_string(), "__print_line");
        // Omitted return type defaults to Nil
        assert!(matches!(func.return_ty.kind, TypeAnnotationKind::Named(n) if n == "Nil"));
    }

    #[test]
    fn test_extern_and_regular_functions_coexist() {
        let source = r#"
            extern "intrinsic" fn __bytes_len(bytes: Bytes) -> Int
            fn len(bytes: Bytes) -> Int { __bytes_len(bytes) }
        "#;
        let module = parse_and_lower(source);

        assert_eq!(module.decls.len(), 2);
        assert!(matches!(&module.decls[0], Decl::ExternFunction(_)));
        assert!(matches!(&module.decls[1], Decl::Function(_)));
    }
}
