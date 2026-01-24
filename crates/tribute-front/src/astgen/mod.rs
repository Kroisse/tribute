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
use crate::source_file::SourceCst;
use crate::tirgen::ParsedCst;
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
}

/// Lower a parsed CST to an AST Module (internal, non-Salsa).
///
/// Returns both the module and the span builder for creating a SpanMap.
fn lower_cst_to_ast_internal(source: &Rope, cst: &ParsedCst) -> LoweringResult {
    let mut ctx = AstLoweringCtx::new(source.clone());
    let root = cst.root_node();
    let module = lower_module(&mut ctx, root);
    let span_builder = ctx.into_span_builder();
    LoweringResult {
        module,
        span_builder,
    }
}

/// Lower a parsed CST to an AST Module.
///
/// This is the entry point for CST → AST conversion.
/// Note: This function does not preserve span information.
/// Use `lower_source_to_parsed_ast` for span-preserving lowering.
pub fn lower_cst_to_ast(source: &Rope, cst: &ParsedCst) -> Module<UnresolvedName> {
    lower_cst_to_ast_internal(source, cst).module
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
    use crate::tirgen::parse_cst;

    let cst = parse_cst(db, source)?;
    let text = source.text(db);
    let result = lower_cst_to_ast_internal(text, &cst);
    let span_map = result.span_builder.finish();
    Some(ParsedAst::new(db, result.module, span_map))
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
    use crate::ast::{Decl, ExprKind, PatternKind, Stmt};
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
}
