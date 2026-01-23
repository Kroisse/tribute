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

use crate::ast::{Module, UnresolvedName};
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

/// Lower a parsed CST to an AST Module.
///
/// This is the entry point for CST → AST conversion.
pub fn lower_cst_to_ast(source: &Rope, cst: &ParsedCst) -> Module<UnresolvedName> {
    let mut ctx = AstLoweringCtx::new(source.clone());
    let root = cst.root_node();
    lower_module(&mut ctx, root)
}

/// Lower a source file to an AST Module.
///
/// Convenience function that extracts the CST from the source file.
pub fn lower_source_to_ast(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<Module<UnresolvedName>> {
    use crate::tirgen::parse_cst;

    let cst = parse_cst(db, source)?;
    let text = source.text(db);
    Some(lower_cst_to_ast(text, &cst))
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
        let ExprKind::Block(stmts) = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        assert_eq!(stmts.len(), 1);
        let Stmt::Return { expr, .. } = &stmts[0] else {
            panic!("Expected return statement");
        };
        let ExprKind::IntLit(42) = expr.kind.as_ref() else {
            panic!("Expected int literal 42");
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
        let ExprKind::Block(stmts) = func.body.kind.as_ref() else {
            panic!("Expected block");
        };

        assert_eq!(stmts.len(), 2);
        let Stmt::Let { pattern, value, .. } = &stmts[0] else {
            panic!("Expected let binding");
        };
        let PatternKind::Bind { name } = pattern.kind.as_ref() else {
            panic!("Expected bind pattern");
        };
        assert_eq!(name.to_string(), "x");
        let ExprKind::IntLit(10) = value.kind.as_ref() else {
            panic!("Expected int literal 10");
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
        let ExprKind::Block(_) = func.body.kind.as_ref() else {
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
        let ExprKind::Block(stmts) = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let Stmt::Return { expr, .. } = &stmts[0] else {
            panic!("Expected return");
        };
        let ExprKind::Case { arms, .. } = expr.kind.as_ref() else {
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
        let ExprKind::Block(stmts) = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let Stmt::Return { expr, .. } = &stmts[0] else {
            panic!("Expected return");
        };
        let ExprKind::Lambda { params, .. } = expr.kind.as_ref() else {
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
        let ExprKind::Block(stmts) = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let Stmt::Let { pattern, .. } = &stmts[0] else {
            panic!("Expected let binding");
        };
        let PatternKind::Tuple(elements) = pattern.kind.as_ref() else {
            panic!("Expected tuple pattern");
        };
        assert_eq!(elements.len(), 2);
    }

    #[test]
    fn test_list_expression() {
        let source = "fn main() { [1, 2, 3] }";
        let module = parse_and_lower(source);

        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block(stmts) = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let Stmt::Return { expr, .. } = &stmts[0] else {
            panic!("Expected return");
        };
        let ExprKind::List(elements) = expr.kind.as_ref() else {
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
