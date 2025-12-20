//! TrunkIR generation from CST.
//!
//! This pass converts Tree-sitter CST directly to TrunkIR operations,
//! bypassing the AST intermediate representation.
//! At this stage, names are unresolved (using `src` dialect ops).
//!
//! ## Pipeline
//!
//! The lowering is split into two Salsa-tracked stages:
//! 1. `parse_cst` - Parse source to CST (cached by Salsa)
//! 2. `lower_cst` - Lower CST to TrunkIR module
//!
//! This allows Salsa to cache the CST independently from the TrunkIR output.

mod context;
mod declarations;
mod expressions;
mod helpers;
mod literals;
mod statements;

use crate::SourceFile;
use tree_sitter::{Node, Parser};
use trunk_ir::dialect::core;
use trunk_ir::{Location, PathId, Span, Symbol};

pub use helpers::ParsedCst;

use context::CstLoweringCtx;
use declarations::{
    lower_ability_decl, lower_const_decl, lower_enum_decl, lower_function, lower_mod_decl,
    lower_struct_decl, lower_use_decl,
};
use helpers::{is_comment, span_from_node};

// =============================================================================
// Entry Points
// =============================================================================

/// Parse a source file into a CST.
///
/// This is the first stage of the compilation pipeline. The resulting
/// `ParsedCst` is cached by Salsa and will only be recomputed when
/// the source file changes.
#[salsa::tracked]
pub fn parse_cst(db: &dyn salsa::Database, source: SourceFile) -> Option<ParsedCst> {
    let text = source.text(db);

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");

    parser.parse(text, None).map(ParsedCst::new)
}

/// Lower a parsed CST to TrunkIR module.
///
/// This is the second stage of the compilation pipeline. It takes
/// the parsed CST and source file (for text extraction) and produces
/// a TrunkIR module.
#[salsa::tracked]
pub fn lower_cst<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
    cst: ParsedCst,
) -> core::Module<'db> {
    let path = PathId::new(db, source.uri(db).as_str().to_owned());
    let text = source.text(db);
    let root = cst.root_node();
    let location = Location::new(path, span_from_node(&root));

    lower_cst_impl(db, path, text, root, location)
}

/// Lower a source file directly from CST to TrunkIR module.
///
/// This is a convenience function that combines `parse_cst` and `lower_cst`.
/// For fine-grained caching control, use the two functions separately.
#[salsa::tracked]
pub fn lower_source_file<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
) -> core::Module<'db> {
    let path = PathId::new(db, source.uri(db).as_str().to_owned());

    match parse_cst(db, source) {
        Some(cst) => lower_cst(db, source, cst),
        None => {
            // Return empty module on parse failure
            let location = Location::new(path, Span::new(0, 0));
            core::Module::build(db, location, Symbol::new("main"), |_| {})
        }
    }
}

/// Internal implementation of CST lowering.
fn lower_cst_impl<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    text: &str,
    root: Node<'_>,
    location: Location<'db>,
) -> core::Module<'db> {
    core::Module::build(db, location, Symbol::new("main"), |top| {
        let mut cursor = root.walk();
        let mut ctx = CstLoweringCtx::new(db, path, text);

        for child in root.named_children(&mut cursor) {
            if is_comment(child.kind()) {
                continue;
            }
            match child.kind() {
                "function_definition" => {
                    if let Some(func) = lower_function(&mut ctx, child) {
                        top.op(func);
                    }
                }
                "struct_declaration" => {
                    if let Some(struct_op) = lower_struct_decl(&mut ctx, child) {
                        top.op(struct_op);
                    }
                }
                "enum_declaration" => {
                    if let Some(enum_op) = lower_enum_decl(&mut ctx, child) {
                        top.op(enum_op);
                    }
                }
                "const_declaration" => {
                    if let Some(const_op) = lower_const_decl(&mut ctx, top, child) {
                        top.op(const_op);
                    }
                }
                "ability_declaration" => {
                    if let Some(ability_op) = lower_ability_decl(&mut ctx, child) {
                        top.op(ability_op);
                    }
                }
                "mod_declaration" => {
                    if let Some(mod_op) = lower_mod_decl(&mut ctx, child) {
                        top.op(mod_op);
                    }
                }
                "use_declaration" => {
                    lower_use_decl(&mut ctx, child, top);
                }
                _ => {}
            }
        }
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::{InputEdit, Parser, Point};
    use trunk_ir::DialectOp;
    use trunk_ir::dialect::{func, src};

    #[salsa::db]
    #[derive(Default, Clone)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    fn lower_and_get_module<'db>(db: &'db TestDb, source: &str) -> core::Module<'db> {
        let file = SourceFile::from_path(db, "test.trb", source.to_string());
        lower_source_file(db, file)
    }

    fn point_for_byte(text: &str, byte: usize) -> Point {
        let mut row = 0usize;
        let mut column = 0usize;
        for b in text.as_bytes().iter().take(byte.min(text.len())) {
            if *b == b'\n' {
                row += 1;
                column = 0;
            } else {
                column += 1;
            }
        }
        Point { row, column }
    }

    fn apply_replace(old: &str, start: usize, old_end: usize, insert: &str) -> (String, InputEdit) {
        let mut new_text = String::with_capacity(old.len() - (old_end - start) + insert.len());
        new_text.push_str(&old[..start]);
        new_text.push_str(insert);
        new_text.push_str(&old[old_end..]);

        let start_point = point_for_byte(old, start);
        let old_end_point = point_for_byte(old, old_end);
        let new_end_byte = start + insert.len();
        let new_end_point = point_for_byte(&new_text, new_end_byte);

        (
            new_text,
            InputEdit {
                start_byte: start,
                old_end_byte: old_end,
                new_end_byte,
                start_position: start_point,
                old_end_position: old_end_point,
                new_end_position: new_end_point,
            },
        )
    }

    #[test]
    fn test_incremental_parse_matches_full_parse() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");

        let old_text = "fn main() {\n  let x = 1;\n  x + 2\n}\n";
        let mut tree = parser.parse(old_text, None).expect("initial parse");

        let start = old_text.find("x + 2").expect("find expression");
        let old_end = start + "x + 2".len();
        let (new_text, edit) = apply_replace(old_text, start, old_end, "x + 2 + 3");

        tree.edit(&edit);
        let incremental = parser
            .parse(&new_text, Some(&tree))
            .expect("incremental parse");
        let full = parser.parse(&new_text, None).expect("full parse");

        assert_eq!(
            incremental.root_node().to_sexp(),
            full.root_node().to_sexp()
        );
    }

    #[test]
    fn test_simple_function() {
        let db = TestDb::default();
        let source = "fn main() { 42 }";
        let module = lower_and_get_module(&db, source);

        // Check module has a function named "main"
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty(), "Module should have at least one block");

        let ops = blocks[0].operations(&db);
        assert!(!ops.is_empty(), "Block should have at least one operation");

        // Check first op is a function
        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "main");
    }

    #[test]
    fn test_nat_literal() {
        let db = TestDb::default();
        let source = "fn main() { 123 }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_binary_expression() {
        let db = TestDb::default();
        let source = "fn main() { 1 + 2 }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_let_binding() {
        let db = TestDb::default();
        let source = "fn main() { let x = 10; x }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_tuple_pattern() {
        let db = TestDb::default();
        let source = "fn main() { let #(a, b) = #(1, 2); a + b }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_list_expression() {
        let db = TestDb::default();
        let source = "fn main() { [1, 2, 3] }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_case_expression() {
        let db = TestDb::default();
        let source = r#"
            fn main() {
                let x = 1;
                case x {
                    0 { "zero" }
                    1 { "one" }
                    _ { "other" }
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_lambda_expression() {
        let db = TestDb::default();
        let source = "fn main() { fn(x) { x + 1 } }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_method_call() {
        let db = TestDb::default();
        let source = "fn main() { [1, 2, 3].len() }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_string_literal() {
        let db = TestDb::default();
        let source = r#"fn main() { "hello" }"#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_wildcard_pattern() {
        let db = TestDb::default();
        let source = "fn main() { let _ = 42; 0 }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_struct_declaration() {
        let db = TestDb::default();
        let source = r#"
            struct Point {
                x: Int,
                y: Int,
            }
            fn main() { 0 }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        // Should have at least 2 operations (struct and function)
        let ops = blocks[0].operations(&db);
        assert!(ops.len() >= 2, "Should have struct and function");
    }

    #[test]
    fn test_enum_declaration() {
        let db = TestDb::default();
        let source = r#"
            enum Option(a) {
                Some(a),
                None,
            }
            fn main() { 0 }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        assert!(ops.len() >= 2, "Should have enum and function");
    }

    #[test]
    fn test_const_declaration() {
        let db = TestDb::default();
        // Test const declaration lowered to src.const
        // Note: uppercase identifiers like PI are parsed as type_identifier by the grammar
        // so we use lowercase for const names
        let source = "const pi = 42";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        // Now only src.const is generated (no separate arith.const for the value)
        assert!(!ops.is_empty(), "Should have at least one operation");

        // The first op should be src.const
        let const_op = src::Const::from_operation(&db, ops[0]).expect("Should be a src.const");
        assert_eq!(const_op.name(&db), "pi");
    }

    #[test]
    fn test_inline_module() {
        let db = TestDb::default();
        let source = r#"
            pub mod math {
                pub fn add(x: Int, y: Int) -> Int {
                    x + y
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        // Top-level module should contain a nested module
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        assert!(!ops.is_empty(), "Should have at least one operation");

        // The first op should be core.module (the nested "math" module)
        let nested_module =
            core::Module::from_operation(&db, ops[0]).expect("Should be a core.module");
        assert_eq!(nested_module.name(&db), "math");

        // The nested module should contain the "add" function
        let nested_body = nested_module.body(&db);
        let nested_blocks = nested_body.blocks(&db);
        assert!(!nested_blocks.is_empty());

        let nested_ops = nested_blocks[0].operations(&db);
        assert!(
            !nested_ops.is_empty(),
            "Nested module should have operations"
        );

        let func_op =
            func::Func::from_operation(&db, nested_ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "add");
    }

    #[test]
    fn test_nested_modules() {
        let db = TestDb::default();
        let source = r#"
            pub mod outer {
                pub mod inner {
                    pub fn value() -> Int { 42 }
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        // Get the outer module
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);
        let outer_module =
            core::Module::from_operation(&db, ops[0]).expect("Should be a core.module");
        assert_eq!(outer_module.name(&db), "outer");

        // Get the inner module
        let outer_body = outer_module.body(&db);
        let outer_blocks = outer_body.blocks(&db);
        let outer_ops = outer_blocks[0].operations(&db);
        let inner_module =
            core::Module::from_operation(&db, outer_ops[0]).expect("Should be a core.module");
        assert_eq!(inner_module.name(&db), "inner");

        // Check the function inside inner
        let inner_body = inner_module.body(&db);
        let inner_blocks = inner_body.blocks(&db);
        let inner_ops = inner_blocks[0].operations(&db);
        let func_op = func::Func::from_operation(&db, inner_ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "value");
    }
}
