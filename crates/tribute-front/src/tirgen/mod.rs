//! TrunkIR generation from CST.
//!
//! This pass converts Tree-sitter CST directly to TrunkIR operations,
//! bypassing the AST intermediate representation.
//! At this stage, names are unresolved (using `src` dialect ops).
//!
//! ## Pipeline
//!
//! The lowering is split into two Salsa-tracked stages:
//! 1. `parse_cst` - Wrap CST from `SourceCst` (cached by Salsa)
//! 2. `lower_cst` - Lower CST to TrunkIR module
//!
//! This allows Salsa to cache the CST independently from the TrunkIR output.

mod context;
mod declarations;
mod expressions;
mod helpers;
mod literals;
mod statements;

use crate::SourceCst;
use ropey::Rope;
use tree_sitter::Node;
use trunk_ir::dialect::core;
use trunk_ir::{Location, PathId, Span, Symbol};

pub use helpers::ParsedCst;
pub use literals::parse_rune_literal;

use context::CstLoweringCtx;
use declarations::{
    lower_ability_decl, lower_const_decl, lower_enum_decl, lower_function, lower_mod_decl,
    lower_struct_decl, lower_use_decl,
};
use helpers::{is_comment, span_from_node};

// =============================================================================
// Entry Points
// =============================================================================

/// Wrap a pre-parsed CST stored in the database.
#[salsa::tracked]
pub fn parse_cst(db: &dyn salsa::Database, source: SourceCst) -> Option<ParsedCst> {
    let tree = source.tree(db).clone()?;
    Some(ParsedCst::new(tree))
}

/// Lower a parsed CST to TrunkIR module.
///
/// This is the second stage of the compilation pipeline. It takes
/// the parsed CST and source file (for text extraction) and produces
/// a TrunkIR module.
#[salsa::tracked]
pub fn lower_cst<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    cst: ParsedCst,
) -> core::Module<'db> {
    let path = PathId::new(db, source.uri(db).as_str().to_owned());
    let text = source.text(db);
    let root = cst.root_node();
    let location = Location::new(path, span_from_node(&root));

    lower_cst_impl(db, path, text.clone(), root, location)
}

/// Lower a pre-parsed CST stored alongside source text to TrunkIR module.
#[salsa::tracked]
pub fn lower_source_cst<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> core::Module<'db> {
    let path = PathId::new(db, source.uri(db).as_str().to_owned());
    let text = source.text(db);
    let Some(cst) = parse_cst(db, source) else {
        let location = Location::new(path, Span::new(0, 0));
        return core::Module::build(db, location, Symbol::new("main"), |_| {});
    };
    let root = cst.root_node();
    let location = Location::new(path, span_from_node(&root));

    lower_cst_impl(db, path, text.clone(), root, location)
}

/// Internal implementation of CST lowering.
fn lower_cst_impl<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    text: Rope,
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
                    if let Some((struct_op, getters_module)) = lower_struct_decl(&mut ctx, child) {
                        top.op(struct_op);
                        top.op(getters_module);
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
    use salsa::Setter;
    use tree_sitter::{InputEdit, Parser, Point};
    use tribute_ir::ModulePathExt as _;
    use tribute_ir::dialect::tribute;
    use trunk_ir::Attribute;
    use trunk_ir::DialectOp;
    use trunk_ir::dialect::{adt, func};

    fn lower_and_get_module<'db>(db: &'db salsa::DatabaseImpl, source: &str) -> core::Module<'db> {
        lower_from_tree(db, source)
    }

    fn lower_from_tree<'db>(db: &'db salsa::DatabaseImpl, source: &str) -> core::Module<'db> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(source, None).expect("tree");
        let file = SourceCst::from_path(db, "test.trb", source.into(), Some(tree));
        lower_source_cst(db, file)
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
    fn test_lower_source_cst_reuses_tree() {
        let db = salsa::DatabaseImpl::default();
        let module = lower_from_tree(&db, "fn main() { 42 }");

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty(), "Module should have at least one block");
    }

    #[test]
    fn test_source_cst_set_tree() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");

        let mut db = salsa::DatabaseImpl::default();
        let tree = parser.parse("fn main() { 1 }", None).expect("tree");
        let source = SourceCst::from_path(&db, "test.trb", "fn main() { 1 }".into(), Some(tree));

        let tree2 = parser.parse("fn main() { 2 }", None).expect("tree2");
        source.set_tree(&mut db).to(Some(tree2));
    }

    #[test]
    fn test_simple_function() {
        let db = salsa::DatabaseImpl::default();
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
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { 123 }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_binary_expression() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { 1 + 2 }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_let_binding() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { let x = 10; x }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_tuple_pattern() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { let #(a, b) = #(1, 2); a + b }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_list_expression() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { [1, 2, 3] }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    fn collect_ops<'db>(
        db: &'db dyn salsa::Database,
        region: trunk_ir::Region<'db>,
        out: &mut Vec<trunk_ir::Operation<'db>>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter().copied() {
                out.push(op);
                for nested in op.regions(db).iter().copied() {
                    collect_ops(db, nested, out);
                }
            }
        }
    }

    fn collect_string_const_values<'db>(
        db: &'db dyn salsa::Database,
        module: core::Module<'db>,
    ) -> Vec<String> {
        let mut ops = Vec::new();
        collect_ops(db, module.body(db), &mut ops);
        ops.iter()
            .filter_map(|op| adt::StringConst::from_operation(db, *op).ok())
            .map(|string_const| string_const.value(db).clone())
            .collect()
    }

    fn collect_bytes_const_values<'db>(
        db: &'db dyn salsa::Database,
        module: core::Module<'db>,
    ) -> Vec<Vec<u8>> {
        let mut ops = Vec::new();
        collect_ops(db, module.body(db), &mut ops);
        ops.iter()
            .filter_map(|op| adt::BytesConst::from_operation(db, *op).ok())
            .filter_map(|bytes_const| {
                let Attribute::Bytes(value) = bytes_const.value(db) else {
                    return None;
                };
                Some(value.clone())
            })
            .collect()
    }

    #[test]
    fn test_case_expression() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"
            fn main() {
                let x = 1;
                case x {
                    0 -> "zero"
                    1 -> "one"
                    _ -> "other"
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let mut ops = Vec::new();
        collect_ops(&db, module.body(&db), &mut ops);

        let case_arms = ops
            .iter()
            .filter(|op| {
                op.dialect(&db) == tribute::DIALECT_NAME() && op.name(&db) == tribute::ARM()
            })
            .count();
        assert_eq!(case_arms, 3, "Expected three case arms");
    }

    #[test]
    fn test_constructor_expression() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { Some(1) }";
        let module = lower_and_get_module(&db, source);

        let mut ops = Vec::new();
        collect_ops(&db, module.body(&db), &mut ops);

        let has_constructor = ops.iter().any(|op| {
            op.dialect(&db) == tribute::DIALECT_NAME() && op.name(&db) == tribute::CONS()
        });
        assert!(has_constructor, "Expected a tribute.cons operation");
    }

    #[test]
    fn test_use_tree() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"
            use std::{io, fmt as f}
            fn main() { 0 }
        "#;
        let module = lower_and_get_module(&db, source);

        let mut ops = Vec::new();
        collect_ops(&db, module.body(&db), &mut ops);

        let use_count = ops
            .iter()
            .filter(|op| {
                op.dialect(&db) == tribute::DIALECT_NAME() && op.name(&db) == tribute::USE()
            })
            .count();
        assert_eq!(use_count, 2, "Expected two use imports");
    }

    #[test]
    fn test_lambda_expression() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { fn(x) { x + 1 } }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_method_call() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { [1, 2, 3].len() }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_string_literal() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"fn main() { "hello" }"#;
        let module = lower_and_get_module(&db, source);

        let values = collect_string_const_values(&db, module);
        assert!(
            values.iter().any(|value| value == "hello"),
            "Expected a string const with value \"hello\""
        );
    }

    #[test]
    fn test_prefixed_string_literal() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"fn main() { s"hello" }"#;
        let module = lower_and_get_module(&db, source);

        let values = collect_string_const_values(&db, module);
        assert!(
            values.iter().any(|value| value == "hello"),
            "Expected a string const with value \"hello\""
        );
    }

    #[test]
    fn test_raw_string_literal() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"fn main() { rs"hello" }"#;
        let module = lower_and_get_module(&db, source);

        let values = collect_string_const_values(&db, module);
        assert!(
            values.iter().any(|value| value == "hello"),
            "Expected a string const with value \"hello\""
        );
    }

    #[test]
    fn test_raw_bytes_literal() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"fn main() { br"hello" }"#;
        let module = lower_and_get_module(&db, source);

        let values = collect_bytes_const_values(&db, module);
        assert!(
            values.iter().any(|value| value.as_slice() == b"hello"),
            "Expected a bytes const with value \"hello\""
        );
    }

    #[test]
    fn test_raw_interpolated_string_literal() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"
            fn main() {
                let name = "World";
                rs"Hello, \{name}!"
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let values = collect_string_const_values(&db, module);
        assert!(
            values.iter().any(|value| value == "Hello, "),
            "Expected a string const prefix"
        );
        assert!(
            values.iter().any(|value| value == "!"),
            "Expected a string const suffix"
        );
    }

    #[test]
    fn test_raw_interpolated_bytes_literal() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"
            fn main() {
                let chunk = "data";
                rb"chunk: \{chunk}"
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let values = collect_bytes_const_values(&db, module);
        assert!(
            values.iter().any(|value| value.as_slice() == b"chunk: "),
            "Expected a bytes const prefix"
        );
    }

    #[test]
    fn test_wildcard_pattern() {
        let db = salsa::DatabaseImpl::default();
        let source = "fn main() { let _ = 42; 0 }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_struct_declaration() {
        let db = salsa::DatabaseImpl::default();
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

        // Should have 3 operations: type.struct, core.module (accessors), func.func (main)
        let ops = blocks[0].operations(&db);
        assert!(
            ops.len() >= 3,
            "Should have struct, accessor module, and function"
        );

        // Verify we have the struct type
        assert_eq!(ops[0].full_name(&db), "tribute.struct_def");

        // Verify we have the accessor module
        assert_eq!(ops[1].full_name(&db), "core.module");

        // Check the accessor module contents
        use trunk_ir::{DialectOp, dialect::core};
        if let Ok(accessor_module) = core::Module::from_operation(&db, ops[1]) {
            // Should contain getter functions and field modules for x and y
            let accessor_body = accessor_module.body(&db);
            let accessor_blocks = accessor_body.blocks(&db);
            assert!(
                !accessor_blocks.is_empty(),
                "Accessor module should have at least one block"
            );

            let accessor_ops = accessor_blocks[0].operations(&db);
            // For 2 fields: 2 getters + 2 field modules = 4 ops
            assert_eq!(
                accessor_ops.len(),
                4,
                "Accessor module should have 2 getters and 2 field modules"
            );

            // Verify operation types: getter, module, getter, module
            assert_eq!(accessor_ops[0].full_name(&db), "func.func");
            assert_eq!(accessor_ops[1].full_name(&db), "core.module");
            assert_eq!(accessor_ops[2].full_name(&db), "func.func");
            assert_eq!(accessor_ops[3].full_name(&db), "core.module");
        }
    }

    #[test]
    fn test_enum_declaration() {
        let db = salsa::DatabaseImpl::default();
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
        let db = salsa::DatabaseImpl::default();
        // Test const declaration lowered to tribute.const
        // Note: uppercase identifiers like PI are parsed as type_identifier by the grammar
        // so we use lowercase for const names
        let source = "const pi = 42";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        // Now only tribute.const is generated (no separate arith.const for the value)
        assert!(!ops.is_empty(), "Should have at least one operation");

        // The first op should be tribute.const
        let const_op =
            tribute::Const::from_operation(&db, ops[0]).expect("Should be a tribute.const");
        assert_eq!(const_op.name(&db), "pi");
    }

    #[test]
    fn test_inline_module() {
        let db = salsa::DatabaseImpl::default();
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
        assert_eq!(func_op.name(&db).last_segment(), "add");
    }

    #[test]
    fn test_nested_modules() {
        let db = salsa::DatabaseImpl::default();
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
        assert_eq!(func_op.name(&db).last_segment(), "value");
    }

    #[test]
    fn test_function_type_with_effects() {
        use trunk_ir::DialectType;
        use trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func as CoreFunc};

        let db = salsa::DatabaseImpl::default();
        // Higher-order function with explicit effect annotation
        let source = r#"
            fn apply(f: fn(Int) ->{Console} Int, x: Int) -> Int {
                f(x)
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "apply");

        // Get the function's declared type to access parameter types
        let func_ty = func_op.r#type(&db);
        let core_func = CoreFunc::from_type(&db, func_ty).expect("Should be a function type");
        let param_types = core_func.params(&db);
        assert_eq!(param_types.len(), 2, "Should have 2 parameters");

        // First param should be fn(Int) ->{Console} Int
        let f_type = param_types[0];
        let func_type = CoreFunc::from_type(&db, f_type).expect("Should be a function type");

        // Check it has an effect annotation
        let effect = func_type
            .effect(&db)
            .expect("Function type should have effect");
        let effect_row = EffectRowType::from_type(&db, effect).expect("Should be an effect row");

        // The effect row should contain Console
        let abilities = effect_row.abilities(&db);
        assert_eq!(abilities.len(), 1, "Should have 1 ability");

        // Verify the ability is Console
        let ability = AbilityRefType::from_type(&db, abilities[0]).expect("Should be ability ref");
        assert_eq!(
            ability.name(&db).map(|s| s.to_string()),
            Some("Console".to_string())
        );
    }

    #[test]
    fn test_function_type_pure() {
        use trunk_ir::DialectType;
        use trunk_ir::dialect::core::{EffectRowType, Func as CoreFunc};

        let db = salsa::DatabaseImpl::default();
        // Pure function type with empty effect row
        let source = r#"
            fn map(f: fn(Int) ->{} Int, x: Int) -> Int {
                f(x)
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        let func_ty = func_op.r#type(&db);
        let core_func = CoreFunc::from_type(&db, func_ty).expect("Should be a function type");
        let param_types = core_func.params(&db);

        // First param should be fn(Int) ->{} Int (pure)
        let f_type = param_types[0];
        let func_type = CoreFunc::from_type(&db, f_type).expect("Should be a function type");

        // Check it has an empty effect row
        let effect = func_type
            .effect(&db)
            .expect("Function type should have effect");
        let effect_row = EffectRowType::from_type(&db, effect).expect("Should be an effect row");
        assert!(
            effect_row.is_empty(&db),
            "Pure function should have empty effect row"
        );
    }

    #[test]
    fn test_function_type_with_row_variable() {
        use trunk_ir::DialectType;
        use trunk_ir::dialect::core::{EffectRowType, Func as CoreFunc};

        let db = salsa::DatabaseImpl::default();
        // Function type with row variable tail
        let source = r#"
            fn run_state(f: fn() ->{State(Int), e} Int) -> Int {
                f()
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        let func_ty = func_op.r#type(&db);
        let core_func = CoreFunc::from_type(&db, func_ty).expect("Should be a function type");
        let param_types = core_func.params(&db);

        // First param should be fn() ->{State(Int), e} Int
        let f_type = param_types[0];
        let func_type = CoreFunc::from_type(&db, f_type).expect("Should be a function type");

        let effect = func_type
            .effect(&db)
            .expect("Function type should have effect");
        let effect_row = EffectRowType::from_type(&db, effect).expect("Should be an effect row");

        // Should have 1 ability (State) and a tail variable
        let abilities = effect_row.abilities(&db);
        assert_eq!(abilities.len(), 1, "Should have 1 ability (State)");
        assert!(
            effect_row.tail_var(&db).is_some(),
            "Should have a tail variable"
        );
    }

    #[test]
    fn test_function_type_row_variable_not_at_end() {
        use trunk_ir::DialectType;
        use trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func as CoreFunc};

        let db = salsa::DatabaseImpl::default();
        // Row variable comes BEFORE the concrete ability
        let source = r#"
            fn apply(f: fn() ->{e, Console} Int) -> Int {
                f()
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        let func_ty = func_op.r#type(&db);
        let core_func = CoreFunc::from_type(&db, func_ty).expect("Should be a function type");
        let param_types = core_func.params(&db);

        // First param should be fn() ->{e, Console} Int
        let f_type = param_types[0];
        let func_type = CoreFunc::from_type(&db, f_type).expect("Should be a function type");

        let effect = func_type
            .effect(&db)
            .expect("Function type should have effect");
        let effect_row = EffectRowType::from_type(&db, effect).expect("Should be an effect row");

        // Should have 1 ability (Console) and a row variable
        let abilities = effect_row.abilities(&db);
        assert_eq!(abilities.len(), 1, "Should have 1 ability (Console)");

        let ability = AbilityRefType::from_type(&db, abilities[0]).expect("Should be ability ref");
        assert_eq!(
            ability.name(&db).map(|s| s.to_string()),
            Some("Console".to_string())
        );

        assert!(
            effect_row.tail_var(&db).is_some(),
            "Should have a row variable even when not at the end"
        );
    }

    #[test]
    fn test_extern_function() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"
            extern "intrinsic" fn int_is_small(x: Int) -> Bool
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "int_is_small");

        // Check the ABI attribute is set
        let abi = func_op
            .operation()
            .attributes(&db)
            .get(&trunk_ir::Symbol::new("abi"));
        assert_eq!(abi, Some(&Attribute::String("intrinsic".to_string())));

        // Check the body contains func.unreachable
        let body = func_op.body(&db);
        let body_blocks = body.blocks(&db);
        let body_ops = body_blocks[0].operations(&db);
        assert_eq!(body_ops.len(), 1);
        assert_eq!(body_ops[0].full_name(&db), "func.unreachable");
    }

    #[test]
    fn test_extern_function_no_abi() {
        let db = salsa::DatabaseImpl::default();
        let source = r#"
            extern fn native_add(a: Int, b: Int) -> Int
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "native_add");

        // No ABI attribute when not specified
        let abi = func_op
            .operation()
            .attributes(&db)
            .get(&trunk_ir::Symbol::new("abi"));
        assert!(abi.is_none());
    }

    #[test]
    fn test_function_declaration_effect_annotation() {
        use trunk_ir::DialectType;
        use trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func as CoreFunc};

        let db = salsa::DatabaseImpl::default();
        // Function declaration with explicit effect annotation
        let source = r#"
            ability State(s) {
                fn get() -> s
            }
            fn get_state() ->{State(Int)} Int {
                State::get()
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        // Find the get_state function (skip ability definition)
        let func_op = ops
            .iter()
            .find_map(|op| {
                func::Func::from_operation(&db, *op)
                    .ok()
                    .filter(|f| f.name(&db) == "get_state")
            })
            .expect("Should have get_state function");

        let func_ty = func_op.r#type(&db);
        let core_func = CoreFunc::from_type(&db, func_ty).expect("Should be a function type");

        // Function should have the effect annotation
        let effect = core_func
            .effect(&db)
            .expect("Function type should have effect annotation");
        let effect_row = EffectRowType::from_type(&db, effect).expect("Should be an effect row");

        // Should have 1 ability (State)
        let abilities = effect_row.abilities(&db);
        assert_eq!(abilities.len(), 1, "Should have 1 ability (State)");

        // Verify the ability is State(Int)
        let ability = AbilityRefType::from_type(&db, abilities[0]).expect("Should be ability ref");
        assert_eq!(
            ability.name(&db).map(|s| s.to_string()),
            Some("State".to_string())
        );

        // Verify the ability has Int parameter
        let params = ability.params(&db);
        assert_eq!(params.len(), 1, "State should have 1 type parameter");
    }

    #[test]
    fn test_extern_function_effect_annotation() {
        use trunk_ir::DialectType;
        use trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func as CoreFunc};

        let db = salsa::DatabaseImpl::default();
        // Extern function with effect annotation
        let source = r#"
            ability IO {
                fn read_line() -> Text
            }
            extern fn do_io() ->{IO} Int
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);

        // Find the do_io extern function
        let func_op = ops
            .iter()
            .find_map(|op| {
                func::Func::from_operation(&db, *op)
                    .ok()
                    .filter(|f| f.name(&db) == "do_io")
            })
            .expect("Should have do_io function");

        let func_ty = func_op.r#type(&db);
        let core_func = CoreFunc::from_type(&db, func_ty).expect("Should be a function type");

        // Function should have the effect annotation
        let effect = core_func
            .effect(&db)
            .expect("Extern function should have effect annotation");
        let effect_row = EffectRowType::from_type(&db, effect).expect("Should be an effect row");

        // Should have 1 ability (IO)
        let abilities = effect_row.abilities(&db);
        assert_eq!(abilities.len(), 1, "Should have 1 ability (IO)");

        // Verify the ability is IO
        let ability = AbilityRefType::from_type(&db, abilities[0]).expect("Should be ability ref");
        assert_eq!(
            ability.name(&db).map(|s| s.to_string()),
            Some("IO".to_string())
        );
    }
}
