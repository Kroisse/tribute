//! CST to AST lowering for patterns.

use tree_sitter::Node;
use trunk_ir::Symbol;

use crate::ast::{FieldPattern, FloatBits, LiteralPattern, Pattern, PatternKind, UnresolvedName};

use super::context::AstLoweringCtx;
use super::expressions::parse_string_literal;
use super::helpers::{is_comment, report_in_node};
use super::numeric::{NumericValue, parse_numeric_literal};

/// Lower a CST pattern node to an AST Pattern.
pub fn lower_pattern(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Pattern<UnresolvedName> {
    let id = ctx.fresh_id_with_span(&node);

    let kind = match node.kind() {
        // === Wildcard ===
        "wildcard_pattern" => PatternKind::Wildcard,

        // === Identifier (binding) ===
        "identifier" | "identifier_pattern" => {
            let name = ctx.node_symbol(&node);
            PatternKind::Bind {
                name,
                local_id: None,
            }
        }

        // === Literal patterns ===
        "nat_literal" | "int_literal" | "float_literal" => {
            let text = ctx.node_text_owned(&node);
            match parse_numeric_literal(&text) {
                Ok(NumericValue::Nat(value)) => PatternKind::Literal(LiteralPattern::Nat(value)),
                Ok(NumericValue::Int(value)) => PatternKind::Literal(LiteralPattern::Int(value)),
                Ok(NumericValue::Float(value)) => {
                    PatternKind::Literal(LiteralPattern::Float(FloatBits::new(value)))
                }
                Err(error) => {
                    report_in_node(ctx, &node, error.range.clone(), error.to_string());
                    PatternKind::Error
                }
            }
        }
        // grammar.js uses "string" for string patterns, not "string_literal"
        "string_literal" | "string" => {
            let text = ctx.node_text_owned(&node);
            match parse_string_literal(&text) {
                Ok(content) => PatternKind::Literal(LiteralPattern::String(content)),
                Err(error) => {
                    report_in_node(ctx, &node, error.range.clone(), error.to_string());
                    PatternKind::Error
                }
            }
        }
        "bool_literal" => {
            let text = ctx.node_text(&node);
            PatternKind::Literal(LiteralPattern::Bool(&*text == "true"))
        }
        // Boolean keywords in patterns: True, False
        "keyword_true" => PatternKind::Literal(LiteralPattern::Bool(true)),
        "keyword_false" => PatternKind::Literal(LiteralPattern::Bool(false)),
        "unit_literal" => PatternKind::Literal(LiteralPattern::Unit),

        // === Constructor/Variant pattern ===
        "constructor_pattern" => lower_constructor_pattern(ctx, node),

        // === Record pattern ===
        "record_pattern" => lower_record_pattern(ctx, node),

        // === Tuple pattern ===
        "tuple_pattern" => lower_tuple_pattern(ctx, node),

        // === List pattern ===
        "list_pattern" => lower_list_pattern(ctx, node),

        // === As pattern ===
        "as_pattern" => lower_as_pattern(ctx, node),

        // === Type identifier (could be unit variant) ===
        "type_identifier" => {
            let name = ctx.node_symbol(&node);
            let ctor_id = ctx.fresh_id_with_span(&node);
            let ctor = UnresolvedName::new(name, ctor_id);
            PatternKind::Variant {
                ctor,
                fields: Vec::new(),
            }
        }

        _ => {
            // Try to find a meaningful child
            if let Some(child) = node.named_child(0) {
                return lower_pattern(ctx, child);
            }
            PatternKind::Error
        }
    };

    Pattern::new(id, kind)
}

fn lower_constructor_pattern(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> PatternKind<UnresolvedName> {
    // grammar.js: field("name", $.type_path)
    //   + optional choice of:
    //     - Tuple-style: field("args", $.pattern_list)
    //     - Struct-style: field("fields", $.pattern_fields)
    let name_node = node.child_by_field_name("name");
    let args_node = node.child_by_field_name("args");
    let fields_node = node.child_by_field_name("fields");

    let Some(name_node) = name_node else {
        return PatternKind::Error;
    };

    let name = ctx.node_symbol(&name_node);
    let ctor_id = ctx.fresh_id_with_span(&name_node);
    let ctor = UnresolvedName::new(name, ctor_id);

    // Handle tuple-style args: Some(x), Pair(a, b)
    if let Some(args) = args_node {
        let fields = lower_pattern_list(ctx, args);
        return PatternKind::Variant { ctor, fields };
    }

    // Handle struct-style fields: Point { x, y }, Ok { value: v }
    if let Some(fields) = fields_node {
        let patterns = lower_constructor_fields(ctx, fields);
        return PatternKind::Variant {
            ctor,
            fields: patterns,
        };
    }

    // No args or fields - unit variant like None
    PatternKind::Variant {
        ctor,
        fields: Vec::new(),
    }
}

/// Lower struct-style pattern fields (pattern_fields) to a list of patterns.
fn lower_constructor_fields(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Vec<Pattern<UnresolvedName>> {
    let mut patterns = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "pattern_field" => {
                // pattern_field: name: pattern or just name (shorthand)
                if let Some(pattern_node) = child.child_by_field_name("pattern") {
                    patterns.push(lower_pattern(ctx, pattern_node));
                } else if let Some(name_node) = child.child_by_field_name("name") {
                    // Shorthand: { x } means bind x
                    let id = ctx.fresh_id_with_span(&name_node);
                    let name = ctx.node_symbol(&name_node);
                    patterns.push(Pattern::new(
                        id,
                        PatternKind::Bind {
                            name,
                            local_id: None,
                        },
                    ));
                }
            }
            "spread" => {
                // Trailing .. to ignore rest - we could track this but for now ignore
            }
            _ => {}
        }
    }

    patterns
}

fn lower_record_pattern(ctx: &mut AstLoweringCtx<'_>, node: Node) -> PatternKind<UnresolvedName> {
    let type_node = node.child_by_field_name("type");
    let fields_node = node.child_by_field_name("fields");

    let type_name = if let Some(n) = type_node {
        let name = ctx.node_symbol(&n);
        let id = ctx.fresh_id_with_span(&n);
        Some(UnresolvedName::new(name, id))
    } else {
        None
    };

    let mut fields = Vec::new();
    let mut rest = false;

    if let Some(fields_node) = fields_node {
        let mut cursor = fields_node.walk();
        for child in fields_node.named_children(&mut cursor) {
            match child.kind() {
                "field_pattern" => {
                    if let Some(field) = lower_field_pattern(ctx, child) {
                        fields.push(field);
                    }
                }
                "shorthand_field_pattern" => {
                    // { name } is shorthand for { name: name }
                    let id = ctx.fresh_id_with_span(&child);
                    let name = ctx.node_symbol(&child);
                    fields.push(FieldPattern {
                        id,
                        name,
                        pattern: None,
                    });
                }
                "rest_pattern" => {
                    rest = true;
                }
                _ => {}
            }
        }
    }

    PatternKind::Record {
        type_name,
        fields,
        rest,
    }
}

fn lower_field_pattern(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<FieldPattern<UnresolvedName>> {
    let name_node = node.child_by_field_name("name")?;
    let pattern_node = node.child_by_field_name("pattern");

    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);
    let pattern = pattern_node.map(|n| lower_pattern(ctx, n));

    Some(FieldPattern { id, name, pattern })
}

fn lower_tuple_pattern(ctx: &mut AstLoweringCtx<'_>, node: Node) -> PatternKind<UnresolvedName> {
    // tuple_pattern has field "elements" which is a pattern_list
    let elements_node = node.child_by_field_name("elements");
    let elements = elements_node
        .map(|n| lower_pattern_list(ctx, n))
        .unwrap_or_default();
    PatternKind::Tuple(elements)
}

fn lower_list_pattern(ctx: &mut AstLoweringCtx<'_>, node: Node) -> PatternKind<UnresolvedName> {
    let mut head = Vec::new();
    let mut has_rest = false;
    let mut rest: Option<Symbol> = None;
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        match child.kind() {
            "rest_pattern" => {
                // [a, b, ..tail] or [a, b, ..]
                // grammar.js uses "name" field for the rest binding
                has_rest = true;
                if let Some(name_node) = child.child_by_field_name("name") {
                    rest = Some(ctx.node_symbol(&name_node));
                }
                // rest stays None for anonymous ".."
            }
            _ => {
                head.push(lower_pattern(ctx, child));
            }
        }
    }

    if has_rest {
        PatternKind::ListRest {
            head,
            rest,
            rest_local_id: None,
        }
    } else {
        PatternKind::List(head)
    }
}

fn lower_as_pattern(ctx: &mut AstLoweringCtx<'_>, node: Node) -> PatternKind<UnresolvedName> {
    let pattern_node = node.child_by_field_name("pattern");
    // grammar.js uses "binding" field for the as-pattern name
    let name_node = node.child_by_field_name("binding");

    let (Some(pattern_node), Some(name_node)) = (pattern_node, name_node) else {
        return PatternKind::Error;
    };

    let pattern = lower_pattern(ctx, pattern_node);
    let name = ctx.node_symbol(&name_node);

    PatternKind::As {
        pattern,
        name,
        local_id: None,
    }
}

fn lower_pattern_list(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Vec<Pattern<UnresolvedName>> {
    let mut patterns = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            patterns.push(lower_pattern(ctx, child));
        }
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Decl, ExprKind, Module, Stmt, UnresolvedName};
    use ropey::Rope;
    use tree_sitter::Parser;

    use crate::astgen::{ParsedCst, lower_cst_to_ast};

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

    fn get_let_pattern(source: &str) -> PatternKind<UnresolvedName> {
        let module = parse_and_lower(source);
        let Decl::Function(func) = &module.decls[0] else {
            panic!("Expected function");
        };
        let ExprKind::Block { stmts, .. } = func.body.kind.as_ref() else {
            panic!("Expected block");
        };
        let Stmt::Let { pattern, .. } = &stmts[0] else {
            panic!("Expected let binding");
        };
        pattern.kind.as_ref().clone()
    }

    fn get_case_pattern(source: &str, arm_index: usize) -> PatternKind<UnresolvedName> {
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
        arms[arm_index].pattern.kind.as_ref().clone()
    }

    // === Wildcard Pattern ===

    #[test]
    fn test_wildcard_pattern() {
        let source = r#"
            fn main() {
                let _ = 42;
                0
            }
        "#;
        let pattern = get_let_pattern(source);
        assert!(matches!(pattern, PatternKind::Wildcard));
    }

    // === Bind (Identifier) Pattern ===

    #[test]
    fn test_bind_pattern() {
        let source = r#"
            fn main() {
                let x = 42;
                x
            }
        "#;
        let pattern = get_let_pattern(source);
        let PatternKind::Bind { name, .. } = pattern else {
            panic!("Expected bind pattern");
        };
        assert_eq!(name.to_string(), "x");
    }

    // === Literal Patterns ===

    #[test]
    fn test_nat_literal_pattern() {
        let source = r#"
            fn main() {
                case x {
                    42 -> 1
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Literal(LiteralPattern::Nat(42)) = pattern else {
            panic!("Expected nat literal pattern 42, got {:?}", pattern);
        };
    }

    #[test]
    fn test_int_literal_pattern_negative() {
        let source = r#"
            fn main() {
                case x {
                    -42 -> 1
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Literal(LiteralPattern::Int(-42)) = pattern else {
            panic!("Expected int literal pattern -42, got {:?}", pattern);
        };
    }

    #[test]
    fn test_float_literal_pattern() {
        let source = r#"
            fn main() {
                case x {
                    2.5 -> 1
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Literal(LiteralPattern::Float(f)) = pattern else {
            panic!("Expected float literal pattern, got {:?}", pattern);
        };
        assert!((f.value() - 2.5_f64).abs() < 0.001);
    }

    #[test]
    fn test_string_literal_pattern() {
        let source = r#"
            fn main() {
                case x {
                    "hello" -> 1
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Literal(LiteralPattern::String(s)) = pattern else {
            panic!("Expected string literal pattern, got {:?}", pattern);
        };
        assert_eq!(s, "hello");
    }

    // Note: In Tribute, `True` and `False` (capitalized) are boolean literals,
    // while lowercase `true` and `false` are valid identifiers. This is an intentional
    // design choice similar to some ML-family languages.
    #[test]
    fn test_lowercase_true_in_pattern_becomes_identifier() {
        let source = r#"
            fn main() {
                case x {
                    true -> 1
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        // Lowercase 'true' is parsed as identifier/bind pattern
        let PatternKind::Bind { name, .. } = pattern else {
            panic!("Expected bind pattern for 'true', got {:?}", pattern);
        };
        assert_eq!(name.to_string(), "true");
    }

    // Capitalized True/False are keywords that become Bool literals in patterns
    #[test]
    fn test_keyword_true_in_pattern_becomes_bool_literal() {
        let source = r#"
            fn main() {
                case x {
                    True -> 1
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Literal(LiteralPattern::Bool(value)) = pattern else {
            panic!(
                "Expected Bool literal pattern for 'True', got {:?}",
                pattern
            );
        };
        assert!(value);
    }

    #[test]
    fn test_keyword_false_in_pattern_becomes_bool_literal() {
        let source = r#"
            fn main() {
                case x {
                    False -> 0
                    _ -> 1
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Literal(LiteralPattern::Bool(value)) = pattern else {
            panic!(
                "Expected Bool literal pattern for 'False', got {:?}",
                pattern
            );
        };
        assert!(!value);
    }

    // === Constructor/Variant Pattern ===

    #[test]
    fn test_constructor_pattern_no_args() {
        let source = r#"
            fn main() {
                case x {
                    None -> 0
                    _ -> 1
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Variant { ctor, fields } = pattern else {
            panic!("Expected variant pattern, got {:?}", pattern);
        };
        assert_eq!(ctor.name().to_string(), "None");
        assert!(fields.is_empty());
    }

    #[test]
    fn test_constructor_pattern_with_args() {
        let source = r#"
            fn main() {
                case x {
                    Some(y) -> y
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Variant { ctor, fields } = pattern else {
            panic!("Expected variant pattern, got {:?}", pattern);
        };
        assert_eq!(ctor.name().to_string(), "Some");
        assert_eq!(fields.len(), 1);
        let PatternKind::Bind { name, .. } = fields[0].kind.as_ref() else {
            panic!("Expected bind pattern in variant");
        };
        assert_eq!(name.to_string(), "y");
    }

    #[test]
    fn test_qualified_constructor_pattern() {
        let source = r#"
            fn main() {
                case x {
                    std::io::ReadLine(bytes) -> bytes
                    _ -> b""
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Variant { ctor, fields } = pattern else {
            panic!("Expected variant pattern, got {:?}", pattern);
        };
        assert_eq!(ctor.qualified.to_string(), "std::io::ReadLine");
        assert_eq!(fields.len(), 1);
    }

    #[test]
    fn test_constructor_pattern_multiple_args() {
        let source = r#"
            fn main() {
                case x {
                    Pair(a, b) -> a
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Variant { ctor, fields } = pattern else {
            panic!("Expected variant pattern, got {:?}", pattern);
        };
        assert_eq!(ctor.name().to_string(), "Pair");
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn test_constructor_pattern_struct_style() {
        let source = r#"
            fn main() {
                case x {
                    Point { x: px, y: py } -> px
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::Variant { ctor, fields } = pattern else {
            panic!("Expected variant pattern, got {:?}", pattern);
        };
        assert_eq!(ctor.name().to_string(), "Point");
        assert_eq!(fields.len(), 2);
    }

    // === Tuple Pattern ===
    // Note: Tuple patterns use #(a, b) syntax

    #[test]
    fn test_tuple_pattern() {
        let source = r#"
            fn main() {
                let #(a, b) = pair;
                a
            }
        "#;
        let pattern = get_let_pattern(source);
        let PatternKind::Tuple(elements) = pattern else {
            panic!("Expected tuple pattern, got {:?}", pattern);
        };
        assert_eq!(elements.len(), 2);
    }

    #[test]
    fn test_tuple_pattern_nested() {
        let source = r#"
            fn main() {
                let #(a, #(b, c)) = nested;
                a
            }
        "#;
        let pattern = get_let_pattern(source);
        let PatternKind::Tuple(elements) = pattern else {
            panic!("Expected tuple pattern, got {:?}", pattern);
        };
        assert_eq!(elements.len(), 2);
        let PatternKind::Tuple(inner) = elements[1].kind.as_ref() else {
            panic!("Expected nested tuple pattern");
        };
        assert_eq!(inner.len(), 2);
    }

    // === List Pattern ===

    #[test]
    fn test_list_pattern_empty() {
        let source = r#"
            fn main() {
                case x {
                    [] -> 0
                    _ -> 1
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::List(elements) = pattern else {
            panic!("Expected list pattern, got {:?}", pattern);
        };
        assert!(elements.is_empty());
    }

    #[test]
    fn test_list_pattern_elements() {
        let source = r#"
            fn main() {
                case x {
                    [a, b, c] -> 0
                    _ -> 1
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::List(elements) = pattern else {
            panic!("Expected list pattern, got {:?}", pattern);
        };
        assert_eq!(elements.len(), 3);
    }

    #[test]
    fn test_list_pattern_with_rest() {
        let source = r#"
            fn main() {
                case x {
                    [head, ..tail] -> head
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::ListRest { head, rest, .. } = pattern else {
            panic!("Expected list rest pattern, got {:?}", pattern);
        };
        assert_eq!(head.len(), 1);
        assert!(rest.is_some());
        assert_eq!(rest.unwrap().to_string(), "tail");
    }

    #[test]
    fn test_list_pattern_with_anonymous_rest() {
        let source = r#"
            fn main() {
                case x {
                    [head, ..] -> head
                    _ -> 0
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::ListRest { head, rest, .. } = pattern else {
            panic!("Expected list rest pattern, got {:?}", pattern);
        };
        assert_eq!(head.len(), 1);
        // Anonymous rest ".." has no binding (None), distinct from explicit ".._ "
        assert!(rest.is_none());
    }

    // === As Pattern ===

    #[test]
    fn test_as_pattern() {
        let source = r#"
            fn main() {
                case x {
                    Some(y) as opt -> opt
                    _ -> None
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::As {
            pattern: inner,
            name,
            ..
        } = pattern
        else {
            panic!("Expected as pattern, got {:?}", pattern);
        };
        assert_eq!(name.to_string(), "opt");
        let PatternKind::Variant { ctor, .. } = inner.kind.as_ref() else {
            panic!("Expected variant pattern inside as");
        };
        assert_eq!(ctor.name().to_string(), "Some");
    }

    #[test]
    fn test_as_pattern_with_wildcard() {
        let source = r#"
            fn main() {
                case x {
                    _ as all -> all
                }
            }
        "#;
        let pattern = get_case_pattern(source, 0);
        let PatternKind::As {
            pattern: inner,
            name,
            ..
        } = pattern
        else {
            panic!("Expected as pattern, got {:?}", pattern);
        };
        assert_eq!(name.to_string(), "all");
        assert!(matches!(inner.kind.as_ref(), PatternKind::Wildcard));
    }

    #[test]
    fn test_parse_string_literal_simple() {
        assert_eq!(parse_string_literal("\"hello\"").unwrap(), "hello");
        assert_eq!(parse_string_literal("\"\"").unwrap(), "");
    }

    #[test]
    fn test_parse_string_literal_no_quotes() {
        // Edge case: if quotes are missing, return as-is
        assert_eq!(parse_string_literal("hello").unwrap(), "hello");
    }

    // === String Escape Sequence Tests ===

    #[test]
    fn test_string_escape_newline() {
        let source = r#"fn test() { case x { "hello\nworld" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "hello\nworld");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_tab() {
        let source = r#"fn test() { case x { "a\tb" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "a\tb");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_backslash() {
        let source = r#"fn test() { case x { "a\\b" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "a\\b");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_unicode() {
        let source = r#"fn test() { case x { "\u{1F600}" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "😀");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_hex() {
        let source = r#"fn test() { case x { "\x41\x42" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "AB");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_carriage_return() {
        let source = r#"fn test() { case x { "a\rb" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "a\rb");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_null() {
        let source = r#"fn test() { case x { "a\0b" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "a\0b");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_string_escape_quote() {
        let source = r#"fn test() { case x { "say \"hello\"" -> 1, _ -> 0 } }"#;
        let pattern = get_case_pattern(source, 0);

        match pattern {
            PatternKind::Literal(LiteralPattern::String(s)) => {
                assert_eq!(s, "say \"hello\"");
            }
            _ => panic!("Expected string literal pattern, got {:?}", pattern),
        }
    }
}
