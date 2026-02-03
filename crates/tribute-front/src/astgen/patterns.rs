//! CST to AST lowering for patterns.

use tree_sitter::Node;
use trunk_ir::Symbol;

use crate::ast::{FieldPattern, FloatBits, LiteralPattern, Pattern, PatternKind, UnresolvedName};

use super::context::AstLoweringCtx;
use super::helpers::is_comment;

/// Lower a CST pattern node to an AST Pattern.
pub fn lower_pattern(ctx: &mut AstLoweringCtx, node: Node) -> Pattern<UnresolvedName> {
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
        "nat_literal" => {
            let text = ctx.node_text(&node);
            match parse_nat_literal(&text) {
                Some(value) => PatternKind::Literal(LiteralPattern::Nat(value)),
                None => PatternKind::Error,
            }
        }
        "int_literal" => {
            let text = ctx.node_text(&node);
            match parse_int_literal(&text) {
                Some(value) => PatternKind::Literal(LiteralPattern::Int(value)),
                None => PatternKind::Error,
            }
        }
        "float_literal" => {
            let text = ctx.node_text(&node);
            match text.parse::<f64>() {
                Ok(value) => PatternKind::Literal(LiteralPattern::Float(FloatBits::new(value))),
                Err(_) => PatternKind::Error,
            }
        }
        // grammar.js uses "string" for string patterns, not "string_literal"
        "string_literal" | "string" => {
            let text = ctx.node_text_owned(&node);
            let content = parse_string_literal(&text);
            PatternKind::Literal(LiteralPattern::String(content))
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
            let ctor = UnresolvedName::simple(name, ctor_id);
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

fn lower_constructor_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
    // grammar.js: field("name", $.type_identifier)
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
    let ctor = UnresolvedName::simple(name, ctor_id);

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
fn lower_constructor_fields(ctx: &mut AstLoweringCtx, node: Node) -> Vec<Pattern<UnresolvedName>> {
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

fn lower_record_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
    let type_node = node.child_by_field_name("type");
    let fields_node = node.child_by_field_name("fields");

    let type_name = if let Some(n) = type_node {
        let name = ctx.node_symbol(&n);
        let id = ctx.fresh_id_with_span(&n);
        Some(UnresolvedName::simple(name, id))
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
    ctx: &mut AstLoweringCtx,
    node: Node,
) -> Option<FieldPattern<UnresolvedName>> {
    let name_node = node.child_by_field_name("name")?;
    let pattern_node = node.child_by_field_name("pattern");

    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);
    let pattern = pattern_node.map(|n| lower_pattern(ctx, n));

    Some(FieldPattern { id, name, pattern })
}

fn lower_tuple_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
    // tuple_pattern has field "elements" which is a pattern_list
    let elements_node = node.child_by_field_name("elements");
    let elements = elements_node
        .map(|n| lower_pattern_list(ctx, n))
        .unwrap_or_default();
    PatternKind::Tuple(elements)
}

fn lower_list_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
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

fn lower_as_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
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

fn lower_pattern_list(ctx: &mut AstLoweringCtx, node: Node) -> Vec<Pattern<UnresolvedName>> {
    let mut patterns = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            patterns.push(lower_pattern(ctx, child));
        }
    }

    patterns
}

// === Literal parsing helpers ===

fn parse_nat_literal(text: &str) -> Option<u64> {
    if text.starts_with("0b") || text.starts_with("0B") {
        u64::from_str_radix(&text[2..].replace('_', ""), 2).ok()
    } else if text.starts_with("0o") || text.starts_with("0O") {
        u64::from_str_radix(&text[2..].replace('_', ""), 8).ok()
    } else if text.starts_with("0x") || text.starts_with("0X") {
        u64::from_str_radix(&text[2..].replace('_', ""), 16).ok()
    } else {
        text.replace('_', "").parse().ok()
    }
}

fn parse_int_literal(text: &str) -> Option<i64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    let (is_negative, rest) = if let Some(rest) = text.strip_prefix('+') {
        (false, rest)
    } else if let Some(rest) = text.strip_prefix('-') {
        (true, rest)
    } else {
        (false, text)
    };

    let value = parse_nat_literal(rest)?;

    if is_negative {
        // For negative numbers, the maximum magnitude is |i64::MIN| = 2^63
        // i64::MAX = 2^63 - 1, so i64::MIN's magnitude is (i64::MAX as u64) + 1
        if value <= i64::MAX as u64 {
            // Safe to convert and negate
            Some(-(value as i64))
        } else if value == (i64::MAX as u64) + 1 {
            // Special case: -9223372036854775808 (i64::MIN)
            Some(i64::MIN)
        } else {
            // Overflow: magnitude too large for i64
            None
        }
    } else {
        // For positive numbers, must fit in i64::MAX
        i64::try_from(value).ok()
    }
}

/// Process escape sequences in a string.
fn unescape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('0') => result.push('\0'),
                Some('x') => {
                    // Hex escape: \xHH
                    let hex: String = chars.by_ref().take(2).collect();
                    if let Ok(code) = u8::from_str_radix(&hex, 16) {
                        result.push(code as char);
                    }
                }
                Some('u') => {
                    // Unicode escape: \uHHHH or \u{H+}
                    if chars.peek() == Some(&'{') {
                        chars.next(); // consume '{'
                        let hex: String = chars.by_ref().take_while(|&c| c != '}').collect();
                        if let Some(ch) =
                            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                        {
                            result.push(ch);
                        }
                    } else {
                        // \uHHHH form
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Some(ch) =
                            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                        {
                            result.push(ch);
                        }
                    }
                }
                Some(c) => result.push(c),
                None => {}
            }
        } else {
            result.push(c);
        }
    }

    result
}

fn parse_string_literal(text: &str) -> String {
    let text = text.trim();

    // Handle raw strings (no escape processing)
    if text.starts_with("r\"") || text.starts_with("r#") {
        // Raw string: strip the r and delimiters
        let content = if text.starts_with("r#") {
            // Find matching number of # at start and end
            let hashes = text[1..].chars().take_while(|&c| c == '#').count();
            let start = 2 + hashes; // r + # + "
            let end = text.len() - hashes - 1; // " + #
            if start < end { &text[start..end] } else { "" }
        } else {
            // r"..."
            &text[2..text.len() - 1]
        };
        return content.to_string();
    }

    // Regular string: process escapes
    let content = if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        &text[1..text.len() - 1]
    } else {
        text
    };

    unescape_string(content)
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
        assert_eq!(ctor.name.to_string(), "None");
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
        assert_eq!(ctor.name.to_string(), "Some");
        assert_eq!(fields.len(), 1);
        let PatternKind::Bind { name, .. } = fields[0].kind.as_ref() else {
            panic!("Expected bind pattern in variant");
        };
        assert_eq!(name.to_string(), "y");
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
        assert_eq!(ctor.name.to_string(), "Pair");
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
        assert_eq!(ctor.name.to_string(), "Point");
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
        assert_eq!(ctor.name.to_string(), "Some");
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

    // === Literal Parsing Helpers ===

    #[test]
    fn test_parse_nat_literal_decimal() {
        assert_eq!(parse_nat_literal("42"), Some(42));
        assert_eq!(parse_nat_literal("0"), Some(0));
        assert_eq!(parse_nat_literal("123_456"), Some(123456));
    }

    #[test]
    fn test_parse_nat_literal_binary() {
        assert_eq!(parse_nat_literal("0b1010"), Some(10));
        assert_eq!(parse_nat_literal("0B1111"), Some(15));
        assert_eq!(parse_nat_literal("0b1010_1010"), Some(170));
    }

    #[test]
    fn test_parse_nat_literal_octal() {
        assert_eq!(parse_nat_literal("0o777"), Some(511));
        assert_eq!(parse_nat_literal("0O10"), Some(8));
    }

    #[test]
    fn test_parse_nat_literal_hex() {
        assert_eq!(parse_nat_literal("0xFF"), Some(255));
        assert_eq!(parse_nat_literal("0xDEAD_BEEF"), Some(0xDEADBEEF));
    }

    #[test]
    fn test_parse_int_literal_positive() {
        assert_eq!(parse_int_literal("42"), Some(42));
        assert_eq!(parse_int_literal("+42"), Some(42));
    }

    #[test]
    fn test_parse_int_literal_negative() {
        assert_eq!(parse_int_literal("-42"), Some(-42));
        assert_eq!(parse_int_literal("-0xFF"), Some(-255));
    }

    #[test]
    fn test_parse_int_literal_bounds() {
        // i64::MAX
        assert_eq!(parse_int_literal("9223372036854775807"), Some(i64::MAX));
        // i64::MIN (special case: magnitude exceeds i64::MAX)
        assert_eq!(parse_int_literal("-9223372036854775808"), Some(i64::MIN));
        // Overflow: positive value exceeds i64::MAX
        assert_eq!(parse_int_literal("9223372036854775808"), None);
        // Overflow: negative magnitude exceeds |i64::MIN|
        assert_eq!(parse_int_literal("-9223372036854775809"), None);
    }

    #[test]
    fn test_parse_string_literal_simple() {
        assert_eq!(parse_string_literal("\"hello\""), "hello");
        assert_eq!(parse_string_literal("\"\""), "");
    }

    #[test]
    fn test_parse_string_literal_no_quotes() {
        // Edge case: if quotes are missing, return as-is
        assert_eq!(parse_string_literal("hello"), "hello");
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
