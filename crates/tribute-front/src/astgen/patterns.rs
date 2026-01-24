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
            PatternKind::Bind { name }
        }

        // === Literal patterns ===
        "nat_literal" => {
            let text = ctx.node_text(&node);
            let value = parse_nat_literal(&text).unwrap_or(0) as i64;
            PatternKind::Literal(LiteralPattern::Int(value))
        }
        "int_literal" => {
            let text = ctx.node_text(&node);
            let value = parse_int_literal(&text).unwrap_or(0);
            PatternKind::Literal(LiteralPattern::Int(value))
        }
        "float_literal" => {
            let text = ctx.node_text(&node);
            let value: f64 = text.parse().unwrap_or(0.0);
            PatternKind::Literal(LiteralPattern::Float(FloatBits::new(value)))
        }
        "string_literal" => {
            let text = ctx.node_text_owned(&node);
            let content = parse_string_literal(&text);
            PatternKind::Literal(LiteralPattern::String(content))
        }
        "bool_literal" => {
            let text = ctx.node_text(&node);
            PatternKind::Literal(LiteralPattern::Bool(&*text == "true"))
        }
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

        // === Or pattern ===
        "or_pattern" => lower_or_pattern(ctx, node),

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

fn lower_constructor_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
    let name_node = node.child_by_field_name("constructor");
    let args_node = node.child_by_field_name("arguments");

    let Some(name_node) = name_node else {
        return PatternKind::Error;
    };

    let name = ctx.node_symbol(&name_node);
    let ctor_id = ctx.fresh_id_with_span(&name_node);
    let ctor = UnresolvedName::new(name, ctor_id);

    let fields = args_node
        .map(|n| lower_pattern_list(ctx, n))
        .unwrap_or_default();

    PatternKind::Variant { ctor, fields }
}

fn lower_record_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
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
                    let name = ctx.node_symbol(&child);
                    fields.push(FieldPattern {
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

    let name = ctx.node_symbol(&name_node);
    let pattern = pattern_node.map(|n| lower_pattern(ctx, n));

    Some(FieldPattern { name, pattern })
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
    let mut rest: Option<Symbol> = None;
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        match child.kind() {
            "rest_pattern" => {
                // [a, b, ..tail] or [a, b, ..]
                if let Some(binding) = child.child_by_field_name("binding") {
                    rest = Some(ctx.node_symbol(&binding));
                } else {
                    rest = Some(Symbol::new("_")); // Anonymous rest
                }
            }
            _ => {
                head.push(lower_pattern(ctx, child));
            }
        }
    }

    if rest.is_some() {
        PatternKind::ListRest { head, rest }
    } else {
        PatternKind::List(head)
    }
}

fn lower_as_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
    let pattern_node = node.child_by_field_name("pattern");
    let name_node = node.child_by_field_name("name");

    let (Some(pattern_node), Some(name_node)) = (pattern_node, name_node) else {
        return PatternKind::Error;
    };

    let pattern = lower_pattern(ctx, pattern_node);
    let name = ctx.node_symbol(&name_node);

    PatternKind::As { pattern, name }
}

fn lower_or_pattern(ctx: &mut AstLoweringCtx, node: Node) -> PatternKind<UnresolvedName> {
    let mut alternatives = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            alternatives.push(lower_pattern(ctx, child));
        }
    }

    PatternKind::Or(alternatives)
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

    let (sign, rest) = if let Some(rest) = text.strip_prefix('+') {
        (1i64, rest)
    } else if let Some(rest) = text.strip_prefix('-') {
        (-1i64, rest)
    } else {
        (1i64, text)
    };

    let value = parse_nat_literal(rest)? as i64;
    Some(sign * value)
}

fn parse_string_literal(text: &str) -> String {
    let text = text.trim();

    let content = if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        &text[1..text.len() - 1]
    } else {
        text
    };

    content.to_string()
}
