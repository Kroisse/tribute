//! CST to AST lowering for expressions.

use tree_sitter::Node;
use tribute_ir::ModulePathExt;
use trunk_ir::Symbol;

use crate::ast::{
    Arm, BinOpKind, Expr, ExprKind, FloatBits, HandlerArm, HandlerKind, Param, Stmt, UnresolvedName,
};

use super::context::AstLoweringCtx;
use super::helpers::is_comment;
use super::patterns::lower_pattern;
use crate::tirgen::parse_rune_literal;

/// Lower a CST expression node to an AST Expr.
pub fn lower_expr(ctx: &mut AstLoweringCtx, node: Node) -> Expr<UnresolvedName> {
    let id = ctx.fresh_id_with_span(&node);

    let kind = match node.kind() {
        // === Literals ===
        "nat_literal" => {
            let text = ctx.node_text(&node);
            let value = parse_nat_literal(&text).unwrap_or(0);
            ExprKind::NatLit(value)
        }
        "int_literal" => {
            let text = ctx.node_text(&node);
            let value = parse_int_literal(&text).unwrap_or(0);
            ExprKind::IntLit(value)
        }
        "float_literal" => {
            let text = ctx.node_text(&node);
            let value: f64 = text.parse().unwrap_or(0.0);
            ExprKind::FloatLit(FloatBits::new(value))
        }
        // String literals: "...", s"...", raw strings, multiline strings
        "string" | "raw_string" | "raw_interpolated_string" | "multiline_string" => {
            let text = ctx.node_text_owned(&node);
            let content = parse_string_literal(&text);
            ExprKind::StringLit(content)
        }
        // Bytes literals: b"...", raw bytes, multiline bytes
        "bytes_string" | "raw_bytes" | "raw_interpolated_bytes" | "multiline_bytes" => {
            let text = ctx.node_text_owned(&node);
            let content = parse_bytes_literal(&text);
            ExprKind::BytesLit(content)
        }
        // Boolean literals: True, False (capitalized keywords)
        "keyword_true" => ExprKind::BoolLit(true),
        "keyword_false" => ExprKind::BoolLit(false),
        // Unit/Nil literal: Nil or ()
        "keyword_nil" => ExprKind::Nil,

        // Rune (character) literal: ?a, ?\n, etc.
        "rune" => {
            let text = ctx.node_text(&node);
            let c = parse_rune_literal(&text).unwrap_or('\0');
            ExprKind::RuneLit(c)
        }

        // Operator as function: (+), (<>), (Int::+), etc.
        "operator_fn" => match node.child_by_field_name("operator") {
            Some(op_node) => match op_node.kind() {
                "qualified_operator" => {
                    // Int::+ → "Int::+"
                    if let (Some(type_node), Some(operator_node)) = (
                        op_node.child_by_field_name("type"),
                        op_node.child_by_field_name("operator"),
                    ) {
                        let type_name = ctx.node_text(&type_node);
                        let operator = ctx.node_text(&operator_node);
                        let text = format!("{}::{}", type_name, operator);
                        let name = Symbol::from_dynamic(&text);
                        let name_id = ctx.fresh_id_with_span(&op_node);
                        ExprKind::Var(UnresolvedName::new(name, name_id))
                    } else {
                        ExprKind::Error
                    }
                }
                _ => {
                    // Simple operator: +, -, <>, ==, etc.
                    let name = ctx.node_symbol(&op_node);
                    let name_id = ctx.fresh_id_with_span(&op_node);
                    ExprKind::Var(UnresolvedName::new(name, name_id))
                }
            },
            None => ExprKind::Error,
        },

        // === Identifiers ===
        "identifier" => {
            let name = ctx.node_symbol(&node);
            let name_id = ctx.fresh_id_with_span(&node);
            ExprKind::Var(UnresolvedName::new(name, name_id))
        }

        // === Binary expressions ===
        "binary_expression" => lower_binary_expr(ctx, node),

        // === Call expressions ===
        "call_expression" => lower_call_expr(ctx, node),

        // === Method call ===
        "method_call_expression" => lower_method_call(ctx, node),

        // === Constructor ===
        "constructor_expression" => lower_constructor_expr(ctx, node),

        // === Record construction ===
        "record_expression" => lower_record_expr(ctx, node),

        // === Field access ===
        "field_access_expression" => lower_field_access(ctx, node),

        // === Block ===
        "block" => lower_block(ctx, node),

        // === Case expression ===
        "case_expression" => lower_case_expr(ctx, node),

        // === Lambda ===
        "lambda_expression" => lower_lambda_expr(ctx, node),

        // === Tuple ===
        "tuple_expression" => lower_tuple_expr(ctx, node),

        // === List ===
        "list_expression" => lower_list_expr(ctx, node),

        // === Handle expression ===
        "handle_expression" => lower_handle_expr(ctx, node),

        // === Parenthesized ===
        "parenthesized_expression" => {
            // Just unwrap the inner expression
            if let Some(inner) = node.named_child(0) {
                return lower_expr(ctx, inner);
            }
            ExprKind::Error
        }

        // === Path expression ===
        "path_expression" | "qualified_identifier" => {
            let name = ctx.node_symbol(&node);
            let name_id = ctx.fresh_id_with_span(&node);
            ExprKind::Var(UnresolvedName::new(name, name_id))
        }

        _ => {
            // Try to find a meaningful child
            if let Some(child) = node.named_child(0) {
                return lower_expr(ctx, child);
            }
            ExprKind::Error
        }
    };

    Expr::new(id, kind)
}

fn lower_binary_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let lhs_node = node.child_by_field_name("left");
    let rhs_node = node.child_by_field_name("right");
    let op_node = node.child_by_field_name("operator");

    let (Some(lhs_node), Some(rhs_node), Some(op_node)) = (lhs_node, rhs_node, op_node) else {
        return ExprKind::Error;
    };

    let lhs = lower_expr(ctx, lhs_node);
    let rhs = lower_expr(ctx, rhs_node);
    let op_text = ctx.node_text(&op_node);

    let op = match &*op_text {
        "+" => BinOpKind::Add,
        "-" => BinOpKind::Sub,
        "*" => BinOpKind::Mul,
        "/" => BinOpKind::Div,
        "%" => BinOpKind::Mod,
        "==" => BinOpKind::Eq,
        "!=" => BinOpKind::Ne,
        "<" => BinOpKind::Lt,
        "<=" => BinOpKind::Le,
        ">" => BinOpKind::Gt,
        ">=" => BinOpKind::Ge,
        "&&" => BinOpKind::And,
        "||" => BinOpKind::Or,
        "<>" => BinOpKind::Concat,
        _ => return ExprKind::Error,
    };

    ExprKind::BinOp { op, lhs, rhs }
}

fn lower_call_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let callee_node = node.child_by_field_name("function");
    let args_node = node.child_by_field_name("arguments");

    let Some(callee_node) = callee_node else {
        return ExprKind::Error;
    };

    let callee = lower_expr(ctx, callee_node);
    let args = args_node
        .map(|args| lower_argument_list(ctx, args))
        .unwrap_or_default();

    ExprKind::Call { callee, args }
}

fn lower_method_call(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let receiver_node = node.child_by_field_name("receiver");
    let method_node = node.child_by_field_name("method");
    let args_node = node.child_by_field_name("arguments");

    let (Some(receiver_node), Some(method_node)) = (receiver_node, method_node) else {
        return ExprKind::Error;
    };

    let receiver = lower_expr(ctx, receiver_node);
    let method = ctx.node_symbol(&method_node);
    let args = args_node
        .map(|args| lower_argument_list(ctx, args))
        .unwrap_or_default();

    ExprKind::MethodCall {
        receiver,
        method,
        args,
    }
}

fn lower_constructor_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let name_node = node.child_by_field_name("constructor");
    let args_node = node.child_by_field_name("arguments");

    let Some(name_node) = name_node else {
        return ExprKind::Error;
    };

    let name = ctx.node_symbol(&name_node);
    let name_id = ctx.fresh_id_with_span(&name_node);
    let ctor = UnresolvedName::new(name, name_id);

    let args = args_node
        .map(|args| lower_argument_list(ctx, args))
        .unwrap_or_default();

    ExprKind::Cons { ctor, args }
}

fn lower_record_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let type_node = node.child_by_field_name("type");
    let fields_node = node.child_by_field_name("fields");

    let Some(type_node) = type_node else {
        return ExprKind::Error;
    };

    let type_name_sym = ctx.node_symbol(&type_node);
    let type_name_id = ctx.fresh_id_with_span(&type_node);
    let type_name = UnresolvedName::new(type_name_sym, type_name_id);

    let mut fields = Vec::new();
    let mut spread = None;

    if let Some(fields_node) = fields_node {
        let mut cursor = fields_node.walk();
        for child in fields_node.named_children(&mut cursor) {
            match child.kind() {
                "field_initializer" => {
                    if let Some((name, value)) = lower_field_initializer(ctx, child) {
                        fields.push((name, value));
                    }
                }
                "spread_expression" => {
                    if let Some(inner) = child.named_child(0) {
                        spread = Some(lower_expr(ctx, inner));
                    }
                }
                _ => {}
            }
        }
    }

    ExprKind::Record {
        type_name,
        fields,
        spread,
    }
}

fn lower_field_initializer(
    ctx: &mut AstLoweringCtx,
    node: Node,
) -> Option<(Symbol, Expr<UnresolvedName>)> {
    let name_node = node.child_by_field_name("name")?;
    let value_node = node.child_by_field_name("value")?;

    let name = ctx.node_symbol(&name_node);
    let value = lower_expr(ctx, value_node);

    Some((name, value))
}

fn lower_field_access(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let expr_node = node.child_by_field_name("value");
    let field_node = node.child_by_field_name("field");

    let (Some(expr_node), Some(field_node)) = (expr_node, field_node) else {
        return ExprKind::Error;
    };

    let expr = lower_expr(ctx, expr_node);
    let field = ctx.node_symbol(&field_node);

    ExprKind::FieldAccess { expr, field }
}

/// Returns true if the node is a statement-like construct that should be
/// executed for side effects rather than used as a value.
/// Note: `expression_statement` is just a wrapper for expressions in the grammar,
/// so only `let_statement` is a true statement that yields Nil.
fn is_statement_node(node: &Node) -> bool {
    node.kind() == "let_statement"
}

fn lower_block(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let mut stmts = Vec::new();
    let mut cursor = node.walk();

    // Collect non-comment, non-error children
    let children: Vec<_> = node
        .named_children(&mut cursor)
        .filter(|c| !is_comment(c.kind()) && c.kind() != "ERROR")
        .collect();

    // Process all but the last child as statements
    for child in children.iter().take(children.len().saturating_sub(1)) {
        if let Some(stmt) = lower_block_item_as_stmt(ctx, child) {
            stmts.push(stmt);
        }
    }

    // For the last child: if it's a statement node, execute it for side effects
    // and return Nil. Otherwise, use it as the block's value expression.
    let value = if let Some(last) = children.last() {
        if is_statement_node(last) {
            // This is a statement - execute for side effects, block returns Nil
            if let Some(stmt) = lower_block_item_as_stmt(ctx, last) {
                stmts.push(stmt);
            }
            let nil_id = ctx.fresh_id_with_span(last);
            Expr::new(nil_id, ExprKind::Nil)
        } else {
            // This is an expression - use as block value
            lower_block_item_as_expr(ctx, last)
        }
    } else {
        // Empty block returns Nil
        let nil_id = ctx.fresh_id_with_span(&node);
        Expr::new(nil_id, ExprKind::Nil)
    };

    ExprKind::Block { stmts, value }
}

fn lower_block_item_as_stmt(
    ctx: &mut AstLoweringCtx,
    child: &Node,
) -> Option<Stmt<UnresolvedName>> {
    match child.kind() {
        "let_statement" => lower_let_statement(ctx, *child),
        "expression_statement" | "statement" => {
            let mut inner_cursor = child.walk();
            let inner = child
                .named_children(&mut inner_cursor)
                .find(|n| !is_comment(n.kind()))?;

            if inner.kind() == "let_statement" {
                lower_let_statement(ctx, inner)
            } else {
                let expr = lower_expr(ctx, inner);
                let stmt_id = ctx.fresh_id_with_span(&inner);
                Some(Stmt::Expr { id: stmt_id, expr })
            }
        }
        _ => {
            let expr = lower_expr(ctx, *child);
            let stmt_id = ctx.fresh_id_with_span(child);
            Some(Stmt::Expr { id: stmt_id, expr })
        }
    }
}

fn lower_block_item_as_expr(ctx: &mut AstLoweringCtx, child: &Node) -> Expr<UnresolvedName> {
    match child.kind() {
        "let_statement" => {
            // A let as the last item - block value is Nil, but we need to add the let as a stmt
            // This shouldn't normally happen with well-formed code
            let nil_id = ctx.fresh_id_with_span(child);
            Expr::new(nil_id, ExprKind::Nil)
        }
        "expression_statement" | "statement" => {
            let mut inner_cursor = child.walk();
            if let Some(inner) = child
                .named_children(&mut inner_cursor)
                .find(|n| !is_comment(n.kind()))
            {
                if inner.kind() == "let_statement" {
                    let nil_id = ctx.fresh_id_with_span(child);
                    Expr::new(nil_id, ExprKind::Nil)
                } else {
                    lower_expr(ctx, inner)
                }
            } else {
                let nil_id = ctx.fresh_id_with_span(child);
                Expr::new(nil_id, ExprKind::Nil)
            }
        }
        _ => lower_expr(ctx, *child),
    }
}

fn lower_let_statement(ctx: &mut AstLoweringCtx, node: Node) -> Option<Stmt<UnresolvedName>> {
    let pattern_node = node.child_by_field_name("pattern")?;
    let value_node = node.child_by_field_name("value")?;

    let id = ctx.fresh_id_with_span(&node);
    let pattern = lower_pattern(ctx, pattern_node);
    let value = lower_expr(ctx, value_node);

    // TODO: type annotation
    let ty = None;

    Some(Stmt::Let {
        id,
        pattern,
        ty,
        value,
    })
}

fn lower_case_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let scrutinee_node = node.child_by_field_name("value");

    let Some(scrutinee_node) = scrutinee_node else {
        return ExprKind::Error;
    };

    let scrutinee = lower_expr(ctx, scrutinee_node);
    let mut arms = Vec::new();

    // case_arm children are direct children of case_expression
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "case_arm"
            && let Some(arm) = lower_case_arm(ctx, child)
        {
            arms.push(arm);
        }
    }

    ExprKind::Case { scrutinee, arms }
}

fn lower_case_arm(ctx: &mut AstLoweringCtx, node: Node) -> Option<Arm<UnresolvedName>> {
    // case_arm has pattern and body as positional named children, not fields
    // Pattern is the first named child, body is the second
    let mut cursor = node.walk();
    let mut pattern_node = None;
    let mut body_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if pattern_node.is_none() {
            pattern_node = Some(child);
        } else if body_node.is_none() {
            body_node = Some(child);
        }
    }

    let pattern_node = pattern_node?;
    let body_node = body_node?;

    let id = ctx.fresh_id_with_span(&node);
    let pattern = lower_pattern(ctx, pattern_node);
    let body = lower_expr(ctx, body_node);

    // TODO: guard
    let guard = None;

    Some(Arm {
        id,
        pattern,
        guard,
        body,
    })
}

fn lower_lambda_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let params_node = node.child_by_field_name("params");
    let body_node = node.child_by_field_name("body");

    let Some(body_node) = body_node else {
        return ExprKind::Error;
    };

    let params = params_node
        .map(|n| lower_param_list(ctx, n))
        .unwrap_or_default();
    let body = lower_expr(ctx, body_node);

    ExprKind::Lambda { params, body }
}

fn lower_param_list(ctx: &mut AstLoweringCtx, node: Node) -> Vec<Param> {
    let mut params = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if child.kind() == "parameter" || child.kind() == "identifier" {
            let id = ctx.fresh_id_with_span(&child);
            let name = ctx.node_symbol(&child);
            // TODO: type annotation
            params.push(Param {
                id,
                name,
                ty: None,
                local_id: None,
            });
        }
    }

    params
}

fn lower_tuple_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let mut elements = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            elements.push(lower_expr(ctx, child));
        }
    }

    ExprKind::Tuple(elements)
}

fn lower_list_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let mut elements = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            elements.push(lower_expr(ctx, child));
        }
    }

    ExprKind::List(elements)
}

fn lower_handle_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    // grammar.js: field("expr", $._expression) for the body
    let expr_node = node.child_by_field_name("expr");

    let Some(expr_node) = expr_node else {
        return ExprKind::Error;
    };

    let body = lower_expr(ctx, expr_node);
    let mut handlers = Vec::new();

    // Handler arms are direct children of handle_expression (no "handlers" field)
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "handler_arm"
            && let Some(handler) = lower_handler_arm(ctx, child)
        {
            handlers.push(handler);
        }
    }

    ExprKind::Handle { body, handlers }
}

fn lower_handler_arm(ctx: &mut AstLoweringCtx, node: Node) -> Option<HandlerArm<UnresolvedName>> {
    let pattern_node = node.child_by_field_name("pattern")?;
    // grammar.js: field("value", $._expression) for the handler body
    let value_node = node.child_by_field_name("value")?;

    let id = ctx.fresh_id_with_span(&node);
    let body = lower_expr(ctx, value_node);

    // Determine handler kind from pattern
    let kind = lower_handler_pattern(ctx, pattern_node)?;

    Some(HandlerArm { id, kind, body })
}

fn lower_handler_pattern(
    ctx: &mut AstLoweringCtx,
    node: Node,
) -> Option<HandlerKind<UnresolvedName>> {
    // grammar.js handler_pattern structure:
    // - { result }              -> field("result", $.identifier)
    // - { op(args) -> k }       -> field("operation", ...), field("args", ...), field("continuation", ...)

    // Check for result pattern: { result }
    if let Some(result_node) = node.child_by_field_name("result") {
        let binding = lower_pattern(ctx, result_node);
        return Some(HandlerKind::Result { binding });
    }

    // Check for effect/suspend pattern: { Path::op(args) -> k }
    if let Some(op_node) = node.child_by_field_name("operation") {
        let args_node = node.child_by_field_name("args");
        let cont_node = node.child_by_field_name("continuation");

        // Parse operation path using ModulePathExt (e.g., State::get or just get)
        let op_text = ctx.node_text(&op_node).to_string();
        let op_symbol = Symbol::from_dynamic(&op_text);
        let (ability, op) = if op_symbol.is_simple() {
            // Unqualified: just op name, ability will be inferred
            let ability_name = Symbol::from_dynamic("_");
            let ability_id = ctx.fresh_id_with_span(&op_node);
            let ability = UnresolvedName::new(ability_name, ability_id);
            (ability, op_symbol)
        } else {
            // Qualified path: State::get -> ability=State, op=get
            let ability_name = op_symbol.parent_path().unwrap();
            let ability_id = ctx.fresh_id_with_span(&op_node);
            let ability = UnresolvedName::new(ability_name, ability_id);
            let op = op_symbol.last_segment();
            (ability, op)
        };

        let params = args_node
            .map(|n| {
                let mut patterns = Vec::new();
                let mut cursor = n.walk();
                for child in n.named_children(&mut cursor) {
                    if !is_comment(child.kind()) {
                        patterns.push(lower_pattern(ctx, child));
                    }
                }
                patterns
            })
            .unwrap_or_default();

        let continuation = cont_node.map(|n| ctx.node_symbol(&n));

        return Some(HandlerKind::Effect {
            ability,
            op,
            params,
            continuation,
        });
    }

    // Fallback: treat node itself as result pattern
    let binding = lower_pattern(ctx, node);
    Some(HandlerKind::Result { binding })
}

fn lower_argument_list(ctx: &mut AstLoweringCtx, node: Node) -> Vec<Expr<UnresolvedName>> {
    let mut args = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            args.push(lower_expr(ctx, child));
        }
    }

    args
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

    let value_u = parse_nat_literal(rest)?;

    if is_negative {
        // Allow values up to i64::MAX + 1 for i64::MIN (-9223372036854775808)
        const MIN_ABS: u64 = (i64::MAX as u64) + 1;
        if value_u > MIN_ABS {
            return None; // Overflow
        }
        if value_u == MIN_ABS {
            Some(i64::MIN)
        } else {
            // Safe: value_u <= i64::MAX, so it fits in i64
            Some(-(value_u as i64))
        }
    } else {
        // Positive: must fit in i64
        i64::try_from(value_u).ok()
    }
}

fn parse_string_literal(text: &str) -> String {
    // Simple implementation: strip quotes and handle basic escapes
    let text = text.trim();

    // Handle different string prefixes with bounds checking
    let content = if text.starts_with("s\"") || text.starts_with("r\"") {
        text.get(2..text.len().saturating_sub(1)).unwrap_or("")
    } else if text.starts_with("rs\"") || text.starts_with("sr\"") {
        text.get(3..text.len().saturating_sub(1)).unwrap_or("")
    } else if text.starts_with('"') {
        text.get(1..text.len().saturating_sub(1)).unwrap_or("")
    } else {
        text
    };

    // TODO: proper escape handling
    content.to_string()
}

fn parse_bytes_literal(text: &str) -> Vec<u8> {
    // Strip quotes and handle basic escapes for byte string literals
    let text = text.trim();

    // Determine prefix and whether it's raw
    let (prefix_len, is_raw) = if text.starts_with("rb") || text.starts_with("br") {
        (2, true)
    } else if text.starts_with('b') {
        (1, false)
    } else {
        return text.as_bytes().to_vec();
    };

    let after_prefix = &text[prefix_len..];

    // Count consecutive '#' characters before the opening quote
    let hash_count = after_prefix.chars().take_while(|&c| c == '#').count();

    // For raw strings with hashes: rb#"..."# or br##"..."##
    // For regular raw strings: rb"..." or br"..."
    // For regular byte strings: b"..."
    if hash_count > 0 {
        // Raw byte string with hashes: b#"..."# or rb##"..."##
        let quote_start = prefix_len + hash_count;
        let expected_end_pattern_len = 1 + hash_count; // closing quote + hashes

        // Validate we have opening quote after hashes
        if text.get(quote_start..quote_start + 1) != Some("\"") {
            return Vec::new();
        }

        // Content starts after the opening quote
        let content_start = quote_start + 1;

        // Find content end: must have closing quote followed by same number of hashes
        let content_end = text.len().saturating_sub(expected_end_pattern_len);
        if content_end <= content_start {
            return Vec::new();
        }

        // Validate closing pattern: " followed by hash_count #'s
        let closing = text.get(content_end..);
        let expected_closing: String = std::iter::once('"')
            .chain(std::iter::repeat_n('#', hash_count))
            .collect();
        if closing != Some(&expected_closing) {
            return Vec::new();
        }

        text.get(content_start..content_end)
            .unwrap_or("")
            .as_bytes()
            .to_vec()
    } else if is_raw {
        // Raw byte string without hashes: rb"..." or br"..."
        text.get(prefix_len + 1..text.len().saturating_sub(1))
            .unwrap_or("")
            .as_bytes()
            .to_vec()
    } else {
        // Regular byte string: b"..."
        // TODO: proper escape handling for non-raw byte strings
        text.get(prefix_len + 1..text.len().saturating_sub(1))
            .unwrap_or("")
            .as_bytes()
            .to_vec()
    }
}
