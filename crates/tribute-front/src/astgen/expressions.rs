//! CST to AST lowering for expressions.

use tree_sitter::Node;
use trunk_ir::Symbol;

use crate::ast::{
    Arm, BinOpKind, Expr, ExprKind, FloatBits, HandlerArm, HandlerKind, Param, Stmt, UnaryOpKind,
    UnresolvedName,
};

use super::context::AstLoweringCtx;
use super::helpers::is_comment;
use super::patterns::lower_pattern;

/// Lower a CST expression node to an AST Expr.
pub fn lower_expr(ctx: &mut AstLoweringCtx, node: Node) -> Expr<UnresolvedName> {
    let id = ctx.fresh_id();

    let kind = match node.kind() {
        // === Literals ===
        "nat_literal" => {
            let text = ctx.node_text(&node);
            let value = parse_nat_literal(text).unwrap_or(0) as i64;
            ExprKind::IntLit(value)
        }
        "int_literal" => {
            let text = ctx.node_text(&node);
            let value = parse_int_literal(text).unwrap_or(0);
            ExprKind::IntLit(value)
        }
        "float_literal" => {
            let text = ctx.node_text(&node);
            let value: f64 = text.parse().unwrap_or(0.0);
            ExprKind::FloatLit(FloatBits::new(value))
        }
        "string_literal" | "interpolated_string" => {
            let text = ctx.node_text_owned(&node);
            let content = parse_string_literal(&text);
            ExprKind::StringLit(content)
        }
        "bool_literal" => {
            let text = ctx.node_text(&node);
            ExprKind::BoolLit(text == "true")
        }
        "unit_literal" => ExprKind::UnitLit,

        // === Identifiers ===
        "identifier" => {
            let name = ctx.node_symbol(&node);
            let name_id = ctx.fresh_id();
            ExprKind::Var(UnresolvedName::new(name, name_id))
        }

        // === Binary expressions ===
        "binary_expression" => lower_binary_expr(ctx, node),

        // === Unary expressions ===
        "unary_expression" => lower_unary_expr(ctx, node),

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

        // === If expression (in case it exists) ===
        "if_expression" => lower_if_expr(ctx, node),

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
            let name_id = ctx.fresh_id();
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

    let op = match op_text {
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

fn lower_unary_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let op_node = node.child_by_field_name("operator");
    let operand_node = node.child_by_field_name("operand");

    let (Some(op_node), Some(operand_node)) = (op_node, operand_node) else {
        return ExprKind::Error;
    };

    let expr = lower_expr(ctx, operand_node);
    let op_text = ctx.node_text(&op_node);

    let op = match op_text {
        "-" => UnaryOpKind::Neg,
        "!" => UnaryOpKind::Not,
        _ => return ExprKind::Error,
    };

    ExprKind::UnaryOp { op, expr }
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
    let name_id = ctx.fresh_id();
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
    let type_name_id = ctx.fresh_id();
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

fn lower_block(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let mut stmts = Vec::new();
    let mut cursor = node.walk();

    // Collect non-comment, non-error children
    let children: Vec<_> = node
        .named_children(&mut cursor)
        .filter(|c| !is_comment(c.kind()) && c.kind() != "ERROR")
        .collect();

    for (i, child) in children.iter().enumerate() {
        let is_last = i == children.len() - 1;

        match child.kind() {
            "let_statement" => {
                if let Some(stmt) = lower_let_statement(ctx, *child) {
                    stmts.push(stmt);
                }
            }
            "expression_statement" | "statement" => {
                // expression_statement contains the actual expression
                let mut inner_cursor = child.walk();
                let inner = child
                    .named_children(&mut inner_cursor)
                    .find(|n| !is_comment(n.kind()));

                if let Some(inner) = inner {
                    if inner.kind() == "let_statement" {
                        if let Some(stmt) = lower_let_statement(ctx, inner) {
                            stmts.push(stmt);
                        }
                    } else {
                        let expr = lower_expr(ctx, inner);
                        let stmt_id = ctx.fresh_id();
                        if is_last {
                            stmts.push(Stmt::Return { id: stmt_id, expr });
                        } else {
                            stmts.push(Stmt::Expr { id: stmt_id, expr });
                        }
                    }
                }
            }
            _ => {
                // Try to lower as expression directly
                let expr = lower_expr(ctx, *child);
                let stmt_id = ctx.fresh_id();
                if is_last {
                    stmts.push(Stmt::Return { id: stmt_id, expr });
                } else {
                    stmts.push(Stmt::Expr { id: stmt_id, expr });
                }
            }
        }
    }

    ExprKind::Block(stmts)
}

fn lower_let_statement(ctx: &mut AstLoweringCtx, node: Node) -> Option<Stmt<UnresolvedName>> {
    let pattern_node = node.child_by_field_name("pattern")?;
    let value_node = node.child_by_field_name("value")?;

    let id = ctx.fresh_id();
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

fn lower_if_expr(ctx: &mut AstLoweringCtx, node: Node) -> ExprKind<UnresolvedName> {
    let cond_node = node.child_by_field_name("condition");
    let then_node = node.child_by_field_name("consequence");
    let else_node = node.child_by_field_name("alternative");

    let (Some(cond_node), Some(then_node)) = (cond_node, then_node) else {
        return ExprKind::Error;
    };

    let cond = lower_expr(ctx, cond_node);
    let then_branch = lower_expr(ctx, then_node);
    let else_branch = else_node.map(|n| lower_expr(ctx, n));

    ExprKind::If {
        cond,
        then_branch,
        else_branch,
    }
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

    let id = ctx.fresh_id();
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
            let id = ctx.fresh_id();
            let name = ctx.node_symbol(&child);
            // TODO: type annotation
            params.push(Param { id, name, ty: None });
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
    let body_node = node.child_by_field_name("body");
    let handlers_node = node.child_by_field_name("handlers");

    let Some(body_node) = body_node else {
        return ExprKind::Error;
    };

    let body = lower_expr(ctx, body_node);
    let mut handlers = Vec::new();

    if let Some(handlers_node) = handlers_node {
        let mut cursor = handlers_node.walk();
        for child in handlers_node.named_children(&mut cursor) {
            if child.kind() == "handler_arm"
                && let Some(handler) = lower_handler_arm(ctx, child)
            {
                handlers.push(handler);
            }
        }
    }

    ExprKind::Handle { body, handlers }
}

fn lower_handler_arm(ctx: &mut AstLoweringCtx, node: Node) -> Option<HandlerArm<UnresolvedName>> {
    let pattern_node = node.child_by_field_name("pattern")?;
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id();
    let body = lower_expr(ctx, body_node);

    // Determine handler kind from pattern
    let kind = lower_handler_pattern(ctx, pattern_node)?;

    Some(HandlerArm { id, kind, body })
}

fn lower_handler_pattern(
    ctx: &mut AstLoweringCtx,
    node: Node,
) -> Option<HandlerKind<UnresolvedName>> {
    match node.kind() {
        "handler_done_pattern" => {
            // { result }
            let binding_node = node.child_by_field_name("binding")?;
            let binding = lower_pattern(ctx, binding_node);
            Some(HandlerKind::Result { binding })
        }
        "handler_suspend_pattern" => {
            // { Ability.op(args) -> k }
            let ability_node = node.child_by_field_name("ability")?;
            let op_node = node.child_by_field_name("operation")?;
            let args_node = node.child_by_field_name("arguments");
            let cont_node = node.child_by_field_name("continuation");

            let ability_name = ctx.node_symbol(&ability_node);
            let ability_id = ctx.fresh_id();
            let ability = UnresolvedName::new(ability_name, ability_id);

            let op = ctx.node_symbol(&op_node);

            let params = args_node
                .map(|n| {
                    let mut patterns = Vec::new();
                    let mut cursor = n.walk();
                    for child in n.named_children(&mut cursor) {
                        patterns.push(lower_pattern(ctx, child));
                    }
                    patterns
                })
                .unwrap_or_default();

            let continuation = cont_node.map(|n| ctx.node_symbol(&n));

            Some(HandlerKind::Effect {
                ability,
                op,
                params,
                continuation,
            })
        }
        _ => {
            // Default: treat as result pattern
            let binding = lower_pattern(ctx, node);
            Some(HandlerKind::Result { binding })
        }
    }
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
    // Simple implementation: strip quotes and handle basic escapes
    let text = text.trim();

    // Handle different string prefixes
    let content = if text.starts_with("s\"") || text.starts_with("r\"") {
        &text[2..text.len() - 1]
    } else if text.starts_with("rs\"") || text.starts_with("sr\"") {
        &text[3..text.len() - 1]
    } else if text.starts_with('"') {
        &text[1..text.len() - 1]
    } else {
        text
    };

    // TODO: proper escape handling
    content.to_string()
}
