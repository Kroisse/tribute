//! CST to AST lowering for expressions.

use tree_sitter::Node;
use tribute_ir::ModulePathExt;
use trunk_ir::Symbol;

use crate::ast::{
    Arm, BinOpKind, Expr, ExprKind, FloatBits, HandlerArm, HandlerKind, Param, Pattern, Stmt,
    UnresolvedName,
};

use super::context::AstLoweringCtx;
use super::helpers::{
    EscapeError, decode_unicode_escape, is_comment, process_escape_sequences, report_in_node,
};
use super::numeric::{NumericValue, parse_numeric_literal};
use super::patterns::lower_pattern;

/// Lower a CST expression node to an AST Expr.
pub fn lower_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Expr<UnresolvedName> {
    let id = ctx.fresh_id_with_span(&node);

    let kind = match node.kind() {
        // === Literals ===
        "number_literal" => {
            let text = ctx.node_text_owned(&node);
            match parse_numeric_literal(&text) {
                Ok(NumericValue::Nat(value)) => ExprKind::NatLit(value),
                Ok(NumericValue::Int(value)) => ExprKind::IntLit(value),
                Ok(NumericValue::Float(value)) => ExprKind::FloatLit(FloatBits::new(value)),
                Err(error) => {
                    report_in_node(ctx, &node, error.range.clone(), error.to_string());
                    ExprKind::Error
                }
            }
        }
        // String literals: "...", s"...", raw strings, multiline strings
        "string" | "raw_string" | "raw_interpolated_string" | "multiline_string" => {
            let text = ctx.node_text_owned(&node);
            match parse_string_literal(&text) {
                Ok(content) => ExprKind::StringLit(content),
                Err(error) => {
                    report_in_node(ctx, &node, error.range.clone(), error.to_string());
                    ExprKind::Error
                }
            }
        }
        // Bytes literals: b"...", raw bytes, multiline bytes
        "bytes_string" | "raw_bytes" | "raw_interpolated_bytes" | "multiline_bytes" => {
            let text = ctx.node_text_owned(&node);
            match parse_bytes_literal(&text) {
                Ok(content) => ExprKind::BytesLit(content),
                Err(error) => {
                    report_in_node(ctx, &node, error.range.clone(), error.to_string());
                    ExprKind::Error
                }
            }
        }
        // Boolean literals: True, False (capitalized keywords)
        "keyword_true" => ExprKind::BoolLit(true),
        "keyword_false" => ExprKind::BoolLit(false),
        // Unit/Nil literal: Nil or ()
        "keyword_nil" => ExprKind::Nil,

        // Rune (character) literal: ?a, ?\n, etc.
        "rune" => {
            let text = ctx.node_text_owned(&node);
            match parse_rune_literal(&text) {
                Ok(Some(c)) => ExprKind::RuneLit(c),
                Ok(None) => ExprKind::Error,
                Err(error) => {
                    report_in_node(ctx, &node, error.range.clone(), error.to_string());
                    ExprKind::Error
                }
            }
        }

        // Operator as function: (+), (<>), (Int::+), etc.
        "operator_fn" => match node.child_by_field_name("operator") {
            Some(op_node) => {
                // Qualified operators such as Int::+ are single tokens, too.
                let name = ctx.node_symbol(&op_node);
                let name_id = ctx.fresh_id_with_span(&op_node);
                ExprKind::Var(UnresolvedName::new(name, name_id))
            }
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
        "resume_expression" => lower_resume_expr(ctx, node),

        // === Parenthesized ===
        "parenthesized_expression" => {
            // Just unwrap the inner expression
            if let Some(inner) = node.named_child(0) {
                return lower_expr(ctx, inner);
            }
            ExprKind::Error
        }

        // === Value path ===
        "value_path" => {
            let sym = ctx.node_symbol(&node);
            let name_id = ctx.fresh_id_with_span(&node);
            ExprKind::Var(UnresolvedName::new(sym, name_id))
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

fn lower_binary_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let lhs_node = node.child_by_field_name("left");
    let rhs_node = node.child_by_field_name("right");
    let op_node = node.child_by_field_name("operator");

    let (Some(lhs_node), Some(rhs_node), Some(op_node)) = (lhs_node, rhs_node, op_node) else {
        return ExprKind::Error;
    };

    let lhs = lower_expr(ctx, lhs_node);
    let rhs = lower_expr(ctx, rhs_node);
    let op_text = ctx.node_text(&op_node);

    // Desugar arithmetic, comparison, and concatenation operators to MethodCall.
    // TDNR resolves these to type-specific intrinsic functions (e.g., Int::(+), Float::(==)).
    // Only boolean operators (&&, ||) remain as BinOp for future short-circuit evaluation.
    let method_sym = match &*op_text {
        "+" => Some(Symbol::new("+")),
        "-" => Some(Symbol::new("-")),
        "*" => Some(Symbol::new("*")),
        "/" => Some(Symbol::new("/")),
        "%" => Some(Symbol::new("%")),
        "==" => Some(Symbol::new("==")),
        "!=" => Some(Symbol::new("!=")),
        "<" => Some(Symbol::new("<")),
        "<=" => Some(Symbol::new("<=")),
        ">" => Some(Symbol::new(">")),
        ">=" => Some(Symbol::new(">=")),
        "<>" => Some(Symbol::new("<>")),
        _ => None,
    };
    if let Some(method) = method_sym {
        return ExprKind::MethodCall {
            receiver: lhs,
            method,
            args: vec![rhs],
        };
    }

    let op = match &*op_text {
        "&&" => BinOpKind::And,
        "||" => BinOpKind::Or,
        _ => return ExprKind::Error,
    };

    ExprKind::BinOp { op, lhs, rhs }
}

fn lower_call_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let callee_node = node.child_by_field_name("function");
    // Note: tree-sitter grammar doesn't define "arguments" field for call_expression,
    // so we find argument_list by kind instead
    let args_node = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "argument_list");

    let Some(callee_node) = callee_node else {
        return ExprKind::Error;
    };

    let callee = lower_expr(ctx, callee_node);
    let args = args_node
        .map(|args| lower_argument_list(ctx, args))
        .unwrap_or_default();

    ExprKind::Call { callee, args }
}

fn lower_method_call(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let receiver_node = node.child_by_field_name("receiver");
    let method_node = node.child_by_field_name("method");
    // Note: tree-sitter grammar doesn't define an "arguments" field for
    // method_call_expression, so we find argument_list by kind instead
    // (same approach as lower_call_expr).
    let args_node = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "argument_list");

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

fn lower_constructor_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let name_node = node.child_by_field_name("constructor");

    // argument_list doesn't have a field name in the grammar, so we find it by kind
    let args_node = node
        .named_children(&mut node.walk())
        .find(|child| child.kind() == "argument_list");

    let Some(name_node) = name_node else {
        return ExprKind::Error;
    };

    let sym = ctx.node_symbol(&name_node);
    let name_id = ctx.fresh_id_with_span(&name_node);
    let ctor = UnresolvedName::new(sym, name_id);

    let args = args_node
        .map(|args| lower_argument_list(ctx, args))
        .unwrap_or_default();

    ExprKind::Cons { ctor, args }
}

fn lower_record_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
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
                "record_field" | "field_initializer" => {
                    // Check if this record_field is a spread (..expr) variant
                    let has_spread = child
                        .named_children(&mut child.walk())
                        .any(|c| c.kind() == "spread");
                    if has_spread {
                        if let Some(value_node) = child.child_by_field_name("value") {
                            spread = Some(lower_expr(ctx, value_node));
                        }
                    } else if let Some((name, value)) = lower_field_initializer(ctx, child) {
                        fields.push((name, value));
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
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<(Symbol, Expr<UnresolvedName>)> {
    let name_node = node.child_by_field_name("name")?;
    let value_node = node.child_by_field_name("value")?;

    let name = ctx.node_symbol(&name_node);
    let value = lower_expr(ctx, value_node);

    Some((name, value))
}

fn lower_field_access(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let expr_node = node.child_by_field_name("value");
    let field_node = node.child_by_field_name("field");

    let (Some(expr_node), Some(field_node)) = (expr_node, field_node) else {
        return ExprKind::Error;
    };

    let expr = lower_expr(ctx, expr_node);
    let field = ctx.node_symbol(&field_node);

    // Field access is syntactic sugar for a zero-arg method call: expr.field → expr.field()
    ExprKind::MethodCall {
        receiver: expr,
        method: field,
        args: vec![],
    }
}

/// Returns true if the node is a statement-like construct that should be
/// executed for side effects rather than used as a value.
/// Note: `expression_statement` is just a wrapper for expressions in the grammar,
/// so only `let_statement` is a true statement that yields Nil.
fn is_statement_node(node: &Node) -> bool {
    node.kind() == "let_statement"
}

fn lower_block(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let mut stmts = Vec::new();
    let mut cursor = node.walk();

    // Collect non-comment children, emitting diagnostics for ERROR nodes
    let children: Vec<_> = node
        .named_children(&mut cursor)
        .filter(|c| {
            if c.kind() == "ERROR" {
                // Skip — diagnostics are emitted by collect_error_nodes
                false
            } else {
                !is_comment(c.kind())
            }
        })
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
    ctx: &mut AstLoweringCtx<'_>,
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

fn lower_block_item_as_expr(ctx: &mut AstLoweringCtx<'_>, child: &Node) -> Expr<UnresolvedName> {
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

fn lower_let_statement(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Option<Stmt<UnresolvedName>> {
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

fn lower_case_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let scrutinee_node = node.child_by_field_name("value");

    let Some(scrutinee_node) = scrutinee_node else {
        return ExprKind::Error;
    };

    let scrutinee = lower_expr(ctx, scrutinee_node);
    let mut arms = Vec::new();

    // case_arm children are direct children of case_expression
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "case_arm" {
            arms.extend(lower_case_arm(ctx, child));
        }
    }

    ExprKind::Case { scrutinee, arms }
}

fn lower_case_arm(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Vec<Arm<UnresolvedName>> {
    let Some(pattern_node) = node.child_by_field_name("pattern") else {
        return vec![];
    };
    let direct_value = node.child_by_field_name("value");
    let pattern = lower_pattern(ctx, pattern_node);

    if let Some(body_node) = direct_value {
        return vec![Arm {
            id: ctx.fresh_id_with_span(&node),
            pattern,
            guard: None,
            body: lower_expr(ctx, body_node),
        }];
    }

    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter(|branch| branch.kind() == "guarded_branch")
        .filter_map(|branch| {
            let guard_node = branch.child_by_field_name("guard")?;
            let body_node = branch.child_by_field_name("value")?;
            Some(Arm {
                id: ctx.fresh_id_with_span(&branch),
                pattern: pattern.clone(),
                guard: Some(lower_expr(ctx, guard_node)),
                body: lower_expr(ctx, body_node),
            })
        })
        .collect()
}

fn lower_lambda_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
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

fn lower_param_list(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Vec<Param> {
    let mut params = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if (child.kind() == "parameter" || child.kind() == "typed_parameter")
            && let Some(name_node) = child.child_by_field_name("name")
        {
            let id = ctx.fresh_id_with_span(&child);
            let name = ctx.node_symbol(&name_node);
            let ty = child
                .child_by_field_name("type")
                .and_then(|n| super::declarations::lower_type_annotation(ctx, n));
            params.push(Param {
                id,
                name,
                ty,
                local_id: None,
            });
        } else if child.kind() == "identifier" {
            // Simple identifier parameter (no type annotation)
            let id = ctx.fresh_id_with_span(&child);
            let name = ctx.node_symbol(&child);
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

fn lower_tuple_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let mut elements = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            elements.push(lower_expr(ctx, child));
        }
    }

    ExprKind::Tuple(elements)
}

fn lower_list_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let mut elements = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            elements.push(lower_expr(ctx, child));
        }
    }

    ExprKind::List(elements)
}

fn lower_handle_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
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

fn lower_handler_arm(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<HandlerArm<UnresolvedName>> {
    // New grammar: handler_arm is a choice of completion_handler, fn_handler, op_handler
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "completion_handler" => return lower_completion_handler(ctx, child),
            "fn_handler" => return lower_fn_handler(ctx, child),
            "op_handler" => return lower_op_handler(ctx, child),
            _ => continue,
        }
    }
    None
}

/// Lower `do result { body }` handler arm.
fn lower_completion_handler(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<HandlerArm<UnresolvedName>> {
    let binding_node = node.child_by_field_name("binding")?;
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id_with_span(&node);
    let binding = lower_pattern(ctx, binding_node);
    let body = lower_expr(ctx, body_node);

    Some(HandlerArm {
        id,
        kind: HandlerKind::Do { binding },
        body,
    })
}

/// Lower `fn Op(args) { body }` handler arm.
fn lower_fn_handler(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<HandlerArm<UnresolvedName>> {
    let (ability, op) = lower_handler_operation_path(ctx, node)?;
    let params = lower_handler_params(ctx, node);
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id_with_span(&node);
    let body = lower_expr(ctx, body_node);

    Some(HandlerArm {
        id,
        kind: HandlerKind::Fn {
            ability,
            op,
            params,
        },
        body,
    })
}

/// Lower `op Op(args) { body }` handler arm.
fn lower_op_handler(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<HandlerArm<UnresolvedName>> {
    let (ability, op) = lower_handler_operation_path(ctx, node)?;
    let params = lower_handler_params(ctx, node);
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id_with_span(&node);
    let body = lower_expr(ctx, body_node);

    Some(HandlerArm {
        id,
        kind: HandlerKind::Op {
            ability,
            op,
            params,
            resume_local_id: None,
        },
        body,
    })
}

/// Parse the operation path from a fn_handler or op_handler node.
/// Returns (ability, op_name).
fn lower_handler_operation_path(
    ctx: &mut AstLoweringCtx<'_>,
    node: Node,
) -> Option<(UnresolvedName, Symbol)> {
    let op_node = node.child_by_field_name("operation")?;
    let op_text = ctx.node_text(&op_node).to_string();
    let op_symbol = Symbol::from_dynamic(&op_text);
    let op_name = op_symbol.last_segment();
    let ability_sym = op_symbol
        .parent_path()
        .unwrap_or_else(|| Symbol::from_dynamic("_"));
    let ability_id = ctx.fresh_id_with_span(&op_node);
    Some((UnresolvedName::new(ability_sym, ability_id), op_name))
}

/// Parse handler parameters (identifiers) from a fn_handler or op_handler node.
fn lower_handler_params(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Vec<Pattern<UnresolvedName>> {
    let Some(params_node) = node.child_by_field_name("params") else {
        return Vec::new();
    };
    let mut patterns = Vec::new();
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        if !is_comment(child.kind()) {
            patterns.push(lower_pattern(ctx, child));
        }
    }
    patterns
}

/// Lower `resume expr` expression.
fn lower_resume_expr(ctx: &mut AstLoweringCtx<'_>, node: Node) -> ExprKind<UnresolvedName> {
    let arg = match node.child_by_field_name("value") {
        Some(value_node) => lower_expr(ctx, value_node),
        None => {
            let id = ctx.fresh_id_with_span(&node);
            Expr {
                id,
                kind: Box::new(ExprKind::Nil),
            }
        }
    };

    ExprKind::Resume {
        arg,
        local_id: None,
    }
}

fn lower_argument_list(ctx: &mut AstLoweringCtx<'_>, node: Node) -> Vec<Expr<UnresolvedName>> {
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

/// Decode a string literal's text into its value.
///
/// Escape errors are reported relative to `text`.
pub(super) fn parse_string_literal(text: &str) -> Result<String, EscapeError> {
    // Strip quotes and handle basic escapes for string literals
    let literal = text;
    let text = text.trim();

    // Determine prefix and whether it's raw
    // Supported prefixes: "", "s", "r", "rs", "sr"
    let (prefix_len, is_raw) = if text.starts_with("rs") || text.starts_with("sr") {
        (2, true)
    } else if text.starts_with('r') {
        (1, true)
    } else if text.starts_with('s') {
        (1, false)
    } else {
        (0, false)
    };

    let after_prefix = &text[prefix_len..];

    // Count consecutive '#' characters before the opening quote (for raw strings)
    let hash_count = if is_raw {
        after_prefix.chars().take_while(|&c| c == '#').count()
    } else {
        0
    };

    // For raw strings with hashes: r#"..."# or rs##"..."##
    // For regular raw strings: r"..." or rs"..."
    // For regular strings: "..." or s"..."
    if hash_count > 0 {
        // Raw string with hashes
        let quote_start = prefix_len + hash_count;
        let expected_end_pattern_len = 1 + hash_count; // closing quote + hashes

        // Validate we have opening quote after hashes
        if text.get(quote_start..quote_start + 1) != Some("\"") {
            return Ok(String::new());
        }

        // Content starts after the opening quote
        let content_start = quote_start + 1;

        // Find content end: must have closing quote followed by same number of hashes
        let content_end = text.len().saturating_sub(expected_end_pattern_len);
        if content_end <= content_start {
            return Ok(String::new());
        }

        // Validate closing pattern: " followed by hash_count #'s
        let closing = text.get(content_end..);
        let expected_closing: String = std::iter::once('"')
            .chain(std::iter::repeat_n('#', hash_count))
            .collect();
        if closing != Some(&expected_closing) {
            return Ok(String::new());
        }

        Ok(text
            .get(content_start..content_end)
            .unwrap_or("")
            .to_string())
    } else if is_raw {
        // Raw string without hashes: r"..." or rs"..."
        let quote_start = prefix_len;
        if text.get(quote_start..quote_start + 1) != Some("\"") {
            return Ok(String::new());
        }
        Ok(text
            .get(quote_start + 1..text.len().saturating_sub(1))
            .unwrap_or("")
            .to_string())
    } else if text.get(prefix_len..prefix_len + 1) == Some("\"") {
        // Regular string: "..." or s"..."
        let content = text
            .get(prefix_len + 1..text.len().saturating_sub(1))
            .unwrap_or("");
        let bytes = process_escape_sequences(content)
            .map_err(|error| error.offset_by(offset_within(literal, content)))?;
        Ok(String::from_utf8(bytes).expect("escape processing produced invalid UTF-8"))
    } else {
        // Fallback
        Ok(text.to_string())
    }
}

/// Decode a bytes literal's text into its value.
///
/// Escape errors are reported relative to `text`.
fn parse_bytes_literal(text: &str) -> Result<Vec<u8>, EscapeError> {
    // Strip quotes and handle basic escapes for byte string literals
    let literal = text;
    let text = text.trim();

    // Determine prefix and whether it's raw
    let (prefix_len, is_raw) = if text.starts_with("rb") || text.starts_with("br") {
        (2, true)
    } else if text.starts_with('b') {
        (1, false)
    } else {
        return Ok(text.as_bytes().to_vec());
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
            return Ok(Vec::new());
        }

        // Content starts after the opening quote
        let content_start = quote_start + 1;

        // Find content end: must have closing quote followed by same number of hashes
        let content_end = text.len().saturating_sub(expected_end_pattern_len);
        if content_end <= content_start {
            return Ok(Vec::new());
        }

        // Validate closing pattern: " followed by hash_count #'s
        let closing = text.get(content_end..);
        let expected_closing: String = std::iter::once('"')
            .chain(std::iter::repeat_n('#', hash_count))
            .collect();
        if closing != Some(&expected_closing) {
            return Ok(Vec::new());
        }

        Ok(text
            .get(content_start..content_end)
            .unwrap_or("")
            .as_bytes()
            .to_vec())
    } else if is_raw {
        // Raw byte string without hashes: rb"..." or br"..."
        Ok(text
            .get(prefix_len + 1..text.len().saturating_sub(1))
            .unwrap_or("")
            .as_bytes()
            .to_vec())
    } else {
        // Regular byte string: b"..."
        let content = text
            .get(prefix_len + 1..text.len().saturating_sub(1))
            .unwrap_or("");
        process_escape_sequences(content)
            .map_err(|error| error.offset_by(offset_within(literal, content)))
    }
}

/// Byte offset of `inner`, a subslice of `outer`, from the start of `outer`.
fn offset_within(outer: &str, inner: &str) -> usize {
    inner.as_ptr() as usize - outer.as_ptr() as usize
}

/// Parse a rune (character) literal.
///
/// Returns `Ok(None)` for text the grammar does not produce, and an
/// [`EscapeError`] relative to `text` for a `\u{…}` escape that is not a
/// Unicode scalar value.
fn parse_rune_literal(text: &str) -> Result<Option<char>, EscapeError> {
    // Format: ?c, ?\n, ?\xHH, ?\u{H…}
    let Some(body) = text.strip_prefix('?') else {
        return Ok(None);
    };

    let Some(escape) = body.strip_prefix('\\') else {
        return Ok(body.chars().next());
    };
    let Some(escape_char) = escape.chars().next() else {
        return Ok(None);
    };
    Ok(match escape_char {
        'n' => Some('\n'),
        'r' => Some('\r'),
        't' => Some('\t'),
        '\\' => Some('\\'),
        '0' => Some('\0'),
        'x' => {
            let hex = &escape[1..];
            u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
        }
        'u' => {
            let Some(hex) = escape[1..]
                .strip_prefix('{')
                .and_then(|rest| rest.strip_suffix('}'))
            else {
                return Ok(None);
            };
            decode_unicode_escape(hex).map_err(|kind| EscapeError {
                range: 1..text.len(),
                kind,
            })?
        }
        _ => None,
    })
}
