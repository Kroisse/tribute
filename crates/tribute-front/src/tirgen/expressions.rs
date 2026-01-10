#![allow(clippy::collapsible_if)]

//! Expression lowering.

use tracing::trace;
use tree_sitter::Node;
use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{adt, list, tribute, tribute_pat};
use trunk_ir::{
    Attribute, Block, BlockBuilder, BlockId, DialectOp, DialectType, IdVec, Operation, Region,
    Symbol, Type, Value,
    dialect::{
        arith,
        core::{self, AbilityRefType},
        func,
    },
    idvec,
};

use super::context::CstLoweringCtx;
use super::declarations::parse_parameter_list;
use super::helpers::{int_const, is_comment, node_text, sym, sym_ref};
use super::literals::{
    parse_bytes_literal, parse_float_literal, parse_int_literal, parse_nat_literal,
    parse_rune_literal, parse_string_literal,
};
use super::statements::{bind_pattern, lower_block_body};

// =============================================================================
// Expression Lowering
// =============================================================================

fn collect_path_segments<'db>(ctx: &CstLoweringCtx<'db>, node: Node) -> Vec<String> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter_map(|child| match child.kind() {
            "identifier" | "type_identifier" => Some(node_text(&child, &ctx.source).to_string()),
            "path_segment" => {
                let mut inner = child.walk();
                child
                    .named_children(&mut inner)
                    .find(|c| c.kind() == "identifier" || c.kind() == "type_identifier")
                    .map(|c| node_text(&c, &ctx.source).to_string())
            }
            _ => None,
        })
        .collect()
}

/// Lower an expression node to TrunkIR operations.
pub fn lower_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let unit_ty = core::Nil::new(ctx.db).as_type();

    trace!("lower_expr: kind={}", node.kind());

    match node.kind() {
        // === Literals ===
        "nat_literal" => {
            let value = parse_nat_literal(&node_text(&node, &ctx.source))?;
            // Natural numbers are represented as Int (arbitrary precision)
            // Phase 1: values must fit in i64; BigInt support will be added later
            let value = i64::try_from(value).ok()?;
            let op = block.op(int_const(ctx.db, location, value));
            Some(op.result(ctx.db))
        }
        "int_literal" => {
            let value = parse_int_literal(&node_text(&node, &ctx.source))?;
            let op = block.op(int_const(ctx.db, location, value));
            Some(op.result(ctx.db))
        }
        "float_literal" => {
            let value = parse_float_literal(&node_text(&node, &ctx.source))?;
            let op = block.op(arith::Const::f64(ctx.db, location, value));
            Some(op.result(ctx.db))
        }
        "true" | "keyword_true" => {
            let op = block.op(arith::r#const(
                ctx.db,
                location,
                core::I1::new(ctx.db).as_type(),
                true.into(),
            ));
            Some(op.result(ctx.db))
        }
        "false" | "keyword_false" => {
            let op = block.op(arith::r#const(
                ctx.db,
                location,
                core::I1::new(ctx.db).as_type(),
                false.into(),
            ));
            Some(op.result(ctx.db))
        }
        "nil" | "keyword_nil" => {
            let op = block.op(arith::r#const(ctx.db, location, unit_ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }
        "rune" => {
            let c = parse_rune_literal(&node_text(&node, &ctx.source))?;
            let op = block.op(arith::r#const(
                ctx.db,
                location,
                core::I32::new(ctx.db).as_type(),
                u64::from(u32::from(c)).into(),
            ));
            Some(op.result(ctx.db))
        }

        // === String literals ===
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(node, &ctx.source);
            let string_ty = core::String::new(ctx.db).as_type();
            let op = block.op(adt::string_const(ctx.db, location, string_ty, s));
            Some(op.result(ctx.db))
        }

        // === Bytes literals ===
        "bytes_string" | "raw_bytes" | "multiline_bytes" => {
            let bytes = parse_bytes_literal(node, &ctx.source);
            let bytes_ty = core::Bytes::new(ctx.db).as_type();
            let op = block.op(adt::bytes_const(
                ctx.db,
                location,
                bytes_ty,
                Attribute::Bytes(bytes),
            ));
            Some(op.result(ctx.db))
        }

        // === Identifiers and paths ===
        "identifier" => {
            // Always create tribute.var for variable references, even for local bindings.
            // This preserves the source span for hover. Resolution will transform
            // local references to identity operations with the correct type.
            let name = node_text(&node, &ctx.source);
            let op = block.op(tribute::var(ctx.db, location, infer_ty, name.into()));
            Some(op.result(ctx.db))
        }
        "path_expression" => {
            let segments = collect_path_segments(ctx, node);
            if segments.is_empty() {
                return None;
            }
            let path = Symbol::from_dynamic(&segments.join("::"));
            let op = block.op(tribute::path(ctx.db, location, infer_ty, path));
            Some(op.result(ctx.db))
        }

        // === Binary expressions ===
        "binary_expression" => lower_binary_expr(ctx, block, node),

        // === Call expressions ===
        "call_expression" => lower_call_expr(ctx, block, node),

        // === Constructor expressions ===
        "constructor_expression" => lower_constructor_expr(ctx, block, node),

        // === Method call ===
        "method_call_expression" => lower_method_call_expr(ctx, block, node),

        // === Lambda ===
        "lambda_expression" => lower_lambda_expr(ctx, block, node),

        // === Match/Case ===
        "case_expression" => lower_case_expr(ctx, block, node),

        // === Block ===
        "block" => lower_block_expr(ctx, block, node),

        // === List ===
        "list_expression" => lower_list_expr(ctx, block, node),

        // === Tuple ===
        "tuple_expression" => lower_tuple_expr(ctx, block, node),

        // === Record ===
        "record_expression" => lower_record_expr(ctx, block, node),

        // === Parenthesized ===
        "parenthesized_expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return lower_expr(ctx, block, child);
                }
            }
            None
        }

        // === Primary expression (wrapper for literals, paths, etc.) ===
        "primary_expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return lower_expr(ctx, block, child);
                }
            }
            None
        }

        // === Expression statement (expression as statement in a block) ===
        "expression_statement" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return lower_expr(ctx, block, child);
                }
            }
            None
        }

        // === Handle expression (ability handling) ===
        "handle_expression" => lower_handle_expr(ctx, block, node),

        // === Interpolated strings ===
        "string_interpolation" | "raw_interpolated_string" => {
            lower_string_interpolation(ctx, block, node)
        }

        // === Interpolated bytes ===
        "bytes_interpolation" | "raw_interpolated_bytes" => {
            lower_bytes_interpolation(ctx, block, node)
        }

        // Unknown expression type - return None
        _ => None,
    }
}

// =============================================================================
// Binary Expression Lowering
// =============================================================================

/// Lower a binary expression.
fn lower_binary_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let bool_ty = core::I1::new(ctx.db).as_type();

    let mut lhs_node = None;
    let mut operator = None;
    let mut rhs_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if lhs_node.is_none() {
            lhs_node = Some(child);
        } else if operator.is_none() && is_operator_node(&child) {
            operator = Some(node_text(&child, &ctx.source).to_string());
        } else if rhs_node.is_none() {
            rhs_node = Some(child);
        }
    }

    // Handle the case where operator is an unnamed token
    if operator.is_none() {
        // Look for operator in unnamed children
        let mut child_cursor = node.walk();
        if child_cursor.goto_first_child() {
            loop {
                let child = child_cursor.node();
                if !child.is_named() && is_operator_text(&node_text(&child, &ctx.source)) {
                    operator = Some(node_text(&child, &ctx.source).to_string());
                    break;
                }
                if !child_cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }

    let lhs_node = lhs_node?;
    let operator = operator?;
    let rhs_node = rhs_node?;

    let lhs = lower_expr(ctx, block, lhs_node)?;
    let rhs = lower_expr(ctx, block, rhs_node)?;

    // Map operator to IR operation
    let result = match operator.as_str() {
        "+" => block
            .op(arith::add(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "-" => block
            .op(arith::sub(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "*" => block
            .op(arith::mul(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "/" => block
            .op(arith::div(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "%" => block
            .op(arith::rem(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "==" => block
            .op(arith::cmp_eq(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "!=" => block
            .op(arith::cmp_ne(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "<" => block
            .op(arith::cmp_lt(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "<=" => block
            .op(arith::cmp_le(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        ">" => block
            .op(arith::cmp_gt(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        ">=" => block
            .op(arith::cmp_ge(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "&&" => block
            .op(arith::and(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "||" => block
            .op(arith::or(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "<>" => {
            // Concatenation operator - resolved by TDNR based on operand type
            block
                .op(tribute::call(
                    ctx.db,
                    location,
                    vec![lhs, rhs],
                    infer_ty,
                    sym("<>"),
                ))
                .result(ctx.db)
        }
        _ => {
            // Unknown operator - emit as tribute.binop
            block
                .op(tribute::binop(
                    ctx.db,
                    location,
                    lhs,
                    rhs,
                    infer_ty,
                    sym(&operator),
                ))
                .result(ctx.db)
        }
    };

    Some(result)
}

fn is_operator_node(node: &Node) -> bool {
    matches!(node.kind(), "operator" | "binary_operator")
}

fn is_operator_text(text: &str) -> bool {
    matches!(
        text,
        "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" | "<>"
    )
}

// =============================================================================
// Call Expression Lowering
// =============================================================================

/// Lower a call expression.
fn lower_call_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    // Use field-based access for function
    let func_node = node.child_by_field_name("function")?;
    match func_node.kind() {
        "identifier" => {
            let name = sym_ref(&node_text(&func_node, &ctx.source));
            let args = collect_argument_list(ctx, block, node);
            if let Some(callee) = ctx.lookup(name) {
                let op = block.op(func::call_indirect(
                    ctx.db, location, callee, args, infer_ty,
                ));
                Some(op.result(ctx.db))
            } else if ctx.is_local(name) {
                let callee_ty = ctx.fresh_type_var();
                let callee_op = block.op(tribute::var(ctx.db, location, callee_ty, name));
                let callee = callee_op.result(ctx.db);
                let op = block.op(func::call_indirect(
                    ctx.db, location, callee, args, infer_ty,
                ));
                Some(op.result(ctx.db))
            } else {
                let op = block.op(tribute::call(ctx.db, location, args, infer_ty, name));
                Some(op.result(ctx.db))
            }
        }
        "path_expression" => {
            let segments = collect_path_segments(ctx, func_node);
            if segments.is_empty() {
                return None;
            }
            let func_path = Symbol::from_dynamic(&segments.join("::"));
            let args = collect_argument_list(ctx, block, node);
            let op = block.op(tribute::call(ctx.db, location, args, infer_ty, func_path));
            Some(op.result(ctx.db))
        }
        _ => None,
    }
}

/// Lower a constructor expression.
fn lower_constructor_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let ctor_node = node.child_by_field_name("constructor")?;
    let ctor_path: Symbol = match ctor_node.kind() {
        "type_identifier" => sym_ref(&node_text(&ctor_node, &ctx.source)),
        "path_expression" => {
            let segments = collect_path_segments(ctx, ctor_node);
            if segments.is_empty() {
                return None;
            }
            Symbol::from_dynamic(&segments.join("::"))
        }
        _ => return None,
    };

    let args = collect_argument_list(ctx, block, node);
    let op = block.op(tribute::cons(ctx.db, location, args, infer_ty, ctor_path));
    Some(op.result(ctx.db))
}

/// Collect arguments from argument_list child node.
fn collect_argument_list<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Vec<Value<'db>> {
    let mut cursor = node.walk();
    let mut args = Vec::new();

    for child in node.named_children(&mut cursor) {
        if child.kind() == "argument_list" {
            let mut arg_cursor = child.walk();
            for arg_child in child.named_children(&mut arg_cursor) {
                if !is_comment(arg_child.kind())
                    && let Some(value) = lower_expr(ctx, block, arg_child)
                {
                    args.push(value);
                }
            }
            break;
        }
    }

    args
}

/// Lower a method call expression (UFCS).
///
/// Handles both simple and qualified method names:
/// - `x.foo()` → `foo(x)`
/// - `x.math::double()` → `math::double(x)`
fn lower_method_call_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    trace!("lower_method_call_expr: node={:?}", node.to_sexp());

    // Use field-based access
    let receiver_node = node.child_by_field_name("receiver")?;
    let method_node = node.child_by_field_name("method")?;
    trace!(
        "  receiver={:?}, method={:?}",
        receiver_node.to_sexp(),
        method_node.to_sexp()
    );

    // Handle method: can be a simple identifier or a path with segments
    // e.g., "x" → just identifier
    // e.g., "math::double" → path with segments
    let method_path: Symbol = match method_node.kind() {
        "identifier" | "type_identifier" => {
            // Simple method name: p.x()
            Symbol::from(node_text(&method_node, &ctx.source))
        }
        _ => {
            // Qualified path: p.math::double()
            let mut cursor = method_node.walk();
            let segments: Vec<_> = method_node
                .named_children(&mut cursor)
                .filter_map(|n| match n.kind() {
                    "identifier" | "type_identifier" => Some(node_text(&n, &ctx.source)),
                    "path_segment" => {
                        // path_segment contains an identifier or type_identifier
                        let mut inner = n.walk();
                        n.named_children(&mut inner)
                            .find(|c| c.kind() == "identifier" || c.kind() == "type_identifier")
                            .map(|c| node_text(&c, &ctx.source))
                    }
                    _ => None,
                })
                .collect();
            if segments.is_empty() {
                return None;
            }
            Symbol::from_dynamic(&segments.join("::"))
        }
    };

    // Lower receiver first
    let receiver = lower_expr(ctx, block, receiver_node)?;

    // Arguments need iteration (not a field)
    let args = collect_argument_list(ctx, block, node);

    // UFCS: x.f(y, z) → f(x, y, z)
    let mut all_args = vec![receiver];
    all_args.extend(args);

    let op = block.op(tribute::call(
        ctx.db,
        location,
        all_args,
        infer_ty,
        method_path,
    ));
    Some(op.result(ctx.db))
}

// =============================================================================
// Lambda Expression Lowering
// =============================================================================

/// Lower a lambda expression.
fn lower_lambda_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    // Use field-based access
    let body_node = node.child_by_field_name("body")?;

    // Optional params field
    let param_names = node
        .child_by_field_name("params")
        .map(|params| parse_parameter_list(ctx, params).0)
        .unwrap_or_default();

    // Build lambda body with bind_name attributes on block args
    let param_types: IdVec<Type<'_>> = std::iter::repeat_n(infer_ty, param_names.len()).collect();
    let result_type = infer_ty;
    let mut body_block = BlockBuilder::new(ctx.db, location);
    for &param_name in &param_names {
        body_block = body_block
            .arg(infer_ty)
            .attr(Symbol::new("bind_name"), param_name);
    }

    let result_value = ctx.scoped(|ctx| {
        // Bind parameters
        for param_name in param_names {
            let param_value = body_block.op(tribute::var(ctx.db, location, infer_ty, param_name));
            ctx.bind(param_name, param_value.result(ctx.db));
        }

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(tribute::r#yield(ctx.db, location, result_value));

    let effect_type = ctx.fresh_effect_row_type();
    let func_type =
        core::Func::with_effect(ctx.db, param_types, result_type, Some(effect_type)).as_type();
    let region = Region::new(ctx.db, location, idvec![body_block.build()]);
    let lambda_op = block.op(tribute::lambda(
        ctx.db, location, infer_ty, func_type, region,
    ));
    Some(lambda_op.result(ctx.db))
}

// =============================================================================
// Case/Match Expression Lowering
// =============================================================================

/// Lower a case/match expression.
fn lower_case_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    // Use field-based access for scrutinee
    let scrutinee_node = node.child_by_field_name("value")?;

    // Collect case arms (need iteration)
    let mut cursor = node.walk();
    let arms: Vec<_> = node
        .named_children(&mut cursor)
        .filter(|child| child.kind() == "case_arm")
        .collect();

    let scrutinee = lower_expr(ctx, block, scrutinee_node)?;

    // Build the body region containing case.arm operations
    let mut body_block = BlockBuilder::new(ctx.db, location);

    for arm in arms {
        if let Some(arm_op) = lower_case_arm(ctx, arm, scrutinee) {
            body_block.op(arm_op);
        }
    }

    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    let op = block.op(tribute::r#case(
        ctx.db,
        location,
        scrutinee,
        infer_ty,
        body_region,
    ));
    Some(op.result(ctx.db))
}

/// Lower a single case arm.
fn lower_case_arm<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
    scrutinee: Value<'db>,
) -> Option<tribute::Arm<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);

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

    // Create arm body
    let mut body_block = BlockBuilder::new(ctx.db, location);

    let result_value = ctx.scoped(|ctx| {
        // Bind pattern variables (including handler patterns)
        bind_pattern(ctx, &mut body_block, pattern_node, scrutinee);

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(tribute::r#yield(ctx.db, location, result_value));

    let pattern_region = pattern_to_region(ctx, pattern_node);
    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    Some(tribute::arm(ctx.db, location, pattern_region, body_region))
}

/// Convert a pattern node to a pattern region for case arms and let bindings.
///
/// Creates a region containing pattern operations from the `tribute_pat` dialect.
pub fn pattern_to_region<'db>(ctx: &CstLoweringCtx<'db>, node: Node) -> Region<'db> {
    let location = ctx.location(&node);

    match node.kind() {
        // Wrapper nodes - unwrap and recurse
        "pattern" | "simple_pattern" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                return pattern_to_region(ctx, child);
            }
            tribute_pat::helpers::wildcard_region(ctx.db, location)
        }
        "identifier" | "identifier_pattern" => {
            // Handle identifier_pattern which may have an inner identifier
            let name = if node.kind() == "identifier_pattern" {
                let mut cursor = node.walk();
                node.named_children(&mut cursor)
                    .next()
                    .map(|child| node_text(&child, &ctx.source))
                    .unwrap_or_else(|| node_text(&node, &ctx.source))
            } else {
                node_text(&node, &ctx.source)
            };
            tribute_pat::helpers::bind_region(ctx.db, location, name.into())
        }
        "wildcard_pattern" => tribute_pat::helpers::wildcard_region(ctx.db, location),
        "literal_pattern" => {
            // Get the literal value
            let mut cursor = node.walk();
            if let Some(child) = node.named_children(&mut cursor).next() {
                pattern_to_region(ctx, child)
            } else {
                tribute_pat::helpers::wildcard_region(ctx.db, location)
            }
        }
        "nat_literal" | "int_literal" => {
            if let Some(n) = parse_int_literal(&node_text(&node, &ctx.source)) {
                tribute_pat::helpers::int_region(ctx.db, location, n)
            } else {
                tribute_pat::helpers::wildcard_region(ctx.db, location)
            }
        }
        "true" | "keyword_true" => tribute_pat::helpers::bool_region(ctx.db, location, true),
        "false" | "keyword_false" => tribute_pat::helpers::bool_region(ctx.db, location, false),
        "nil" | "keyword_nil" => {
            let op = tribute_pat::literal(ctx.db, location, Attribute::Unit);
            tribute_pat::helpers::single_op_region(ctx.db, location, op.as_operation())
        }
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(node, &ctx.source);
            tribute_pat::helpers::string_region(ctx.db, location, &s)
        }
        "constructor_pattern" => {
            let mut cursor = node.walk();
            let mut ctor_name = None;
            let mut field_ops: Vec<Operation<'db>> = Vec::new();

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "type_identifier" => {
                        ctor_name = Some(node_text(&child, &ctx.source).into());
                    }
                    "pattern_list" => {
                        let mut list_cursor = child.walk();
                        for pat_child in child.named_children(&mut list_cursor) {
                            if !is_comment(pat_child.kind()) {
                                let pat_region = pattern_to_region(ctx, pat_child);
                                // Extract the pattern operation from the region
                                if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                                    field_ops.push(op);
                                }
                            }
                        }
                    }
                    "pattern_fields" => {
                        // Named fields - treat as positional for pattern matching
                        let mut fields_cursor = child.walk();
                        for field_child in child.named_children(&mut fields_cursor) {
                            if field_child.kind() == "pattern_field" {
                                let mut field_cursor = field_child.walk();
                                for pat in field_child.named_children(&mut field_cursor) {
                                    if !is_comment(pat.kind()) {
                                        let pat_region = pattern_to_region(ctx, pat);
                                        if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                                            field_ops.push(op);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            let name = ctor_name.unwrap_or_else(|| Symbol::new("_"));
            let fields_region = ops_to_region(ctx.db, location, field_ops);
            tribute_pat::helpers::variant_region(ctx.db, location, name, fields_region)
        }
        "tuple_pattern" => {
            let mut cursor = node.walk();
            let mut elem_ops: Vec<Operation<'db>> = Vec::new();

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "pattern_list" {
                    let mut list_cursor = child.walk();
                    for pat_child in child.named_children(&mut list_cursor) {
                        if !is_comment(pat_child.kind()) {
                            let pat_region = pattern_to_region(ctx, pat_child);
                            if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                                elem_ops.push(op);
                            }
                        }
                    }
                }
            }

            let elements_region = ops_to_region(ctx.db, location, elem_ops);
            tribute_pat::helpers::tuple_region(ctx.db, location, elements_region)
        }
        "list_pattern" => {
            let mut cursor = node.walk();
            let mut elem_ops: Vec<Operation<'db>> = Vec::new();
            let mut rest_name = None;

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "rest_pattern" {
                    let mut rest_cursor = child.walk();
                    for rest_child in child.named_children(&mut rest_cursor) {
                        if rest_child.kind() == "identifier" {
                            rest_name = Some(node_text(&rest_child, &ctx.source).into());
                            break;
                        }
                    }
                } else {
                    let pat_region = pattern_to_region(ctx, child);
                    if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                        elem_ops.push(op);
                    }
                }
            }

            if let Some(name) = rest_name {
                let head_region = ops_to_region(ctx.db, location, elem_ops);
                tribute_pat::helpers::list_rest_region(ctx.db, location, name, head_region)
            } else {
                let elements_region = ops_to_region(ctx.db, location, elem_ops);
                tribute_pat::helpers::list_region(ctx.db, location, elements_region)
            }
        }
        "as_pattern" => {
            let mut cursor = node.walk();
            let mut inner_region = None;
            let mut binding_name = None;

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "identifier" => {
                        binding_name = Some(node_text(&child, &ctx.source).into());
                    }
                    _ if inner_region.is_none() => {
                        inner_region = Some(pattern_to_region(ctx, child));
                    }
                    _ => {}
                }
            }

            let inner = inner_region
                .unwrap_or_else(|| tribute_pat::helpers::wildcard_region(ctx.db, location));
            let name = binding_name.unwrap_or_else(|| Symbol::new("_"));
            // Create as_pat operation with inner region
            let as_op = tribute_pat::as_pat(ctx.db, location, name, inner);
            tribute_pat::helpers::single_op_region(ctx.db, location, as_op.as_operation())
        }
        "handler_pattern" => {
            // Handler patterns are for ability effect handling in case expressions
            // Delegate to the specialized handler pattern converter
            handler_pattern_to_region(ctx, node)
        }
        _ => tribute_pat::helpers::wildcard_region(ctx.db, location),
    }
}

/// Extract the first pattern operation from a region.
fn extract_pattern_op<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Operation<'db>> {
    let blocks = region.blocks(db);
    let block = blocks.first()?;
    let ops = block.operations(db);
    ops.first().copied()
}

/// Create a region from a list of operations.
fn ops_to_region<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    ops: Vec<Operation<'db>>,
) -> Region<'db> {
    let block = Block::new(
        db,
        BlockId::fresh(),
        location,
        IdVec::new(),
        IdVec::from(ops),
    );
    Region::new(db, location, IdVec::from(vec![block]))
}

// =============================================================================
// Collection Expression Lowering
// =============================================================================

/// Lower a block expression.
fn lower_block_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut body_block = BlockBuilder::new(ctx.db, location);
    let result_value = ctx.scoped(|ctx| lower_block_body(ctx, &mut body_block, node));

    let result_value = result_value?;
    body_block.op(tribute::r#yield(ctx.db, location, result_value));

    let region = Region::new(ctx.db, location, idvec![body_block.build()]);
    let block_op = block.op(tribute::block(ctx.db, location, infer_ty, region));
    Some(block_op.result(ctx.db))
}

/// Lower a list expression.
fn lower_list_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let elem_ty = ctx.fresh_type_var();

    let mut elements = Vec::new();
    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind())
            && let Some(value) = lower_expr(ctx, block, child)
        {
            elements.push(value);
        }
    }

    let op = block.op(list::new(ctx.db, location, elements, infer_ty, elem_ty));
    Some(op.result(ctx.db))
}

/// Lower a tuple expression.
fn lower_tuple_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut elements = Vec::new();
    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind())
            && let Some(value) = lower_expr(ctx, block, child)
        {
            elements.push(value);
        }
    }

    if elements.is_empty() {
        return None;
    }

    let tuple_op = block.op(tribute::tuple(ctx.db, location, elements, infer_ty));
    Some(tuple_op.result(ctx.db))
}

/// Lower a record expression.
///
/// Handles three forms:
/// - Full construction: `User { name: "Alice", age: 30 }`
/// - Shorthand: `User { name, age }` (binds to variables with same names)
/// - Spread: `User { ..base, age: 31 }` (copies fields from base, overrides specified fields)
fn lower_record_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut type_name = None;
    let mut spread_base: Option<Value<'db>> = None;
    let mut field_names = Vec::new();
    let mut field_values = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "type_identifier" if type_name.is_none() => {
                type_name = Some(node_text(&child, &ctx.source).to_string());
            }
            "record_fields" => {
                // Container for record_field nodes
                let mut fields_cursor = child.walk();
                for field in child.named_children(&mut fields_cursor) {
                    if is_comment(field.kind()) || field.kind() != "record_field" {
                        continue;
                    }

                    let mut field_cursor = field.walk();
                    let children: Vec<_> = field
                        .named_children(&mut field_cursor)
                        .filter(|c| !is_comment(c.kind()))
                        .collect();

                    // Check for spread: ..expr
                    if children.iter().any(|c| c.kind() == "spread") {
                        // Find the value expression after spread
                        for field_child in &children {
                            if field_child.kind() != "spread" {
                                if let Some(value) = lower_expr(ctx, block, *field_child) {
                                    spread_base = Some(value);
                                }
                            }
                        }
                    } else if children.len() == 1 && children[0].kind() == "identifier" {
                        // Shorthand: { name } means { name: name }
                        let field_name: Symbol = node_text(&children[0], &ctx.source).into();
                        field_names.push(field_name);
                        if let Some(value) = ctx.lookup(field_name) {
                            field_values.push(value);
                        } else {
                            let var_op =
                                block.op(tribute::var(ctx.db, location, infer_ty, field_name));
                            field_values.push(var_op.result(ctx.db));
                        }
                    } else {
                        // Full syntax: { name: value }
                        // Get the field name
                        let field_name: Symbol = children
                            .iter()
                            .find(|c| c.kind() == "identifier")
                            .map(|c| node_text(c, &ctx.source).into())
                            .unwrap_or_else(|| Symbol::new(""));
                        field_names.push(field_name);

                        // Find the value expression (skip the identifier)
                        for field_child in &children {
                            if field_child.kind() != "identifier" {
                                if let Some(value) = lower_expr(ctx, block, *field_child) {
                                    field_values.push(value);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let type_name = type_name?;

    // Build fields region containing tribute.field_arg operations
    // Each field_arg pairs the field name with its value
    let mut fields_block = BlockBuilder::new(ctx.db, location);
    for (field_name, field_value) in field_names.iter().zip(field_values.iter()) {
        fields_block.op(tribute::field_arg(
            ctx.db,
            location,
            *field_value,
            *field_name,
        ));
    }
    let fields_region = Region::new(ctx.db, location, idvec![fields_block.build()]);

    // Convert spread_base from Option<Value> to Vec<Value>
    let base: Vec<Value<'db>> = spread_base.into_iter().collect();

    // Create the record operation
    let op = block.op(tribute::record(
        ctx.db,
        location,
        base,
        infer_ty,
        sym_ref(&type_name),
        fields_region,
    ));
    Some(op.result(ctx.db))
}

// =============================================================================
// Ability Handling Expression Lowering
// =============================================================================

/// Lower a handle expression (ability handling).
///
/// Fused handler syntax: `handle expr { handler_arms }`.
///
/// Lowers to:
/// ```text
/// %result = tribute.handle { expr } { arms }
/// ```
///
/// The handler arms are evaluated when:
/// - Body completes with a value → matches `{ result }` handler pattern
/// - Body performs an ability operation → matches `{ Op(args) -> k }` handler pattern
fn lower_handle_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let result_ty = ctx.fresh_type_var();

    // Find the expression to handle (field: "expr")
    let expr_node = node.child_by_field_name("expr")?;

    // Collect handler arms
    let mut cursor = node.walk();
    let handler_arms: Vec<_> = node
        .named_children(&mut cursor)
        .filter(|child| child.kind() == "handler_arm")
        .collect();

    // Build body region with the expression
    let mut body_block = BlockBuilder::new(ctx.db, location);
    let body_result = ctx.scoped(|ctx| lower_expr(ctx, &mut body_block, expr_node));

    if let Some(value) = body_result {
        body_block.op(tribute::r#yield(ctx.db, location, value));
    }

    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    // Build arms region with handler arms
    let mut arms_block = BlockBuilder::new(ctx.db, location);

    for arm in handler_arms {
        if let Some(arm_op) = lower_handler_arm(ctx, arm) {
            arms_block.op(arm_op);
        }
    }

    let arms_region = Region::new(ctx.db, location, idvec![arms_block.build()]);

    // Create tribute.handle with body and arms
    let handle_op = block.op(tribute::handle(
        ctx.db,
        location,
        result_ty,
        body_region,
        arms_region,
    ));
    Some(handle_op.result(ctx.db))
}

/// Lower a single handler arm.
///
/// Handler arms have patterns like `{ result }` or `{ State::get() -> k }`.
fn lower_handler_arm<'db>(ctx: &mut CstLoweringCtx<'db>, node: Node) -> Option<tribute::Arm<'db>> {
    let location = ctx.location(&node);

    // Get pattern and body from fields
    let pattern_node = node.child_by_field_name("pattern")?;
    let body_node = node.child_by_field_name("value")?;

    // Create the pattern region (handler pattern)
    let pattern_region = handler_pattern_to_region(ctx, pattern_node);

    // Create arm body, binding pattern variables
    let mut body_block = BlockBuilder::new(ctx.db, location);

    let result_value = ctx.scoped(|ctx| {
        // Bind handler pattern variables (result for done, args + k for suspend)
        bind_handler_pattern(ctx, &mut body_block, pattern_node);

        // Lower body expression
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(tribute::r#yield(ctx.db, location, result_value));

    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    Some(tribute::arm(ctx.db, location, pattern_region, body_region))
}

/// Extract binding names from a pattern and mark them as local.
/// This is used for handler pattern arguments where the actual values
/// are provided later by cont::get_shift_value in tribute_to_scf.
fn extract_and_mark_pattern_bindings<'db>(ctx: &mut CstLoweringCtx<'db>, pattern: Node) {
    match pattern.kind() {
        "identifier" | "identifier_pattern" => {
            let name: Symbol = node_text(&pattern, &ctx.source).into();
            ctx.mark_local(name);
        }
        "wildcard_pattern" => {
            // No binding needed
        }
        "as_pattern" => {
            // Mark the binding name
            if let Some(binding_node) = pattern.child_by_field_name("binding") {
                let name: Symbol = node_text(&binding_node, &ctx.source).into();
                ctx.mark_local(name);
            }
            // Recurse on inner pattern
            if let Some(inner_pattern) = pattern.child_by_field_name("pattern") {
                extract_and_mark_pattern_bindings(ctx, inner_pattern);
            }
        }
        "constructor_pattern" => {
            // Recurse on positional fields
            let mut cursor = pattern.walk();
            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "pattern_list" => {
                        let mut list_cursor = child.walk();
                        for pat_child in child.named_children(&mut list_cursor) {
                            if !is_comment(pat_child.kind()) {
                                extract_and_mark_pattern_bindings(ctx, pat_child);
                            }
                        }
                    }
                    "pattern_fields" => {
                        let mut fields_cursor = child.walk();
                        for field_child in child.named_children(&mut fields_cursor) {
                            if field_child.kind() == "pattern_field" {
                                let mut field_cursor = field_child.walk();
                                for pat in field_child.named_children(&mut field_cursor) {
                                    if !is_comment(pat.kind()) && pat.kind() != "identifier" {
                                        extract_and_mark_pattern_bindings(ctx, pat);
                                        break;
                                    } else if pat.kind() == "identifier" {
                                        // Shorthand: { name }
                                        let name: Symbol = node_text(&pat, &ctx.source).into();
                                        ctx.mark_local(name);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "tuple_pattern" => {
            // Recurse on tuple elements
            let mut cursor = pattern.walk();
            for child in pattern.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    extract_and_mark_pattern_bindings(ctx, child);
                }
            }
        }
        _ => {}
    }
}

/// Bind variables from a handler pattern.
fn bind_handler_pattern<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    _block: &mut BlockBuilder<'db>,
    node: Node,
) {
    let mut cursor = node.walk();

    let mut result_name: Option<Symbol> = None;
    let mut has_operation = false;
    let mut args_node: Option<Node> = None;
    let mut continuation_name: Option<Symbol> = None;

    // Parse handler pattern children
    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "identifier" => {
                if has_operation {
                    // After operation, this is the continuation
                    continuation_name = Some(node_text(&child, &ctx.source).into());
                } else if result_name.is_none() {
                    // First identifier without operation is result binding
                    result_name = Some(node_text(&child, &ctx.source).into());
                }
            }
            "path_expression" => {
                has_operation = true;
            }
            "pattern_list" => {
                args_node = Some(child);
            }
            _ => {}
        }
    }

    if has_operation {
        // Suspend pattern: mark args and continuation as local.
        // The actual values are bound later in tribute_to_scf via cont::get_shift_value.
        if let Some(args) = args_node {
            let mut args_cursor = args.walk();
            for arg_child in args.named_children(&mut args_cursor) {
                if is_comment(arg_child.kind()) {
                    continue;
                }
                // Extract binding names from the pattern and mark as local
                extract_and_mark_pattern_bindings(ctx, arg_child);
            }
        }
        if let Some(k_name) = continuation_name {
            // Continuation binding - mark as local for indirect call resolution
            ctx.mark_local(k_name);
        }
    } else if let Some(name) = result_name {
        // Done pattern: mark the result as local binding
        ctx.mark_local(name);
    }
}

/// Convert a handler pattern to a pattern region.
///
/// Handler patterns are for ability effect handling:
/// - `{ result }` - Done pattern, binds the result value
/// - `{ Path::op(args) -> k }` - Suspend pattern, binds args and continuation
fn handler_pattern_to_region<'db>(ctx: &CstLoweringCtx<'db>, node: Node) -> Region<'db> {
    let location = ctx.location(&node);
    let mut cursor = node.walk();

    let mut result_name = None;
    let mut operation_path: Option<Node> = None;
    let mut args_node: Option<Node> = None;
    let mut continuation_name = None;

    // Parse handler pattern children
    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "identifier" => {
                // Could be result binding or continuation
                if operation_path.is_some() {
                    // After operation, this is the continuation
                    continuation_name = Some(node_text(&child, &ctx.source).into());
                } else if result_name.is_none() {
                    // First identifier without operation is result binding
                    result_name = Some(node_text(&child, &ctx.source).into());
                }
            }
            "path_expression" => {
                // This is the operation path (e.g., State::get)
                operation_path = Some(child);
            }
            "pattern_list" => {
                // Arguments to the operation
                args_node = Some(child);
            }
            _ => {}
        }
    }

    // Determine pattern type: Done or Suspend
    if let Some(op_path) = operation_path {
        // Suspend pattern: { Path::op(args) -> k }
        let (ability_ref, op_name) = parse_operation_path(ctx, op_path);

        // Build args pattern region
        let args_region = if let Some(args) = args_node {
            let mut ops = Vec::new();
            let mut args_cursor = args.walk();
            for arg_child in args.named_children(&mut args_cursor) {
                if is_comment(arg_child.kind()) {
                    continue;
                }
                let pat_region = pattern_to_region(ctx, arg_child);
                if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                    ops.push(op);
                }
            }
            ops_to_region(ctx.db, location, ops)
        } else {
            tribute_pat::helpers::empty_region(ctx.db, location)
        };

        // Continuation pattern: bind or wildcard
        let continuation_region = match continuation_name {
            Some(name) => tribute_pat::helpers::bind_region(ctx.db, location, name),
            None => tribute_pat::helpers::wildcard_region(ctx.db, location),
        };

        // If ability_ref is None, use a placeholder for inference
        // The type checker will resolve this later
        let ability_ref = ability_ref
            .unwrap_or_else(|| AbilityRefType::simple(ctx.db, Symbol::new("?")).as_type());

        tribute_pat::helpers::handler_suspend_region(
            ctx.db,
            location,
            ability_ref,
            op_name,
            args_region,
            continuation_region,
        )
    } else {
        // Done pattern: { result }
        let result_region = match result_name {
            Some(name) => tribute_pat::helpers::bind_region(ctx.db, location, name),
            None => tribute_pat::helpers::wildcard_region(ctx.db, location),
        };

        tribute_pat::helpers::handler_done_region(ctx.db, location, result_region)
    }
}

/// Parse an operation path like `State::get` into (ability_ref, op_name).
/// Returns (None, op_name) if the ability should be inferred.
///
/// The ability_ref is returned as a `core.ability_ref` Type to support
/// parameterized abilities like `State(Int)`.
fn parse_operation_path<'db>(ctx: &CstLoweringCtx<'db>, node: Node) -> (Option<Type<'db>>, Symbol) {
    let mut cursor = node.walk();

    // Collect all path components
    let segments: Vec<_> = node
        .named_children(&mut cursor)
        .filter_map(|child| {
            if is_comment(child.kind()) {
                return None;
            }
            match child.kind() {
                "identifier" | "type_identifier" => Some(node_text(&child, &ctx.source)),
                "path_segment" => {
                    let mut inner = child.walk();
                    child
                        .named_children(&mut inner)
                        .find(|c| c.kind() == "identifier" || c.kind() == "type_identifier")
                        .map(|c| node_text(&c, &ctx.source))
                }
                _ => None,
            }
        })
        .collect();

    let path = if segments.is_empty() {
        sym_ref("unknown")
    } else {
        Symbol::from_dynamic(&segments.join("::"))
    };

    // Convert the parent path (ability name) to an AbilityRefType
    // Note: Type parameters will be inferred/resolved later by the type checker
    let ability_ref = path.parent_path().map(|parent| {
        // For now, use just the final component as the ability name
        // Multi-module support would require preserving the full qualified name
        AbilityRefType::simple(ctx.db, parent.last_segment()).as_type()
    });

    (ability_ref, path.last_segment())
}

// =============================================================================
// String/Bytes Interpolation Lowering
// =============================================================================

/// Lower a string interpolation expression.
fn lower_string_interpolation<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let string_ty = core::String::new(ctx.db).as_type();
    let infer_ty = ctx.fresh_type_var();

    let mut cursor = node.walk();
    let mut result: Option<Value<'db>> = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        let value = match child.kind() {
            "string_segment" | "multiline_string_segment" | "raw_interpolated_string_segment" => {
                // Regular string content
                let content = node_text(&child, &ctx.source);
                let op = block.op(adt::string_const(
                    ctx.db,
                    location,
                    string_ty,
                    content.to_string(),
                ));
                Some(op.result(ctx.db))
            }
            "interpolation" | "multiline_interpolation" => {
                // Interpolated expression ${expr}
                let mut interp_cursor = child.walk();
                let mut expr_value = None;
                for interp_child in child.named_children(&mut interp_cursor) {
                    if !is_comment(interp_child.kind()) {
                        expr_value = lower_expr(ctx, block, interp_child);
                        break;
                    }
                }

                // Convert to string using to_string
                expr_value.map(|v| {
                    block
                        .op(tribute::call(
                            ctx.db,
                            location,
                            vec![v],
                            string_ty,
                            sym_ref("to_string"),
                        ))
                        .result(ctx.db)
                })
            }
            _ => lower_expr(ctx, block, child),
        };

        if let Some(v) = value {
            result = Some(match result {
                None => v,
                Some(r) => block
                    .op(tribute::binop(
                        ctx.db,
                        location,
                        r,
                        v,
                        infer_ty,
                        sym("concat"),
                    ))
                    .result(ctx.db),
            });
        }
    }

    result.or_else(|| {
        // Empty string
        let op = block.op(adt::string_const(
            ctx.db,
            location,
            string_ty,
            String::new(),
        ));
        Some(op.result(ctx.db))
    })
}

/// Lower a bytes interpolation expression.
fn lower_bytes_interpolation<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let bytes_ty = core::Bytes::new(ctx.db).as_type();
    let infer_ty = ctx.fresh_type_var();

    let mut cursor = node.walk();
    let mut result: Option<Value<'db>> = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        let value = match child.kind() {
            "bytes_segment" | "multiline_bytes_segment" | "raw_interpolated_bytes_segment" => {
                // Regular bytes content
                let content = node_text(&child, &ctx.source);
                let op = block.op(adt::bytes_const(
                    ctx.db,
                    location,
                    bytes_ty,
                    Attribute::Bytes(content.as_bytes().to_vec()),
                ));
                Some(op.result(ctx.db))
            }
            "bytes_interpolation" | "multiline_bytes_interpolation" => {
                // Interpolated expression ${expr}
                let mut interp_cursor = child.walk();
                let mut expr_value = None;
                for interp_child in child.named_children(&mut interp_cursor) {
                    if !is_comment(interp_child.kind()) {
                        expr_value = lower_expr(ctx, block, interp_child);
                        break;
                    }
                }

                // Convert to bytes using to_bytes
                expr_value.map(|v| {
                    block
                        .op(tribute::call(
                            ctx.db,
                            location,
                            vec![v],
                            bytes_ty,
                            sym_ref("to_bytes"),
                        ))
                        .result(ctx.db)
                })
            }
            _ => lower_expr(ctx, block, child),
        };

        if let Some(v) = value {
            result = Some(match result {
                None => v,
                Some(r) => block
                    .op(tribute::binop(
                        ctx.db,
                        location,
                        r,
                        v,
                        infer_ty,
                        sym("concat"),
                    ))
                    .result(ctx.db),
            });
        }
    }

    result.or_else(|| {
        // Empty bytes
        let op = block.op(adt::bytes_const(
            ctx.db,
            location,
            bytes_ty,
            Attribute::Bytes(vec![]),
        ));
        Some(op.result(ctx.db))
    })
}
