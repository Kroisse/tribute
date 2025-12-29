#![allow(clippy::collapsible_if)]

//! Expression lowering.

use tracing::trace;
use tree_sitter::Node;
use trunk_ir::{
    Attribute, Block, BlockBuilder, BlockId, DialectOp, DialectType, IdVec, Operation,
    QualifiedName, Region, Symbol, Type, Value,
    dialect::{
        ability, adt, arith, case,
        core::{self, AbilityRefType},
        list, pat, src,
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
            // Always create src.var for variable references, even for local bindings.
            // This preserves the source span for hover. Resolution will transform
            // local references to identity operations with the correct type.
            let name = node_text(&node, &ctx.source);
            let op = block.op(src::var(ctx.db, location, infer_ty, name.into()));
            Some(op.result(ctx.db))
        }
        "path_expression" => {
            let mut cursor = node.walk();
            let path = node
                .named_children(&mut cursor)
                .filter(|n| n.kind() == "identifier" || n.kind() == "type_identifier")
                .map(|n| Symbol::from(node_text(&n, &ctx.source)))
                .collect::<Option<_>>()?;
            let op = block.op(src::path(ctx.db, location, infer_ty, path));
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
            // String concatenation - use src.binop
            block
                .op(src::binop(
                    ctx.db,
                    location,
                    lhs,
                    rhs,
                    infer_ty,
                    sym("concat"),
                ))
                .result(ctx.db)
        }
        _ => {
            // Unknown operator - emit as src.binop
            block
                .op(src::binop(
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
    let func_path: QualifiedName = match func_node.kind() {
        "identifier" => sym_ref(&node_text(&func_node, &ctx.source)),
        "path_expression" => {
            let mut cursor = func_node.walk();
            func_node
                .named_children(&mut cursor)
                .filter(|n| n.kind() == "identifier" || n.kind() == "type_identifier")
                .map(|n| node_text(&n, &ctx.source).into())
                .collect::<Option<_>>()?
        }
        _ => return None,
    };

    // Arguments need iteration (not a field)
    let args = collect_argument_list(ctx, block, node);

    let op = block.op(src::call(ctx.db, location, args, infer_ty, func_path));
    Some(op.result(ctx.db))
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
    let ctor_path: QualifiedName = match ctor_node.kind() {
        "type_identifier" => sym_ref(&node_text(&ctor_node, &ctx.source)),
        "path_expression" => {
            let mut cursor = ctor_node.walk();
            ctor_node
                .named_children(&mut cursor)
                .filter(|n| n.kind() == "identifier" || n.kind() == "type_identifier")
                .map(|n| node_text(&n, &ctx.source).into())
                .collect::<Option<_>>()?
        }
        _ => return None,
    };

    let args = collect_argument_list(ctx, block, node);
    let op = block.op(src::cons(ctx.db, location, args, infer_ty, ctor_path));
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
    let method_path: QualifiedName = match method_node.kind() {
        "identifier" | "type_identifier" => {
            // Simple method name: p.x()
            let name = node_text(&method_node, &ctx.source);
            std::iter::once(Symbol::from(name)).collect::<Option<_>>()?
        }
        _ => {
            // Qualified path: p.math::double()
            let mut cursor = method_node.walk();
            method_node
                .named_children(&mut cursor)
                .filter_map(|n| match n.kind() {
                    "identifier" | "type_identifier" => Some(node_text(&n, &ctx.source).into()),
                    "path_segment" => {
                        // path_segment contains an identifier or type_identifier
                        let mut inner = n.walk();
                        n.named_children(&mut inner)
                            .find(|c| c.kind() == "identifier" || c.kind() == "type_identifier")
                            .map(|c| node_text(&c, &ctx.source).into())
                    }
                    _ => None,
                })
                .collect::<Option<_>>()?
        }
    };

    // Lower receiver first
    let receiver = lower_expr(ctx, block, receiver_node)?;

    // Arguments need iteration (not a field)
    let args = collect_argument_list(ctx, block, node);

    // UFCS: x.f(y, z) → f(x, y, z)
    let mut all_args = vec![receiver];
    all_args.extend(args);

    let op = block.op(src::call(ctx.db, location, all_args, infer_ty, method_path));
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

    // Build lambda body
    let param_types: IdVec<Type<'_>> = std::iter::repeat_n(infer_ty, param_names.len()).collect();
    let result_type = infer_ty;
    let mut body_block = BlockBuilder::new(ctx.db, location).args(param_types.clone());

    let result_value = ctx.scoped(|ctx| {
        // Bind parameters
        for param_name in param_names {
            let param_value = body_block.op(src::var(ctx.db, location, infer_ty, param_name));
            ctx.bind(param_name, param_value.result(ctx.db));
        }

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(src::r#yield(ctx.db, location, result_value));

    let effect_type = ctx.fresh_effect_row_type();
    let func_type =
        core::Func::with_effect(ctx.db, param_types, result_type, Some(effect_type)).as_type();
    let region = Region::new(ctx.db, location, idvec![body_block.build()]);
    let lambda_op = block.op(src::lambda(ctx.db, location, infer_ty, func_type, region));
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

    let op = block.op(case::r#case(
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
) -> Option<case::Arm<'db>> {
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

    // Check if pattern is a handler_pattern (for ability effect handling)
    let is_handler_pattern = find_handler_pattern(pattern_node).is_some();

    let result_value = ctx.scoped(|ctx| {
        // Bind pattern - use appropriate binding function based on pattern type
        if is_handler_pattern {
            bind_handler_pattern(ctx, &mut body_block, pattern_node, scrutinee);
        } else {
            bind_pattern(ctx, &mut body_block, pattern_node, scrutinee);
        }

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(case::r#yield(ctx.db, location, result_value));

    let pattern_region = pattern_to_region(ctx, pattern_node);
    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    Some(case::arm(ctx.db, location, pattern_region, body_region))
}

/// Find a handler_pattern node within a pattern (unwrapping wrapper nodes).
fn find_handler_pattern(node: Node) -> Option<Node> {
    match node.kind() {
        "handler_pattern" => Some(node),
        "pattern" | "simple_pattern" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if let Some(found) = find_handler_pattern(child) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

/// Convert a pattern node to a pattern region for case arms and let bindings.
///
/// Creates a region containing pattern operations from the `pat` dialect.
pub fn pattern_to_region<'db>(ctx: &CstLoweringCtx<'db>, node: Node) -> Region<'db> {
    let location = ctx.location(&node);

    match node.kind() {
        // Wrapper nodes - unwrap and recurse
        "pattern" | "simple_pattern" => {
            let mut cursor = node.walk();
            if let Some(child) = node.named_children(&mut cursor).next() {
                return pattern_to_region(ctx, child);
            }
            pat::helpers::wildcard_region(ctx.db, location)
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
            pat::helpers::bind_region(ctx.db, location, name.into())
        }
        "wildcard_pattern" => pat::helpers::wildcard_region(ctx.db, location),
        "literal_pattern" => {
            // Get the literal value
            let mut cursor = node.walk();
            if let Some(child) = node.named_children(&mut cursor).next() {
                pattern_to_region(ctx, child)
            } else {
                pat::helpers::wildcard_region(ctx.db, location)
            }
        }
        "nat_literal" | "int_literal" => {
            if let Some(n) = parse_int_literal(&node_text(&node, &ctx.source)) {
                pat::helpers::int_region(ctx.db, location, n)
            } else {
                pat::helpers::wildcard_region(ctx.db, location)
            }
        }
        "true" | "keyword_true" => pat::helpers::bool_region(ctx.db, location, true),
        "false" | "keyword_false" => pat::helpers::bool_region(ctx.db, location, false),
        "nil" | "keyword_nil" => {
            let op = pat::literal(ctx.db, location, Attribute::Unit);
            pat::helpers::single_op_region(ctx.db, location, op.as_operation())
        }
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(node, &ctx.source);
            pat::helpers::string_region(ctx.db, location, &s)
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
            let variant_path = QualifiedName::simple(name);
            let fields_region = ops_to_region(ctx.db, location, field_ops);
            pat::helpers::variant_region(ctx.db, location, variant_path, fields_region)
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
            pat::helpers::tuple_region(ctx.db, location, elements_region)
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
                pat::helpers::list_rest_region(ctx.db, location, name, head_region)
            } else {
                let elements_region = ops_to_region(ctx.db, location, elem_ops);
                pat::helpers::list_region(ctx.db, location, elements_region)
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

            let inner =
                inner_region.unwrap_or_else(|| pat::helpers::wildcard_region(ctx.db, location));
            let name = binding_name.unwrap_or_else(|| Symbol::new("_"));
            // Create as_pat operation with inner region
            let as_op = pat::as_pat(ctx.db, location, name, inner);
            pat::helpers::single_op_region(ctx.db, location, as_op.as_operation())
        }
        "handler_pattern" => {
            // Handler patterns are for ability effect handling in case expressions
            // Delegate to the specialized handler pattern converter
            handler_pattern_to_region(ctx, node)
        }
        _ => pat::helpers::wildcard_region(ctx.db, location),
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
    body_block.op(src::r#yield(ctx.db, location, result_value));

    let region = Region::new(ctx.db, location, idvec![body_block.build()]);
    let block_op = block.op(src::block(ctx.db, location, infer_ty, region));
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

    let tuple_op = block.op(src::tuple(ctx.db, location, elements, infer_ty));
    Some(tuple_op.result(ctx.db))
}

/// Lower a record expression.
fn lower_record_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut type_name = None;
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
                    // record_field has: identifier (field name) and optionally an expression (value)
                    // For `x: 10`, there's identifier("x") and literal(10)
                    // For shorthand `{ x }`, there's only identifier("x")
                    let mut field_cursor = field.walk();
                    let children: Vec<_> = field
                        .named_children(&mut field_cursor)
                        .filter(|c| !is_comment(c.kind()))
                        .collect();

                    if children.len() == 1 && children[0].kind() == "identifier" {
                        // Shorthand: { name } means { name: name }
                        let field_name = node_text(&children[0], &ctx.source).into();
                        if let Some(value) = ctx.lookup(field_name) {
                            field_values.push(value);
                        } else {
                            let var_op = block.op(src::var(ctx.db, location, infer_ty, field_name));
                            field_values.push(var_op.result(ctx.db));
                        }
                    } else {
                        // Full syntax: { name: value }
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
    let op = block.op(src::cons(
        ctx.db,
        location,
        field_values,
        infer_ty,
        sym_ref(&type_name),
    ));
    Some(op.result(ctx.db))
}

// =============================================================================
// Ability Handling Expression Lowering
// =============================================================================

/// Lower a handle expression (ability handling).
///
/// Source: `handle expr`
/// Lowers to:
/// ```text
/// %request = ability.prompt { expr }
/// ```
///
/// The `handle expr` expression returns a Request value that can be pattern-matched
/// to handle both normal completion (`{ result }`) and suspended effects
/// (`{ State::get() -> k }`).
///
/// Typically used with case expressions:
/// ```text
/// case handle expr {
///     { result } -> ...
///     { State::get() -> k } -> ...
/// }
/// ```
fn lower_handle_expr<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let request_ty = ctx.fresh_type_var(); // Type for Request value

    // Find the expression to handle
    let mut expr_node = None;
    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) || child.kind() == "keyword_handle" {
            continue;
        }
        expr_node = Some(child);
        break;
    }

    let expr_node = expr_node?;

    // Build body region with the expression
    let mut body_block = BlockBuilder::new(ctx.db, location);
    let result_value = ctx.scoped(|ctx| lower_expr(ctx, &mut body_block, expr_node));

    if let Some(value) = result_value {
        body_block.op(src::r#yield(ctx.db, location, value));
    }

    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    // Create ability.prompt to run body and get Request
    let prompt_op = block.op(ability::prompt(ctx.db, location, request_ty, body_region));
    Some(prompt_op.request(ctx.db))
}

/// Bind handler pattern variables.
fn bind_handler_pattern<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
    request: Value<'db>,
) {
    let location = ctx.location(&node);
    let mut cursor = node.walk();

    match node.kind() {
        // Unwrap wrapper nodes
        "pattern" | "simple_pattern" => {
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    bind_handler_pattern(ctx, block, child, request);
                    return;
                }
            }
        }
        "handler_pattern" => {
            // Handler patterns: { value } or { Op(args) -> k }
            let mut value_name = None;
            let mut op_name = None;
            let mut continuation_name = None;

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "identifier" if op_name.is_none() && value_name.is_none() => {
                        // Could be value binding or operation name
                        let name = node_text(&child, &ctx.source).into();
                        // Check if this is followed by -> (continuation)
                        // For now, treat single identifier as value binding
                        value_name = Some(name);
                    }
                    "type_identifier" => {
                        // This is an ability operation name
                        op_name = Some(node_text(&child, &ctx.source));
                    }
                    "identifier" if op_name.is_some() => {
                        // This is the continuation binding
                        continuation_name = Some(node_text(&child, &ctx.source).into());
                    }
                    _ => {}
                }
            }

            // Bind the value (for { result } pattern)
            if let Some(name) = value_name
                && op_name.is_none()
            {
                // Simple value binding: { result }
                // The request's Done payload is bound to name
                let bind_op = block.op(case::bind(ctx.db, location, ctx.fresh_type_var(), name));
                ctx.bind(name, bind_op.result(ctx.db));
            }

            // Bind continuation (for { Op(args) -> k } pattern)
            if let Some(cont_name) = continuation_name {
                let cont_bind = block.op(case::bind(
                    ctx.db,
                    location,
                    ctx.fresh_type_var(),
                    cont_name,
                ));
                ctx.bind(cont_name, cont_bind.result(ctx.db));
            }

            // TODO: Bind operation arguments
            let _ = request; // Will be used for extracting payload
        }
        _ => {
            // Unknown pattern type
        }
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
            pat::helpers::empty_region(ctx.db, location)
        };

        // Continuation name (empty Symbol for wildcard/discard)
        let cont_symbol = continuation_name.unwrap_or_else(|| Symbol::new("_"));

        // If ability_ref is None, use a placeholder for inference
        // The type checker will resolve this later
        let ability_ref = ability_ref
            .unwrap_or_else(|| AbilityRefType::simple(ctx.db, Symbol::new("?")).as_type());

        pat::helpers::handler_suspend_region(
            ctx.db,
            location,
            ability_ref,
            op_name,
            args_region,
            cont_symbol,
        )
    } else {
        // Done pattern: { result }
        let result_region = match result_name {
            Some(name) => pat::helpers::bind_region(ctx.db, location, name),
            None => pat::helpers::wildcard_region(ctx.db, location),
        };

        pat::helpers::handler_done_region(ctx.db, location, result_region)
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
    let path = node
        .named_children(&mut cursor)
        .filter_map(|child| {
            if is_comment(child.kind()) {
                return None;
            }
            (child.kind() == "identifier" || child.kind() == "type_identifier")
                .then(|| Symbol::from_dynamic(&node_text(&child, &ctx.source)))
        })
        .collect::<Option<QualifiedName>>()
        .unwrap_or_else(|| sym_ref("unknown"));

    // Convert the parent path (ability name) to an AbilityRefType
    // Note: Type parameters will be inferred/resolved later by the type checker
    let ability_ref = path.to_parent().map(|parent| {
        // For now, use just the final component as the ability name
        // Multi-module support would require preserving the full qualified name
        AbilityRefType::simple(ctx.db, parent.name()).as_type()
    });

    (ability_ref, path.name())
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
                        .op(src::call(
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
                    .op(src::binop(ctx.db, location, r, v, infer_ty, sym("concat")))
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
                        .op(src::call(
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
                    .op(src::binop(ctx.db, location, r, v, infer_ty, sym("concat")))
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
