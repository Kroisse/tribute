//! Block and statement lowering.

use tree_sitter::Node;
use tribute_ir::dialect::{adt, list, tribute};
use trunk_ir::{BlockBuilder, Symbol, Value, dialect::arith};

use super::context::CstLoweringCtx;
use super::expressions::{lower_expr, pattern_to_region};
use super::helpers::{is_comment, node_text, sym_ref};

// =============================================================================
// Block and Statement Lowering
// =============================================================================

/// Lower block body statements, returning the last expression value.
pub fn lower_block_body<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let mut last_value = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "let_statement" => {
                lower_let_statement(ctx, block, child);
                last_value = None; // Let doesn't produce a value
            }
            _ => {
                // Try to lower as expression
                if let Some(value) = lower_expr(ctx, block, child) {
                    last_value = Some(value);
                }
            }
        }
    }

    last_value
}

/// Lower a let statement.
///
/// Generates a `src.let` operation with a pattern region for name resolution,
/// and also calls `bind_pattern` to register bindings in the lowering context.
pub fn lower_let_statement<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) {
    let location = ctx.location(&node);

    // Use field-based access
    let Some(pattern_node) = node.child_by_field_name("pattern") else {
        return;
    };
    let Some(value_node) = node.child_by_field_name("value") else {
        return;
    };

    if let Some(value) = lower_expr(ctx, block, value_node) {
        // Generate tribute.let with pattern region for resolver
        let pattern_region = pattern_to_region(ctx, pattern_node);
        block.op(tribute::r#let(ctx.db, location, value, pattern_region));

        // Also bind in context for tirgen's own use during lowering
        bind_pattern(ctx, block, pattern_node, value);
    }
}

/// Bind a pattern to a value, emitting extraction operations as needed.
pub fn bind_pattern<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    block: &mut BlockBuilder<'db>,
    pattern: Node,
    value: Value<'db>,
) {
    let location = ctx.location(&pattern);
    let infer_ty = ctx.fresh_type_var();

    match pattern.kind() {
        "identifier" | "identifier_pattern" => {
            let name = node_text(&pattern, &ctx.source).into();
            ctx.bind(name, value);
        }
        "wildcard_pattern" => {
            // Discard - no binding
        }
        "as_pattern" => {
            // Bind the whole value to the name, then recurse on inner pattern
            // Use field-based access
            if let Some(binding_node) = pattern.child_by_field_name("binding") {
                let name = node_text(&binding_node, &ctx.source).into();
                ctx.bind(name, value);
            }
            if let Some(inner_pattern) = pattern.child_by_field_name("pattern") {
                bind_pattern(ctx, block, inner_pattern, value);
            }
        }
        "literal_pattern" => {
            // Literal patterns in let bindings don't introduce bindings
            // They just assert the value matches - no action needed
        }
        "constructor_pattern" => {
            // Destructure variant: let Some(x) = opt
            let mut cursor = pattern.walk();
            let mut idx = 0;

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "type_identifier" => {
                        // Skip the constructor name
                    }
                    "pattern_list" => {
                        // Positional fields: Some(x, y)
                        let mut list_cursor = child.walk();
                        for pat_child in child.named_children(&mut list_cursor) {
                            if is_comment(pat_child.kind()) {
                                continue;
                            }
                            let field_value = block
                                .op(adt::variant_get(
                                    ctx.db,
                                    location,
                                    value,
                                    infer_ty,
                                    (idx as u64).into(),
                                ))
                                .result(ctx.db);
                            bind_pattern(ctx, block, pat_child, field_value);
                            idx += 1;
                        }
                    }
                    "pattern_fields" => {
                        // Named fields: Point { x, y }
                        let mut fields_cursor = child.walk();
                        for field_child in child.named_children(&mut fields_cursor) {
                            if field_child.kind() == "pattern_field" {
                                let field_value = block
                                    .op(adt::variant_get(
                                        ctx.db,
                                        location,
                                        value,
                                        infer_ty,
                                        (idx as u64).into(),
                                    ))
                                    .result(ctx.db);

                                // Get the pattern inside the field
                                let mut field_cursor = field_child.walk();
                                for pat in field_child.named_children(&mut field_cursor) {
                                    if !is_comment(pat.kind()) && pat.kind() != "identifier" {
                                        bind_pattern(ctx, block, pat, field_value);
                                        break;
                                    } else if pat.kind() == "identifier" {
                                        // Shorthand: { name } means { name: name }
                                        let field_name = node_text(&pat, &ctx.source).into();
                                        ctx.bind(field_name, field_value);
                                        break;
                                    }
                                }
                                idx += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "tuple_pattern" => {
            // Destructure tuple: let #(a, b, c) = tuple
            let mut cursor = pattern.walk();

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "pattern_list" {
                    let mut list_cursor = child.walk();
                    let mut idx = 0;
                    for pat_child in child.named_children(&mut list_cursor) {
                        if is_comment(pat_child.kind()) {
                            continue;
                        }
                        let elem_value = block
                            .op(tribute::call(
                                ctx.db,
                                location,
                                vec![value],
                                infer_ty,
                                sym_ref(&format!("tuple_get_{}", idx)),
                            ))
                            .result(ctx.db);
                        bind_pattern(ctx, block, pat_child, elem_value);
                        idx += 1;
                    }
                }
            }
        }
        "list_pattern" => {
            // Destructure list: let [a, b, ..rest] = list
            let mut cursor = pattern.walk();
            let mut idx = 0;
            let mut rest_pattern = None;

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "rest_pattern" {
                    rest_pattern = Some(child);
                } else {
                    // Regular element pattern
                    let index_value = block
                        .op(arith::Const::i64(ctx.db, location, idx))
                        .result(ctx.db);
                    let elem_ty = ctx.fresh_type_var();
                    let elem_value = block
                        .op(list::get(
                            ctx.db,
                            location,
                            value,
                            index_value,
                            infer_ty,
                            elem_ty,
                        ))
                        .result(ctx.db);
                    bind_pattern(ctx, block, child, elem_value);
                    idx += 1;
                }
            }

            // Handle rest pattern: ..rest binds to list[n..]
            if let Some(rest_node) = rest_pattern {
                let mut rest_cursor = rest_node.walk();
                for rest_child in rest_node.named_children(&mut rest_cursor) {
                    if rest_child.kind() == "identifier" {
                        let rest_name = node_text(&rest_child, &ctx.source).into();
                        let start_value = block
                            .op(arith::Const::i64(ctx.db, location, idx))
                            .result(ctx.db);
                        let len_value = block
                            .op(list::len(ctx.db, location, value, infer_ty))
                            .result(ctx.db);
                        let elem_ty = ctx.fresh_type_var();
                        let rest_value = block
                            .op(list::slice(
                                ctx.db,
                                location,
                                value,
                                start_value,
                                len_value,
                                infer_ty,
                                elem_ty,
                            ))
                            .result(ctx.db);
                        ctx.bind(rest_name, rest_value);
                        break;
                    }
                }
            }
        }
        "handler_pattern" => {
            // Handler patterns: { value } or { Op(args) -> k }
            let location = ctx.location(&pattern);
            let mut handler_cursor = pattern.walk();

            let mut value_name = None;
            let mut has_operation = false;
            let mut args_node = None;
            let mut continuation_name = None;

            for child in pattern.named_children(&mut handler_cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "identifier" if !has_operation && value_name.is_none() => {
                        value_name = Some(node_text(&child, &ctx.source).into());
                    }
                    "type_identifier" | "path_expression" => {
                        has_operation = true;
                    }
                    "pattern_list" => {
                        args_node = Some(child);
                    }
                    "identifier" if has_operation => {
                        continuation_name = Some(node_text(&child, &ctx.source).into());
                    }
                    _ => {}
                }
            }

            // Bind the value (for { result } pattern)
            if let Some(name) = value_name
                && !has_operation
            {
                let bind_op = block.op(tribute::bind(ctx.db, location, ctx.fresh_type_var(), name));
                ctx.bind(name, bind_op.result(ctx.db));
            }

            // Bind operation arguments (for { Op(args) -> k } pattern)
            if let Some(args) = args_node {
                let mut args_cursor = args.walk();
                for arg_child in args.named_children(&mut args_cursor) {
                    if is_comment(arg_child.kind()) {
                        continue;
                    }
                    let arg_bind = block.op(tribute::bind(
                        ctx.db,
                        location,
                        ctx.fresh_type_var(),
                        Symbol::new("_arg"),
                    ));
                    let arg_value = arg_bind.result(ctx.db);
                    bind_pattern(ctx, block, arg_child, arg_value);
                }
            }

            // Bind continuation (for { Op(args) -> k } pattern)
            if let Some(cont_name) = continuation_name {
                let cont_bind = block.op(tribute::bind(
                    ctx.db,
                    location,
                    ctx.fresh_type_var(),
                    cont_name,
                ));
                ctx.bind(cont_name, cont_bind.result(ctx.db));
            }
        }
        _ => {
            // Unknown pattern - try to handle child patterns
            let mut cursor = pattern.walk();
            for child in pattern.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    bind_pattern(ctx, block, child, value);
                    break;
                }
            }
        }
    }
}
