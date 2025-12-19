//! Block and statement lowering.

use tree_sitter::Node;
use tribute_trunk_ir::{
    BlockBuilder, Value,
    dialect::{adt, arith, list, src},
};

use super::context::CstLoweringCtx;
use super::expressions::lower_expr;
use super::helpers::{is_comment, node_text, sym_ref};

// =============================================================================
// Block and Statement Lowering
// =============================================================================

/// Lower block body statements, returning the last expression value.
pub fn lower_block_body<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
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
pub fn lower_let_statement<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) {
    // Use field-based access
    let Some(pattern_node) = node.child_by_field_name("pattern") else {
        return;
    };
    let Some(value_node) = node.child_by_field_name("value") else {
        return;
    };

    if let Some(value) = lower_expr(ctx, block, value_node) {
        bind_pattern(ctx, block, pattern_node, value);
    }
}

/// Bind a pattern to a value, emitting extraction operations as needed.
pub fn bind_pattern<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    pattern: Node,
    value: Value<'db>,
) {
    let location = ctx.location(&pattern);
    let infer_ty = ctx.fresh_type_var();

    match pattern.kind() {
        "identifier" | "identifier_pattern" => {
            let name = node_text(&pattern, ctx.source).to_string();
            ctx.bind(name, value);
        }
        "wildcard_pattern" => {
            // Discard - no binding
        }
        "as_pattern" => {
            // Bind the whole value to the name, then recurse on inner pattern
            // Use field-based access
            if let Some(binding_node) = pattern.child_by_field_name("binding") {
                let name = node_text(&binding_node, ctx.source).to_string();
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
                                        let field_name = node_text(&pat, ctx.source).to_string();
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
                            .op(src::call(
                                ctx.db,
                                location,
                                vec![value],
                                infer_ty,
                                sym_ref(ctx.db, &format!("tuple_get_{}", idx)),
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
                        let rest_name = node_text(&rest_child, ctx.source).to_string();
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
            // Handler patterns are for ability effect handling in match expressions
            // They don't make sense in let bindings - handled in case arms
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
