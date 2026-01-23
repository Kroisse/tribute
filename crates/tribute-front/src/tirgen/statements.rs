//! Block and statement lowering.

use tree_sitter::Node;
use tribute_ir::dialect::tribute;
use trunk_ir::{BlockBuilder, Value};

use super::context::CstLoweringCtx;
use super::expressions::{collect_pattern_bindings, lower_expr, pattern_to_region};
use super::helpers::is_comment;

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
            "statement" | "expression_statement" => {
                let mut stmt_cursor = child.walk();
                let inner = child
                    .named_children(&mut stmt_cursor)
                    .find(|n| !is_comment(n.kind()));
                if let Some(inner) = inner {
                    if inner.kind() == "let_statement" {
                        lower_let_statement(ctx, block, inner);
                        last_value = None;
                    } else if let Some(value) = lower_expr(ctx, block, inner) {
                        last_value = Some(value);
                    }
                }
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
/// Generates a `tribute.let` operation that returns one result per binding.
/// The downstream pass (tribute_to_scf) extracts values from the input and
/// produces the actual result values.
///
/// Example:
/// ```text
/// let #(x, y) = pair;
/// // generates:
/// %x, %y = tribute.let %pair {
///     tribute_pat.tuple { tribute_pat.bind("x"), tribute_pat.bind("y") }
/// }
/// ```
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

    let Some(value) = lower_expr(ctx, block, value_node) else {
        return;
    };

    // 1. Collect bindings from pattern
    let bindings = collect_pattern_bindings(ctx, pattern_node);

    // 2. Generate result types (one type variable per binding)
    let result_types: Vec<_> = bindings.iter().map(|_| ctx.fresh_type_var()).collect();

    // 3. Generate tribute.let with pattern region and result types
    let pattern_region = pattern_to_region(ctx, pattern_node);
    let let_op = block.op(tribute::r#let(
        ctx.db,
        location,
        value,
        result_types.iter().copied(),
        pattern_region,
    ));

    // 4. Bind each result to its corresponding name
    for (i, binding) in bindings.iter().enumerate() {
        let result_value = let_op.result(ctx.db, i);
        ctx.bind(binding.name, result_value);
    }
}
