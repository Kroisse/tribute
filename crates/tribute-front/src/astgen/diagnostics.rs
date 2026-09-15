//! CST syntax error collection and source-oriented diagnostics.

use super::{AstLoweringCtx, truncate_token_preview};

/// Recursively collect ERROR and MISSING nodes from the CST and emit parse error diagnostics.
pub(super) fn collect_error_nodes(ctx: &mut AstLoweringCtx<'_>, node: tree_sitter::Node) {
    if node.is_missing() {
        let span = trunk_ir::Span::new(node.start_byte(), node.end_byte());
        ctx.parse_error(span, format!("syntax error: expected '{}'", node.kind()));
        return;
    }

    if node.kind() == "ERROR" {
        let span = trunk_ir::Span::new(node.start_byte(), node.end_byte());
        let text = ctx.node_text(&node);

        // Check for unmatched delimiters first
        if let Some(msg) = detect_unmatched_delimiter(&text) {
            ctx.parse_error(span, msg);
        } else {
            let parent_ctx = node
                .parent()
                .map(|p| describe_parent_context(p.kind()))
                .unwrap_or_default();
            let msg = format!(
                "syntax error: unexpected `{}`{parent_ctx}",
                truncate_token_preview(&text)
            );
            ctx.parse_error(span, msg);
        }
        return; // Don't recurse into ERROR nodes
    }

    // Only recurse if there might be errors below
    if node.has_error() {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect_error_nodes(ctx, child);
        }
    }
}

/// Describe the parent context for error messages.
fn describe_parent_context(parent_kind: &str) -> &'static str {
    match parent_kind {
        "source_file" => "; expected a declaration (fn, struct, enum, ability, mod, or use)",
        "block" => "; expected a statement or expression",
        "function_definition" => " in function definition",
        "struct_declaration" => " in struct declaration",
        "enum_declaration" => " in enum declaration",
        "ability_declaration" => " in ability declaration",
        "parameters" | "param_list" => " in parameter list",
        "arguments" | "arg_list" => " in argument list",
        "type_annotation" => " in type annotation",
        "case_expression" => " in case expression",
        "handle_expression" => " in handle expression",
        _ => "",
    }
}

/// Detect unmatched delimiters in ERROR node text.
///
/// Scans for `(`, `)`, `[`, `]`, `{`, `}` and reports the first
/// delimiter whose matching pair is missing.
fn detect_unmatched_delimiter(text: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            // Skip string literals
            '"' => {
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        chars.next(); // skip escaped char
                    } else if c == '"' {
                        break;
                    }
                }
            }
            // Skip line comments
            '/' if chars.peek() == Some(&'/') => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '(' | '[' | '{' => stack.push(ch),
            ')' | ']' | '}' => {
                let open = match ch {
                    ')' => '(',
                    ']' => '[',
                    '}' => '{',
                    _ => unreachable!(),
                };
                if stack.pop() != Some(open) {
                    return Some(format!("syntax error: unmatched `{ch}`"));
                }
            }
            _ => {}
        }
    }

    // Report the innermost (most recent) unclosed opener
    if let Some(&open) = stack.last() {
        let close = match open {
            '(' => ')',
            '[' => ']',
            '{' => '}',
            _ => unreachable!(),
        };
        return Some(format!(
            "syntax error: unmatched `{open}`, expected `{close}`"
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::detect_unmatched_delimiter;

    #[test]
    fn delimiter_diagnostics_preserve_nesting_and_ignored_text() {
        for (text, expected) in [
            (")", Some("syntax error: unmatched `)`")),
            ("]", Some("syntax error: unmatched `]`")),
            ("}", Some("syntax error: unmatched `}`")),
            ("{]", Some("syntax error: unmatched `]`")),
            ("({[", Some("syntax error: unmatched `[`, expected `]`")),
            ("([]{})", None),
            ("\"}\" // ]\n()", None),
        ] {
            assert_eq!(
                detect_unmatched_delimiter(text).as_deref(),
                expected,
                "{text:?}"
            );
        }
    }
}
