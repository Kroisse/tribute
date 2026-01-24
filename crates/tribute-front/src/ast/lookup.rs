//! AST node lookup utilities.
//!
//! Provides utilities for finding AST nodes at specific byte offsets,
//! enabling reverse lookup from source positions to AST nodes.

use tree_sitter::Tree;

use super::{NodeId, SpanMap};
use trunk_ir::Span;

/// Result of finding an AST node at a byte offset.
#[derive(Clone, Copy, Debug)]
pub struct AstNodeLookup {
    /// The NodeId of the found AST node.
    pub node_id: NodeId,
    /// The span of the AST node.
    pub span: Span,
}

/// Find the AST node at a given byte offset.
///
/// This performs a reverse lookup from source position to AST node:
/// 1. Uses Tree-sitter to find the CST node at the offset
/// 2. Walks up the tree until finding a node whose ID is in the SpanMap
/// 3. Returns the NodeId and Span of the AST node
///
/// Returns `None` if no AST node contains the offset.
///
/// # Arguments
///
/// * `tree` - The parsed Tree-sitter tree
/// * `span_map` - The SpanMap mapping NodeIds to Spans
/// * `offset` - The byte offset to look up
///
/// # Example
///
/// ```ignore
/// let tree = source.tree(&db)?;
/// let span_map = query::span_map(db, source)?;
/// if let Some(lookup) = find_ast_node_at(&tree, &span_map, cursor_offset) {
///     println!("Found AST node {:?} at {:?}", lookup.node_id, lookup.span);
/// }
/// ```
pub fn find_ast_node_at(tree: &Tree, span_map: &SpanMap, offset: usize) -> Option<AstNodeLookup> {
    // Find the smallest CST node containing the offset
    let mut node = tree.root_node().descendant_for_byte_range(offset, offset)?;

    // Walk up the tree until we find an AST node (one that's in the SpanMap)
    loop {
        let node_id = NodeId::from_cst(&node);
        if span_map.contains(node_id) {
            return Some(AstNodeLookup {
                node_id,
                span: span_map.get_or_default(node_id),
            });
        }
        node = node.parent()?;
    }
}

/// Find all AST nodes containing a given byte offset, from innermost to outermost.
///
/// This is useful when you need to consider multiple enclosing nodes,
/// such as finding both the expression and the containing statement.
pub fn find_ast_nodes_at(tree: &Tree, span_map: &SpanMap, offset: usize) -> Vec<AstNodeLookup> {
    let mut results = Vec::new();

    // Find the smallest CST node containing the offset
    let Some(mut node) = tree.root_node().descendant_for_byte_range(offset, offset) else {
        return results;
    };

    // Walk up the tree, collecting all AST nodes
    loop {
        let node_id = NodeId::from_cst(&node);
        if span_map.contains(node_id) {
            results.push(AstNodeLookup {
                node_id,
                span: span_map.get_or_default(node_id),
            });
        }
        match node.parent() {
            Some(parent) => node = parent,
            None => break,
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::SpanMapBuilder;
    use tree_sitter::Parser;

    fn parse_source(source: &str) -> Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        parser.parse(source, None).expect("Failed to parse")
    }

    #[test]
    fn test_find_ast_node_at_simple() {
        let source = "fn main() { 42 }";
        let tree = parse_source(source);

        // Build a span map with the function definition node
        let mut builder = SpanMapBuilder::new();

        // Find the function_definition node in the tree
        let root = tree.root_node();
        let func_node = root.child(0).expect("Should have function node");
        assert_eq!(func_node.kind(), "function_definition");

        let func_id = NodeId::from_cst(&func_node);
        builder.insert(func_id, Span::new(0, 16));

        let span_map = builder.finish();

        // Looking up any position in the source should find the function
        let result = find_ast_node_at(&tree, &span_map, 0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().node_id, func_id);
    }

    #[test]
    fn test_find_ast_node_at_nested() {
        let source = "fn main() { 42 }";
        let tree = parse_source(source);

        let mut builder = SpanMapBuilder::new();

        // Register both function and the literal
        let root = tree.root_node();
        let func_node = root.child(0).expect("Should have function node");
        let func_id = NodeId::from_cst(&func_node);
        builder.insert(func_id, Span::new(0, 16));

        // Find the int_literal node by walking down the tree
        // Tree-sitter uses "int_literal" or "nat_literal" for integers
        let literal_node = root.descendant_for_byte_range(12, 13);
        let literal_registered = if let Some(node) = literal_node {
            let mut current = node;
            loop {
                if current.kind() == "int_literal" || current.kind() == "nat_literal" {
                    let literal_id = NodeId::from_cst(&current);
                    builder.insert(literal_id, Span::new(12, 14));
                    break true;
                }
                match current.parent() {
                    Some(parent) => current = parent,
                    None => break false,
                }
            }
        } else {
            false
        };

        assert!(
            literal_registered,
            "Should have found and registered int_literal/nat_literal"
        );

        let span_map = builder.finish();

        // Looking up at position 12-14 should find the literal (innermost)
        let result = find_ast_node_at(&tree, &span_map, 12);
        assert!(result.is_some());
        // The span should be for the literal
        let lookup = result.unwrap();
        assert_eq!(lookup.span.start, 12);
        assert_eq!(lookup.span.end, 14);
    }

    #[test]
    fn test_find_ast_nodes_at() {
        let source = "fn main() { 42 }";
        let tree = parse_source(source);

        let mut builder = SpanMapBuilder::new();

        let root = tree.root_node();
        let func_node = root.child(0).expect("Should have function node");
        let func_id = NodeId::from_cst(&func_node);
        builder.insert(func_id, Span::new(0, 16));

        // Find the int_literal node by walking up from position 12
        let literal_node = root.descendant_for_byte_range(12, 13);
        if let Some(node) = literal_node {
            let mut current = node;
            loop {
                if current.kind() == "int_literal" || current.kind() == "nat_literal" {
                    let literal_id = NodeId::from_cst(&current);
                    builder.insert(literal_id, Span::new(12, 14));
                    break;
                }
                match current.parent() {
                    Some(parent) => current = parent,
                    None => break,
                }
            }
        }

        let span_map = builder.finish();

        // Should find both literal (inner) and function (outer)
        let results = find_ast_nodes_at(&tree, &span_map, 12);
        assert_eq!(results.len(), 2);
        // First should be the innermost (literal)
        assert_eq!(results[0].span.start, 12);
        // Second should be the function
        assert_eq!(results[1].span.start, 0);
    }

    #[test]
    fn test_find_ast_node_at_not_found() {
        let source = "fn main() { 42 }";
        let tree = parse_source(source);

        // Empty span map
        let span_map = SpanMapBuilder::new().finish();

        let result = find_ast_node_at(&tree, &span_map, 5);
        assert!(result.is_none());
    }
}
