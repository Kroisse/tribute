use tree_sitter::{Language, Parser, Tree, Node};
use crate::ast::{Expr, Identifier};

extern "C" {
    fn tree_sitter_tribute() -> *const tree_sitter::ffi::TSLanguage;
}

fn get_language() -> Language {
    unsafe { tree_sitter::Language::from_raw(tree_sitter_tribute()) }
}

pub struct TreeSitterParser {
    parser: Parser,
}

impl TreeSitterParser {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut parser = Parser::new();
        let language = get_language();
        parser.set_language(&language)?;
        Ok(TreeSitterParser { parser })
    }

    pub fn parse(&mut self, source: &str) -> Result<Vec<(Expr, chumsky::span::SimpleSpan)>, Box<dyn std::error::Error>> {
        let tree = self.parser.parse(source, None)
            .ok_or("Failed to parse")?;
        
        let root_node = tree.root_node();
        let mut expressions = Vec::new();
        
        for i in 0..root_node.child_count() {
            if let Some(child) = root_node.child(i) {
                if let Some((expr, span)) = self.node_to_expr_with_span(child, source) {
                    expressions.push((expr, span));
                }
            }
        }
        
        Ok(expressions)
    }

    fn node_to_expr_with_span(&self, node: Node, source: &str) -> Option<(Expr, chumsky::span::SimpleSpan)> {
        let span = chumsky::span::SimpleSpan::new(node.start_byte(), node.end_byte());
        let expr = self.node_to_expr(node, source)?;
        Some((expr, span))
    }

    fn node_to_expr(&self, node: Node, source: &str) -> Option<Expr> {
        match node.kind() {
            "number" => {
                let text = node.utf8_text(source.as_bytes()).ok()?;
                let num = text.parse::<i64>().ok()?;
                Some(Expr::Number(num))
            }
            "string" => {
                let text = node.utf8_text(source.as_bytes()).ok()?;
                // Remove quotes
                let content = &text[1..text.len()-1];
                Some(Expr::String(content.to_string()))
            }
            "identifier" => {
                let text = node.utf8_text(source.as_bytes()).ok()?;
                Some(Expr::Identifier(text.to_string()))
            }
            "list" => {
                let mut children = Vec::new();
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        if child.kind() == "(" || child.kind() == ")" {
                            continue;
                        }
                        if let Some((expr, span)) = self.node_to_expr_with_span(child, source) {
                            children.push((expr, span));
                        }
                    }
                }
                Some(Expr::List(children))
            }
            _ => None
        }
    }
}