use tree_sitter::{Parser, Node};
use crate::ast::{Expr, SimpleSpan};

pub struct TributeParser {
    parser: Parser,
}

impl TributeParser {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut parser = Parser::new();
        let language = tree_sitter_tribute::language();
        parser.set_language(&language)?;
        Ok(TributeParser { parser })
    }

    pub fn parse(&mut self, source: &str) -> Result<Vec<(Expr, SimpleSpan)>, Box<dyn std::error::Error>> {
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

    fn node_to_expr_with_span(&self, node: Node, source: &str) -> Option<(Expr, SimpleSpan)> {
        let span = SimpleSpan::new(node.start_byte(), node.end_byte());
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