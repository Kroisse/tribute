pub mod ast;
pub mod eval;
pub mod parser;
pub mod tree_sitter_parser;

pub use crate::parser::parse as parse_chumsky;
pub use crate::tree_sitter_parser::TreeSitterParser;

// Default to tree-sitter parser
pub fn parse(source: &str) -> Vec<(ast::Expr, chumsky::span::SimpleSpan)> {
    let mut parser = TreeSitterParser::new().expect("Failed to create tree-sitter parser");
    parser.parse(source).expect("Failed to parse source")
}
