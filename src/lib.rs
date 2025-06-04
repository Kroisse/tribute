pub mod ast;
pub mod eval;
pub mod parser;

pub use crate::parser::TributeParser;

// Parse using tree-sitter
pub fn parse(source: &str) -> Vec<(ast::Expr, ast::SimpleSpan)> {
    let mut parser = TributeParser::new().expect("Failed to create parser");
    parser.parse(source).expect("Failed to parse source")
}
