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
                // Remove quotes and process escape sequences
                let content = &text[1..text.len()-1];
                let processed = process_escape_sequences(content)?;
                Some(Expr::String(processed))
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

/// Process escape sequences in a string literal
fn process_escape_sequences(input: &str) -> Option<String> {
    let mut result = String::new();
    let mut chars = input.chars();
    
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('0') => result.push('\0'),
                Some(other) => {
                    // For unknown escape sequences, preserve the backslash and character
                    result.push('\\');
                    result.push(other);
                }
                None => {
                    // Trailing backslash - preserve it
                    result.push('\\');
                }
            }
        } else {
            result.push(ch);
        }
    }
    
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_escape_sequences_basic() {
        assert_eq!(process_escape_sequences("hello"), Some("hello".to_string()));
        assert_eq!(process_escape_sequences(""), Some("".to_string()));
    }

    #[test]
    fn test_process_escape_sequences_quotes() {
        assert_eq!(
            process_escape_sequences(r#"Hello \"World\""#),
            Some(r#"Hello "World""#.to_string())
        );
        assert_eq!(
            process_escape_sequences(r#"\""#),
            Some(r#"""#.to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_backslash() {
        assert_eq!(
            process_escape_sequences(r"C:\\Users\\name"),
            Some(r"C:\Users\name".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"\\"),
            Some(r"\".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_whitespace() {
        assert_eq!(
            process_escape_sequences(r"Line 1\nLine 2"),
            Some("Line 1\nLine 2".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Tab\there"),
            Some("Tab\there".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Carriage\rReturn"),
            Some("Carriage\rReturn".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_null() {
        assert_eq!(
            process_escape_sequences(r"Null\0char"),
            Some("Null\0char".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_unknown() {
        // Unknown escape sequences should be preserved as-is
        assert_eq!(
            process_escape_sequences(r"Unknown\x escape"),
            Some(r"Unknown\x escape".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"\z"),
            Some(r"\z".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_trailing_backslash() {
        assert_eq!(
            process_escape_sequences(r"trailing\"),
            Some(r"trailing\".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_mixed() {
        assert_eq!(
            process_escape_sequences(r#"Mixed: \"quote\" and \\backslash\n"#),
            Some("Mixed: \"quote\" and \\backslash\n".to_string())
        );
    }

    #[test]
    fn test_string_parsing_with_escape_sequences() {
        let mut parser = TributeParser::new().unwrap();
        
        // Test basic quote escaping
        let result = parser.parse(r#""Hello \"World\"""#).unwrap();
        assert_eq!(result.len(), 1);
        if let Expr::String(s) = &result[0].0 {
            assert_eq!(s, "Hello \"World\"");
        } else {
            panic!("Expected string expression");
        }

        // Test mixed escape sequences
        let result = parser.parse(r#""Line1\nTab\tQuote\"""#).unwrap();
        assert_eq!(result.len(), 1);
        if let Expr::String(s) = &result[0].0 {
            assert_eq!(s, "Line1\nTab\tQuote\"");
        } else {
            panic!("Expected string expression");
        }
    }
}