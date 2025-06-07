use crate::ast::{Expr, SimpleSpan};
use tree_sitter::{Node, Parser};

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

    pub fn parse(
        &mut self,
        source: &str,
    ) -> Result<Vec<(Expr, SimpleSpan)>, Box<dyn std::error::Error>> {
        let tree = self.parser.parse(source, None).ok_or("Failed to parse")?;

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
                let content = &text[1..text.len() - 1];
                match process_escape_sequences(content) {
                    Ok(processed) => Some(Expr::String(processed)),
                    Err(_) => None, // Parsing fails for invalid escape sequences
                }
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
            _ => None,
        }
    }
}

/// Error type for string literal processing
#[derive(Debug, Clone, PartialEq, derive_more::Display, derive_more::Error)]
pub enum StringLiteralError {
    #[display("Invalid UTF-8 sequence in processed string")]
    InvalidUtf8Sequence,
    #[display("Incomplete hex escape sequence")]
    IncompleteHexEscape,
    #[display("Invalid hex digits in escape sequence: {hex_str}")]
    InvalidHexDigits { hex_str: String },
    #[display("Unknown escape sequence: \\{char}")]
    UnknownEscapeSequence { char: char },
    #[display("Trailing backslash in string literal")]
    TrailingBackslash,
}

/// Process escape sequences in a string literal
fn process_escape_sequences(input: &str) -> Result<String, StringLiteralError> {
    let mut result = Vec::<u8>::new();
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => result.extend_from_slice("\"".as_bytes()),
                Some('\\') => result.extend_from_slice("\\".as_bytes()),
                Some('n') | Some('N') => result.extend_from_slice("\n".as_bytes()),
                Some('t') | Some('T') => result.extend_from_slice("\t".as_bytes()),
                Some('r') | Some('R') => result.extend_from_slice("\r".as_bytes()),
                Some('0') => result.push(0),
                Some('x') | Some('X') => {
                    // Hex escape sequence: \xHH
                    let hex1 = chars
                        .next()
                        .ok_or(StringLiteralError::IncompleteHexEscape)?;
                    let hex2 = chars
                        .next()
                        .ok_or(StringLiteralError::IncompleteHexEscape)?;

                    let hex_str = format!("{}{}", hex1, hex2);
                    let byte_value = u8::from_str_radix(&hex_str, 16).map_err(|_| {
                        StringLiteralError::InvalidHexDigits {
                            hex_str: hex_str.clone(),
                        }
                    })?;

                    // Push the raw byte - this might create invalid UTF-8
                    result.push(byte_value);
                }
                Some(other) => {
                    // Unknown escape sequences are now errors
                    return Err(StringLiteralError::UnknownEscapeSequence { char: other });
                }
                None => {
                    // Trailing backslash is now an error
                    return Err(StringLiteralError::TrailingBackslash);
                }
            }
        } else {
            result.extend_from_slice(ch.to_string().as_bytes());
        }
    }

    // Validate that the final result is valid UTF-8
    match String::from_utf8(result) {
        Ok(valid_string) => Ok(valid_string),
        Err(_) => Err(StringLiteralError::InvalidUtf8Sequence),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_escape_sequences_basic() {
        assert_eq!(process_escape_sequences("hello"), Ok("hello".to_string()));
        assert_eq!(process_escape_sequences(""), Ok("".to_string()));
    }

    #[test]
    fn test_process_escape_sequences_quotes() {
        assert_eq!(
            process_escape_sequences(r#"Hello \"World\""#),
            Ok(r#"Hello "World""#.to_string())
        );
        assert_eq!(process_escape_sequences(r#"\""#), Ok(r#"""#.to_string()));
    }

    #[test]
    fn test_process_escape_sequences_backslash() {
        assert_eq!(
            process_escape_sequences(r"C:\\Users\\name"),
            Ok(r"C:\Users\name".to_string())
        );
        assert_eq!(process_escape_sequences(r"\\"), Ok(r"\".to_string()));
    }

    #[test]
    fn test_process_escape_sequences_whitespace() {
        assert_eq!(
            process_escape_sequences(r"Line 1\nLine 2"),
            Ok("Line 1\nLine 2".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Tab\there"),
            Ok("Tab\there".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Carriage\rReturn"),
            Ok("Carriage\rReturn".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_case_insensitive() {
        // Test uppercase escape sequences
        assert_eq!(
            process_escape_sequences(r"Line 1\NLine 2"),
            Ok("Line 1\nLine 2".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Tab\There"),
            Ok("Tab\there".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Carriage\RReturn"),
            Ok("Carriage\rReturn".to_string())
        );

        // Test mixed case
        assert_eq!(
            process_escape_sequences(r"Mixed\n\T\r\N"),
            Ok("Mixed\n\t\r\n".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_null() {
        assert_eq!(
            process_escape_sequences(r"Null\0char"),
            Ok("Null\0char".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_unknown() {
        // Unknown escape sequences should now be errors
        assert_eq!(
            process_escape_sequences(r"Unknown\z escape"),
            Err(StringLiteralError::UnknownEscapeSequence { char: 'z' })
        );
        assert_eq!(
            process_escape_sequences(r"\z"),
            Err(StringLiteralError::UnknownEscapeSequence { char: 'z' })
        );
    }

    #[test]
    fn test_process_escape_sequences_trailing_backslash() {
        assert_eq!(
            process_escape_sequences(r"trailing\"),
            Err(StringLiteralError::TrailingBackslash)
        );
    }

    #[test]
    fn test_process_escape_sequences_mixed() {
        assert_eq!(
            process_escape_sequences(r#"Mixed: \"quote\" and \\backslash\n"#),
            Ok("Mixed: \"quote\" and \\backslash\n".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_hex() {
        // Valid hex escapes
        assert_eq!(process_escape_sequences(r"\x41"), Ok("A".to_string()));
        assert_eq!(
            process_escape_sequences(r"Hello\x20World"),
            Ok("Hello World".to_string())
        );

        // Invalid hex escapes should return errors
        assert_eq!(
            process_escape_sequences(r"\x4"),
            Err(StringLiteralError::IncompleteHexEscape)
        );
        assert_eq!(
            process_escape_sequences(r"\xGG"),
            Err(StringLiteralError::InvalidHexDigits {
                hex_str: "GG".to_string()
            })
        );
        assert_eq!(
            process_escape_sequences(r"\x"),
            Err(StringLiteralError::IncompleteHexEscape)
        );
    }

    #[test]
    fn test_process_escape_sequences_invalid_utf8() {
        // Create invalid UTF-8 sequences using hex escapes
        // 0xFF is not valid UTF-8
        assert_eq!(
            process_escape_sequences(r"\xFF"),
            Err(StringLiteralError::InvalidUtf8Sequence)
        );

        // 0x80 without proper leading byte is invalid UTF-8
        assert_eq!(
            process_escape_sequences(r"\x80"),
            Err(StringLiteralError::InvalidUtf8Sequence)
        );

        // Valid UTF-8 should work
        assert_eq!(
            process_escape_sequences(r"\x41\x42\x43"),
            Ok("ABC".to_string())
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
