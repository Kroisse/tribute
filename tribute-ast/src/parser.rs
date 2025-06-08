use crate::ast::*;
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

    pub fn parse_internal<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        source: &'db str,
    ) -> Result<Program<'db>, Box<dyn std::error::Error>> {
        let tree = self.parser.parse(source, None).ok_or("Failed to parse")?;

        let root_node = tree.root_node();
        let mut items = Vec::new();

        for i in 0..root_node.child_count() {
            if let Some(child) = root_node.child(i) {
                // Skip comments and whitespace
                if child.kind() == "line_comment"
                    || child.kind() == "block_comment"
                    || child.kind() == "ERROR"
                {
                    continue;
                }
                match self.node_to_item(db, child, source) {
                    Ok(item) => items.push(item),
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(Program::new(db, items))
    }
}

#[salsa::tracked]
pub fn parse_source_file<'db>(db: &'db dyn salsa::Database, source: crate::SourceFile) -> Program<'db> {
    let mut parser = match TributeParser::new() {
        Ok(p) => p,
        Err(_) => return Program::new(db, Vec::new()),
    };

    parser
        .parse_internal(db, source.text(db))
        .unwrap_or_else(|_| Program::new(db, Vec::new()))
}

impl TributeParser {
    fn node_to_item<'db>(
        &self,
        db: &'db dyn salsa::Database,
        node: Node,
        source: &'db str,
    ) -> Result<Item<'db>, Box<dyn std::error::Error>> {
        match node.kind() {
            "function_definition" => {
                let mut name = None;
                let mut parameters = Vec::new();
                let mut body = None;

                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        match child.kind() {
                            "identifier" => {
                                if name.is_none() {
                                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                                }
                            }
                            "parameter_list" => {
                                parameters = self.parse_parameter_list(child, source)?;
                            }
                            "block" => {
                                body = Some(self.parse_block(child, source)?);
                            }
                            _ => {} // Skip other tokens like 'fn', '(', ')'
                        }
                    }
                }

                let name = name.ok_or("Missing function name")?;
                let body = body.ok_or("Missing function body")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Function(FunctionDefinition::new(db, name, parameters, body, span)),
                    span,
                ))
            }
            _ => Err(format!("Unknown item kind: {}", node.kind()).into()),
        }
    }

    fn parse_parameter_list(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Identifier>, Box<dyn std::error::Error>> {
        let mut parameters = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "identifier" {
                    parameters.push(child.utf8_text(source.as_bytes())?.to_string());
                }
            }
        }

        Ok(parameters)
    }

    fn parse_block(&self, node: Node, source: &str) -> Result<Block, Box<dyn std::error::Error>> {
        let mut statements = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "let_statement" => {
                        statements.push(Statement::Let(self.parse_let_statement(child, source)?));
                    }
                    "expression_statement" => {
                        statements.push(Statement::Expression(
                            self.parse_expression_statement(child, source)?,
                        ));
                    }
                    "line_comment" | "block_comment" | "{" | "}" => {
                        // Skip comments and block delimiters
                    }
                    _ => {
                        // Try to parse as expression statement
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            statements.push(Statement::Expression(expr));
                        }
                    }
                }
            }
        }

        Ok(Block { statements })
    }

    fn parse_let_statement(
        &self,
        node: Node,
        source: &str,
    ) -> Result<LetStatement, Box<dyn std::error::Error>> {
        let mut name = None;
        let mut value = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => {
                        if name.is_none() {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                    }
                    _ => {
                        // Try to parse as expression
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            value = Some(expr);
                        }
                    }
                }
            }
        }

        let name = name.ok_or("Missing let variable name")?;
        let value = value.ok_or("Missing let value")?;

        Ok(LetStatement { name, value })
    }

    fn parse_expression_statement(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Spanned<Expr>, Box<dyn std::error::Error>> {
        // expression_statement should have one child which is the expression
        if let Some(child) = node.child(0) {
            self.node_to_expr_with_span(child, source)
        } else {
            Err("Empty expression statement".into())
        }
    }

    fn node_to_expr_with_span(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Spanned<Expr>, Box<dyn std::error::Error>> {
        let span = Span::new(node.start_byte(), node.end_byte());
        let expr = self.node_to_expr(node, source)?;
        Ok((expr, span))
    }

    fn node_to_expr(&self, node: Node, source: &str) -> Result<Expr, Box<dyn std::error::Error>> {
        match node.kind() {
            "number" => {
                let text = node.utf8_text(source.as_bytes())?;
                let num = text.parse::<i64>()?;
                Ok(Expr::Number(num))
            }
            "string" => {
                // Check if this is an interpolated string by looking for child nodes
                let has_interpolation = (0..node.child_count())
                    .any(|i| node.child(i).is_some_and(|child| child.kind() == "interpolation"));
                
                if has_interpolation {
                    self.parse_interpolated_string(node, source)
                } else {
                    let text = node.utf8_text(source.as_bytes())?;
                    // Remove quotes and process escape sequences
                    let content = &text[1..text.len() - 1];
                    let processed = process_escape_sequences(content)?;
                    Ok(Expr::String(processed))
                }
            }
            "identifier" => {
                let text = node.utf8_text(source.as_bytes())?;
                Ok(Expr::Identifier(text.to_string()))
            }
            "binary_expression" => self.parse_binary_expression(node, source),
            "call_expression" => self.parse_call_expression(node, source),
            "match_expression" => self.parse_match_expression(node, source),
            "primary_expression" => {
                // primary_expression should have one child
                if let Some(child) = node.child(0) {
                    self.node_to_expr(child, source)
                } else {
                    Err("Empty primary expression".into())
                }
            }
            "parenthesized_expression" => {
                // Find the expression inside parentheses
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        if child.kind() != "(" && child.kind() != ")" {
                            return self.node_to_expr(child, source);
                        }
                    }
                }
                Err("Empty parenthesized expression".into())
            }
            _ => Err(format!("Unknown expression kind: {}", node.kind()).into()),
        }
    }

    fn parse_binary_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut left = None;
        let mut operator = None;
        let mut right = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "+" => operator = Some(BinaryOperator::Add),
                    "-" => operator = Some(BinaryOperator::Subtract),
                    "*" => operator = Some(BinaryOperator::Multiply),
                    "/" => operator = Some(BinaryOperator::Divide),
                    _ => {
                        // Try to parse as expression
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            if left.is_none() {
                                left = Some(Box::new(expr));
                            } else if right.is_none() {
                                right = Some(Box::new(expr));
                            }
                        }
                    }
                }
            }
        }

        let left = left.ok_or("Missing left operand")?;
        let operator = operator.ok_or("Missing operator")?;
        let right = right.ok_or("Missing right operand")?;

        Ok(Expr::Binary(BinaryExpression {
            left,
            operator,
            right,
        }))
    }

    fn parse_call_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut function = None;
        let mut arguments = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => {
                        if function.is_none() {
                            function = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                    }
                    "argument_list" => {
                        arguments = self.parse_argument_list(child, source)?;
                    }
                    _ => {} // Skip other tokens like '(', ')'
                }
            }
        }

        let function = function.ok_or("Missing function name")?;

        Ok(Expr::Call(CallExpression {
            function,
            arguments,
        }))
    }

    fn parse_argument_list(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Spanned<Expr>>, Box<dyn std::error::Error>> {
        let mut arguments = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() != "," {
                    if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                        arguments.push(expr);
                    }
                }
            }
        }

        Ok(arguments)
    }

    fn parse_match_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut value = None;
        let mut arms = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "match_arm" => {
                        arms.push(self.parse_match_arm(child, source)?);
                    }
                    "match" | "{" | "}" => {
                        // Skip keywords and delimiters
                    }
                    _ => {
                        // Try to parse as the value expression
                        if value.is_none() {
                            if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                                value = Some(Box::new(expr));
                            }
                        }
                    }
                }
            }
        }

        let value = value.ok_or("Missing match value")?;

        Ok(Expr::Match(MatchExpression { value, arms }))
    }

    fn parse_match_arm(
        &self,
        node: Node,
        source: &str,
    ) -> Result<MatchArm, Box<dyn std::error::Error>> {
        let mut pattern = None;
        let mut value = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "pattern" => {
                        pattern = Some(self.parse_pattern(child, source)?);
                    }
                    "=>" | "," => {
                        // Skip tokens
                    }
                    _ => {
                        // Try to parse as value expression
                        if value.is_none() {
                            if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                                value = Some(expr);
                            }
                        }
                    }
                }
            }
        }

        let pattern = pattern.ok_or("Missing pattern")?;
        let value = value.ok_or("Missing match arm value")?;

        Ok(MatchArm { pattern, value })
    }

    fn parse_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Pattern, Box<dyn std::error::Error>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "literal_pattern" => {
                        return Ok(Pattern::Literal(self.parse_literal_pattern(child, source)?));
                    }
                    "wildcard_pattern" => {
                        return Ok(Pattern::Wildcard);
                    }
                    "identifier_pattern" => {
                        if let Some(id_child) = child.child(0) {
                            let text = id_child.utf8_text(source.as_bytes())?;
                            return Ok(Pattern::Identifier(text.to_string()));
                        }
                    }
                    _ => {}
                }
            }
        }
        Err("Invalid pattern".into())
    }

    fn parse_interpolated_string(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut segments = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "string_segment" => {
                        let text = child.utf8_text(source.as_bytes())?;
                        let processed = process_escape_sequences(text)?;
                        segments.push(StringSegment::Text(processed));
                    }
                    "interpolation" => {
                        // Find the expression inside the interpolation
                        for j in 0..child.child_count() {
                            if let Some(expr_node) = child.child(j) {
                                if expr_node.kind() != "{" && expr_node.kind() != "}" {
                                    let expr = self.node_to_expr_with_span(expr_node, source)?;
                                    segments.push(StringSegment::Interpolation(Box::new(expr)));
                                    break;
                                }
                            }
                        }
                    }
                    "\"" => {
                        // Skip quote delimiters
                    }
                    _ => {}
                }
            }
        }

        Ok(Expr::StringInterpolation(StringInterpolation { segments }))
    }

    fn parse_literal_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<LiteralPattern, Box<dyn std::error::Error>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "number" => {
                        let text = child.utf8_text(source.as_bytes())?;
                        let num = text.parse::<i64>()?;
                        return Ok(LiteralPattern::Number(num));
                    }
                    "string" => {
                        // Check if this is an interpolated string by looking for child nodes
                        let has_interpolation = (0..child.child_count())
                            .any(|i| child.child(i).is_some_and(|grandchild| grandchild.kind() == "interpolation"));
                        
                        if has_interpolation {
                            // Parse as Expr first, then extract StringInterpolation
                            if let Ok(Expr::StringInterpolation(interp)) = self.parse_interpolated_string(child, source) {
                                return Ok(LiteralPattern::StringInterpolation(interp));
                            }
                        } else {
                            let text = child.utf8_text(source.as_bytes())?;
                            let content = &text[1..text.len() - 1];
                            let processed = process_escape_sequences(content)?;
                            return Ok(LiteralPattern::String(processed));
                        }
                    }
                    _ => {}
                }
            }
        }
        Err("Invalid literal pattern".into())
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

/// Process a hex escape sequence (\xHH) and return the byte value
fn process_hex_escape(chars: &mut std::str::Chars) -> Result<u8, StringLiteralError> {
    let hex1 = chars
        .next()
        .ok_or(StringLiteralError::IncompleteHexEscape)?;
    let hex2 = chars
        .next()
        .ok_or(StringLiteralError::IncompleteHexEscape)?;

    let hex_str = format!("{}{}", hex1, hex2);
    u8::from_str_radix(&hex_str, 16).map_err(|_| StringLiteralError::InvalidHexDigits {
        hex_str: hex_str.clone(),
    })
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
                    let byte_value = process_hex_escape(&mut chars)?;
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
            let mut buffer = [0; 4];
            let encoded = ch.encode_utf8(&mut buffer);
            result.extend_from_slice(encoded.as_bytes());
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
    fn test_simple_function() {
        use crate::TributeDatabaseImpl;
        use salsa::Database;

        TributeDatabaseImpl::default().attach(|db| {
            let source_file = crate::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn main() {
    print_line("Hello, world!")
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db);
            assert_eq!(func.name(db), "main");
            assert_eq!(func.parameters(db).len(), 0);
            assert_eq!(func.body(db).statements.len(), 1);
        });
    }

    #[test]
    fn test_function_with_parameters() {
        use crate::TributeDatabaseImpl;
        use salsa::Database;

        TributeDatabaseImpl::default().attach(|db| {
            let source_file = crate::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn add(a, b) {
    a + b
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db);
            assert_eq!(func.name(db), "add");
            assert_eq!(func.parameters(db), vec!["a".to_string(), "b".to_string()]);
        });
    }

    #[test]
    fn test_match_expression() {
        use crate::TributeDatabaseImpl;
        use salsa::Database;

        TributeDatabaseImpl::default().attach(|db| {
            let source_file = crate::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn test(n) {
    match n {
        0 => "zero",
        1 => "one",
        _ => "other"
    }
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db);
            if let Statement::Expression((Expr::Match(_), _)) = &func.body(db).statements[0] {
                // Match expression parsed successfully
            } else {
                panic!("Expected match expression");
            }
        });
    }

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
        use crate::TributeDatabaseImpl;
        use salsa::Database;

        TributeDatabaseImpl::default().attach(|db| {
            // Test basic quote escaping
            let source_file = crate::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn test() {
    "Hello \"World\""
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);
            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db);
            if let Statement::Expression((Expr::String(s), _)) = &func.body(db).statements[0] {
                assert_eq!(s, "Hello \"World\"");
            } else {
                panic!("Expected string expression");
            }

            // Test mixed escape sequences
            let source_file2 = crate::SourceFile::new(
                db,
                std::path::PathBuf::from("test2.trb"),
                r#"
fn test() {
    "Line1\nTab\tQuote\""
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file2);
            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db);
            if let Statement::Expression((Expr::String(s), _)) = &func.body(db).statements[0] {
                assert_eq!(s, "Line1\nTab\tQuote\"");
            } else {
                panic!("Expected string expression");
            }
        });
    }
}
