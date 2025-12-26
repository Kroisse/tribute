//! Literal parsing utilities.

use ropey::Rope;
use tree_sitter::Node;

use super::helpers::node_text;

// =============================================================================
// Literal Parsing
// =============================================================================

/// Parse a natural number literal (unsigned).
pub fn parse_nat_literal(text: &str) -> Option<u64> {
    let text = text.replace('_', "");
    if text.starts_with("0x") || text.starts_with("0X") {
        u64::from_str_radix(&text[2..], 16).ok()
    } else if text.starts_with("0b") || text.starts_with("0B") {
        u64::from_str_radix(&text[2..], 2).ok()
    } else if text.starts_with("0o") || text.starts_with("0O") {
        u64::from_str_radix(&text[2..], 8).ok()
    } else {
        text.parse().ok()
    }
}

/// Parse an integer literal (signed).
/// Phase 1: values must fit in i64; BigInt support will be added later.
pub fn parse_int_literal(text: &str) -> Option<i64> {
    let text = text.replace('_', "");
    // Handle explicit + or - prefix
    if let Some(rest) = text.strip_prefix('+') {
        parse_nat_literal(rest).and_then(|n| i64::try_from(n).ok())
    } else if let Some(rest) = text.strip_prefix('-') {
        // Handle i64::MIN edge case: -9223372036854775808
        parse_nat_literal(rest).and_then(|n| {
            if n == (i64::MAX as u64) + 1 {
                Some(i64::MIN)
            } else {
                i64::try_from(n).ok().map(|v| -v)
            }
        })
    } else {
        text.parse().ok()
    }
}

/// Parse a float literal.
pub fn parse_float_literal(text: &str) -> Option<f64> {
    let text = text.replace('_', "");
    text.parse().ok()
}

/// Parse a rune (character) literal.
pub fn parse_rune_literal(text: &str) -> Option<char> {
    // Format: ?c, ?\n, ?\xHH, ?\uHHHH
    let text = text.strip_prefix('?')?;

    if let Some(escape) = text.strip_prefix('\\') {
        match escape.chars().next()? {
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            '\\' => Some('\\'),
            '0' => Some('\0'),
            'x' => {
                let hex = &escape[1..];
                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
            }
            'u' => {
                let hex = &escape[1..];
                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
            }
            _ => None,
        }
    } else {
        text.chars().next()
    }
}

fn strip_prefixes<'a>(text: &'a str, prefixes: &[&str]) -> &'a str {
    for prefix in prefixes {
        if let Some(rest) = text.strip_prefix(prefix) {
            return rest;
        }
    }
    text
}

/// Parse a string literal (handling escapes).
pub fn parse_string_literal(node: Node, source: &Rope) -> String {
    let text = node_text(&node, source);
    let text = text.as_ref();

    match node.kind() {
        "raw_string" => {
            // r"...", rs"...", sr"..."
            let content = strip_prefixes(text, &["rs", "sr", "r"]);
            extract_raw_string_content(content)
        }
        "raw_interpolated_string" => {
            // rs"...", sr"..."
            let content = strip_prefixes(text, &["rs", "sr"]);
            extract_raw_string_content(content)
        }
        "multiline_string" => {
            // #"..."# or s#"..."#
            let content = text.strip_prefix('s').unwrap_or(text);
            extract_multiline_string_content(content)
        }
        "string" => {
            // Regular "..." or s"..."
            let content = text.strip_prefix('s').unwrap_or(text);
            extract_string_content(content)
        }
        _ => text.to_owned(),
    }
}

/// Parse a bytes literal.
pub fn parse_bytes_literal(node: Node, source: &Rope) -> Vec<u8> {
    let text = node_text(&node, source);

    match node.kind() {
        "raw_bytes" => {
            // rb"...", br"..."
            let content = text.strip_prefix("rb").or_else(|| text.strip_prefix("br"));
            content
                .map(|value| extract_raw_string_content(value).into_bytes())
                .unwrap_or_default()
        }
        "multiline_bytes" => {
            // b#"..."#
            if let Some(content) = text.strip_prefix("b") {
                extract_multiline_string_content(content).into_bytes()
            } else {
                Vec::new()
            }
        }
        "bytes_string" => {
            // b"..."
            if let Some(content) = text.strip_prefix("b") {
                extract_string_content(content).into_bytes()
            } else {
                Vec::new()
            }
        }
        _ => text.as_bytes().to_vec(),
    }
}

/// Extract content from a regular string literal.
pub fn extract_string_content(text: &str) -> String {
    // Remove quotes
    let inner = text
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(text);

    // Process escapes
    let mut result = String::new();
    let mut chars = inner.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('0') => result.push('\0'),
                Some('x') => {
                    // \xHH
                    let hex: String = chars.by_ref().take(2).collect();
                    if let Ok(code) = u8::from_str_radix(&hex, 16) {
                        result.push(code as char);
                    }
                }
                Some('u') => {
                    // \uHHHH
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16)
                        && let Some(c) = char::from_u32(code)
                    {
                        result.push(c);
                    }
                }
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Extract content from a raw string literal.
pub fn extract_raw_string_content(text: &str) -> String {
    // r"..." or r#"..."#
    if text.starts_with('#') {
        // Count opening hashes
        let hash_count = text.chars().take_while(|&c| c == '#').count();
        let delimiter = "#".repeat(hash_count);
        let open = format!("{}\"", delimiter);
        let close = format!("\"{}", delimiter);

        if let Some(start) = text.find(&open)
            && let Some(end) = text.rfind(&close)
        {
            return text[start + open.len()..end].to_string();
        }
        text.to_string()
    } else {
        // r"..."
        text.strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(text)
            .to_string()
    }
}

/// Extract content from a multiline string literal.
pub fn extract_multiline_string_content(text: &str) -> String {
    // #"..."#
    let hash_count = text.chars().take_while(|&c| c == '#').count();
    let open_delim = format!("{}\"", "#".repeat(hash_count));
    let close_delim = format!("\"{}", "#".repeat(hash_count));

    if let Some(start) = text.find(&open_delim)
        && let Some(end) = text.rfind(&close_delim)
    {
        return text[start + open_delim.len()..end].to_string();
    }
    text.to_string()
}
