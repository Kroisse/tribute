//! Tokens of the TrunkIR text format.
//!
//! These winnow functions read one token from the text and build no IR.
//! [`AsmParser`](super::AsmParser) and [`OpAsmParser`](super::OpAsmParser)
//! read the text through them, and
//! [`AsmParser::parse_token`](super::AsmParser::parse_token) runs one for a
//! format that writes a token of its own.

use winnow::ascii;
use winnow::combinator::{alt, delimited, opt, preceded, repeat, separated};
use winnow::prelude::*;
use winnow::token::{any, one_of, take, take_while};

/// Skip whitespace.
pub fn ws(input: &mut &str) -> ModalResult<()> {
    take_while(0.., |c: char| c.is_ascii_whitespace())
        .void()
        .parse_next(input)
}

/// Parse an identifier: `[a-zA-Z_][a-zA-Z0-9_]*`
pub fn ident<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    (
        one_of(|c: char| c.is_ascii_alphabetic() || c == '_'),
        take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
        .take()
        .parse_next(input)
}

/// Parse a value name: `%name` or `%number`, without the `%`.
pub fn value_ref<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    preceded(
        '%',
        take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
    .parse_next(input)
}

/// Parse a symbol reference: @name or @"quoted name"
///
/// Quoted symbols use the same escape sequences as string literals
/// (`\\`, `\"`, `\n`, `\t`, `\r`, `\0`, `\xNN`).
pub fn symbol_ref(input: &mut &str) -> ModalResult<String> {
    '@'.parse_next(input)?;
    if input.starts_with('"') {
        string_lit.parse_next(input)
    } else {
        take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_')
            .map(|s: &str| s.to_owned())
            .parse_next(input)
    }
}

/// Parse a symbol path: one or more symbol references joined by `::`, such
/// as `@outer::@inner::@name`. Each component is one name; a quoted
/// component may itself contain `::`.
pub fn symbol_path(input: &mut &str) -> ModalResult<Vec<String>> {
    let mut components = vec![symbol_ref.parse_next(input)?];
    while input.starts_with("::@") {
        *input = &input[2..];
        components.push(symbol_ref.parse_next(input)?);
    }
    Ok(components)
}

/// Parse a name: a bare identifier or a quoted string.
pub fn name_token(input: &mut &str) -> ModalResult<String> {
    if input.starts_with('"') {
        string_lit.parse_next(input)
    } else {
        ident.map(str::to_owned).parse_next(input)
    }
}

/// Parse a block label: `^bbN` or `^name`, without the `^`.
pub fn block_label<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    preceded(
        '^',
        take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
    .parse_next(input)
}

/// Parse a dialect-qualified name, `dialect.name`, as one slice.
pub fn qualified_name<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    (ident, '.', ident).take().parse_next(input)
}

/// Parse an integer literal (signed or unsigned).
pub fn integer_lit(input: &mut &str) -> ModalResult<i128> {
    let negative = opt('-').parse_next(input)?.is_some();
    let digits = take_while(1.., |c: char| c.is_ascii_digit()).parse_next(input)?;
    let magnitude: u128 = digits
        .parse()
        .map_err(|_| winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new()))?;
    if negative {
        let max_neg = (i128::MAX as u128) + 1; // 1<<127
        match magnitude.cmp(&max_neg) {
            std::cmp::Ordering::Greater => Err(winnow::error::ErrMode::Backtrack(
                winnow::error::ContextError::new(),
            )),
            std::cmp::Ordering::Equal => Ok(i128::MIN),
            std::cmp::Ordering::Less => Ok(-(magnitude as i128)),
        }
    } else {
        i128::try_from(magnitude)
            .map_err(|_| winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new()))
    }
}

/// Parse a float literal that MUST contain a decimal point.
/// Accepts optional exponent notation: `3.14`, `-1.0e10`, `2.5e-3`.
/// This prevents `42` from being parsed as a float.
pub fn float_with_dot(input: &mut &str) -> ModalResult<f64> {
    let s = (
        opt('-'),
        take_while(1.., |c: char| c.is_ascii_digit()),
        '.',
        take_while(1.., |c: char| c.is_ascii_digit()),
        opt((
            one_of(['e', 'E']),
            opt(one_of(['+', '-'])),
            take_while(1.., |c: char| c.is_ascii_digit()),
        )),
    )
        .take()
        .parse_next(input)?;
    s.parse::<f64>()
        .map_err(|_| winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new()))
}

/// Parse a string literal: "content"
pub fn string_lit(input: &mut &str) -> ModalResult<String> {
    '"'.parse_next(input)?;
    let mut result = String::new();
    loop {
        let c = any.parse_next(input)?;
        match c {
            '"' => break,
            '\\' => {
                let escaped = any.parse_next(input)?;
                match escaped {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    'n' => result.push('\n'),
                    't' => result.push('\t'),
                    'r' => result.push('\r'),
                    '0' => result.push('\0'),
                    'x' => {
                        // \xNN hex escape
                        let h1 = any.parse_next(input)?;
                        let h2 = any.parse_next(input)?;
                        let hex_str = format!("{}{}", h1, h2);
                        if let Ok(code) = u8::from_str_radix(&hex_str, 16) {
                            result.push(code as char);
                        } else {
                            // Invalid hex: pass through literally
                            result.push('\\');
                            result.push('x');
                            result.push(h1);
                            result.push(h2);
                        }
                    }
                    _ => {
                        result.push('\\');
                        result.push(escaped);
                    }
                }
            }
            _ => result.push(c),
        }
    }
    Ok(result)
}

/// Parse a type alias name after its `!`: bare `ident` or quoted `"string"`.
pub fn type_alias_name(input: &mut &str) -> ModalResult<String> {
    name_token(input)
}

/// Parse `loc("path" start:end)`.
pub fn location(input: &mut &str) -> ModalResult<(String, usize, usize)> {
    "loc".parse_next(input)?;
    ws.parse_next(input)?;
    '('.parse_next(input)?;
    ws.parse_next(input)?;
    let path = string_lit.parse_next(input)?;
    ws.parse_next(input)?;
    let start: usize = ascii::dec_uint(input)?;
    ':'.parse_next(input)?;
    let end: usize = ascii::dec_uint(input)?;
    ws.parse_next(input)?;
    ')'.parse_next(input)?;
    Ok((path, start, end))
}

/// Parse b"..." bytes string literal.
///
/// Unlike `string_lit`, this interprets `\xNN` escapes as single raw bytes
/// (0x00–0xFF) instead of Unicode code points, so the result is a `Vec<u8>`
/// that faithfully represents arbitrary byte sequences.
pub fn bytes_lit(input: &mut &str) -> ModalResult<Vec<u8>> {
    preceded('b', delimited('"', bytes_content, '"')).parse_next(input)
}

/// Collect bytes content fragments into a `Vec<u8>`.
fn bytes_content(input: &mut &str) -> ModalResult<Vec<u8>> {
    enum Fragment<'a> {
        Literal(&'a str),
        Escape(u8),
    }

    repeat(
        0..,
        alt((
            preceded('\\', byte_escape).map(Fragment::Escape),
            take_while(1.., |c: char| c != '"' && c != '\\').map(Fragment::Literal),
        )),
    )
    .fold(Vec::new, |mut acc, frag| {
        match frag {
            Fragment::Literal(s) => acc.extend_from_slice(s.as_bytes()),
            Fragment::Escape(b) => acc.push(b),
        }
        acc
    })
    .parse_next(input)
}

/// Parse a single byte escape sequence (after the leading `\`).
fn byte_escape(input: &mut &str) -> ModalResult<u8> {
    alt((
        '"'.value(b'"'),
        '\\'.value(b'\\'),
        'n'.value(b'\n'),
        't'.value(b'\t'),
        'r'.value(b'\r'),
        '0'.value(b'\0'),
        preceded('x', take(2usize)).try_map(|hex: &str| u8::from_str_radix(hex, 16)),
    ))
    .parse_next(input)
}

/// Parse an attribute key.
///
/// Attribute symbols may be namespaced with dots (for example,
/// `test.marker`). The printer has always emitted the complete
/// symbol, so accepting the same spelling here restores generic round-trips.
/// Keys that are not dot-separated identifiers use the quoted string form.
pub fn attribute_key(input: &mut &str) -> ModalResult<String> {
    if input.starts_with('"') {
        return string_lit.parse_next(input);
    }
    let key = take_while(1.., |c: char| {
        c.is_ascii_alphanumeric() || c == '_' || c == '.'
    })
    .parse_next(input)?;
    if is_bare_attribute_key(key) {
        Ok(key.to_owned())
    } else {
        Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ))
    }
}

/// Whether `key` can be written without quotes: dot-separated identifier
/// segments. Other keys use the quoted string form.
pub(crate) fn is_bare_attribute_key(key: &str) -> bool {
    key.split('.').all(|segment| {
        let mut chars = segment.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// Parse an operation's result names before its `=`: `%0 =` or `%0, %1 =`.
/// Each name comes with its position in `input`'s remaining length.
pub(crate) fn result_list<'a>(input: &mut &'a str) -> ModalResult<Vec<(&'a str, usize)>> {
    let results: Vec<(&str, usize)> = separated(
        1..,
        (
            ws,
            |input: &mut &'a str| {
                let remaining = input.len();
                value_ref.map(|name| (name, remaining)).parse_next(input)
            },
            ws,
        )
            .map(|(_, v, _)| v),
        ',',
    )
    .parse_next(input)?;
    ws.parse_next(input)?;
    '='.parse_next(input)?;
    Ok(results)
}

/// Whether `input` starts with a type alias definition (`!name =` or
/// `!"quoted" =`).
pub(crate) fn is_type_alias_def(input: &str) -> bool {
    if !input.starts_with('!') {
        return false;
    }
    let rest = &input[1..];
    let after_name = if let Some(quoted) = rest.strip_prefix('"') {
        // Quoted: skip to closing quote (simplified — no escape handling needed for lookahead)
        match quoted.find('"') {
            Some(end) => &quoted[end + 1..],
            None => return false,
        }
    } else {
        let after_ident = rest.trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_');
        if after_ident.len() == rest.len() {
            return false;
        }
        after_ident
    };
    after_name.trim_start().starts_with('=')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_string_escapes() {
        let cases = [
            (r#""hello""#, "hello"),
            (r#""a\nb""#, "a\nb"),
            (r#""a\tb""#, "a\tb"),
            (r#""a\rb""#, "a\rb"),
            (r#""a\0b""#, "a\0b"),
            (r#""a\\b""#, "a\\b"),
            (r#""a\"b""#, "a\"b"),
            (r#""a\x01b""#, "a\x01b"),
            (r#""a\x7fb""#, "a\x7fb"),
        ];
        for (input_str, expected) in &cases {
            let mut input = *input_str;
            let result = string_lit.parse_next(&mut input).expect("should parse");
            assert_eq!(&result, *expected, "failed for input: {}", input_str);
        }
    }

    #[test]
    fn test_parse_integer_lit_signed() {
        for (text, expected) in [
            ("-9223372036854775808", i64::MIN as i128),
            ("-9223372036854775809", -9223372036854775809i128),
            ("9223372036854775807", i64::MAX as i128),
            ("18446744073709551615", u64::MAX as i128),
            ("-42", -42),
            ("-170141183460469231731687303715884105728", i128::MIN),
            ("170141183460469231731687303715884105727", i128::MAX),
        ] {
            let mut input = text;
            assert_eq!(integer_lit.parse_next(&mut input).ok(), Some(expected));
        }
        for text in [
            "170141183460469231731687303715884105728",
            "-170141183460469231731687303715884105729",
        ] {
            let mut input = text;
            assert!(
                integer_lit.parse_next(&mut input).is_err(),
                "{text} is out of range"
            );
        }
    }

    #[test]
    fn test_parse_float_exponent() {
        for (text, expected) in [
            ("1.5e10", 1.5e10),
            ("2.0E-3", 2.0e-3),
            ("-3.14e+2", -3.14e2),
            ("3.25", 3.25),
        ] {
            let mut input = text;
            assert_eq!(float_with_dot.parse_next(&mut input).ok(), Some(expected));
        }
    }

    #[test]
    fn test_parse_bytes_string_lit_hex_escape() {
        // \xNN escapes decode to single raw bytes, including above 0x7F
        let mut input = r#"b"\x00\x7f\x80\xff""#;
        let bytes = bytes_lit
            .parse_next(&mut input)
            .expect("should parse bytes with hex escapes");
        assert_eq!(bytes, [0x00, 0x7f, 0x80, 0xff]);
    }
}
