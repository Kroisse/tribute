//! Raw (unresolved) parse structures and winnow combinators for TrunkIR text format.
//!
//! This module contains the "stage 1" parser: text → `Raw*` structs.
//! It is shared by both the Salsa IR builder and the Arena IR builder.

use winnow::ascii;
use winnow::combinator::{alt, delimited, opt, preceded, separated};
use winnow::prelude::*;
use winnow::token::{any, one_of, take_while};

// ============================================================================
// Error type
// ============================================================================

/// Parse error for IR text format.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at offset {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

// ============================================================================
// Raw (unresolved) AST structures
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) struct RawOperation<'a> {
    pub results: Vec<&'a str>,
    pub dialect: &'a str,
    pub op_name: &'a str,
    /// Optional symbol name parsed from `@name` after `dialect.op`.
    pub sym_name: Option<String>,
    /// Optional function-style parameters: `(%arg: type, ...)`.
    pub func_params: Vec<(&'a str, RawType<'a>)>,
    /// Optional return type from `-> type`.
    pub return_type: Option<RawType<'a>>,
    /// Optional effect type from `effects type`.
    pub effect_type: Option<RawType<'a>>,
    pub operands: Vec<&'a str>,
    pub attributes: Vec<(&'a str, RawAttribute<'a>)>,
    pub result_types: Vec<RawType<'a>>,
    pub regions: Vec<RawRegion<'a>>,
    /// Optional successor list from `[^bb0, ^bb1]`.
    pub successors: Vec<&'a str>,
}

#[derive(Debug, Clone)]
pub(crate) struct RawRegion<'a> {
    pub blocks: Vec<RawBlock<'a>>,
}

#[derive(Debug, Clone)]
pub(crate) struct RawBlock<'a> {
    pub label: &'a str,
    pub args: Vec<(&'a str, RawType<'a>)>,
    pub ops: Vec<RawOperation<'a>>,
}

#[derive(Debug, Clone)]
pub(crate) struct RawType<'a> {
    pub dialect: &'a str,
    pub name: &'a str,
    pub params: Vec<RawType<'a>>,
    pub attrs: Vec<(&'a str, RawAttribute<'a>)>,
}

#[derive(Debug, Clone)]
pub(crate) enum RawAttribute<'a> {
    Bool(bool),
    Int(u64),
    Float(f64),
    String(String),
    Symbol(String),
    Type(RawType<'a>),
    List(Vec<RawAttribute<'a>>),
    Unit,
    Location(String, usize, usize),
    Bytes(Vec<u8>),
}

// ============================================================================
// Winnow parsers
// ============================================================================

/// Skip whitespace.
pub(crate) fn ws(input: &mut &str) -> ModalResult<()> {
    take_while(0.., |c: char| c.is_ascii_whitespace())
        .void()
        .parse_next(input)
}

/// Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*
pub(crate) fn ident<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    (
        one_of(|c: char| c.is_ascii_alphabetic() || c == '_'),
        take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
        .take()
        .parse_next(input)
}

/// Parse a value reference: %name or %number
pub(crate) fn value_ref<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
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
pub(crate) fn symbol_ref(input: &mut &str) -> ModalResult<String> {
    '@'.parse_next(input)?;
    if input.starts_with('"') {
        // Quoted symbol — reuse string_lit which handles all escapes
        string_lit.parse_next(input)
    } else {
        // Bare symbol
        take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_')
            .map(|s: &str| s.to_owned())
            .parse_next(input)
    }
}

/// Parse a block label: ^bbN or ^name
pub(crate) fn block_label<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    preceded(
        '^',
        take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
    .parse_next(input)
}

/// Parse a dialect-qualified name: dialect.name
pub(crate) fn qualified_name<'a>(input: &mut &'a str) -> ModalResult<(&'a str, &'a str)> {
    (ident, '.', ident)
        .map(|(d, _, n)| (d, n))
        .parse_next(input)
}

/// Parse an integer literal (unsigned or negative via two's complement).
pub(crate) fn integer_lit(input: &mut &str) -> ModalResult<u64> {
    let negative = opt('-').parse_next(input)?.is_some();
    let value: u64 = ascii::dec_uint(input)?;
    if negative {
        // Two's complement: the magnitude must fit in i64 range.
        // i64::MIN magnitude (9223372036854775808) needs special handling.
        let i64_min_magnitude = i64::MAX as u64 + 1; // 9223372036854775808
        if value > i64_min_magnitude {
            return Err(winnow::error::ErrMode::Backtrack(
                winnow::error::ContextError::new(),
            ));
        }
        if value == i64_min_magnitude {
            // Exactly i64::MIN
            Ok(u64::from_ne_bytes(i64::MIN.to_ne_bytes()))
        } else {
            let signed = -(value as i64);
            Ok(u64::from_ne_bytes(signed.to_ne_bytes()))
        }
    } else {
        Ok(value)
    }
}

/// Parse a float literal that MUST contain a decimal point.
/// Accepts optional exponent notation: `3.14`, `-1.0e10`, `2.5e-3`.
/// This prevents `42` from being parsed as a float.
pub(crate) fn float_with_dot(input: &mut &str) -> ModalResult<f64> {
    // Match: [-]digits.digits[e[+-]digits]
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
pub(crate) fn string_lit(input: &mut &str) -> ModalResult<String> {
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

/// Parse a type: `dialect.name`, `dialect.name(params)`, or
/// `dialect.name(params) {key = value, ...}`.
///
/// The optional `{...}` block carries type-level attributes (e.g., the
/// `effect` attribute on function types).  Type attributes are only parsed
/// when explicit parentheses `()` are present to avoid ambiguity with the
/// opening `{` of operation body regions.
pub(crate) fn raw_type<'a>(input: &mut &'a str) -> ModalResult<RawType<'a>> {
    let (dialect, name) = qualified_name.parse_next(input)?;

    // Optional type parameters
    let opt_params = opt(delimited(
        ('(', ws),
        separated(0.., (ws, raw_type, ws).map(|(_, t, _)| t), ','),
        (ws, ')'),
    ))
    .parse_next(input)?;
    let has_parens = opt_params.is_some();
    let params = opt_params.unwrap_or_default();

    // Optional type attributes: {key = value, ...}
    // Only attempted after explicit parens `()` to avoid ambiguity with
    // the opening `{` of operation body regions (e.g. `-> core.nil { ... }`).
    let attrs = if has_parens {
        opt(preceded(ws, raw_attr_dict))
            .parse_next(input)?
            .unwrap_or_default()
    } else {
        vec![]
    };

    Ok(RawType {
        dialect,
        name,
        params,
        attrs,
    })
}

/// Parse an attribute value.
pub(crate) fn raw_attr_value<'a>(input: &mut &'a str) -> ModalResult<RawAttribute<'a>> {
    alt((
        // Keywords
        "true".value(RawAttribute::Bool(true)),
        "false".value(RawAttribute::Bool(false)),
        "unit".value(RawAttribute::Unit),
        // Location: loc("path" start:end)
        raw_location_attr,
        // Bytes: bytes(1, 2, 3)
        raw_bytes_attr,
        // String literal
        string_lit.map(RawAttribute::String),
        // Symbol reference
        symbol_ref.map(RawAttribute::Symbol),
        // List
        delimited(
            ('[', ws),
            separated(0.., (ws, raw_attr_value, ws).map(|(_, a, _)| a), ','),
            (ws, ']'),
        )
        .map(RawAttribute::List),
        // Float (requires dot: 3.14, -1.0)
        float_with_dot.map(RawAttribute::Float),
        // Integer (42, -1)
        integer_lit.map(RawAttribute::Int),
        // Type (dialect.name...)
        raw_type.map(RawAttribute::Type),
    ))
    .parse_next(input)
}

/// Parse loc("path" start:end)
fn raw_location_attr<'a>(input: &mut &'a str) -> ModalResult<RawAttribute<'a>> {
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
    Ok(RawAttribute::Location(path, start, end))
}

/// Parse bytes(1, 2, 3)
fn raw_bytes_attr<'a>(input: &mut &'a str) -> ModalResult<RawAttribute<'a>> {
    "bytes".parse_next(input)?;
    ws.parse_next(input)?;
    let bytes: Vec<u8> = delimited(
        ('(', ws),
        separated(
            0..,
            (ws, ascii::dec_uint::<_, u8, _>, ws).map(|(_, b, _)| b),
            ',',
        ),
        (ws, ')'),
    )
    .parse_next(input)?;
    Ok(RawAttribute::Bytes(bytes))
}

/// Parse an attribute dict: {key = value, ...}
pub(crate) fn raw_attr_dict<'a>(
    input: &mut &'a str,
) -> ModalResult<Vec<(&'a str, RawAttribute<'a>)>> {
    delimited(
        ('{', ws),
        separated(
            0..,
            (ws, ident, ws, '=', ws, raw_attr_value, ws).map(|(_, k, _, _, _, v, _)| (k, v)),
            ',',
        ),
        (ws, '}'),
    )
    .parse_next(input)
}

/// Parse result list: %0 = or %0, %1 =
fn result_list<'a>(input: &mut &'a str) -> ModalResult<Vec<&'a str>> {
    let results: Vec<&str> =
        separated(1.., (ws, value_ref, ws).map(|(_, v, _)| v), ',').parse_next(input)?;
    ws.parse_next(input)?;
    '='.parse_next(input)?;
    Ok(results)
}

/// Parse operand list: %a, %b, ...
fn operand_list<'a>(input: &mut &'a str) -> ModalResult<Vec<&'a str>> {
    separated(1.., (ws, value_ref, ws).map(|(_, v, _)| v), ',').parse_next(input)
}

/// Parse type annotation: : type1, type2
fn type_annotation<'a>(input: &mut &'a str) -> ModalResult<Vec<RawType<'a>>> {
    preceded(
        (ws, ':', ws),
        separated(1.., (ws, raw_type, ws).map(|(_, t, _)| t), ','),
    )
    .parse_next(input)
}

/// Parse function-style parameter list: (%arg: type, ...)
pub(crate) fn func_params<'a>(input: &mut &'a str) -> ModalResult<Vec<(&'a str, RawType<'a>)>> {
    delimited(
        ('(', ws),
        separated(
            0..,
            (ws, value_ref, ws, ':', ws, raw_type, ws).map(|(_, name, _, _, _, ty, _)| (name, ty)),
            ',',
        ),
        (ws, ')'),
    )
    .parse_next(input)
}

/// Parse empty parens: ()
fn empty_parens(input: &mut &str) -> ModalResult<()> {
    ('(', ws, ')').void().parse_next(input)
}

/// Parse return type: -> type
fn return_type<'a>(input: &mut &'a str) -> ModalResult<RawType<'a>> {
    preceded((ws, '-', '>', ws), raw_type).parse_next(input)
}

/// Parse a successor list: [^bb0, ^bb1]
fn successor_list<'a>(input: &mut &'a str) -> ModalResult<Vec<&'a str>> {
    delimited(
        ('[', ws),
        separated(1.., (ws, block_label, ws).map(|(_, l, _)| l), ','),
        (ws, ']'),
    )
    .parse_next(input)
}

/// Parse a single operation.
///
/// Grammar:
/// ```text
/// [results =] dialect.op [@symbol] [(%arg: type, ...) | () | operands]
///   [-> type] [effects type] [{attrs}] [: types] [[successors]] [regions]
/// ```
pub(crate) fn raw_operation<'a>(input: &mut &'a str) -> ModalResult<RawOperation<'a>> {
    ws.parse_next(input)?;

    // Try to parse results: %0 = or %0, %1 =
    let results = opt(result_list).parse_next(input)?.unwrap_or_default();
    ws.parse_next(input)?;

    // dialect.op
    let (dialect, op_name) = qualified_name.parse_next(input)?;

    // Optional @symbol (e.g., core.module @test, func.func @main)
    let sym_name = opt(preceded(ws, symbol_ref)).parse_next(input)?;

    // Parse either func-style params (%arg: type, ...) or operands (%val, %val)
    // or empty parens ()
    let mut func_params_parsed = Vec::new();
    let mut operands = Vec::new();

    ws.parse_next(input)?;
    if input.starts_with('(') {
        // Try func_params first (which includes empty parens)
        if let Ok(params) = func_params.parse_next(input) {
            func_params_parsed = params;
        } else if empty_parens.parse_next(input).is_ok() {
            // Empty parens: no params, no operands
        }
    } else if input.starts_with('%') {
        // Operand list
        operands = opt(operand_list).parse_next(input)?.unwrap_or_default();
    }

    // Successor list (optional): [^bb0, ^bb1]
    // Must be parsed before attrs/types to match arena printer output order.
    let successors = opt(preceded(ws, successor_list))
        .parse_next(input)?
        .unwrap_or_default();

    // Optional return type: -> type
    let return_ty = opt(return_type).parse_next(input)?;

    // Optional effect type: effects type
    let effect_ty = opt(preceded((ws, "effects", ws), raw_type)).parse_next(input)?;

    // Attributes (optional)
    let attributes = opt(preceded(ws, raw_attr_dict))
        .parse_next(input)?
        .unwrap_or_default();

    // Type annotation (optional): : type1, type2
    let result_types = opt(type_annotation).parse_next(input)?.unwrap_or_default();

    // Regions (optional, zero or more)
    let mut regions = Vec::new();
    loop {
        ws.parse_next(input)?;
        if input.starts_with('{') {
            let region = raw_region.parse_next(input)?;
            regions.push(region);
        } else {
            break;
        }
    }

    Ok(RawOperation {
        results,
        dialect,
        op_name,
        sym_name,
        func_params: func_params_parsed,
        return_type: return_ty,
        effect_type: effect_ty,
        operands,
        attributes,
        result_types,
        regions,
        successors,
    })
}

/// Parse a block: ^label(args): ops...
pub(crate) fn raw_block<'a>(input: &mut &'a str) -> ModalResult<RawBlock<'a>> {
    ws.parse_next(input)?;
    let label = block_label.parse_next(input)?;

    // Optional arguments: (%arg0: type, ...)
    let args = opt(delimited(
        ('(', ws),
        separated(
            0..,
            (ws, value_ref, ws, ':', ws, raw_type, ws).map(|(_, name, _, _, _, ty, _)| (name, ty)),
            ',',
        ),
        (ws, ')'),
    ))
    .parse_next(input)?
    .unwrap_or_default();

    ws.parse_next(input)?;
    ':'.parse_next(input)?;

    // Operations until next block label or closing brace
    let mut ops = Vec::new();
    loop {
        ws.parse_next(input)?;
        // Stop at block label or closing brace
        if input.starts_with('^') || input.starts_with('}') || input.is_empty() {
            break;
        }
        let op = raw_operation.parse_next(input)?;
        ops.push(op);
    }

    Ok(RawBlock { label, args, ops })
}

/// Parse a region: { blocks... } or { ops... } (single implicit block)
pub(crate) fn raw_region<'a>(input: &mut &'a str) -> ModalResult<RawRegion<'a>> {
    '{'.parse_next(input)?;
    ws.parse_next(input)?;

    let mut blocks = Vec::new();

    // Check if there are explicit blocks (starting with ^)
    if input.starts_with('^') {
        // Explicit blocks
        loop {
            ws.parse_next(input)?;
            if input.starts_with('}') {
                break;
            }
            let block = raw_block.parse_next(input)?;
            blocks.push(block);
        }
    } else if !input.starts_with('}') {
        // Implicit single block (operations without a block label)
        let mut ops = Vec::new();
        loop {
            ws.parse_next(input)?;
            if input.starts_with('}') || input.is_empty() {
                break;
            }
            let op = raw_operation.parse_next(input)?;
            ops.push(op);
        }
        blocks.push(RawBlock {
            label: "bb0",
            args: vec![],
            ops,
        });
    }

    ws.parse_next(input)?;
    '}'.parse_next(input)?;

    Ok(RawRegion { blocks })
}

// ============================================================================
// Tests (pure combinator tests)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_type() {
        let mut input = "core.i32";
        let raw = raw_type.parse_next(&mut input).expect("should parse type");
        assert_eq!(raw.dialect, "core");
        assert_eq!(raw.name, "i32");
        assert!(raw.params.is_empty());
    }

    #[test]
    fn test_parse_parameterized_type() {
        let mut input = "core.func(core.nil, core.i32, core.i32)";
        let raw = raw_type.parse_next(&mut input).expect("should parse type");
        assert_eq!(raw.dialect, "core");
        assert_eq!(raw.name, "func");
        assert_eq!(raw.params.len(), 3);
    }

    #[test]
    fn test_parse_attribute_values() {
        // Integer
        let mut input = "42";
        let attr = raw_attr_value
            .parse_next(&mut input)
            .expect("should parse int");
        assert!(matches!(attr, RawAttribute::Int(42)));

        // Float
        let mut input = "3.25";
        let attr = raw_attr_value
            .parse_next(&mut input)
            .expect("should parse float");
        assert!(matches!(attr, RawAttribute::Float(f) if (f - 3.25).abs() < 1e-10));

        // Bool
        let mut input = "true";
        let attr = raw_attr_value
            .parse_next(&mut input)
            .expect("should parse bool");
        assert!(matches!(attr, RawAttribute::Bool(true)));

        // String
        let mut input = r#""hello""#;
        let attr = raw_attr_value
            .parse_next(&mut input)
            .expect("should parse string");
        assert!(matches!(attr, RawAttribute::String(ref s) if s == "hello"));

        // Symbol
        let mut input = "@foo";
        let attr = raw_attr_value
            .parse_next(&mut input)
            .expect("should parse symbol");
        assert!(matches!(attr, RawAttribute::Symbol(ref s) if s == "foo"));
    }

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
    fn test_parse_symbol_ref_escapes() {
        // Bare symbol
        let mut input = "@foo";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "foo");

        // Quoted symbol with colons
        let mut input = "@\"std::List::map\"";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "std::List::map");

        // Quoted symbol with escaped backslash
        let mut input = "@\"a\\\\b\"";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "a\\b");

        // Quoted symbol with escaped quote
        let mut input = "@\"say\\\"hi\\\"\"";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "say\"hi\"");

        // Quoted symbol with newline escape
        let mut input = "@\"line1\\nline2\"";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "line1\nline2");

        // Quoted symbol with hex escape
        let mut input = "@\"x\\x01y\"";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "x\x01y");

        // Empty quoted symbol
        let mut input = "@\"\"";
        let result = symbol_ref.parse_next(&mut input).expect("should parse");
        assert_eq!(result, "");
    }

    #[test]
    fn test_parse_integer_lit_overflow() {
        // i64::MIN magnitude is valid
        let mut input = "-9223372036854775808";
        let val = integer_lit
            .parse_next(&mut input)
            .expect("i64::MIN should parse");
        assert_eq!(val, u64::from_ne_bytes(i64::MIN.to_ne_bytes()));

        // One beyond i64::MIN magnitude should fail
        let mut input = "-9223372036854775809";
        let result = integer_lit.parse_next(&mut input);
        assert!(result.is_err(), "beyond i64::MIN should fail");

        // Positive i64::MAX should be fine
        let mut input = "9223372036854775807";
        let val = integer_lit
            .parse_next(&mut input)
            .expect("i64::MAX should parse");
        assert_eq!(val, i64::MAX as u64);
    }

    #[test]
    fn test_parse_float_exponent() {
        let mut input = "1.5e10";
        let val = float_with_dot.parse_next(&mut input).expect("should parse");
        assert_eq!(val, 1.5e10);

        let mut input = "2.0E-3";
        let val = float_with_dot.parse_next(&mut input).expect("should parse");
        assert_eq!(val, 2.0e-3);

        let mut input = "-3.14e+2";
        let val = float_with_dot.parse_next(&mut input).expect("should parse");
        assert_eq!(val, -3.14e2);

        // Plain float still works
        let mut input = "3.25";
        let val = float_with_dot.parse_next(&mut input).expect("should parse");
        assert_eq!(val, 3.25);
    }

    #[test]
    fn test_parse_type_with_attrs() {
        let mut input = "core.func(core.nil, core.i32) {effect = core.nil}";
        let result = raw_type.parse_next(&mut input).expect("should parse");
        assert_eq!(result.dialect, "core");
        assert_eq!(result.name, "func");
        assert_eq!(result.params.len(), 2);
        assert_eq!(result.attrs.len(), 1);
        assert_eq!(result.attrs[0].0, "effect");
    }

    #[test]
    fn test_parse_successor_list() {
        let mut input = "[^bb0, ^bb1]";
        let result = successor_list.parse_next(&mut input).expect("should parse");
        assert_eq!(result, vec!["bb0", "bb1"]);
    }
}
