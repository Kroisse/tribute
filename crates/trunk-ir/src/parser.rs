//! TrunkIR text format parser.
//!
//! Parses the textual IR format produced by [`super::printer`] back into
//! Salsa-tracked TrunkIR structures. Uses winnow for parsing.
//!
//! # Two-stage parsing
//!
//! 1. **Raw parse**: winnow combinators parse text into `Raw*` structures
//!    (strings, not Salsa-tracked).
//! 2. **IR build**: `Raw*` structures are converted to Salsa `Operation`,
//!    `Block`, `Region`, etc., resolving SSA value names and block labels.

use std::collections::BTreeMap;

use winnow::ascii;
use winnow::combinator::{alt, delimited, opt, preceded, separated};
use winnow::prelude::*;
use winnow::token::{any, one_of, take_while};

use crate::{
    Attribute, Block, BlockArg, BlockId, IdVec, Location, Operation, PathId, Region, Span, Symbol,
    Type, Value, ValueDef,
};

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
struct RawOperation<'a> {
    results: Vec<&'a str>,
    dialect: &'a str,
    op_name: &'a str,
    /// Optional symbol name parsed from `@name` after `dialect.op`.
    sym_name: Option<String>,
    /// Optional function-style parameters: `(%arg: type, ...)`.
    func_params: Vec<(&'a str, RawType<'a>)>,
    /// Optional return type from `-> type`.
    return_type: Option<RawType<'a>>,
    /// Optional effect type from `effects type`.
    effect_type: Option<RawType<'a>>,
    operands: Vec<&'a str>,
    attributes: Vec<(&'a str, RawAttribute<'a>)>,
    result_types: Vec<RawType<'a>>,
    regions: Vec<RawRegion<'a>>,
}

#[derive(Debug, Clone)]
struct RawRegion<'a> {
    blocks: Vec<RawBlock<'a>>,
}

#[derive(Debug, Clone)]
struct RawBlock<'a> {
    label: &'a str,
    args: Vec<(&'a str, RawType<'a>)>,
    ops: Vec<RawOperation<'a>>,
}

#[derive(Debug, Clone)]
struct RawType<'a> {
    dialect: &'a str,
    name: &'a str,
    params: Vec<RawType<'a>>,
    attrs: Vec<(&'a str, RawAttribute<'a>)>,
}

#[derive(Debug, Clone)]
enum RawAttribute<'a> {
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
fn ws(input: &mut &str) -> ModalResult<()> {
    take_while(0.., |c: char| c.is_ascii_whitespace())
        .void()
        .parse_next(input)
}

/// Parse an identifier: [a-zA-Z_][a-zA-Z0-9_]*
fn ident<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    (
        one_of(|c: char| c.is_ascii_alphabetic() || c == '_'),
        take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
        .take()
        .parse_next(input)
}

/// Parse a value reference: %name or %number
fn value_ref<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
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
fn symbol_ref(input: &mut &str) -> ModalResult<String> {
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
fn block_label<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    preceded(
        '^',
        take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
    .parse_next(input)
}

/// Parse a dialect-qualified name: dialect.name
fn qualified_name<'a>(input: &mut &'a str) -> ModalResult<(&'a str, &'a str)> {
    (ident, '.', ident)
        .map(|(d, _, n)| (d, n))
        .parse_next(input)
}

/// Parse an integer literal (unsigned or negative via two's complement).
fn integer_lit(input: &mut &str) -> ModalResult<u64> {
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
fn float_with_dot(input: &mut &str) -> ModalResult<f64> {
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
fn string_lit(input: &mut &str) -> ModalResult<String> {
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
fn raw_type<'a>(input: &mut &'a str) -> ModalResult<RawType<'a>> {
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
fn raw_attr_value<'a>(input: &mut &'a str) -> ModalResult<RawAttribute<'a>> {
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
fn raw_attr_dict<'a>(input: &mut &'a str) -> ModalResult<Vec<(&'a str, RawAttribute<'a>)>> {
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
fn func_params<'a>(input: &mut &'a str) -> ModalResult<Vec<(&'a str, RawType<'a>)>> {
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

/// Parse a single operation.
///
/// Grammar:
/// ```text
/// [results =] dialect.op [@symbol] [(%arg: type, ...) | () | operands]
///   [-> type] [effects type] [{attrs}] [: types] [regions]
/// ```
fn raw_operation<'a>(input: &mut &'a str) -> ModalResult<RawOperation<'a>> {
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
    })
}

/// Parse a block: ^label(args): ops...
fn raw_block<'a>(input: &mut &'a str) -> ModalResult<RawBlock<'a>> {
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
fn raw_region<'a>(input: &mut &'a str) -> ModalResult<RawRegion<'a>> {
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
// IR Builder (Raw -> Salsa-tracked IR)
// ============================================================================

struct IrBuilder<'db> {
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    /// Maps value name (without %) -> Value
    value_map: std::collections::HashMap<String, Value<'db>>,
    /// Maps block label (without ^) -> BlockId
    block_id_map: std::collections::HashMap<String, BlockId>,
}

impl<'db> IrBuilder<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        let path = PathId::new(db, "textual-ir".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        Self {
            db,
            location,
            value_map: std::collections::HashMap::new(),
            block_id_map: std::collections::HashMap::new(),
        }
    }

    fn build_type(&self, raw: &RawType<'_>) -> Type<'db> {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let name = Symbol::from_dynamic(raw.name);
        let params: IdVec<Type<'db>> = raw.params.iter().map(|p| self.build_type(p)).collect();
        let attrs: BTreeMap<Symbol, Attribute<'db>> = raw
            .attrs
            .iter()
            .map(|(k, v)| (Symbol::from_dynamic(k), self.build_attribute(v)))
            .collect();
        Type::new(self.db, dialect, name, params, attrs)
    }

    fn build_attribute(&self, raw: &RawAttribute<'_>) -> Attribute<'db> {
        match raw {
            RawAttribute::Bool(b) => Attribute::Bool(*b),
            RawAttribute::Int(n) => Attribute::IntBits(*n),
            RawAttribute::Float(f) => Attribute::FloatBits(f.to_bits()),
            RawAttribute::String(s) => Attribute::String(s.clone()),
            RawAttribute::Symbol(s) => Attribute::Symbol(Symbol::from_dynamic(s.as_str())),
            RawAttribute::Type(t) => Attribute::Type(self.build_type(t)),
            RawAttribute::List(items) => {
                Attribute::List(items.iter().map(|a| self.build_attribute(a)).collect())
            }
            RawAttribute::Unit => Attribute::Unit,
            RawAttribute::Location(path, start, end) => {
                let path_id = PathId::new(self.db, path.clone());
                Attribute::Location(Location::new(path_id, Span::new(*start, *end)))
            }
            RawAttribute::Bytes(bytes) => Attribute::Bytes(bytes.clone()),
        }
    }

    fn resolve_value(&self, name: &str) -> Option<Value<'db>> {
        self.value_map.get(name).copied()
    }

    /// Pre-assign BlockIds for all blocks in a region.
    ///
    /// Returns an error if the same label appears more than once *within the
    /// same region*. Labels in different (possibly nested) regions are
    /// independent and may legitimately share the same name (e.g. `^bb0`).
    fn pre_assign_blocks(&mut self, raw_region: &RawRegion<'_>) -> Result<(), ParseError> {
        let mut seen_in_region = std::collections::HashSet::new();
        for block in &raw_region.blocks {
            let label = block.label.to_string();
            if !seen_in_region.insert(label.clone()) {
                return Err(ParseError {
                    message: format!("duplicate block label '^{}'", label),
                    offset: 0,
                });
            }
            let block_id = BlockId::fresh();
            self.block_id_map.insert(label, block_id);
        }
        Ok(())
    }

    /// Save the current `block_id_map` entries for the given labels, so they
    /// can be restored after an inner region is built.  Returns the entries
    /// that existed before (if any) for each label.
    fn save_block_labels(&self, raw_region: &RawRegion<'_>) -> Vec<(String, Option<BlockId>)> {
        raw_region
            .blocks
            .iter()
            .map(|b| {
                let label = b.label.to_string();
                let prev = self.block_id_map.get(&label).copied();
                (label, prev)
            })
            .collect()
    }

    /// Restore previously saved `block_id_map` entries, undoing the insertions
    /// made by `pre_assign_blocks` for an inner region.
    fn restore_block_labels(&mut self, saved: Vec<(String, Option<BlockId>)>) {
        for (label, prev) in saved {
            match prev {
                Some(id) => {
                    self.block_id_map.insert(label, id);
                }
                None => {
                    self.block_id_map.remove(&label);
                }
            }
        }
    }

    fn build_region(&mut self, raw: &RawRegion<'_>) -> Result<Region<'db>, ParseError> {
        let saved = self.save_block_labels(raw);
        self.pre_assign_blocks(raw)?;
        let blocks: IdVec<Block<'db>> = raw
            .blocks
            .iter()
            .map(|b| self.build_block(b))
            .collect::<Result<_, _>>()?;
        let region = Region::new(self.db, self.location, blocks);
        self.restore_block_labels(saved);
        Ok(region)
    }

    fn build_block_with_extra_args(
        &mut self,
        raw: &RawBlock<'_>,
        extra_args: &[(&str, RawType<'_>)],
    ) -> Result<Block<'db>, ParseError> {
        let block_id = self
            .block_id_map
            .get(raw.label)
            .copied()
            .unwrap_or_else(BlockId::fresh);

        // Build block args: merge extra_args (from func params) with block's own args.
        // If the block already has args that match the extra_args, use the block's args.
        // When both are present, validate that they agree in count and types.
        let all_args = if raw.args.is_empty() && !extra_args.is_empty() {
            extra_args.to_vec()
        } else if !raw.args.is_empty() && !extra_args.is_empty() {
            // Both present — validate arity and types match
            if raw.args.len() != extra_args.len() {
                return Err(ParseError {
                    message: format!(
                        "entry block has {} args but function signature has {} params",
                        raw.args.len(),
                        extra_args.len()
                    ),
                    offset: 0,
                });
            }
            for (i, ((_, block_ty), (_, param_ty))) in
                raw.args.iter().zip(extra_args.iter()).enumerate()
            {
                let bt = self.build_type(block_ty);
                let pt = self.build_type(param_ty);
                if bt != pt {
                    return Err(ParseError {
                        message: format!(
                            "entry block arg {} type mismatch: block has {}.{} but function param has {}.{}",
                            i,
                            bt.dialect(self.db),
                            bt.name(self.db),
                            pt.dialect(self.db),
                            pt.name(self.db),
                        ),
                        offset: 0,
                    });
                }
            }
            raw.args.clone()
        } else {
            raw.args.clone()
        };

        let mut block_args = IdVec::new();
        let mut seen_names = std::collections::HashSet::new();
        for (i, (name, raw_ty)) in all_args.iter().enumerate() {
            // Detect duplicate argument names
            if !seen_names.insert(name.to_string()) {
                return Err(ParseError {
                    message: format!("duplicate block argument name '%{}' at index {}", name, i),
                    offset: 0,
                });
            }

            let ty = self.build_type(raw_ty);
            let is_default_name = name
                .strip_prefix("arg")
                .is_some_and(|rest| rest.parse::<usize>().is_ok());
            let arg = if !is_default_name {
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    Symbol::new("bind_name"),
                    Attribute::Symbol(Symbol::from_dynamic(name)),
                );
                BlockArg::new(self.db, ty, attrs)
            } else {
                BlockArg::of_type(self.db, ty)
            };
            block_args.push(arg);

            let value = Value::new(self.db, ValueDef::BlockArg(block_id), i);
            self.value_map.insert(name.to_string(), value);
        }

        // Build operations
        let ops: IdVec<Operation<'db>> = raw
            .ops
            .iter()
            .map(|op| self.build_operation(op))
            .collect::<Result<_, _>>()?;

        Ok(Block::new(
            self.db,
            block_id,
            self.location,
            block_args,
            ops,
        ))
    }

    fn build_block(&mut self, raw: &RawBlock<'_>) -> Result<Block<'db>, ParseError> {
        self.build_block_with_extra_args(raw, &[])
    }

    fn build_operation(&mut self, raw: &RawOperation<'_>) -> Result<Operation<'db>, ParseError> {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let op_name = Symbol::from_dynamic(raw.op_name);

        // Resolve operands
        let operands: IdVec<Value<'db>> = raw
            .operands
            .iter()
            .map(|name| {
                self.resolve_value(name).ok_or_else(|| ParseError {
                    message: format!(
                        "undefined value '%{}' in operation '{}.{}'",
                        name, raw.dialect, raw.op_name
                    ),
                    offset: 0,
                })
            })
            .collect::<Result<_, _>>()?;

        // Build result types
        let results: IdVec<Type<'db>> = raw
            .result_types
            .iter()
            .map(|t| self.build_type(t))
            .collect();

        // Build attributes from explicit attr dict
        let mut attributes: BTreeMap<Symbol, Attribute<'db>> = raw
            .attributes
            .iter()
            .map(|(k, v)| (Symbol::from_dynamic(k), self.build_attribute(v)))
            .collect();

        // Add sym_name attribute if present
        if let Some(ref name) = raw.sym_name {
            attributes.insert(
                Symbol::new("sym_name"),
                Attribute::Symbol(Symbol::from_dynamic(name.as_str())),
            );
        }

        // Handle func-style signature: build function type and inject into attributes
        let has_func_signature = raw.return_type.is_some() || !raw.func_params.is_empty();
        if has_func_signature {
            let return_ty = raw
                .return_type
                .as_ref()
                .map(|t| self.build_type(t))
                .unwrap_or_else(|| {
                    Type::new(
                        self.db,
                        Symbol::new("core"),
                        Symbol::new("nil"),
                        IdVec::new(),
                        BTreeMap::new(),
                    )
                });

            // Build function type: core.func(return_type, param_types...)
            let mut func_params: IdVec<Type<'db>> = IdVec::new();
            func_params.push(return_ty);
            for (_, raw_ty) in &raw.func_params {
                func_params.push(self.build_type(raw_ty));
            }

            let mut func_attrs = BTreeMap::new();
            if let Some(effect_raw) = &raw.effect_type {
                let effect_ty = self.build_type(effect_raw);
                func_attrs.insert(Symbol::new("effect"), Attribute::Type(effect_ty));
            }

            let func_ty = Type::new(
                self.db,
                Symbol::new("core"),
                Symbol::new("func"),
                func_params,
                func_attrs,
            );
            attributes.insert(Symbol::new("type"), Attribute::Type(func_ty));
        }

        // Build regions, injecting func params as entry block args if needed
        let regions: IdVec<Region<'db>> = raw
            .regions
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i == 0 && !raw.func_params.is_empty() {
                    // First region: inject func_params as entry block args.
                    // Save/restore block_id_map so inner labels don't clobber
                    // outer mappings.
                    let saved = self.save_block_labels(r);
                    self.pre_assign_blocks(r)?;
                    let blocks: IdVec<Block<'db>> = r
                        .blocks
                        .iter()
                        .enumerate()
                        .map(|(j, b)| {
                            if j == 0 {
                                self.build_block_with_extra_args(b, &raw.func_params)
                            } else {
                                self.build_block(b)
                            }
                        })
                        .collect::<Result<_, _>>()?;
                    let region = Region::new(self.db, self.location, blocks);
                    self.restore_block_labels(saved);
                    Ok(region)
                } else {
                    self.build_region(r)
                }
            })
            .collect::<Result<_, _>>()?;

        // Validate result name count matches declared result types.
        // This catches both mismatches (2 names vs 1 type) and names-without-
        // types (1 name vs 0 types), preventing semantically invalid Values.
        if !raw.results.is_empty() && raw.results.len() != results.len() {
            return Err(ParseError {
                message: format!(
                    "operation '{}.{}' declares {} result names but {} result types",
                    raw.dialect,
                    raw.op_name,
                    raw.results.len(),
                    results.len()
                ),
                offset: 0,
            });
        }

        let op = Operation::of(self.db, self.location, dialect, op_name)
            .operands(operands)
            .results(results)
            .attrs(attributes)
            .regions(regions)
            .build();

        // Register result values
        for (i, name) in raw.results.iter().enumerate() {
            let value = op.result(self.db, i);
            self.value_map.insert(name.to_string(), value);
        }

        Ok(op)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parse a TrunkIR module from its textual representation.
pub fn parse_module<'db>(
    db: &'db dyn salsa::Database,
    input: &str,
) -> Result<Operation<'db>, ParseError> {
    let mut remaining = input;
    ws.parse_next(&mut remaining).map_err(|e| ParseError {
        message: format!("lexer error: {}", e),
        offset: input.len() - remaining.len(),
    })?;

    // Parse the top-level operation
    let raw_op = raw_operation
        .parse_next(&mut remaining)
        .map_err(|e| ParseError {
            message: format!("parse error: {}", e),
            offset: input.len() - remaining.len(),
        })?;

    // Reject trailing input
    ws.parse_next(&mut remaining).map_err(|e| ParseError {
        message: format!("lexer error: {}", e),
        offset: input.len() - remaining.len(),
    })?;
    if !remaining.is_empty() {
        return Err(ParseError {
            message: "trailing input after top-level operation".to_string(),
            offset: input.len() - remaining.len(),
        });
    }

    // Build IR
    let mut builder = IrBuilder::new(db);
    builder.build_operation(&raw_op)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DialectOp, dialect::core, printer::print_op};
    use salsa::Database;
    use salsa_test_macros::salsa_test;

    // Salsa input to hold text for tracked function parsing
    #[salsa::input]
    struct TextInput {
        #[returns(ref)]
        text: String,
    }

    // Tracked wrapper: Salsa requires tracked struct creation inside tracked functions
    #[salsa::tracked]
    fn do_parse(db: &dyn salsa::Database, input: TextInput) -> Operation<'_> {
        parse_module(db, input.text(db)).expect("should parse")
    }

    #[salsa_test]
    fn test_parse_simple_module(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"core.module @test {
  func.func @main() -> core.i32 {
    ^bb0:
      %0 = arith.const {value = 40} : core.i32
      %1 = arith.const {value = 2} : core.i32
      %2 = arith.add %0, %1 : core.i32
      func.return %2
  }
}"#
            .to_string(),
        );

        let op = do_parse(db, input);
        let module = core::Module::from_operation(db, op).expect("should be a module");
        assert_eq!(module.name(db).to_string(), "test");

        // Verify round-trip: print -> re-parse -> print -> compare
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "round-trip failed");
    }

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

    #[salsa_test]
    fn test_roundtrip_with_params(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"core.module @test {
  func.func @add(%x: core.i32, %y: core.i32) -> core.i32 {
    ^bb0(%x: core.i32, %y: core.i32):
      %0 = arith.add %x, %y : core.i32
      func.return %0
  }
}"#
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "round-trip failed");
    }

    #[test]
    fn test_parse_trailing_input() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.nil {\n",
                    "    func.return\n",
                    "  }\n",
                    "} garbage",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on trailing input");
        assert!(
            err.message.contains("trailing input"),
            "unexpected error: {}",
            err.message
        );
    }

    #[salsa_test]
    fn test_parse_trailing_whitespace_ok(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    func.return\n",
                "  }\n",
                "}  \n  \n",
            )
            .to_string(),
        );
        // Should succeed — do_parse calls .expect() internally
        let _ = do_parse(db, input);
    }

    #[test]
    fn test_parse_string_escapes() {
        use winnow::prelude::*;

        // Basic escapes
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

    #[salsa_test]
    fn test_string_escape_roundtrip(db: &salsa::DatabaseImpl) {
        // Build an operation with a string attribute containing special chars
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    %0 = test.op {msg = \"line1\\nline2\\ttab\\r\\0end\"} : core.nil\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "string escape round-trip failed");
    }

    #[test]
    fn test_parse_symbol_ref_escapes() {
        use winnow::prelude::*;

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

    #[salsa_test]
    fn test_symbol_escape_roundtrip(db: &salsa::DatabaseImpl) {
        // Module with a function whose name contains special characters
        // that need escaping in quoted symbols
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @\"has\\\\slash\"() -> core.nil {\n",
                "    func.return\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "symbol escape round-trip failed");
    }

    #[test]
    fn test_parse_undefined_operand() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.i32 {\n",
                    "    ^bb0:\n",
                    "      %0 = arith.add %x, %y : core.i32\n",
                    "      func.return %0\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on undefined operand");
        assert!(
            err.message.contains("%x"),
            "error should mention the undefined value: {}",
            err.message
        );
        assert!(
            err.message.contains("arith.add"),
            "error should mention the operation: {}",
            err.message
        );
    }

    // ---- Fix 4: type attributes round-trip ----

    #[salsa_test]
    fn test_type_attrs_roundtrip(db: &salsa::DatabaseImpl) {
        // A function type with an effect attribute: core.func(...) {effect = ...}
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.i32 effects core.nil {\n",
                "    %0 = arith.const {value = 1} : core.i32\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "type attrs round-trip failed");
    }

    #[test]
    fn test_parse_type_with_attrs() {
        use winnow::prelude::*;

        let mut input = "core.func(core.nil, core.i32) {effect = core.nil}";
        let result = raw_type.parse_next(&mut input).expect("should parse");
        assert_eq!(result.dialect, "core");
        assert_eq!(result.name, "func");
        assert_eq!(result.params.len(), 2);
        assert_eq!(result.attrs.len(), 1);
        assert_eq!(result.attrs[0].0, "effect");
    }

    // ---- Fix 2: duplicate block labels ----

    #[test]
    fn test_parse_duplicate_block_label() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.i32 {\n",
                    "    ^bb0:\n",
                    "      %0 = arith.const {value = 1} : core.i32\n",
                    "      func.return %0\n",
                    "    ^bb0:\n",
                    "      %1 = arith.const {value = 2} : core.i32\n",
                    "      func.return %1\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on duplicate block label");
        assert!(
            err.message.contains("duplicate block label"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix 1: entry-block arg validation ----

    #[test]
    fn test_parse_entry_block_arity_mismatch() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // func declares 2 params but entry block has 1 arg
                concat!(
                    "core.module @test {\n",
                    "  func.func @f(%x: core.i32, %y: core.i32) -> core.i32 {\n",
                    "    ^bb0(%x: core.i32):\n",
                    "      func.return %x\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on arity mismatch");
        assert!(
            err.message.contains("args") && err.message.contains("params"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_entry_block_type_mismatch() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // func param is i32, block arg is f64
                concat!(
                    "core.module @test {\n",
                    "  func.func @f(%x: core.i32) -> core.i32 {\n",
                    "    ^bb0(%x: core.f64):\n",
                    "      func.return %x\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on type mismatch");
        assert!(
            err.message.contains("type mismatch"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_duplicate_block_arg_name() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f(%x: core.i32, %x: core.i32) -> core.i32 {\n",
                    "    func.return %x\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on duplicate arg name");
        assert!(
            err.message.contains("duplicate block argument name"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix 3: result count arity ----

    #[test]
    fn test_parse_result_count_mismatch() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // 2 result names but only 1 type
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.nil {\n",
                    "    %0, %1 = test.op : core.i32\n",
                    "    func.return %0\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on result count mismatch");
        assert!(
            err.message.contains("result names") && err.message.contains("result types"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix 5: integer_lit overflow ----

    #[test]
    fn test_parse_integer_lit_overflow() {
        use winnow::prelude::*;

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

    // ---- Fix 7: float exponent notation ----

    #[test]
    fn test_parse_float_exponent() {
        use winnow::prelude::*;

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

    // ---- Fix: result names without types ----

    #[test]
    fn test_parse_result_names_without_types() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // Result name but no `: type` annotation
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.nil {\n",
                    "    %0 = test.op\n",
                    "    func.return\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail: result name without type");
        assert!(
            err.message.contains("result names") && err.message.contains("result types"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix: nested regions with shared block labels ----

    #[salsa_test]
    fn test_nested_regions_shared_block_labels(db: &salsa::DatabaseImpl) {
        // Two sibling functions inside a module, each with ^bb0.
        // After parsing the first func, ^bb0 must NOT be clobbered for the
        // second func (or vice versa).  Round-trip verifies correctness.
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.i32 {\n",
                "    ^bb0:\n",
                "      %0 = arith.const {value = 1} : core.i32\n",
                "      func.return %0\n",
                "  }\n",
                "  func.func @g() -> core.i32 {\n",
                "    ^bb0:\n",
                "      %0 = arith.const {value = 2} : core.i32\n",
                "      func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "nested-region round-trip failed");
    }

    // ---- Fix: FloatBits round-trip (exponent forms) ----

    #[salsa_test]
    fn test_float_bits_roundtrip(db: &salsa::DatabaseImpl) {
        // Use a value that Rust's Display formats with exponent notation.
        // The printer must ensure a decimal point so the parser accepts it.
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    %0 = test.op {big = 1.0e300, small = 1.0e-300} : core.nil\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        // The printed text must be re-parseable
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "float-bits round-trip failed");
    }

    // ---- Fix: type with attrs + empty parens round-trip ----

    #[salsa_test]
    fn test_type_empty_parens_attrs_roundtrip(db: &salsa::DatabaseImpl) {
        // Type with attrs but no params: printer emits `custom.t() {k = v}`.
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    %0 = test.op {ty = custom.t() {size = 4}} : core.nil\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(
            printed, printed2,
            "type empty-parens attrs round-trip failed"
        );
    }

    // ========================================================================
    // Property-based tests (proptest)
    // ========================================================================

    mod proptest_fuzz {
        use super::{TextInput, parse_module, raw_attr_value, raw_type};
        use crate::printer::print_op;
        use proptest::prelude::*;
        use salsa::Database;
        use winnow::prelude::*;

        /// Tracked wrapper: attempts parse and returns whether it succeeded.
        /// Panics from Salsa tracked struct creation are avoided because
        /// this function IS a tracked function.
        #[salsa::tracked]
        fn try_parse<'db>(
            db: &'db dyn salsa::Database,
            input: TextInput,
        ) -> Option<crate::Operation<'db>> {
            parse_module(db, input.text(db)).ok()
        }

        /// Parse and print, returning the printed text (or None on parse failure).
        #[salsa::tracked]
        fn parse_and_print(db: &dyn salsa::Database, input: TextInput) -> Option<String> {
            let op = parse_module(db, input.text(db)).ok()?;
            Some(print_op(db, op))
        }

        /// Valid IR texts used as seed corpus for mutation.
        fn seed_corpus() -> Vec<&'static str> {
            vec![
                // Simple module with no-arg function
                concat!(
                    "core.module @test {\n",
                    "  func.func @main() -> core.i32 {\n",
                    "    %0 = arith.const {value = 40} : core.i32\n",
                    "    %1 = arith.const {value = 2} : core.i32\n",
                    "    %2 = arith.add %0, %1 : core.i32\n",
                    "    func.return %2\n",
                    "  }\n",
                    "}",
                ),
                // Function with named params
                concat!(
                    "core.module @test {\n",
                    "  func.func @add(%x: core.i32, %y: core.i32) -> core.i32 {\n",
                    "    %0 = arith.add %x, %y : core.i32\n",
                    "    func.return %0\n",
                    "  }\n",
                    "}",
                ),
                // Multiple functions
                concat!(
                    "core.module @test {\n",
                    "  func.func @foo() -> core.nil {\n",
                    "    %0 = arith.const {value = 42} : core.i32\n",
                    "    func.return %0\n",
                    "  }\n",
                    "  func.func @bar(%a: core.f64) -> core.f64 {\n",
                    "    func.return %a\n",
                    "  }\n",
                    "}",
                ),
                // Explicit block labels
                concat!(
                    "core.module @m {\n",
                    "  func.func @f() -> core.i32 {\n",
                    "    ^bb0:\n",
                    "      %0 = arith.const {value = 1} : core.i32\n",
                    "      func.return %0\n",
                    "  }\n",
                    "}",
                ),
                // Various attribute types
                concat!(
                    "core.module @test {\n",
                    "  func.func @g() -> core.nil {\n",
                    "    %0 = test.op {a = true, b = 99, c = \"hello\", d = @sym, e = unit} : core.nil\n",
                    "    func.return %0\n",
                    "  }\n",
                    "}",
                ),
            ]
        }

        /// Strategy: pick a seed and apply a random mutation.
        fn mutated_ir() -> impl Strategy<Value = String> {
            let seeds = seed_corpus();
            let n = seeds.len();
            (0..n, 0..1000usize, 0..5u8, proptest::num::u8::ANY).prop_map(
                move |(seed_idx, pos_raw, mutation_kind, random_byte)| {
                    let text = seeds[seed_idx];
                    let mut bytes = text.as_bytes().to_vec();
                    if bytes.is_empty() {
                        return String::new();
                    }
                    let pos = pos_raw % bytes.len();

                    match mutation_kind {
                        0 => {
                            // Replace byte
                            bytes[pos] = random_byte;
                        }
                        1 => {
                            // Delete byte
                            bytes.remove(pos);
                        }
                        2 => {
                            // Insert byte
                            bytes.insert(pos, random_byte);
                        }
                        3 => {
                            // Delete a chunk (up to 8 bytes)
                            let end = (pos + 8).min(bytes.len());
                            bytes.drain(pos..end);
                        }
                        _ => {
                            // Duplicate a chunk
                            let end = (pos + 8).min(bytes.len());
                            let chunk: Vec<u8> = bytes[pos..end].to_vec();
                            bytes.splice(pos..pos, chunk);
                        }
                    }

                    String::from_utf8(bytes).unwrap_or_default()
                },
            )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(2000))]

            /// Parser must never panic on arbitrary mutated input.
            #[test]
            fn parser_never_panics(input in mutated_ir()) {
                let db = salsa::DatabaseImpl::default();
                db.attach(|db| {
                    let ti = TextInput::new(db, input.clone());
                    let _ = try_parse(db, ti);
                });
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]

            /// Completely random strings must not panic the parser.
            #[test]
            fn parser_handles_random_strings(input in "\\PC{0,200}") {
                let db = salsa::DatabaseImpl::default();
                db.attach(|db| {
                    let ti = TextInput::new(db, input.clone());
                    let _ = try_parse(db, ti);
                });
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]

            /// Valid seed texts must round-trip: print(parse(text)) → parse → print = same.
            #[test]
            fn seed_corpus_round_trips(seed_idx in 0..3usize) {
                let seeds = seed_corpus();
                let text = seeds[seed_idx].to_string();
                let db = salsa::DatabaseImpl::default();
                db.attach(|db| {
                    let ti = TextInput::new(db, text.clone());
                    if let Some(printed) = parse_and_print(db, ti) {
                        let ti2 = TextInput::new(db, printed.clone());
                        if let Some(printed2) = parse_and_print(db, ti2) {
                            prop_assert_eq!(printed, printed2, "round-trip mismatch");
                        }
                    }
                    Ok(())
                })?;
            }
        }

        // Individual combinator fuzzing: type parser.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            #[test]
            fn type_parser_never_panics(input in "[a-z_.()0-9, ]{0,80}") {
                let mut s = input.as_str();
                let _ = raw_type.parse_next(&mut s);
            }
        }

        // Individual combinator fuzzing: attribute value parser.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            #[test]
            fn attr_parser_never_panics(input in "[a-z0-9_.@\"\\[\\]{}, =truefalsnui]{0,60}") {
                let mut s = input.as_str();
                let _ = raw_attr_value.parse_next(&mut s);
            }
        }
    }
}
