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
    sym_name: Option<&'a str>,
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
    Symbol(&'a str),
    Type(RawType<'a>),
    List(Vec<RawAttribute<'a>>),
    Unit,
    Location(String, usize, usize),
    Bytes(Vec<u8>),
}

// ============================================================================
// Winnow parsers
// ============================================================================

/// Skip whitespace and comments.
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
fn symbol_ref<'a>(input: &mut &'a str) -> ModalResult<&'a str> {
    preceded(
        '@',
        alt((
            // Quoted: @"name with special chars"
            delimited('"', take_while(0.., |c: char| c != '"'), '"'),
            // Bare: @name
            take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
        )),
    )
    .parse_next(input)
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
        // Store as two's complement for negative values
        let signed = -(value as i64);
        Ok(u64::from_ne_bytes(signed.to_ne_bytes()))
    } else {
        Ok(value)
    }
}

/// Parse a float literal that MUST contain a decimal point.
/// This prevents `42` from being parsed as a float.
fn float_with_dot(input: &mut &str) -> ModalResult<f64> {
    // Match: [-]digits.digits
    let s = (
        opt('-'),
        take_while(1.., |c: char| c.is_ascii_digit()),
        '.',
        take_while(1.., |c: char| c.is_ascii_digit()),
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

/// Parse a type: dialect.name or dialect.name(params)
///
/// Note: type attributes `{...}` are NOT parsed here to avoid ambiguity with
/// operation attribute dicts. Type attrs are only used in specific contexts
/// (e.g., effect rows) and are handled by the printer, but for parsing we
/// rely on the type parameters to carry all necessary info.
fn raw_type<'a>(input: &mut &'a str) -> ModalResult<RawType<'a>> {
    let (dialect, name) = qualified_name.parse_next(input)?;

    // Optional type parameters
    let params = opt(delimited(
        ('(', ws),
        separated(0.., (ws, raw_type, ws).map(|(_, t, _)| t), ','),
        (ws, ')'),
    ))
    .parse_next(input)?
    .unwrap_or_default();

    Ok(RawType {
        dialect,
        name,
        params,
        attrs: vec![],
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
            RawAttribute::Symbol(s) => Attribute::Symbol(Symbol::from_dynamic(s)),
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
    fn pre_assign_blocks(&mut self, raw_region: &RawRegion<'_>) {
        for block in &raw_region.blocks {
            let block_id = BlockId::fresh();
            self.block_id_map.insert(block.label.to_string(), block_id);
        }
    }

    fn build_region(&mut self, raw: &RawRegion<'_>) -> Region<'db> {
        self.pre_assign_blocks(raw);
        let blocks: IdVec<Block<'db>> = raw.blocks.iter().map(|b| self.build_block(b)).collect();
        Region::new(self.db, self.location, blocks)
    }

    fn build_block_with_extra_args(
        &mut self,
        raw: &RawBlock<'_>,
        extra_args: &[(&str, RawType<'_>)],
    ) -> Block<'db> {
        let block_id = self
            .block_id_map
            .get(raw.label)
            .copied()
            .unwrap_or_else(BlockId::fresh);

        // Build block args: merge extra_args (from func params) with block's own args.
        // If the block already has args that match the extra_args, use the block's args.
        let all_args = if raw.args.is_empty() && !extra_args.is_empty() {
            extra_args.to_vec()
        } else {
            raw.args.clone()
        };

        let mut block_args = IdVec::new();
        for (i, (name, raw_ty)) in all_args.iter().enumerate() {
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
        let ops: IdVec<Operation<'db>> =
            raw.ops.iter().map(|op| self.build_operation(op)).collect();

        Block::new(self.db, block_id, self.location, block_args, ops)
    }

    fn build_block(&mut self, raw: &RawBlock<'_>) -> Block<'db> {
        self.build_block_with_extra_args(raw, &[])
    }

    fn build_operation(&mut self, raw: &RawOperation<'_>) -> Operation<'db> {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let op_name = Symbol::from_dynamic(raw.op_name);

        // Resolve operands
        let operands: IdVec<Value<'db>> = raw
            .operands
            .iter()
            .filter_map(|name| self.resolve_value(name))
            .collect();

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
        if let Some(name) = raw.sym_name {
            attributes.insert(
                Symbol::new("sym_name"),
                Attribute::Symbol(Symbol::from_dynamic(name)),
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
                    // First region: inject func_params as entry block args
                    self.pre_assign_blocks(r);
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
                        .collect();
                    Region::new(self.db, self.location, blocks)
                } else {
                    self.build_region(r)
                }
            })
            .collect();

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

        op
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

    // Build IR
    let mut builder = IrBuilder::new(db);
    Ok(builder.build_operation(&raw_op))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DialectOp, dialect::core, printer::print_op};
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
        let mut input = "3.14";
        let attr = raw_attr_value
            .parse_next(&mut input)
            .expect("should parse float");
        assert!(matches!(attr, RawAttribute::Float(f) if (f - 3.14).abs() < 1e-10));

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
        assert!(matches!(attr, RawAttribute::Symbol("foo")));
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
}
