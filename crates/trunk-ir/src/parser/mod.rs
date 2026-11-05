//! IR text format parser.
//!
//! The parser reads text straight into an [`IrContext`],
//! as MLIR's does. [`AsmParser`] reads types, attributes, and tokens;
//! [`OpAsmParser`] extends it with operations, regions, and blocks, and is
//! what an operation's custom format reads with. [`lexer`] holds the token
//! functions both read the text through.

mod asm_parser;
pub mod lexer;
mod op_parser;
#[cfg(test)]
mod tests;

pub use asm_parser::AsmParser;
pub(crate) use asm_parser::is_keyword;
pub(crate) use op_parser::Operand;
pub use op_parser::{Argument, OpAsmParser, OperationState};

use crate::context::IrContext;
use crate::refs::OpRef;
use crate::rewrite::Module;

/// Parse error for IR text format.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    /// The byte offset in the parsed text where the error is.
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at offset {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

/// The result of a parser piece.
pub type ParseResult<T> = Result<T, ParseError>;

/// Parse a TrunkIR operation from its textual representation into arena IR.
pub fn parse_module(ctx: &mut IrContext, input: &str) -> Result<OpRef, ParseError> {
    OpAsmParser::new(ctx, input).parse_top_level()
}

/// Parse a type on its own, as [`print_type`](crate::printer::print_type)
/// writes it.
pub fn parse_type(ctx: &mut IrContext, input: &str) -> Result<crate::TypeRef, ParseError> {
    let mut p = AsmParser::new(ctx, input);
    let ty = p.parse_type()?;
    if !p.at_end() {
        return Err(p.error("trailing input after type"));
    }
    Ok(ty)
}

/// Parse an attribute on its own, in generic attribute syntax.
pub fn parse_attribute(
    ctx: &mut IrContext,
    input: &str,
) -> Result<crate::types::Attribute, ParseError> {
    let mut p = AsmParser::new(ctx, input);
    let attr = p.parse_attribute()?;
    if !p.at_end() {
        return Err(p.error("trailing input after attribute"));
    }
    Ok(attr)
}

/// Parse textual IR into an [`Module`], panicking on failure.
///
/// Convenience wrapper around [`parse_module`] for tests.
pub fn parse_test_module(ctx: &mut IrContext, input: &str) -> Module {
    let op = parse_module(ctx, input).unwrap_or_else(|e| {
        panic!(
            "Failed to parse test IR at offset {}:\n  {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    Module::new(ctx, op).unwrap_or_else(|| {
        panic!(
            "Parsed operation is not a core.module.\n\nInput:\n{}",
            input
        );
    })
}
