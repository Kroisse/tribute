//! The custom syntax that writes the position an `adt` update replaces
//! right after its source, as `%p[1, 2]` and `%a[%i]`.

use std::fmt::{self, Write as _};

use itertools::Itertools as _;

use trunk_ir::IrContext;
use trunk_ir::asm_format::{CustomEntity, OpAsmDirective, ParsedEntity};
use trunk_ir::parser::{OpAsmParser, ParseResult};
use trunk_ir::printer::OpAsmPrinter;
use trunk_ir::refs::ValueRef;
use trunk_ir::types::Attribute;

/// A struct source with the fields an update replaces: `%p[1, 2]`.
pub struct FieldSubscript;

impl OpAsmDirective for FieldSubscript {
    type Parts<'a> = (ValueRef, &'a [Attribute]);

    fn decompose<'a>(
        _ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        let [
            CustomEntity::Operands([source]),
            CustomEntity::Attribute(Some(Attribute::List(fields))),
        ] = entities
        else {
            return None;
        };
        fields
            .iter()
            .all(|field| matches!(field, Attribute::Int(field) if u32::try_from(*field).is_ok()))
            .then_some((*source, fields.as_slice()))
    }

    fn print(p: &mut OpAsmPrinter<'_>, (source, fields): Self::Parts<'_>) -> fmt::Result {
        let s = p.state();
        let fields = fields.iter().format_with(", ", |field, f| match field {
            Attribute::Int(field) => f(field),
            _ => unreachable!("decompose admits only integer fields"),
        });
        write!(p, "{}[{fields}]", s.value(source))
    }

    fn parse(p: &mut OpAsmParser<'_>) -> ParseResult<Vec<ParsedEntity>> {
        let source = p.parse_operand()?;
        let fields = p.parse_delimited("[", "]", |p| {
            let offset = p.offset();
            let field = p.parse_integer()?;
            u32::try_from(field)
                .map(|field| Attribute::Int(field.into()))
                .map_err(|_| p.error_at(offset, "a field index is a `u32`"))
        })?;
        Ok(vec![
            ParsedEntity::Operands(vec![source]),
            ParsedEntity::Attribute(Some(Attribute::List(fields))),
        ])
    }
}

/// An array source with the index of the element an update replaces:
/// `%a[%i]`.
pub struct ElementSubscript;

impl OpAsmDirective for ElementSubscript {
    type Parts<'a> = (ValueRef, ValueRef);

    fn decompose<'a>(
        _ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        let [
            CustomEntity::Operands([source]),
            CustomEntity::Operands([index]),
        ] = entities
        else {
            return None;
        };
        Some((*source, *index))
    }

    fn print(p: &mut OpAsmPrinter<'_>, (source, index): Self::Parts<'_>) -> fmt::Result {
        let s = p.state();
        write!(p, "{}[{}]", s.value(source), s.value(index))
    }

    fn parse(p: &mut OpAsmParser<'_>) -> ParseResult<Vec<ParsedEntity>> {
        let source = p.parse_operand()?;
        p.parse_punct("[")?;
        let index = p.parse_operand()?;
        p.parse_punct("]")?;
        Ok(vec![
            ParsedEntity::Operands(vec![source]),
            ParsedEntity::Operands(vec![index]),
        ])
    }
}
