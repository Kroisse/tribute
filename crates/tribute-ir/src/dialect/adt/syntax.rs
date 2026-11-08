//! The custom syntax that writes a nominal declaration's names as the
//! positional list they name: `(x, y)` for a struct's fields and
//! `[Dot, Rect]` for an enum's variants.

use std::fmt::{self, Write as _};

use itertools::Itertools as _;

use trunk_ir::IrContext;
use trunk_ir::asm_format::{CustomEntity, OpAsmDirective, ParsedEntity};
use trunk_ir::parser::lexer::name_token;
use trunk_ir::parser::{OpAsmParser, ParseResult};
use trunk_ir::printer::{OpAsmPrinter, display_name};
use trunk_ir::types::Attribute;

/// `adt.struct_decl`'s field names: `(x, y)`.
pub struct FieldNames;

/// `adt.enum_decl`'s variant names: `[Dot, Rect]`.
pub struct VariantNames;

impl OpAsmDirective for FieldNames {
    type Parts<'a> = Vec<&'a str>;

    fn decompose<'a>(
        ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        decompose_names(ctx, entities)
    }

    fn print(p: &mut OpAsmPrinter<'_>, names: Self::Parts<'_>) -> fmt::Result {
        print_names(p, ("(", ")"), names)
    }

    fn parse(p: &mut OpAsmParser<'_>) -> ParseResult<Vec<ParsedEntity>> {
        parse_names(p, ("(", ")"), "a field name")
    }
}

impl OpAsmDirective for VariantNames {
    type Parts<'a> = Vec<&'a str>;

    fn decompose<'a>(
        ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        decompose_names(ctx, entities)
    }

    fn print(p: &mut OpAsmPrinter<'_>, names: Self::Parts<'_>) -> fmt::Result {
        print_names(p, ("[", "]"), names)
    }

    fn parse(p: &mut OpAsmParser<'_>) -> ParseResult<Vec<ParsedEntity>> {
        parse_names(p, ("[", "]"), "a variant name")
    }
}

fn decompose_names<'a>(
    ctx: &'a IrContext,
    entities: &'a [CustomEntity<'a>],
) -> Option<Vec<&'a str>> {
    let [CustomEntity::Attribute(Some(Attribute::List(names)))] = entities else {
        return None;
    };
    names
        .iter()
        .map(|name| match name {
            Attribute::String(name) => Some(ctx.str(*name)),
            _ => None,
        })
        .collect()
}

fn print_names(
    p: &mut OpAsmPrinter<'_>,
    (open, close): (&str, &str),
    names: Vec<&str>,
) -> fmt::Result {
    let names = names.into_iter().map(display_name).format(", ");
    write!(p, "{open}{names}{close}")
}

fn parse_names(
    p: &mut OpAsmParser<'_>,
    (open, close): (&str, &str),
    expected: &'static str,
) -> ParseResult<Vec<ParsedEntity>> {
    let names = p.parse_delimited(open, close, |p| {
        let name = p.parse_token(expected, name_token)?;
        Ok(p.ctx_mut().string_attr(&name))
    })?;
    Ok(vec![ParsedEntity::Attribute(Some(Attribute::List(names)))])
}
