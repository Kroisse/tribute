//! The custom syntax that writes each element of an `ability` operation's
//! list attribute together with the operand it pairs with.

use std::fmt::{self, Write as _};

use itertools::Itertools as _;

use trunk_ir::asm_format::{CustomEntity, OpAsmDirective, RawCustomEntity};
use trunk_ir::parser::raw::{RawAttribute, raw_type, symbol_path, value_ref, ws};
use trunk_ir::printer::{OpAsmPrinter, display_symbol_ref};
use trunk_ir::refs::ValueRef;
use trunk_ir::types::Attribute;
use trunk_ir::{IrContext, SymbolPath};

use super::HandlerBinding;

/// `ability.handle`'s bindings, each with its arm closure, one per line:
/// `[op @State @get : core.i32 = %get, ..]`.
pub struct HandlerArms;

impl OpAsmDirective for HandlerArms {
    type Parts<'a> = Vec<(HandlerBinding, ValueRef)>;

    fn decompose<'a>(
        ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        let [
            CustomEntity::Attribute(Some(Attribute::List(handlers))),
            CustomEntity::Operands(arms),
        ] = entities
        else {
            return None;
        };
        if handlers.len() != arms.len() {
            return None;
        }
        handlers
            .iter()
            .zip(arms.iter())
            .map(|(binding, &arm)| Some((HandlerBinding::from_attribute(ctx, binding).ok()?, arm)))
            .collect()
    }

    fn print(p: &mut OpAsmPrinter<'_>, arms: Self::Parts<'_>) -> fmt::Result {
        let s = p.state();
        // Each arm on its own line; the printer indents each line.
        let entries = arms.iter().format_with(",", |(binding, arm), f| {
            f(&format_args!(
                "\n  {} : {} = {}",
                binding.header(s.ctx()),
                s.ty(binding.operation_result_type),
                s.value(*arm)
            ))
        });
        let end = if arms.is_empty() { "" } else { "\n" };
        write!(p, "[{entries}{end}]")
    }

    fn parse<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<RawCustomEntity<'a>>> {
        use winnow::Parser as _;
        use winnow::combinator::{delimited, separated};

        let arms: Vec<(RawAttribute<'a>, &'a str)> = delimited(
            ('[', ws),
            separated(
                0..,
                (
                    ws,
                    HandlerBinding::parse_header,
                    ws,
                    ':',
                    ws,
                    raw_type,
                    ws,
                    '=',
                    ws,
                    value_ref,
                    ws,
                )
                    .map(|(_, header, _, _, _, ty, _, _, _, arm, _)| {
                        (header.into_attribute(ty), arm)
                    }),
                ',',
            ),
            (ws, ']'),
        )
        .parse_next(input)?;
        let (handlers, arms) = arms.into_iter().unzip();
        Ok(vec![
            RawCustomEntity::Attribute(Some(RawAttribute::List(handlers))),
            RawCustomEntity::Operands(arms),
        ])
    }
}

/// `ability.handle_dispatch`'s abilities, each with its dispatcher:
/// `[@State: %d0, @Console: %d1]`.
pub struct Dispatchers;

impl OpAsmDirective for Dispatchers {
    type Parts<'a> = Vec<(&'a SymbolPath, ValueRef)>;

    fn decompose<'a>(
        _ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        let [
            CustomEntity::Attribute(Some(Attribute::List(abilities))),
            CustomEntity::Operands(dispatchers),
        ] = entities
        else {
            return None;
        };
        if abilities.len() != dispatchers.len() {
            return None;
        }
        abilities
            .iter()
            .zip(dispatchers.iter())
            .map(|(ability, &dispatcher)| match ability {
                Attribute::SymbolRef(ability) => Some((ability, dispatcher)),
                _ => None,
            })
            .collect()
    }

    fn print(p: &mut OpAsmPrinter<'_>, dispatchers: Self::Parts<'_>) -> fmt::Result {
        let s = p.state();
        let dispatchers = dispatchers
            .into_iter()
            .format_with(", ", |(ability, dispatcher), f| {
                f(&format_args!(
                    "{}: {}",
                    display_symbol_ref(ability),
                    s.value(dispatcher)
                ))
            });
        write!(p, "[{dispatchers}]")
    }

    fn parse<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<RawCustomEntity<'a>>> {
        use winnow::Parser as _;
        use winnow::combinator::{delimited, separated};

        let dispatchers: Vec<(Vec<String>, &'a str)> = delimited(
            ('[', ws),
            separated(
                0..,
                (ws, symbol_path, ws, ':', ws, value_ref, ws)
                    .map(|(_, ability, _, _, _, dispatcher, _)| (ability, dispatcher)),
                ',',
            ),
            (ws, ']'),
        )
        .parse_next(input)?;
        let (abilities, dispatchers): (Vec<_>, _) = dispatchers.into_iter().unzip();
        Ok(vec![
            RawCustomEntity::Attribute(Some(RawAttribute::List(
                abilities.into_iter().map(RawAttribute::SymbolRef).collect(),
            ))),
            RawCustomEntity::Operands(dispatchers),
        ])
    }
}
