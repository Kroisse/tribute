//! The custom syntax that writes each element of an `ability` operation's
//! list attribute together with the operand it pairs with.

use trunk_ir::asm_format::syntax::{
    Syntax, SyntaxDirective, delimited, lit, map, operand, optional, preceded, separated,
    symbol_ref, terminated, ty,
};
use trunk_ir::asm_format::{CustomEntity, ParsedEntity};
use trunk_ir::refs::ValueRef;
use trunk_ir::types::Attribute;
use trunk_ir::{IrContext, SymbolPath};

use super::HandlerBinding;

/// `ability.handle`'s bindings, each with its arm closure, one per line:
/// `[op @State @get : core.i32 = %get, ..]`.
pub struct HandlerArms;

impl SyntaxDirective for HandlerArms {
    type Value = Vec<(HandlerBinding, ValueRef)>;

    fn syntax() -> impl Syntax<Value = Self::Value> {
        // Each arm on its own line; the printer indents each line.
        let arm = map(
            (
                preceded(lit("\n  "), HandlerBinding::header_syntax()),
                preceded(lit(" : "), ty()),
                preceded(lit(" = "), operand()),
            ),
            |(header, result, arm)| (HandlerBinding::from_header(header, result), arm),
            |(binding, arm)| {
                let (header, result) = binding.header();
                (header, result, *arm)
            },
        );
        map(
            delimited(
                lit("["),
                optional(terminated(separated(arm, ","), lit("\n"))),
                lit("]"),
            ),
            Option::unwrap_or_default,
            |arms| (!arms.is_empty()).then(|| arms.clone()),
        )
    }

    fn from_entities(ctx: &IrContext, entities: &[CustomEntity<'_>]) -> Option<Self::Value> {
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

    fn into_entities(ctx: &mut IrContext, arms: Self::Value) -> Vec<ParsedEntity> {
        let (handlers, arms) = arms
            .into_iter()
            .map(|(binding, arm)| (binding.to_attribute(ctx), arm))
            .unzip();
        vec![
            ParsedEntity::Attribute(Some(Attribute::List(handlers))),
            ParsedEntity::Operands(arms),
        ]
    }
}

trunk_ir::syntax_directive!(HandlerArms);

/// `ability.handle_dispatch`'s abilities, each with its dispatcher:
/// `[@State: %d0, @Console: %d1]`.
pub struct Dispatchers;

impl SyntaxDirective for Dispatchers {
    type Value = Vec<(SymbolPath, ValueRef)>;

    fn syntax() -> impl Syntax<Value = Self::Value> {
        delimited(
            lit("["),
            separated((terminated(symbol_ref(), lit(": ")), operand()), ", "),
            lit("]"),
        )
    }

    fn from_entities(_ctx: &IrContext, entities: &[CustomEntity<'_>]) -> Option<Self::Value> {
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
                Attribute::SymbolRef(ability) => Some((ability.clone(), dispatcher)),
                _ => None,
            })
            .collect()
    }

    fn into_entities(_ctx: &mut IrContext, dispatchers: Self::Value) -> Vec<ParsedEntity> {
        let (abilities, dispatchers) = dispatchers
            .into_iter()
            .map(|(ability, dispatcher)| (Attribute::SymbolRef(ability), dispatcher))
            .unzip();
        vec![
            ParsedEntity::Attribute(Some(Attribute::List(abilities))),
            ParsedEntity::Operands(dispatchers),
        ]
    }
}

trunk_ir::syntax_directive!(Dispatchers);
