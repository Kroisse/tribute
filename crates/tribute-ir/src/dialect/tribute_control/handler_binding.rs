//! The operation a handler arm handles, and the syntax that writes each arm
//! of `tribute_control.handle` together with its binding.

use std::fmt::{self, Write as _};

use trunk_ir::asm_format::{CustomEntity, OpAsmDirective, ParsedEntity};
use trunk_ir::dialect::core::Never;
use trunk_ir::ops::DialectType;
use trunk_ir::parser::{AsmParser, OpAsmParser, ParseResult, lexer};
use trunk_ir::printer::{OpAsmPrinter, display_symbol_ref, display_symbol_text};
use trunk_ir::refs::{RegionRef, TypeRef};
use trunk_ir::types::{Attribute, AttributeMap, StringRef};
use trunk_ir::{IrContext, SymbolPath};

use super::OperationKind;

/// The operation one handler arm handles: one element of the `handlers` of
/// `tribute_control.handle` and of `ability.handle`.
///
/// Stored as `{ability = @State, op_name = "get", kind = "op",
/// operation_result_type = core.i32}`, and written by those operations'
/// formats as `op @State @get` with the result type after the arm.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HandlerBinding {
    pub ability: SymbolPath,
    pub op_name: StringRef,
    pub kind: OperationKind,
    pub operation_result_type: TypeRef,
}

impl HandlerBinding {
    /// Decode one binding, or say why it is malformed.
    pub fn from_attribute(ctx: &IrContext, attr: &Attribute) -> Result<Self, String> {
        let Attribute::Dict(entries) = attr else {
            return Err("handlers element must be a dictionary".into());
        };
        if entries.len() != 4 {
            return Err(format!(
                "handlers element must have exactly ability, op_name, kind, and \
                 operation_result_type, found {} entries",
                entries.len()
            ));
        }
        let entries = entries.view(ctx);
        let ability = match entries.get("ability") {
            Some(Attribute::SymbolRef(ability)) => ability.clone(),
            _ => return Err("handlers ability must be a symbol reference".into()),
        };
        let op_name = match entries.get("op_name") {
            Some(Attribute::String(name)) => *name,
            _ => return Err("handlers op_name must be a string".into()),
        };
        let kind = match entries.get("kind") {
            Some(Attribute::String(kind)) => OperationKind::from_keyword(ctx.str(*kind))
                .ok_or_else(|| "handlers kind must be fn or op".to_string())?,
            _ => return Err("handlers kind must be fn or op".into()),
        };
        let operation_result_type = match entries.get("operation_result_type") {
            Some(Attribute::Type(ty)) => *ty,
            _ => return Err("handlers operation_result_type must be a type".into()),
        };
        Ok(Self {
            ability,
            op_name,
            kind,
            operation_result_type,
        })
    }

    /// Decode every binding of a `handlers` list, or say why one is
    /// malformed.
    pub fn list(ctx: &IrContext, attr: Option<&Attribute>) -> Result<Vec<Self>, String> {
        let Some(Attribute::List(items)) = attr else {
            return Err("requires a handlers list".into());
        };
        items
            .iter()
            .map(|item| Self::from_attribute(ctx, item))
            .collect()
    }

    pub fn to_attribute(&self, ctx: &mut IrContext) -> Attribute {
        let kind = ctx.string_attr(self.kind.keyword());
        Attribute::Dict(AttributeMap::from_entries(
            ctx,
            [
                ("ability", Attribute::SymbolRef(self.ability.clone())),
                ("op_name", Attribute::String(self.op_name)),
                ("kind", kind),
                (
                    "operation_result_type",
                    Attribute::Type(self.operation_result_type),
                ),
            ],
        ))
    }

    /// Whether the arm receives a resumption: a general (`op`) arm whose
    /// operation result is not `core.never`.
    pub fn is_resumptive(&self, ctx: &IrContext) -> bool {
        self.kind == OperationKind::Op
            && Never::from_type_ref(ctx, self.operation_result_type).is_none()
    }

    /// The header the handle formats write for this binding,
    /// `op @Ability @name`.
    pub fn header<'a>(&'a self, ctx: &'a IrContext) -> impl fmt::Display + 'a {
        fmt::from_fn(move |f| {
            write!(
                f,
                "{} {} {}",
                self.kind.keyword(),
                display_symbol_ref(&self.ability),
                display_symbol_text(ctx.str(self.op_name))
            )
        })
    }

    /// Parse the header [`Self::header`] writes, or `None`, leaving the
    /// text as it was, when no kind keyword is next, so a list of arms ends
    /// there.
    pub fn parse_header(p: &mut AsmParser<'_>) -> ParseResult<Option<BindingHeader>> {
        use winnow::Parser as _;

        let Some(kind) = p.parse_optional_token("an operation kind", |input| {
            lexer::ident
                .verify_map(OperationKind::from_keyword)
                .parse_next(input)
        })?
        else {
            return Ok(None);
        };
        let ability = p.parse_symbol_ref()?;
        let op_name = p.parse_symbol_name()?;
        Ok(Some(BindingHeader {
            kind,
            ability,
            op_name,
        }))
    }
}

/// A binding header as [`HandlerBinding::parse_header`] reads it, still
/// without its result type.
#[derive(Clone, Debug)]
pub struct BindingHeader {
    pub kind: OperationKind,
    pub ability: SymbolPath,
    pub op_name: String,
}

impl BindingHeader {
    /// The binding, with the result type written after it.
    pub fn with_result_type(
        self,
        ctx: &mut IrContext,
        operation_result_type: TypeRef,
    ) -> HandlerBinding {
        HandlerBinding {
            ability: self.ability,
            op_name: ctx.intern_str(&self.op_name),
            kind: self.kind,
            operation_result_type,
        }
    }
}

impl trunk_ir::attr_kind::AttrKind for HandlerBinding {
    const KIND: trunk_ir::op_schema::AttributeKind =
        trunk_ir::op_schema::AttributeKind::Dict(&trunk_ir::op_schema::AttributeKind::Any);
    type Out<'ctx> = HandlerBinding;
    type In = HandlerBinding;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> HandlerBinding {
        Self::from_attribute(ctx, attr)
            .unwrap_or_else(|error| panic!("unverified handlers element: {error}"))
    }

    fn write(ctx: &mut IrContext, value: HandlerBinding) -> Attribute {
        value.to_attribute(ctx)
    }
}

/// Each arm of a `tribute_control.handle` after its completion, as
/// ` op @State @get(%x: A) : R { .. }`: the binding, the arm's entry
/// block arguments, the operation result type, and the arm region.
pub struct HandleArms;

impl OpAsmDirective for HandleArms {
    type Parts<'a> = Vec<(HandlerBinding, RegionRef)>;

    const SPACED: bool = false;

    fn decompose<'a>(
        ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        let [
            CustomEntity::Attribute(Some(Attribute::List(handlers))),
            CustomEntity::Regions(arms),
        ] = entities
        else {
            return None;
        };
        // An arm without an entry block has no arguments to write, so the
        // handle prints generically.
        if handlers.len() != arms.len() || arms.iter().any(|&arm| ctx.region(arm).blocks.is_empty())
        {
            return None;
        }
        handlers
            .iter()
            .zip(arms)
            .map(|(binding, &arm)| Some((HandlerBinding::from_attribute(ctx, binding).ok()?, arm)))
            .collect()
    }

    fn print(p: &mut OpAsmPrinter<'_>, arms: Self::Parts<'_>) -> fmt::Result {
        let s = p.state();
        for (binding, arm) in arms {
            write!(
                p,
                " {}{} : {} ",
                binding.header(s.ctx()),
                s.entry_args(arm),
                s.ty(binding.operation_result_type)
            )?;
            p.print_region_without_entry(arm)?;
        }
        Ok(())
    }

    fn parse(p: &mut OpAsmParser<'_>) -> ParseResult<Vec<ParsedEntity>> {
        let mut handlers = Vec::new();
        let mut regions = Vec::new();
        while let Some(header) = HandlerBinding::parse_header(p)? {
            let args = p.parse_entry_args()?;
            p.parse_punct(":")?;
            let result = p.parse_type()?;
            regions.push(p.parse_region_with_entry(&args)?);
            let binding = header.with_result_type(p.ctx_mut(), result);
            handlers.push(binding.to_attribute(p.ctx_mut()));
        }
        Ok(vec![
            ParsedEntity::Attribute(Some(Attribute::List(handlers))),
            ParsedEntity::Regions(regions),
        ])
    }
}
