//! The operation a handler arm handles, and the syntax that writes each arm
//! of `tribute_control.handle` together with its binding.

use std::fmt::{self, Write as _};

use trunk_ir::asm_format::{
    CustomEntity, OpAsmDirective, RawCustomEntity, attach_entry_args, parse_entry_args,
    write_entry_args,
};
use trunk_ir::dialect::core::Never;
use trunk_ir::ops::DialectType;
use trunk_ir::parser::raw::{
    RawAttribute, RawType, ident, raw_region, raw_type, symbol_path, symbol_ref, ws,
};
use trunk_ir::printer::OpPrintHelper;
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

    /// Write `prefix`, then `op @Ability @name`.
    pub fn write_header(&self, h: &mut OpPrintHelper<'_, '_>, prefix: &str) -> fmt::Result {
        write!(h, "{prefix}{} ", self.kind.keyword())?;
        h.write_attribute(&Attribute::SymbolRef(self.ability.clone()))?;
        h.write_char(' ')?;
        h.write_symbol_text(h.ctx().str(self.op_name))
    }

    /// Parse the header [`Self::write_header`] writes. It backtracks when
    /// the input does not start with a kind keyword, so a list of arms ends
    /// there.
    pub fn parse_header(input: &mut &str) -> winnow::ModalResult<RawHeader> {
        use winnow::Parser as _;

        let checkpoint = *input;
        let Some(kind) = winnow::combinator::opt(ident)
            .parse_next(input)?
            .and_then(OperationKind::from_keyword)
        else {
            *input = checkpoint;
            return Err(winnow::error::ErrMode::Backtrack(
                winnow::error::ContextError::new(),
            ));
        };
        let (_, ability, _, op_name) = (ws, symbol_path, ws, symbol_ref)
            .parse_next(input)
            .map_err(|e| e.cut())?;
        Ok(RawHeader {
            kind,
            ability,
            op_name,
        })
    }
}

/// A binding header as [`HandlerBinding::parse_header`] reads it, still
/// without its result type.
#[derive(Clone, Debug)]
pub struct RawHeader {
    pub kind: OperationKind,
    pub ability: Vec<String>,
    pub op_name: String,
}

impl RawHeader {
    /// The binding as a raw `handlers` element, with its result type.
    pub fn into_attribute(self, operation_result_type: RawType<'_>) -> RawAttribute<'_> {
        RawAttribute::Dict(vec![
            ("ability".into(), RawAttribute::SymbolRef(self.ability)),
            ("op_name".into(), RawAttribute::String(self.op_name)),
            (
                "kind".into(),
                RawAttribute::String(self.kind.keyword().to_owned()),
            ),
            (
                "operation_result_type".into(),
                RawAttribute::Type(operation_result_type),
            ),
        ])
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
/// ` op @State @get(%x: A, %k: K) : R { .. }`: the binding, the arm's entry
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

    fn print(h: &mut OpPrintHelper<'_, '_>, arms: Self::Parts<'_>) -> fmt::Result {
        for (binding, arm) in arms {
            binding.write_header(h, " ")?;
            write_entry_args(h, arm)?;
            h.write_str(" : ")?;
            h.write_type(binding.operation_result_type)?;
            h.write_char(' ')?;
            h.print_region_without_entry(arm)?;
        }
        Ok(())
    }

    fn parse<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<RawCustomEntity<'a>>> {
        use winnow::Parser as _;
        use winnow::combinator::{preceded, repeat};

        let arms: Vec<_> = repeat(0.., preceded(ws, parse_arm)).parse_next(input)?;
        let (handlers, regions) = arms.into_iter().unzip();
        Ok(vec![
            RawCustomEntity::Attribute(Some(RawAttribute::List(handlers))),
            RawCustomEntity::Regions(regions),
        ])
    }
}

fn parse_arm<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<(RawAttribute<'a>, trunk_ir::parser::raw::RawRegion<'a>)> {
    use winnow::Parser as _;

    let header = HandlerBinding::parse_header(input)?;
    let (args, _, _, _, result, _, mut region) =
        (parse_entry_args, ws, ':', ws, raw_type, ws, raw_region)
            .parse_next(input)
            .map_err(|e| e.cut())?;
    attach_entry_args(&mut region, args)?;
    Ok((header.into_attribute(result), region))
}
