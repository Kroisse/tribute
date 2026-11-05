//! The operation a handler arm handles, and the syntax that writes each arm
//! of `tribute_control.handle` together with its binding.

use trunk_ir::asm_format::syntax::{
    Syntax, SyntaxDirective, delimited, keyword_of, lit, many, map, preceded, region_with_entry,
    symbol_name, symbol_ref, ty,
};
use trunk_ir::asm_format::{CustomEntity, ParsedEntity};
use trunk_ir::dialect::core::Never;
use trunk_ir::ops::DialectType;
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

    /// The syntax of the header the handle formats write for a binding,
    /// `op @Ability @name`. It declines when no kind keyword is next, so a
    /// list of arms ends there.
    pub fn header_syntax() -> impl Syntax<Value = BindingHeader> {
        map(
            (
                keyword_of(
                    "an operation kind",
                    OperationKind::from_keyword,
                    |kind: &OperationKind| kind.keyword(),
                ),
                preceded(lit(" "), symbol_ref()),
                preceded(lit(" "), symbol_name()),
            ),
            |(kind, ability, op_name)| BindingHeader {
                kind,
                ability,
                op_name,
            },
            |header| (header.kind, header.ability.clone(), header.op_name),
        )
    }

    /// The binding with `header` and the result type written after it.
    pub fn from_header(header: BindingHeader, operation_result_type: TypeRef) -> Self {
        Self {
            ability: header.ability,
            op_name: header.op_name,
            kind: header.kind,
            operation_result_type,
        }
    }

    /// The header and the result type a format writes for this binding.
    pub fn header(&self) -> (BindingHeader, TypeRef) {
        let header = BindingHeader {
            kind: self.kind,
            ability: self.ability.clone(),
            op_name: self.op_name,
        };
        (header, self.operation_result_type)
    }
}

/// A binding header, `op @Ability @name`: the binding without its result
/// type.
#[derive(Clone, Debug)]
pub struct BindingHeader {
    pub kind: OperationKind,
    pub ability: SymbolPath,
    pub op_name: StringRef,
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

impl SyntaxDirective for HandleArms {
    type Value = Vec<(HandlerBinding, RegionRef)>;

    fn syntax() -> impl Syntax<Value = Self::Value> {
        many(map(
            (
                preceded(lit(" "), HandlerBinding::header_syntax()),
                region_with_entry(delimited(lit(" : "), ty(), lit(" "))),
            ),
            |(header, (arm, result))| (HandlerBinding::from_header(header, result), arm),
            |(binding, arm)| {
                let (header, result) = binding.header();
                (header, (*arm, result))
            },
        ))
    }

    fn from_entities(ctx: &IrContext, entities: &[CustomEntity<'_>]) -> Option<Self::Value> {
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

    fn into_entities(ctx: &mut IrContext, arms: Self::Value) -> Vec<ParsedEntity> {
        let (handlers, regions) = arms
            .into_iter()
            .map(|(binding, arm)| (binding.to_attribute(ctx), arm))
            .unzip();
        vec![
            ParsedEntity::Attribute(Some(Attribute::List(handlers))),
            ParsedEntity::Regions(regions),
        ]
    }
}

trunk_ir::syntax_directive!(HandleArms, spaced = false);
