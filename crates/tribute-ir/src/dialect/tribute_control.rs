//! Verified source-logical callable and direct-style control dialect.
//!
//! `tribute_control` is the target-independent boundary between typed frontend
//! lowering and shared CPS legalization. It deliberately contains no physical
//! evidence, closure-environment, continuation, or backend carrier layout.

use std::collections::{HashMap, HashSet};
use std::fmt;

use itertools::Itertools;
use trunk_ir::dialect::{adt, arith, core};
use trunk_ir::op_interface::{RegionBranchOps, RegionBranchPoint, RegionSuccessor};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::{IrContext, Symbol};

use super::list;

/// Exact type-attribute key used by logical callable types.
pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";

/// Exact semantic identity copied from the trusted compiler-intrinsic registry.
pub const COMPILER_INTRINSIC_ATTR: &str = "tribute.compiler_intrinsic";

/// Source-logical callable convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CallingConvention {
    Direct = 0,
    EvidenceDirect = 1,
    Cps = 2,
}

impl CallingConvention {
    fn keyword(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::EvidenceDirect => "evidence_direct",
            Self::Cps => "cps",
        }
    }

    fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword {
            "direct" => Some(Self::Direct),
            "evidence_direct" => Some(Self::EvidenceDirect),
            "cps" => Some(Self::Cps),
            _ => None,
        }
    }
}

impl TryFrom<i128> for CallingConvention {
    type Error = i128;

    fn try_from(code: i128) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Direct),
            1 => Ok(Self::EvidenceDirect),
            2 => Ok(Self::Cps),
            _ => Err(code),
        }
    }
}

#[trunk_ir::dialect]
mod tribute_control {
    // Types
    struct ResumeToken<Input, Answer>;

    // Callable operations
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)]
        {}
    }

    fn lambda(#[rest] captures: ()) -> result {
        #[region(body)]
        {}
    }

    #[attr(func_ref: Symbol)]
    fn func_ref() -> result {}

    #[attr(callee: Symbol)]
    fn call(#[rest] args: ()) -> result {}

    fn call_indirect(callee: (), #[rest] args: ()) -> result {}

    fn r#return(value: ()) {}

    // Direct-style control operations
    #[attr(ability_ref: Type, op_name: Symbol, operation_kind: Symbol)]
    fn perform(#[rest] args: ()) -> result {}

    fn handle() -> result {
        #[region(body)]
        {}
        #[region(completion)]
        {}
        #[region(handlers)]
        {}
    }

    #[attr(
        ability_ref: Type,
        op_name: Symbol,
        kind: Symbol,
        operation_result_type: Type
    )]
    fn handler() {
        #[region(body)]
        {}
    }

    fn resume(resume_token: (), value: ()) -> result {}

    fn r#yield(value: ()) {}
}

/// Typed wrapper for `tribute_control.callable(Result, Params...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Callable(TypeRef);

impl DialectType for Callable {
    const DIALECT_NAME: &'static str = "tribute_control";
    const TYPE_NAME: &'static str = "callable";

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        (Self::matches(ctx, ty) && !ctx.types.get(ty).params.is_empty()).then_some(Self(ty))
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl From<Callable> for TypeRef {
    fn from(value: Callable) -> Self {
        value.0
    }
}

impl Callable {
    pub fn as_type_ref(self) -> TypeRef {
        self.0
    }

    pub fn result(self, ctx: &IrContext) -> TypeRef {
        ctx.types.get(self.0).params[0]
    }

    pub fn params(self, ctx: &IrContext) -> &[TypeRef] {
        &ctx.types.get(self.0).params[1..]
    }
}

/// Build a verified-shape logical callable type.
pub fn callable(
    ctx: &mut IrContext,
    result: TypeRef,
    params: impl IntoIterator<Item = TypeRef>,
    convention: CallingConvention,
) -> Callable {
    let data = TypeDataBuilder::new(Symbol::new("tribute_control"), Symbol::new("callable"))
        .param(result)
        .params(params)
        .attr(CALLING_CONVENTION_ATTR, Attribute::Int(convention as i128))
        .build();
    let ty = ctx.types.intern(data);
    Callable::from_type_ref(ctx, ty).expect("newly built callable type")
}

/// Read the exact convention metadata from a logical callable type.
pub fn callable_convention(ctx: &IrContext, ty: TypeRef) -> Option<CallingConvention> {
    let callable = Callable::from_type_ref(ctx, ty)?;
    ctx.types
        .get(callable.as_type_ref())
        .attrs
        .get_i128(CALLING_CONVENTION_ATTR)
        .and_then(|code| CallingConvention::try_from(code).ok())
}

/// Read the exact input and answer types carried by a resume token.
pub fn resume_token_parts(ctx: &IrContext, ty: TypeRef) -> Option<(TypeRef, TypeRef)> {
    if !ResumeToken::matches(ctx, ty) {
        return None;
    }
    let [input, answer] = ctx.types.get(ty).params.as_slice() else {
        return None;
    };
    Some((*input, *answer))
}

/// Build a body-less `tribute_control.func` declaration.
pub fn func_declaration(
    ctx: &mut IrContext,
    location: Location,
    sym_name: Symbol,
    callable_type: TypeRef,
) -> Func {
    let data = trunk_ir::OperationDataBuilder::new(
        location,
        Symbol::new("tribute_control"),
        Symbol::new("func"),
    )
    .attr("sym_name", Attribute::Symbol(sym_name))
    .attr("type", Attribute::Type(callable_type))
    .build(ctx);
    let op = ctx.create_op(data);
    Func::from_op(ctx, op).expect("newly built tribute_control.func")
}

impl Func {
    /// Return the optional semantic identity attached by the trusted frontend registry.
    pub fn compiler_intrinsic_identity(&self, ctx: &IrContext) -> Option<Symbol> {
        ctx.op(self.op_ref())
            .attributes
            .get_symbol(COMPILER_INTRINSIC_ATTR)
    }
}

// Only the named function is isolated. Lambda intentionally remains
// non-isolated so its exact external-reference/capture set can be validated.
inventory::submit! {
    trunk_ir::op_interface::IsolatedFromAboveOps::register("tribute_control", "func")
}

// These operations only create/refer to values and are safe for DCE.
inventory::submit! {
    trunk_ir::op_interface::PureOps::register("tribute_control", "func_ref")
}
inventory::submit! {
    trunk_ir::op_interface::PureOps::register("tribute_control", "lambda")
}

// === Custom assembly: tribute_control.func ===

fn print_symbol(h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>, symbol: Symbol) -> fmt::Result {
    h.write_attribute(&Attribute::Symbol(symbol))
}

fn callable_parts(
    ctx: &IrContext,
    ty: TypeRef,
) -> Option<(TypeRef, Vec<TypeRef>, CallingConvention)> {
    let callable = Callable::from_type_ref(ctx, ty)?;
    let data = ctx.types.get(callable.as_type_ref());
    let (&result, params) = data.params.split_first()?;
    let convention = callable_convention(ctx, ty)?;
    Some((result, params.to_vec(), convention))
}

fn print_extra_attributes(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    excluded: &[&str],
) -> fmt::Result {
    use fmt::Write;
    let excluded: HashSet<Symbol> = excluded
        .iter()
        .map(|key| Symbol::from_dynamic(key))
        .collect();
    let attrs: Vec<_> = h
        .ctx()
        .op(op)
        .attributes
        .iter()
        .filter(|(key, _)| !excluded.contains(key))
        .map(|(key, value)| (*key, value.clone()))
        .collect();
    if attrs.is_empty() {
        return Ok(());
    }
    write!(h, " attributes {{")?;
    for (index, (key, value)) in attrs.iter().enumerate() {
        if index > 0 {
            write!(h, ", ")?;
        }
        write!(h, "{key} = ")?;
        h.write_attribute(value)?;
    }
    write!(h, "}}")
}

fn print_signature_params(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    region: Option<RegionRef>,
    param_types: &[TypeRef],
) -> fmt::Result {
    use fmt::Write;
    write!(h, "(")?;
    if let Some(region) = region {
        let args = h
            .ctx()
            .region(region)
            .blocks
            .first()
            .map(|block| h.ctx().block_args(*block).to_vec())
            .unwrap_or_default();
        for (index, arg) in args.iter().copied().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            let name = h.assign_value_name(arg);
            write!(h, "{name}: ")?;
            h.write_type(h.ctx().value_ty(arg))?;
        }
    } else {
        for (index, ty) in param_types.iter().copied().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            write!(h, "%arg{index}: ")?;
            h.write_type(ty)?;
        }
    }
    write!(h, ")")
}

fn print_func(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    use fmt::Write;

    let data = h.ctx().op(op);
    let symbol = data.attributes.get_symbol("sym_name");
    let callable_ty = data.attributes.get_type("type");
    let region = data.regions.first().copied();
    let parts = callable_ty.and_then(|ty| callable_parts(h.ctx(), ty));

    write!(h, "{}tribute_control.func ", " ".repeat(indent))?;
    if let Some(symbol) = symbol {
        print_symbol(h, symbol)?;
    } else {
        write!(h, "@<missing>")?;
    }
    h.reset_numbering();

    let param_types = parts
        .as_ref()
        .map(|(_, params, _)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(h, region, param_types)?;

    if let Some((result, _, convention)) = parts {
        write!(h, " -> ")?;
        h.write_type(result)?;
        write!(h, " convention({})", convention.keyword())?;
    }
    print_extra_attributes(h, op, &["sym_name", "type", CALLING_CONVENTION_ATTR])?;

    if let Some(region) = region {
        writeln!(h, " {{")?;
        h.print_region_eliding_entry(region, indent + 2)?;
        writeln!(h, "{}}}", " ".repeat(indent))
    } else {
        writeln!(h)
    }
}

fn parse_convention(input: &mut &str) -> winnow::ModalResult<CallingConvention> {
    use trunk_ir::parser::raw::{ident, ws};
    use winnow::prelude::*;

    ws.parse_next(input)?;
    "convention".parse_next(input)?;
    ws.parse_next(input)?;
    '('.parse_next(input)?;
    ws.parse_next(input)?;
    let keyword = ident.parse_next(input)?;
    let Some(convention) = CallingConvention::from_keyword(keyword) else {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    };
    ws.parse_next(input)?;
    ')'.parse_next(input)?;
    Ok(convention)
}

fn callable_raw_type<'a>(
    result: trunk_ir::parser::raw::RawType<'a>,
    params: &[(&'a str, trunk_ir::parser::raw::RawType<'a>)],
    convention: CallingConvention,
) -> trunk_ir::parser::raw::RawType<'a> {
    use trunk_ir::parser::raw::{RawAttribute, RawType};
    let mut type_params = vec![result];
    type_params.extend(params.iter().map(|(_, ty)| ty.clone()));
    RawType::Concrete {
        dialect: "tribute_control",
        name: "callable",
        params: type_params,
        attrs: vec![(
            CALLING_CONVENTION_ATTR,
            RawAttribute::Int(convention as i128),
        )],
    }
}

fn has_duplicate_convention_attr(
    attrs: &[(&str, trunk_ir::parser::raw::RawAttribute<'_>)],
) -> bool {
    attrs.iter().any(|(key, _)| *key == CALLING_CONVENTION_ATTR)
}

fn parse_func<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    if !results.is_empty() || sym_name.is_none() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let params = func_params.parse_next(input)?;
    let result = return_type.parse_next(input)?;
    let convention = parse_convention(input)?;
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();
    if has_duplicate_convention_attr(&attributes)
        || attributes
            .iter()
            .any(|(key, _)| *key == "sym_name" || *key == "type")
    {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let mut regions = Vec::new();
    if input.starts_with('{') {
        let mut region = raw_region.parse_next(input)?;
        if !region.blocks.is_empty() {
            let mut merged = params.clone();
            merged.append(&mut region.blocks[0].args);
            region.blocks[0].args = merged;
        }
        regions.push(region);
    }

    let callable = callable_raw_type(result, &params, convention);
    let mut attributes = attributes;
    attributes.push(("type", RawAttribute::Type(callable)));

    Ok(RawOperation {
        results,
        dialect: "tribute_control",
        op_name: "func",
        sym_name,
        has_func_signature: false,
        func_params: vec![],
        return_type: None,
        operands: vec![],
        attributes,
        result_types: vec![],
        regions,
        successors: vec![],
    })
}

inventory::submit! {
    trunk_ir::op_interface::OpAsmFormat {
        dialect: "tribute_control",
        op_name: "func",
        print_fn: print_func,
        parse_fn: parse_func,
    }
}

// === Custom assembly: tribute_control.lambda ===

fn print_lambda(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    use fmt::Write;

    let result = h.ctx().op_results(op).first().copied();
    let result_name = result.map(|value| h.assign_value_name(value));
    let callable_ty = h.ctx().op_result_types(op).first().copied();
    let parts = callable_ty.and_then(|ty| callable_parts(h.ctx(), ty));
    let region = h.ctx().op(op).regions.first().copied();

    write!(h, "{}", " ".repeat(indent))?;
    if let Some(name) = result_name {
        write!(h, "{name} = ")?;
    }
    write!(h, "tribute_control.lambda")?;

    let param_types = parts
        .as_ref()
        .map(|(_, params, _)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(h, region, param_types)?;
    if let Some((result, _, convention)) = parts {
        write!(h, " -> ")?;
        h.write_type(result)?;
        write!(h, " convention({})", convention.keyword())?;
    }

    write!(h, " captures [")?;
    let captures: Vec<String> = h
        .ctx()
        .op_operands(op)
        .iter()
        .map(|value| h.get_value_name(*value).to_owned())
        .collect();
    for (index, capture) in captures.iter().enumerate() {
        if index > 0 {
            write!(h, ", ")?;
        }
        write!(h, "{capture}")?;
    }
    write!(h, "]")?;
    print_extra_attributes(h, op, &[CALLING_CONVENTION_ATTR])?;

    if let Some(region) = region {
        writeln!(h, " {{")?;
        h.print_region_eliding_entry(region, indent + 2)?;
        writeln!(h, "{}}}", " ".repeat(indent))
    } else {
        writeln!(h)
    }
}

fn parse_captures<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<&'a str>> {
    use trunk_ir::parser::raw::{value_ref, ws};
    use winnow::combinator::{delimited, separated};
    use winnow::prelude::*;

    ws.parse_next(input)?;
    "captures".parse_next(input)?;
    ws.parse_next(input)?;
    delimited(
        ('[', ws),
        separated(0.., (ws, value_ref, ws).map(|(_, value, _)| value), ','),
        (ws, ']'),
    )
    .parse_next(input)
}

fn parse_lambda<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    if results.len() != 1 || sym_name.is_some() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let params = func_params.parse_next(input)?;
    let result = return_type.parse_next(input)?;
    let convention = parse_convention(input)?;
    let captures = parse_captures(input)?;
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();
    if has_duplicate_convention_attr(&attributes) {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let mut region = raw_region.parse_next(input)?;
    if !region.blocks.is_empty() {
        let mut merged = params.clone();
        merged.append(&mut region.blocks[0].args);
        region.blocks[0].args = merged;
    }

    Ok(RawOperation {
        results,
        dialect: "tribute_control",
        op_name: "lambda",
        sym_name,
        has_func_signature: false,
        func_params: vec![],
        return_type: None,
        operands: captures,
        attributes,
        result_types: vec![callable_raw_type(result, &params, convention)],
        regions: vec![region],
        successors: vec![],
    })
}

inventory::submit! {
    trunk_ir::op_interface::OpAsmFormat {
        dialect: "tribute_control",
        op_name: "lambda",
        print_fn: print_lambda,
        parse_fn: parse_lambda,
    }
}

// === Explicit Tribute validation entry point ===

/// One resolved source ability-operation declaration used by symbol-aware
/// whole-IR validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationDeclaration {
    pub ability_ref: TypeRef,
    pub op_name: Symbol,
    pub kind: Symbol,
    pub parameter_types: Vec<TypeRef>,
    pub result_type: TypeRef,
}

/// One exact compiler-intrinsic declaration supplied by the typed frontend.
///
/// This metadata is deliberately kept outside textual TrunkIR.  The symbol and
/// operation attribute are accepted only when they match this declaration's
/// registered identity and complete logical callable type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerIntrinsicDeclaration {
    pub symbol: Symbol,
    pub identity: Symbol,
    pub callable_type: TypeRef,
}

impl CompilerIntrinsicDeclaration {
    pub fn new(symbol: Symbol, identity: Symbol, callable_type: TypeRef) -> Self {
        Self {
            symbol,
            identity,
            callable_type,
        }
    }
}

impl OperationDeclaration {
    pub fn new(
        ability_ref: TypeRef,
        op_name: Symbol,
        kind: Symbol,
        parameter_types: impl IntoIterator<Item = TypeRef>,
        result_type: TypeRef,
    ) -> Self {
        Self {
            ability_ref,
            op_name,
            kind,
            parameter_types: parameter_types.into_iter().collect(),
            result_type,
        }
    }
}

/// A focused `tribute_control` validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    pub op: Option<OpRef>,
    pub location: Option<Location>,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(op) = self.op {
            write!(f, "{op}: ")?;
        }
        write!(f, "{}", self.message)
    }
}

/// Result returned by the explicit Tribute validator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ok() {
            return write!(f, "tribute_control validation passed");
        }
        writeln!(
            f,
            "{} tribute_control validation error(s):",
            self.errors.len()
        )?;
        for error in &self.errors {
            writeln!(f, "  - {error}")?;
        }
        Ok(())
    }
}

fn push_op_error(
    ctx: &IrContext,
    op: OpRef,
    errors: &mut Vec<ValidationError>,
    message: impl Into<String>,
) {
    errors.push(ValidationError {
        op: Some(op),
        location: Some(ctx.op(op).location),
        message: message.into(),
    });
}

fn push_type_error(errors: &mut Vec<ValidationError>, message: impl Into<String>) {
    errors.push(ValidationError {
        op: None,
        location: None,
        message: message.into(),
    });
}

fn is_control_op(ctx: &IrContext, op: OpRef, name: &str) -> bool {
    let data = ctx.op(op);
    data.dialect == Symbol::new("tribute_control") && data.name == Symbol::from_dynamic(name)
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    let region = ctx.block(block).parent_region?;
    ctx.region(region).parent_op
}

fn parent_region(ctx: &IrContext, op: OpRef) -> Option<RegionRef> {
    let block = ctx.op(op).parent_block?;
    ctx.block(block).parent_region
}

fn is_never(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("never")
}

fn is_unresolved_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    let unresolved_name = data.name == Symbol::new("var")
        || data.name == Symbol::new("infer")
        || data.name == Symbol::new("unresolved");
    (data.dialect == Symbol::new("type") || data.dialect == Symbol::new("tribute"))
        && unresolved_name
}

fn contains_unresolved_type(ctx: &IrContext, ty: TypeRef, visiting: &mut HashSet<TypeRef>) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let data = ctx.types.get(ty);
    let unresolved = is_unresolved_type(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_unresolved_type(ctx, param, visiting));
    visiting.remove(&ty);
    unresolved
}

fn contains_forbidden_logical_component(
    ctx: &IrContext,
    ty: TypeRef,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    if !visiting.insert(ty) {
        return true;
    }
    let data = ctx.types.get(ty);
    let forbidden = (data.dialect == Symbol::new("func") && data.name == Symbol::new("func_sig"))
        || (data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure"))
        || ResumeToken::matches(ctx, ty)
        || is_unresolved_type(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_forbidden_logical_component(ctx, param, visiting));
    visiting.remove(&ty);
    forbidden
}

fn validate_control_types(ctx: &IrContext, errors: &mut Vec<ValidationError>) {
    for (ty, data) in ctx.types.iter() {
        if data.dialect != Symbol::new("tribute_control") {
            continue;
        }
        if data.name == Symbol::new("callable") {
            if data.params.is_empty() {
                push_type_error(
                    errors,
                    format!("{ty}: tribute_control.callable requires one result type"),
                );
                continue;
            }
            match data.attrs.get_i128(CALLING_CONVENTION_ATTR) {
                Some(code) if CallingConvention::try_from(code).is_ok() => {}
                Some(code) => push_type_error(
                    errors,
                    format!(
                        "{ty}: tribute_control.callable has unsupported {CALLING_CONVENTION_ATTR} code {code}; expected 0, 1, or 2"
                    ),
                ),
                None => push_type_error(
                    errors,
                    format!(
                        "{ty}: tribute_control.callable requires integer {CALLING_CONVENTION_ATTR}"
                    ),
                ),
            }
            for component in data.params.iter().copied() {
                if contains_forbidden_logical_component(ctx, component, &mut HashSet::new()) {
                    push_type_error(
                        errors,
                        format!(
                            "{ty}: callable component {component} is unresolved, recursive, physical, or contains a resume token"
                        ),
                    );
                }
            }
        } else if data.name == Symbol::new("resume_token") {
            if data.params.len() != 2 {
                push_type_error(
                    errors,
                    format!(
                        "{ty}: tribute_control.resume_token requires exactly input and answer types"
                    ),
                );
            } else {
                for component in data.params.iter().copied() {
                    if contains_forbidden_logical_component(ctx, component, &mut HashSet::new()) {
                        push_type_error(
                            errors,
                            format!(
                                "{ty}: resume token component {component} is not a resolved logical value type"
                            ),
                        );
                    }
                }
            }
        } else {
            push_type_error(
                errors,
                format!("{ty}: unsupported tribute_control type '{}'", data.name),
            );
        }
    }
}

fn validate_attr_keys(
    ctx: &IrContext,
    op: OpRef,
    required: &[&str],
    allowed_extra: bool,
    errors: &mut Vec<ValidationError>,
) {
    let data = ctx.op(op);
    for key in required {
        if data.attributes.get(*key).is_none() {
            push_op_error(ctx, op, errors, format!("requires '{key}' attribute"));
        }
    }
    if data.attributes.get(CALLING_CONVENTION_ATTR).is_some() {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "{CALLING_CONVENTION_ATTR} belongs only on tribute_control.callable, not on the operation"
            ),
        );
    }
    if !allowed_extra {
        let allowed: HashSet<Symbol> = required
            .iter()
            .map(|key| Symbol::from_dynamic(key))
            .collect();
        for key in data.attributes.keys() {
            if !allowed.contains(key) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    format!("has unsupported attribute '{key}'"),
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
enum AttributeKind {
    Symbol,
    Type,
}

impl AttributeKind {
    fn name(self) -> &'static str {
        match self {
            Self::Symbol => "Symbol",
            Self::Type => "Type",
        }
    }

    fn matches(self, attribute: &Attribute) -> bool {
        matches!(
            (self, attribute),
            (Self::Symbol, Attribute::Symbol(_)) | (Self::Type, Attribute::Type(_))
        )
    }
}

fn validate_attr_types(
    ctx: &IrContext,
    op: OpRef,
    required: &[(&str, AttributeKind)],
    errors: &mut Vec<ValidationError>,
) {
    for (key, kind) in required {
        if let Some(attribute) = ctx.op(op).attributes.get(*key)
            && !kind.matches(attribute)
        {
            push_op_error(
                ctx,
                op,
                errors,
                format!("attribute '{key}' must be {}", kind.name()),
            );
        }
    }
}

fn validate_arity(
    ctx: &IrContext,
    op: OpRef,
    operands: Option<usize>,
    results: Option<usize>,
    regions: Option<usize>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(expected) = operands
        && ctx.op_operands(op).len() != expected
    {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "expects {expected} operand(s), found {}",
                ctx.op_operands(op).len()
            ),
        );
    }
    if let Some(expected) = results
        && ctx.op_results(op).len() != expected
    {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "expects {expected} result(s), found {}",
                ctx.op_results(op).len()
            ),
        );
    }
    if let Some(expected) = regions
        && ctx.op(op).regions.len() != expected
    {
        push_op_error(
            ctx,
            op,
            errors,
            format!(
                "expects {expected} region(s), found {}",
                ctx.op(op).regions.len()
            ),
        );
    }
}

fn single_block(
    ctx: &IrContext,
    owner: OpRef,
    region: RegionRef,
    name: &str,
    errors: &mut Vec<ValidationError>,
) -> Option<BlockRef> {
    let blocks = &ctx.region(region).blocks;
    if let [block] = blocks.as_slice() {
        Some(*block)
    } else {
        push_op_error(
            ctx,
            owner,
            errors,
            format!(
                "{name} region expects exactly one block, found {}",
                blocks.len()
            ),
        );
        None
    }
}

fn terminator(
    ctx: &IrContext,
    owner: OpRef,
    block: BlockRef,
    dialect: &str,
    name: &str,
    errors: &mut Vec<ValidationError>,
) -> Option<OpRef> {
    let Some(&op) = ctx.block(block).ops.last() else {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("region must terminate with {dialect}.{name}"),
        );
        return None;
    };
    let data = ctx.op(op);
    if data.dialect != Symbol::from_dynamic(dialect) || data.name != Symbol::from_dynamic(name) {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("region must terminate with {dialect}.{name}"),
        );
        None
    } else {
        Some(op)
    }
}

fn check_types_equal(
    ctx: &IrContext,
    owner: OpRef,
    actual: &[TypeRef],
    expected: &[TypeRef],
    what: &str,
    errors: &mut Vec<ValidationError>,
) {
    if actual != expected {
        push_op_error(
            ctx,
            owner,
            errors,
            format!("{what} types do not match the logical signature"),
        );
    }
}

fn block_arg_types(ctx: &IrContext, block: BlockRef) -> Vec<TypeRef> {
    ctx.block_args(block)
        .iter()
        .map(|value| ctx.value_ty(*value))
        .collect()
}

fn value_types(ctx: &IrContext, values: &[ValueRef]) -> Vec<TypeRef> {
    values.iter().map(|value| ctx.value_ty(*value)).collect()
}

fn validate_callable_body(
    ctx: &IrContext,
    owner: OpRef,
    callable_ty: TypeRef,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let Some((result_ty, params, _)) = callable_parts(ctx, callable_ty) else {
        push_op_error(
            ctx,
            owner,
            errors,
            "requires a valid tribute_control.callable signature",
        );
        return;
    };
    let Some(block) = single_block(ctx, owner, body, "body", errors) else {
        return;
    };
    check_types_equal(
        ctx,
        owner,
        &block_arg_types(ctx, block),
        &params,
        "entry block argument",
        errors,
    );
    if let Some(ret) = terminator(ctx, owner, block, "tribute_control", "return", errors)
        && ctx.op_operands(ret).len() == 1
        && ctx.value_ty(ctx.op_operands(ret)[0]) != result_ty
    {
        push_op_error(
            ctx,
            owner,
            errors,
            "return operand type does not match callable result type",
        );
    }
}

fn collect_region_values(ctx: &IrContext, region: RegionRef, values: &mut HashSet<ValueRef>) {
    for block in ctx.region(region).blocks.iter().copied() {
        values.extend(ctx.block_args(block).iter().copied());
        for op in ctx.block(block).ops.iter().copied() {
            values.extend(ctx.op_results(op).iter().copied());
            for nested in ctx.op(op).regions.iter().copied() {
                collect_region_values(ctx, nested, values);
            }
        }
    }
}

fn walk_region_ops(ctx: &IrContext, region: RegionRef, callback: &mut impl FnMut(OpRef)) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            callback(op);
            for nested in ctx.op(op).regions.iter().copied() {
                walk_region_ops(ctx, nested, callback);
            }
        }
    }
}

fn validate_func_isolation(
    ctx: &IrContext,
    op: OpRef,
    body: RegionRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut defined = HashSet::new();
    collect_region_values(ctx, body, &mut defined);
    let mut invalid = Vec::new();
    walk_region_ops(ctx, body, &mut |nested| {
        for operand in ctx.op_operands(nested).iter().copied() {
            if !defined.contains(&operand) {
                invalid.push((nested, operand));
            }
        }
    });
    for (consumer, value) in invalid {
        push_op_error(
            ctx,
            op,
            errors,
            format!("isolated func body operation {consumer} references external value {value}"),
        );
    }
}

fn validate_return_or_yield_shape(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, Some(1), Some(0), Some(0), errors);
    validate_attr_keys(ctx, op, &[], false, errors);
    let Some(block) = ctx.op(op).parent_block else {
        push_op_error(ctx, op, errors, "must be attached to a block");
        return;
    };
    if ctx.block(block).ops.last().copied() != Some(op) {
        push_op_error(ctx, op, errors, "must be the final operation in its block");
    }
}

fn validate_func(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, Some(0), Some(0), None, errors);
    validate_attr_keys(ctx, op, &["sym_name", "type"], true, errors);
    validate_attr_types(
        ctx,
        op,
        &[
            ("sym_name", AttributeKind::Symbol),
            ("type", AttributeKind::Type),
        ],
        errors,
    );
    let Some(callable_ty) = ctx.op(op).attributes.get_type("type") else {
        return;
    };
    if !Callable::matches(ctx, callable_ty) {
        push_op_error(
            ctx,
            op,
            errors,
            "type attribute must be tribute_control.callable",
        );
    }
    match ctx.op(op).regions.as_slice() {
        [] => {}
        [body] => {
            validate_callable_body(ctx, op, callable_ty, *body, errors);
            validate_func_isolation(ctx, op, *body, errors);
        }
        regions => push_op_error(
            ctx,
            op,
            errors,
            format!("expects zero or one body region, found {}", regions.len()),
        ),
    }
}

fn validate_lambda(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, None, Some(1), Some(1), errors);
    validate_attr_keys(ctx, op, &[], true, errors);
    let Some(&callable_ty) = ctx.op_result_types(op).first() else {
        return;
    };
    if !Callable::matches(ctx, callable_ty) {
        push_op_error(
            ctx,
            op,
            errors,
            "result must have tribute_control.callable type",
        );
    }
    if let Some(&body) = ctx.op(op).regions.first() {
        validate_callable_body(ctx, op, callable_ty, body, errors);
    }
}

fn validate_func_ref(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, Some(0), Some(1), Some(0), errors);
    validate_attr_keys(ctx, op, &["func_ref"], false, errors);
    validate_attr_types(ctx, op, &[("func_ref", AttributeKind::Symbol)], errors);
    if let Some(&ty) = ctx.op_result_types(op).first()
        && !Callable::matches(ctx, ty)
    {
        push_op_error(
            ctx,
            op,
            errors,
            "result must have tribute_control.callable type",
        );
    }
}

fn validate_call(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, None, Some(1), Some(0), errors);
    validate_attr_keys(ctx, op, &["callee"], false, errors);
    validate_attr_types(ctx, op, &[("callee", AttributeKind::Symbol)], errors);
    let has_unresolved_type = value_types(ctx, ctx.op_operands(op))
        .into_iter()
        .chain(ctx.op_result_types(op).iter().copied())
        .any(|ty| contains_unresolved_type(ctx, ty, &mut HashSet::new()));
    if has_unresolved_type {
        push_op_error(ctx, op, errors, "operands and result must be resolved");
    }
}

fn validate_call_indirect(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, None, Some(1), Some(0), errors);
    validate_attr_keys(ctx, op, &[], false, errors);
    let Some((&callee, args)) = ctx.op_operands(op).split_first() else {
        push_op_error(ctx, op, errors, "requires a callable callee operand");
        return;
    };
    let callee_ty = ctx.value_ty(callee);
    let Some((result, params, _)) = callable_parts(ctx, callee_ty) else {
        push_op_error(
            ctx,
            op,
            errors,
            "callee operand must have tribute_control.callable type",
        );
        return;
    };
    check_types_equal(
        ctx,
        op,
        &value_types(ctx, args),
        &params,
        "argument",
        errors,
    );
    check_types_equal(
        ctx,
        op,
        ctx.op_result_types(op),
        &[result],
        "result",
        errors,
    );
}

fn validate_return(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    let Some(owner) = parent_op(ctx, op) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a tribute_control.func or lambda body",
        );
        return;
    };
    let callable_ty = if is_control_op(ctx, owner, "func") {
        ctx.op(owner).attributes.get_type("type")
    } else if is_control_op(ctx, owner, "lambda") {
        ctx.op_result_types(owner).first().copied()
    } else {
        None
    };
    let Some((result, _, _)) = callable_ty.and_then(|ty| callable_parts(ctx, ty)) else {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a tribute_control.func or lambda body",
        );
        return;
    };
    if let Some(&value) = ctx.op_operands(op).first()
        && ctx.value_ty(value) != result
    {
        push_op_error(
            ctx,
            op,
            errors,
            "operand type does not match enclosing callable result",
        );
    }
}

fn validate_perform(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, None, Some(1), Some(0), errors);
    validate_attr_keys(
        ctx,
        op,
        &["ability_ref", "op_name", "operation_kind"],
        false,
        errors,
    );
    validate_attr_types(
        ctx,
        op,
        &[
            ("ability_ref", AttributeKind::Type),
            ("op_name", AttributeKind::Symbol),
            ("operation_kind", AttributeKind::Symbol),
        ],
        errors,
    );
    match ctx.op(op).attributes.get_symbol("operation_kind") {
        Some(kind) if kind == Symbol::new("fn") || kind == Symbol::new("op") => {}
        Some(kind) => push_op_error(
            ctx,
            op,
            errors,
            format!("operation_kind must be @fn or @op, found @{kind}"),
        ),
        None => {}
    }
    let has_unresolved_type = value_types(ctx, ctx.op_operands(op))
        .into_iter()
        .chain(ctx.op_result_types(op).iter().copied())
        .any(|ty| contains_unresolved_type(ctx, ty, &mut HashSet::new()));
    if has_unresolved_type {
        push_op_error(ctx, op, errors, "operands and result must be resolved");
    }
}

fn validate_handle(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, Some(0), Some(1), Some(3), errors);
    validate_attr_keys(ctx, op, &[], false, errors);
    let [body_region, completion_region, handlers_region] = ctx.op(op).regions.as_slice() else {
        return;
    };
    let Some(body_block) = single_block(ctx, op, *body_region, "body", errors) else {
        return;
    };
    let Some(completion_block) = single_block(ctx, op, *completion_region, "completion", errors)
    else {
        return;
    };
    let Some(handlers_block) = single_block(ctx, op, *handlers_region, "handlers", errors) else {
        return;
    };
    if !ctx.block_args(body_block).is_empty() {
        push_op_error(ctx, op, errors, "body block must not have arguments");
    }
    if ctx.block_args(completion_block).len() != 1 {
        push_op_error(
            ctx,
            op,
            errors,
            "completion block must have exactly one argument",
        );
    }
    if !ctx.block_args(handlers_block).is_empty() {
        push_op_error(
            ctx,
            op,
            errors,
            "handlers table block must not have arguments",
        );
    }
    let body_yield = terminator(ctx, op, body_block, "tribute_control", "yield", errors);
    let completion_yield = terminator(
        ctx,
        op,
        completion_block,
        "tribute_control",
        "yield",
        errors,
    );
    if let (Some(body_yield), Some(&completion_arg)) =
        (body_yield, ctx.block_args(completion_block).first())
        && let Some(&body_value) = ctx.op_operands(body_yield).first()
        && ctx.value_ty(body_value) != ctx.value_ty(completion_arg)
    {
        push_op_error(
            ctx,
            op,
            errors,
            "completion argument type must match the body yield type",
        );
    }
    let handle_result = ctx.op_result_types(op).first().copied();
    if let (Some(result), Some(completion_yield)) = (handle_result, completion_yield)
        && let Some(&value) = ctx.op_operands(completion_yield).first()
        && ctx.value_ty(value) != result
    {
        push_op_error(
            ctx,
            op,
            errors,
            "completion yield type must match handle result type",
        );
    }
    let mut clauses = HashSet::new();
    for child in ctx.block(handlers_block).ops.iter().copied() {
        if !is_control_op(ctx, child, "handler") {
            push_op_error(
                ctx,
                op,
                errors,
                format!("handlers block may contain only tribute_control.handler, found {child}"),
            );
            continue;
        }
        let data = ctx.op(child);
        if let (Some(ability), Some(name)) = (
            data.attributes.get_type("ability_ref"),
            data.attributes.get_symbol("op_name"),
        ) && !clauses.insert((ability, name))
        {
            push_op_error(
                ctx,
                child,
                errors,
                format!("duplicate handler clause for {ability}::{name}"),
            );
        }
        if let Some(result) = handle_result
            && let Some(&region) = data.regions.first()
            && let Some(&block) = ctx.region(region).blocks.first()
            && let Some(&yield_op) = ctx.block(block).ops.last()
            && is_control_op(ctx, yield_op, "yield")
            && data.attributes.get_symbol("kind") == Some(Symbol::new("op"))
            && let Some(&value) = ctx.op_operands(yield_op).first()
            && ctx.value_ty(value) != result
        {
            push_op_error(
                ctx,
                child,
                errors,
                "general handler yield type must match enclosing handle result type",
            );
        }
    }
}

fn region_contains_resume(ctx: &IrContext, region: RegionRef) -> bool {
    let mut found = false;
    walk_region_ops(ctx, region, &mut |op| {
        found |= is_control_op(ctx, op, "resume");
    });
    found
}

fn validate_handler(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, Some(0), Some(0), Some(1), errors);
    validate_attr_keys(
        ctx,
        op,
        &["ability_ref", "op_name", "kind", "operation_result_type"],
        false,
        errors,
    );
    validate_attr_types(
        ctx,
        op,
        &[
            ("ability_ref", AttributeKind::Type),
            ("op_name", AttributeKind::Symbol),
            ("kind", AttributeKind::Symbol),
            ("operation_result_type", AttributeKind::Type),
        ],
        errors,
    );
    let kind = ctx.op(op).attributes.get_symbol("kind");
    if !matches!(kind, Some(k) if k == Symbol::new("fn") || k == Symbol::new("op"))
        && let Some(kind) = kind
    {
        push_op_error(
            ctx,
            op,
            errors,
            format!("kind must be @fn or @op, found @{kind}"),
        );
    }
    let Some(&body) = ctx.op(op).regions.first() else {
        return;
    };
    let Some(block) = single_block(ctx, op, body, "body", errors) else {
        return;
    };
    let yield_op = terminator(ctx, op, block, "tribute_control", "yield", errors);
    let result_type = ctx.op(op).attributes.get_type("operation_result_type");
    let handle_result = parent_op(ctx, op)
        .filter(|parent| is_control_op(ctx, *parent, "handle"))
        .and_then(|parent| ctx.op_result_types(parent).first().copied());
    let args = ctx.block_args(block);
    let last_token = args
        .last()
        .copied()
        .and_then(|value| resume_token_parts(ctx, ctx.value_ty(value)));
    match (kind, result_type) {
        (Some(kind), Some(operation_result))
            if kind == Symbol::new("op") && !is_never(ctx, operation_result) =>
        {
            let Some((token_input, token_answer)) = last_token else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resumptive @op handler requires a final resume_token block argument",
                );
                return;
            };
            if token_input != operation_result
                || handle_result.is_some_and(|answer| token_answer != answer)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resume token input/answer types do not match operation and handle results",
                );
            }
            if let (Some(yield_op), Some(answer)) = (yield_op, handle_result)
                && let Some(&value) = ctx.op_operands(yield_op).first()
                && ctx.value_ty(value) != answer
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "general handler yield type must match handle result type",
                );
            }
        }
        (Some(kind), Some(operation_result)) => {
            if last_token.is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "fn and op -> core.never handlers must not receive a resume token",
                );
            }
            if kind == Symbol::new("op")
                && is_never(ctx, operation_result)
                && region_contains_resume(ctx, body)
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "op -> core.never handler must not contain tribute_control.resume",
                );
            }
            let expected = if kind == Symbol::new("fn") {
                Some(operation_result)
            } else {
                handle_result
            };
            if let (Some(yield_op), Some(expected)) = (yield_op, expected)
                && let Some(&value) = ctx.op_operands(yield_op).first()
                && ctx.value_ty(value) != expected
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "handler yield type does not match its kind-specific result",
                );
            }
        }
        _ => {}
    }

    let placed = parent_op(ctx, op).is_some_and(|parent| {
        is_control_op(ctx, parent, "handle")
            && ctx
                .op(parent)
                .regions
                .get(2)
                .copied()
                .is_some_and(|handlers| parent_region(ctx, op) == Some(handlers))
    });
    if !placed {
        push_op_error(
            ctx,
            op,
            errors,
            "must be a direct child of a tribute_control.handle handlers region",
        );
    }
}

fn validate_resume(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_arity(ctx, op, Some(2), Some(1), Some(0), errors);
    validate_attr_keys(ctx, op, &[], false, errors);
    let [token_value, input_value] = ctx.op_operands(op) else {
        return;
    };
    let Some((token_input, token_answer)) = resume_token_parts(ctx, ctx.value_ty(*token_value))
    else {
        push_op_error(
            ctx,
            op,
            errors,
            "first operand must have tribute_control.resume_token type",
        );
        return;
    };
    if ctx.value_ty(*input_value) != token_input
        || ctx.op_result_types(op).first().copied() != Some(token_answer)
    {
        push_op_error(
            ctx,
            op,
            errors,
            "resume input/result types do not match the token input/answer types",
        );
    }
}

fn validate_yield(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    validate_return_or_yield_shape(ctx, op, errors);
    if let Some(&value) = ctx.op_operands(op).first()
        && ResumeToken::matches(ctx, ctx.value_ty(value))
    {
        push_op_error(ctx, op, errors, "must not yield a resume token");
    }
    let valid_owner = parent_op(ctx, op).is_some_and(|owner| {
        if is_control_op(ctx, owner, "handler") {
            true
        } else if is_control_op(ctx, owner, "handle") {
            let region = parent_region(ctx, op);
            ctx.op(owner).regions.first().copied() == region
                || ctx.op(owner).regions.get(1).copied() == region
        } else {
            false
        }
    });
    if !valid_owner {
        push_op_error(
            ctx,
            op,
            errors,
            "must terminate a handle body/completion or handler body",
        );
    }
}

fn validate_local_operation(ctx: &IrContext, op: OpRef, errors: &mut Vec<ValidationError>) {
    let data = ctx.op(op);
    if data.dialect != Symbol::new("tribute_control") {
        return;
    }
    match data.name.with_str(|name| name.to_owned()).as_str() {
        "func" => validate_func(ctx, op, errors),
        "lambda" => validate_lambda(ctx, op, errors),
        "func_ref" => validate_func_ref(ctx, op, errors),
        "call" => validate_call(ctx, op, errors),
        "call_indirect" => validate_call_indirect(ctx, op, errors),
        "return" => validate_return(ctx, op, errors),
        "perform" => validate_perform(ctx, op, errors),
        "handle" => validate_handle(ctx, op, errors),
        "handler" => validate_handler(ctx, op, errors),
        "resume" => validate_resume(ctx, op, errors),
        "yield" => validate_yield(ctx, op, errors),
        name => push_op_error(
            ctx,
            op,
            errors,
            format!("unsupported tribute_control operation '{name}'"),
        ),
    }
}

/// Validate only operation-local and type-local `tribute_control` invariants.
pub fn validate_local(ctx: &IrContext, module: Module) -> ValidationResult {
    let mut errors = Vec::new();
    validate_control_types(ctx, &mut errors);
    if let Some(body) = module.body(ctx) {
        walk_region_ops(ctx, body, &mut |op| {
            validate_local_operation(ctx, op, &mut errors);
        });
    }
    ValidationResult { errors }
}

fn is_core_module(ctx: &IrContext, op: OpRef) -> bool {
    let data = ctx.op(op);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("module")
}

fn walk_symbol_scope_ops(ctx: &IrContext, region: RegionRef, callback: &mut impl FnMut(OpRef)) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            if is_core_module(ctx, op) {
                continue;
            }
            callback(op);
            for nested in ctx.op(op).regions.iter().copied() {
                walk_symbol_scope_ops(ctx, nested, callback);
            }
        }
    }
}

fn collect_funcs(
    ctx: &IrContext,
    region: RegionRef,
    funcs: &mut HashMap<Symbol, OpRef>,
    errors: &mut Vec<ValidationError>,
) {
    walk_symbol_scope_ops(ctx, region, &mut |op| {
        if is_control_op(ctx, op, "func")
            && let Some(symbol) = ctx.op(op).attributes.get_symbol("sym_name")
            && let Some(previous) = funcs.insert(symbol, op)
        {
            push_op_error(
                ctx,
                op,
                errors,
                format!("duplicate function symbol @{symbol}; first defined by {previous}"),
            );
        }
    });
}

fn same_source_signature(ctx: &IrContext, left: TypeRef, right: TypeRef) -> bool {
    let Some((left_result, left_params, _)) = callable_parts(ctx, left) else {
        return false;
    };
    let Some((right_result, right_params, _)) = callable_parts(ctx, right) else {
        return false;
    };
    left_result == right_result && left_params == right_params
}

fn validate_symbol_uses(
    ctx: &IrContext,
    body: RegionRef,
    funcs: &HashMap<Symbol, OpRef>,
    errors: &mut Vec<ValidationError>,
) {
    walk_symbol_scope_ops(ctx, body, &mut |op| {
        if is_control_op(ctx, op, "func_ref") {
            let Some(symbol) = ctx.op(op).attributes.get_symbol("func_ref") else {
                return;
            };
            let Some(target) = funcs.get(&symbol).copied() else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    format!("unresolved function symbol @{symbol}"),
                );
                return;
            };
            let Some(target_ty) = ctx.op(target).attributes.get_type("type") else {
                return;
            };
            let Some(result_ty) = ctx.op_result_types(op).first().copied() else {
                return;
            };
            if !same_source_signature(ctx, target_ty, result_ty) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "func_ref result source signature does not match its target",
                );
            } else if let (Some(target_cc), Some(result_cc)) = (
                callable_convention(ctx, target_ty),
                callable_convention(ctx, result_ty),
            ) && result_cc < target_cc
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "func_ref result convention must be at least as strong as its target",
                );
            }
        } else if is_control_op(ctx, op, "call") {
            let Some(symbol) = ctx.op(op).attributes.get_symbol("callee") else {
                return;
            };
            let Some(target) = funcs.get(&symbol).copied() else {
                push_op_error(ctx, op, errors, format!("unresolved callee @{symbol}"));
                return;
            };
            let Some(target_ty) = ctx.op(target).attributes.get_type("type") else {
                return;
            };
            let Some((result, params, _)) = callable_parts(ctx, target_ty) else {
                return;
            };
            check_types_equal(
                ctx,
                op,
                &value_types(ctx, ctx.op_operands(op)),
                &params,
                "call argument",
                errors,
            );
            check_types_equal(
                ctx,
                op,
                ctx.op_result_types(op),
                &[result],
                "call result",
                errors,
            );
        }
    });
}

fn validate_module_symbol_scopes(
    ctx: &IrContext,
    module_op: OpRef,
    errors: &mut Vec<ValidationError>,
) {
    let regions = ctx.op(module_op).regions.to_vec();
    let mut funcs = HashMap::new();
    for region in regions.iter().copied() {
        collect_funcs(ctx, region, &mut funcs, errors);
    }
    for region in regions.iter().copied() {
        validate_symbol_uses(ctx, region, &funcs, errors);
    }

    fn visit_nested_modules(ctx: &IrContext, region: RegionRef, errors: &mut Vec<ValidationError>) {
        for block in ctx.region(region).blocks.iter().copied() {
            for op in ctx.block(block).ops.iter().copied() {
                if is_core_module(ctx, op) {
                    validate_module_symbol_scopes(ctx, op, errors);
                } else {
                    for nested in ctx.op(op).regions.iter().copied() {
                        visit_nested_modules(ctx, nested, errors);
                    }
                }
            }
        }
    }

    for region in regions {
        visit_nested_modules(ctx, region, errors);
    }
}

fn collect_external_references(
    ctx: &IrContext,
    region: RegionRef,
    defined: &HashSet<ValueRef>,
    external: &mut HashSet<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            for operand in ctx.op_operands(op).iter().copied() {
                if !defined.contains(&operand) {
                    external.insert(operand);
                }
            }
            if is_control_op(ctx, op, "func") {
                continue;
            }
            for nested in ctx.op(op).regions.iter().copied() {
                collect_external_references(ctx, nested, defined, external);
            }
        }
    }
}

fn validate_lambda_captures(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        if !is_control_op(ctx, op, "lambda") {
            return;
        }
        let Some(&region) = ctx.op(op).regions.first() else {
            return;
        };
        let captures = ctx.op_operands(op);
        let capture_set: HashSet<ValueRef> = captures.iter().copied().collect();
        if capture_set.len() != captures.len() {
            push_op_error(ctx, op, errors, "capture list contains duplicate values");
        }
        let mut defined = HashSet::new();
        collect_region_values(ctx, region, &mut defined);
        let mut external = HashSet::new();
        collect_external_references(ctx, region, &defined, &mut external);
        for missing in external.difference(&capture_set) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("capture list is missing external value {missing}"),
            );
        }
        for excess in capture_set.difference(&external) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("capture list contains unused external value {excess}"),
            );
        }
    });
}

fn declaration_map<'a>(
    declarations: &'a [OperationDeclaration],
    errors: &mut Vec<ValidationError>,
) -> HashMap<(TypeRef, Symbol), &'a OperationDeclaration> {
    let mut map = HashMap::new();
    for declaration in declarations {
        let key = (declaration.ability_ref, declaration.op_name);
        if map.insert(key, declaration).is_some() {
            push_type_error(
                errors,
                format!(
                    "duplicate operation declaration for {}::{}",
                    declaration.ability_ref, declaration.op_name
                ),
            );
        }
    }
    map
}

fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("adt") && data.name == Symbol::new("typeref")
}

fn contains_adt_typeref(ctx: &IrContext, ty: TypeRef, visiting: &mut HashSet<TypeRef>) -> bool {
    if !visiting.insert(ty) {
        return false;
    }
    let data = ctx.types.get(ty);
    let contains = is_adt_typeref(ctx, ty)
        || data
            .params
            .iter()
            .copied()
            .any(|param| contains_adt_typeref(ctx, param, visiting))
        || data
            .attrs
            .values()
            .any(|attribute| attribute_contains_adt_typeref(ctx, attribute, visiting));
    visiting.remove(&ty);
    contains
}

fn attribute_contains_adt_typeref(
    ctx: &IrContext,
    attribute: &Attribute,
    visiting: &mut HashSet<TypeRef>,
) -> bool {
    match attribute {
        Attribute::Type(ty) => contains_adt_typeref(ctx, *ty, visiting),
        Attribute::List(values) => values
            .iter()
            .any(|value| attribute_contains_adt_typeref(ctx, value, visiting)),
        _ => false,
    }
}

fn nominal_identity(ctx: &IrContext, ty: TypeRef) -> Option<Symbol> {
    let data = ctx.types.get(ty);
    (data.dialect == Symbol::new("adt")
        && matches!(
            data.name,
            name if name == Symbol::new("typeref")
                || name == Symbol::new("struct")
                || name == Symbol::new("enum")
        ))
    .then(|| data.attrs.get_symbol("name"))
    .flatten()
}

fn canonical_nominal_layouts(
    ctx: &IrContext,
    reachable_types: &HashSet<TypeRef>,
    errors: &mut Vec<ValidationError>,
) -> HashMap<Symbol, TypeRef> {
    let mut layouts = HashMap::new();
    let mut referenced_names = HashSet::new();
    let mut sorted_reachable_types = reachable_types.iter().copied().collect::<Vec<_>>();
    sorted_reachable_types.sort_unstable();
    for ty in sorted_reachable_types {
        let data = ctx.types.get(ty);
        if is_adt_typeref(ctx, ty) {
            if let Some(name) = data.attrs.get_symbol("name") {
                referenced_names.insert(name);
            }
            continue;
        }
        if data.dialect != Symbol::new("adt")
            || !(data.name == Symbol::new("struct") || data.name == Symbol::new("enum"))
        {
            continue;
        }
        let Some(name) = data.attrs.get_symbol("name") else {
            continue;
        };
        if let Some(previous) = layouts.get(&name).copied() {
            if previous != ty {
                push_type_error(
                    errors,
                    format!(
                        "nominal layout @{name} is declared more than once ({previous} and {ty})"
                    ),
                );
            }
        } else {
            layouts.insert(name, ty);
        }
    }
    for name in referenced_names {
        if layouts.contains_key(&name) {
            continue;
        }
        let candidates = ctx
            .types
            .iter()
            .filter_map(|(ty, data)| {
                (data.dialect == Symbol::new("adt")
                    && (data.name == Symbol::new("struct") || data.name == Symbol::new("enum"))
                    && data.attrs.get_symbol("name") == Some(name))
                .then_some(ty)
            })
            .collect::<Vec<_>>();
        if let [layout] = candidates.as_slice() {
            layouts.insert(name, *layout);
        } else if let [first, second, ..] = candidates.as_slice() {
            push_type_error(
                errors,
                format!("nominal layout @{name} is declared more than once ({first} and {second})"),
            );
        }
    }
    layouts
}

fn is_core_ptr(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("ptr")
}

fn collect_attribute_types(ctx: &IrContext, attribute: &Attribute, types: &mut HashSet<TypeRef>) {
    match attribute {
        Attribute::Type(ty) => collect_reachable_type(ctx, *ty, types),
        Attribute::List(values) => {
            for value in values {
                collect_attribute_types(ctx, value, types);
            }
        }
        _ => {}
    }
}

fn collect_reachable_type(ctx: &IrContext, ty: TypeRef, types: &mut HashSet<TypeRef>) {
    if !types.insert(ty) {
        return;
    }
    let data = ctx.types.get(ty);
    for parameter in data.params.iter().copied() {
        collect_reachable_type(ctx, parameter, types);
    }
    for attribute in data.attrs.values() {
        collect_attribute_types(ctx, attribute, types);
    }
}

/// Collect semantic types reachable from a module operation and its nested IR.
///
/// Context-wide type aliases are printer/parser conveniences, not IR roots:
/// an alias whose type is never used by an operation, operand, result, block
/// argument, or type-bearing attribute must not affect validation of this
/// module. A reachable `adt.typeref` may resolve one unique layout alias; an
/// ambiguous nominal name remains invalid.
fn collect_reachable_ir_types(ctx: &IrContext, module: OpRef) -> HashSet<TypeRef> {
    fn collect_op_types(ctx: &IrContext, op: OpRef, types: &mut HashSet<TypeRef>) {
        for value in ctx.op_operands(op).iter().chain(ctx.op_results(op).iter()) {
            collect_reachable_type(ctx, ctx.value_ty(*value), types);
        }
        for attribute in ctx.op(op).attributes.values() {
            collect_attribute_types(ctx, attribute, types);
        }
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for argument in ctx.block_args(block) {
                    collect_reachable_type(ctx, ctx.value_ty(*argument), types);
                }
            }
        }
    }

    let mut types = HashSet::new();
    collect_op_types(ctx, module, &mut types);
    for region in ctx.op(module).regions.iter().copied() {
        walk_region_ops(ctx, region, &mut |op| collect_op_types(ctx, op, &mut types));
    }
    types
}

fn raw_pointer_cast_origin(
    ctx: &IrContext,
    value: ValueRef,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    if !visiting.insert(value) {
        return false;
    }
    if is_core_ptr(ctx, ctx.value_ty(value)) {
        return true;
    }
    let ValueDef::OpResult(producer, _) = ctx.value_def(value) else {
        return false;
    };
    let transparent_cast = core::UnrealizedConversionCast::from_op(ctx, producer).is_ok()
        || adt::RefCast::from_op(ctx, producer).is_ok()
        || arith::Cast::from_op(ctx, producer).is_ok();
    transparent_cast
        && ctx
            .op_operands(producer)
            .first()
            .copied()
            .is_some_and(|operand| raw_pointer_cast_origin(ctx, operand, visiting))
}

fn validate_managed_reference_boundaries(
    ctx: &IrContext,
    body: RegionRef,
    reachable_types: &HashSet<TypeRef>,
    nominal_layouts: &HashMap<Symbol, TypeRef>,
    errors: &mut Vec<ValidationError>,
) {
    for ty in reachable_types.iter().copied() {
        if !is_adt_typeref(ctx, ty) {
            continue;
        }
        let data = ctx.types.get(ty);
        let Some(name) = data.attrs.get_symbol("name") else {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref requires nominal name metadata"),
            );
            continue;
        };
        if !data.params.is_empty() {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref cannot have type parameters"),
            );
        }
        if !nominal_layouts.contains_key(&name) {
            push_type_error(
                errors,
                format!("{ty}: adt.typeref @{name} has no verified nominal declaration"),
            );
        }
    }

    walk_region_ops(ctx, body, &mut |op| {
        let data = ctx.op(op);
        if data.dialect == Symbol::new("adt") && data.name == Symbol::new("ref_null") {
            let ([], [result], Some(declared)) = (
                ctx.op_operands(op),
                ctx.op_result_types(op),
                data.attributes.get_type("type"),
            ) else {
                push_op_error(ctx, op, errors, "adt.ref_null requires one typed result");
                return;
            };
            if *result != declared || !is_adt_typeref(ctx, *result) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_null result and type attribute must be the same managed adt.typeref",
                );
            }
        }
        if data.dialect == Symbol::new("adt") && data.name == Symbol::new("ref_cast") {
            let ([operand], [result], Some(declared)) = (
                ctx.op_operands(op),
                ctx.op_result_types(op),
                data.attributes.get_type("type"),
            ) else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_cast requires one typed operand and result",
                );
                return;
            };
            let operand_type = ctx.value_ty(*operand);
            let source = nominal_identity(ctx, operand_type);
            let target = nominal_identity(ctx, *result);
            if declared != *result
                || !is_adt_typeref(ctx, operand_type)
                || !is_adt_typeref(ctx, *result)
                || source.is_none()
                || source != target
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "adt.ref_cast requires compatible managed nominal reference types",
                );
            }
        }
        for result in ctx.op_results(op) {
            if is_adt_typeref(ctx, ctx.value_ty(*result))
                && raw_pointer_cast_origin(ctx, *result, &mut HashSet::new())
            {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "core.ptr cast chain cannot produce a managed adt.typeref",
                );
            }
        }
    });
}

fn compiler_intrinsic_map<'a>(
    ctx: &IrContext,
    declarations: &'a [CompilerIntrinsicDeclaration],
    errors: &mut Vec<ValidationError>,
) -> HashMap<Symbol, &'a CompilerIntrinsicDeclaration> {
    let mut map = HashMap::new();
    let mut previous = None;
    for declaration in declarations {
        let key = (declaration.symbol, declaration.identity);
        if previous.is_some_and(|previous| previous > key) {
            push_type_error(
                errors,
                "compiler intrinsic declarations are not deterministically ordered",
            );
        }
        previous = Some(key);
        if callable_convention(ctx, declaration.callable_type) != Some(CallingConvention::Direct) {
            push_type_error(
                errors,
                format!(
                    "registered compiler intrinsic @{} must use Direct calling convention",
                    declaration.symbol
                ),
            );
        }
        if map.insert(declaration.symbol, declaration).is_some() {
            push_type_error(
                errors,
                format!(
                    "duplicate compiler intrinsic declaration for @{}",
                    declaration.symbol
                ),
            );
        }
    }
    map
}

fn adt_projection_has_exact_callable_type(
    ctx: &IrContext,
    producer: OpRef,
    result_type: TypeRef,
    nominal_layouts: &HashMap<Symbol, TypeRef>,
) -> bool {
    if ctx.op_result_types(producer) != [result_type] {
        return false;
    }
    if let Ok(get) = adt::StructGet::from_op(ctx, producer) {
        let layout = get.r#type(ctx);
        return projection_source_matches_layout(ctx, get.r#ref(ctx), layout, nominal_layouts)
            && struct_field_type(ctx, layout, get.field(ctx)) == Some(result_type);
    }
    if let Ok(get) = adt::VariantGet::from_op(ctx, producer) {
        let layout = get.r#type(ctx);
        return projection_source_matches_layout(ctx, get.r#ref(ctx), layout, nominal_layouts)
            && variant_field_type(ctx, layout, get.tag(ctx), get.field(ctx)) == Some(result_type);
    }
    false
}

fn projection_source_matches_layout(
    ctx: &IrContext,
    source: ValueRef,
    layout: TypeRef,
    nominal_layouts: &HashMap<Symbol, TypeRef>,
) -> bool {
    let layout_identity = nominal_identity(ctx, layout);
    layout_identity.is_some_and(|identity| nominal_layouts.get(&identity) == Some(&layout))
        && nominal_identity(ctx, ctx.value_ty(source)) == layout_identity
}

fn struct_field_type(ctx: &IrContext, layout: TypeRef, field: u32) -> Option<TypeRef> {
    let data = ctx.types.get(layout);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("struct") {
        return None;
    }
    let Attribute::List(fields) = data.attrs.get("fields")? else {
        return None;
    };
    let Attribute::List(pair) = fields.get(field as usize)? else {
        return None;
    };
    match pair.as_slice() {
        [Attribute::Symbol(_), Attribute::Type(field_type)] => Some(*field_type),
        _ => None,
    }
}

fn variant_field_type(
    ctx: &IrContext,
    layout: TypeRef,
    tag: Symbol,
    field: u32,
) -> Option<TypeRef> {
    let data = ctx.types.get(layout);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("enum") {
        return None;
    }
    let Attribute::List(variants) = data.attrs.get("variants")? else {
        return None;
    };
    variants.iter().find_map(|variant| {
        let Attribute::List(pair) = variant else {
            return None;
        };
        let [Attribute::Symbol(variant_tag), Attribute::List(fields)] = pair.as_slice() else {
            return None;
        };
        (*variant_tag == tag).then(|| match fields.get(field as usize) {
            Some(Attribute::Type(field_type)) => Some(*field_type),
            _ => None,
        })?
    })
}

struct CallableProvenance<'a> {
    registered: &'a HashMap<Symbol, &'a CompilerIntrinsicDeclaration>,
    declarations: &'a HashMap<(TypeRef, Symbol), &'a OperationDeclaration>,
    nominal_layouts: &'a HashMap<Symbol, TypeRef>,
}

fn callable_block_arg_has_source_contract(
    ctx: &IrContext,
    block: BlockRef,
    index: u32,
    value: ValueRef,
    provenance: &CallableProvenance<'_>,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    let Some(region) = ctx.block(block).parent_region else {
        return false;
    };
    let Some(owner) = ctx.region(region).parent_op else {
        return false;
    };
    let value_type = ctx.value_ty(value);

    let callable_type = if is_control_op(ctx, owner, "func") {
        (ctx.op(owner).regions.as_slice() == [region])
            .then(|| ctx.op(owner).attributes.get_type("type"))
            .flatten()
    } else if is_control_op(ctx, owner, "lambda") {
        (ctx.op(owner).regions.as_slice() == [region])
            .then(|| ctx.op_result_types(owner).first().copied())
            .flatten()
    } else {
        None
    };
    if let Some((_, parameters, _)) = callable_type.and_then(|ty| callable_parts(ctx, ty)) {
        return parameters.get(index as usize) == Some(&value_type);
    }

    if is_control_op(ctx, owner, "handler") && ctx.op(owner).regions.as_slice() == [region] {
        let data = ctx.op(owner);
        if let (Some(ability), Some(name), Some(kind)) = (
            data.attributes.get_type("ability_ref"),
            data.attributes.get_symbol("op_name"),
            data.attributes.get_symbol("kind"),
        ) {
            return provenance
                .declarations
                .get(&(ability, name))
                .is_some_and(|declaration| {
                    declaration.kind == kind
                        && declaration.parameter_types.get(index as usize) == Some(&value_type)
                });
        }
    }

    let [body_region, completion_region, _] = ctx.op(owner).regions.as_slice() else {
        return false;
    };
    if !is_control_op(ctx, owner, "handle") || completion_region != &region || index != 0 {
        return false;
    }
    let [body_block] = ctx.region(*body_region).blocks.as_slice() else {
        return false;
    };
    let Some(body_yield) = ctx.block(*body_block).ops.last().copied() else {
        return false;
    };
    let [body_value] = ctx.op_operands(body_yield) else {
        return false;
    };
    is_control_op(ctx, body_yield, "yield")
        && ctx.value_ty(*body_value) == value_type
        && callable_has_semantic_provenance(ctx, *body_value, provenance, visiting)
}

fn callable_has_semantic_provenance(
    ctx: &IrContext,
    value: ValueRef,
    provenance: &CallableProvenance<'_>,
    visiting: &mut HashSet<ValueRef>,
) -> bool {
    if !visiting.insert(value) || Callable::from_type_ref(ctx, ctx.value_ty(value)).is_none() {
        return false;
    }
    match ctx.value_def(value) {
        ValueDef::BlockArg(block, index) => {
            callable_block_arg_has_source_contract(ctx, block, index, value, provenance, visiting)
        }
        ValueDef::OpResult(producer, _) => {
            is_control_op(ctx, producer, "lambda")
                || ((is_control_op(ctx, producer, "func_ref")
                    || is_control_op(ctx, producer, "call"))
                    && ctx
                        .op(producer)
                        .attributes
                        .get_symbol(if is_control_op(ctx, producer, "func_ref") {
                            "func_ref"
                        } else {
                            "callee"
                        })
                        .and_then(|symbol| resolve_visible_function(ctx, producer, symbol))
                        .is_some_and(|function| {
                            verified_callable_declaration(ctx, function, provenance.registered)
                        }))
                || (is_control_op(ctx, producer, "call_indirect")
                    && ctx.op_operands(producer).first().is_some_and(|callee| {
                        callable_has_semantic_provenance(ctx, *callee, provenance, visiting)
                    }))
                || list::Head::from_op(ctx, producer).is_ok_and(|head| {
                    head.element_type(ctx) == ctx.value_ty(value)
                        && ctx.op_result_types(producer) == [ctx.value_ty(value)]
                })
                || adt_projection_has_exact_callable_type(
                    ctx,
                    producer,
                    ctx.value_ty(value),
                    provenance.nominal_layouts,
                )
        }
    }
}

fn resolve_visible_function(ctx: &IrContext, use_op: OpRef, symbol: Symbol) -> Option<OpRef> {
    let mut owner = Some(use_op);
    let scope = loop {
        let current = owner?;
        if is_core_module(ctx, current) {
            break current;
        }
        owner = parent_op(ctx, current);
    };
    let body = ctx.op(scope).regions.first().copied()?;
    let mut resolved = None;
    let mut duplicate = false;
    walk_symbol_scope_ops(ctx, body, &mut |op| {
        if is_control_op(ctx, op, "func")
            && ctx.op(op).attributes.get_symbol("sym_name") == Some(symbol)
        {
            duplicate |= resolved.replace(op).is_some();
        }
    });
    (!duplicate).then_some(resolved).flatten()
}

fn verified_callable_declaration(
    ctx: &IrContext,
    function: OpRef,
    registered: &HashMap<Symbol, &CompilerIntrinsicDeclaration>,
) -> bool {
    let data = ctx.op(function);
    if !data.regions.is_empty() {
        return true;
    }
    let (Some(symbol), Some(identity), Some(callable_type)) = (
        data.attributes.get_symbol("sym_name"),
        Func::from_op(ctx, function)
            .ok()
            .and_then(|function| function.compiler_intrinsic_identity(ctx)),
        data.attributes.get_type("type"),
    ) else {
        return false;
    };
    registered.get(&symbol).is_some_and(|declaration| {
        declaration.identity == identity && declaration.callable_type == callable_type
    })
}

fn validate_callable_origins(
    ctx: &IrContext,
    body: RegionRef,
    declarations: &[CompilerIntrinsicDeclaration],
    operation_declarations: &HashMap<(TypeRef, Symbol), &OperationDeclaration>,
    nominal_layouts: &HashMap<Symbol, TypeRef>,
    errors: &mut Vec<ValidationError>,
) {
    let registered = compiler_intrinsic_map(ctx, declarations, errors);
    let provenance = CallableProvenance {
        registered: &registered,
        declarations: operation_declarations,
        nominal_layouts,
    };
    let mut seen = HashSet::new();
    walk_region_ops(ctx, body, &mut |op| {
        if is_control_op(ctx, op, "func") {
            let data = ctx.op(op);
            let (Some(symbol), Some(callable_type)) = (
                data.attributes.get_symbol("sym_name"),
                data.attributes.get_type("type"),
            ) else {
                return;
            };
            let intrinsic_identity = Func::from_op(ctx, op)
                .ok()
                .and_then(|function| function.compiler_intrinsic_identity(ctx));
            let bodyless = data.regions.is_empty();
            if !bodyless && intrinsic_identity.is_some() {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "defined Tribute function cannot claim compiler intrinsic identity",
                );
            }
            if bodyless {
                let exact_intrinsic = match (registered.get(&symbol), intrinsic_identity) {
                    (Some(expected), Some(actual))
                        if expected.identity == actual
                            && expected.callable_type == callable_type =>
                    {
                        seen.insert(symbol);
                        true
                    }
                    (Some(_), _) => {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "compiler intrinsic identity or complete signature does not match the registered declaration",
                        );
                        false
                    }
                    (None, Some(_)) => {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "unregistered declaration cannot claim compiler intrinsic identity",
                        );
                        false
                    }
                    (None, None) => false,
                };
                if !exact_intrinsic && contains_adt_typeref(ctx, callable_type, &mut HashSet::new())
                {
                    push_op_error(
                        ctx,
                        op,
                        errors,
                        "unclassified bodyless external declaration cannot use managed adt.typeref",
                    );
                }
            }
        } else if is_control_op(ctx, op, "call_indirect")
            && let Some(callee) = ctx.op_operands(op).first().copied()
            && !callable_has_semantic_provenance(ctx, callee, &provenance, &mut HashSet::new())
        {
            push_op_error(
                ctx,
                op,
                errors,
                "indirect callee lacks verified source-logical callable provenance",
            );
        }
    });
    for symbol in registered.keys() {
        if !seen.contains(symbol) {
            push_type_error(
                errors,
                format!(
                    "registered compiler intrinsic @{symbol} has no exact bodyless declaration"
                ),
            );
        }
    }
}

fn validate_return_contracts(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        if !is_control_op(ctx, op, "return") {
            return;
        }
        let mut owner = parent_op(ctx, op);
        let callable = loop {
            let Some(current) = owner else {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "return has no containing semantic callable",
                );
                return;
            };
            if is_control_op(ctx, current, "func") {
                break ctx.op(current).attributes.get_type("type");
            }
            if is_control_op(ctx, current, "lambda") {
                break ctx.op_result_types(current).first().copied();
            }
            owner = parent_op(ctx, current);
        };
        let Some((expected, _, _)) = callable.and_then(|ty| callable_parts(ctx, ty)) else {
            return;
        };
        if let [value] = ctx.op_operands(op)
            && ctx.value_ty(*value) != expected
        {
            push_op_error(
                ctx,
                op,
                errors,
                "return operand type does not match containing semantic callable",
            );
        }
    });
}

fn validate_declaration_uses(
    ctx: &IrContext,
    body: RegionRef,
    declarations: &HashMap<(TypeRef, Symbol), &OperationDeclaration>,
    errors: &mut Vec<ValidationError>,
) {
    walk_region_ops(ctx, body, &mut |op| {
        let (kind_attr, args, result_ty) = if is_control_op(ctx, op, "perform") {
            (
                "operation_kind",
                ctx.op_operands(op),
                ctx.op_result_types(op).first().copied(),
            )
        } else if is_control_op(ctx, op, "handler") {
            let Some(&region) = ctx.op(op).regions.first() else {
                return;
            };
            let Some(&block) = ctx.region(region).blocks.first() else {
                return;
            };
            let mut args = ctx.block_args(block);
            if args
                .last()
                .is_some_and(|value| ResumeToken::matches(ctx, ctx.value_ty(*value)))
            {
                args = &args[..args.len() - 1];
            }
            (
                "kind",
                args,
                ctx.op(op).attributes.get_type("operation_result_type"),
            )
        } else {
            return;
        };
        let data = ctx.op(op);
        let (Some(ability), Some(name)) = (
            data.attributes.get_type("ability_ref"),
            data.attributes.get_symbol("op_name"),
        ) else {
            return;
        };
        let Some(declaration) = declarations.get(&(ability, name)) else {
            push_op_error(
                ctx,
                op,
                errors,
                format!("no resolved operation declaration for {ability}::{name}"),
            );
            return;
        };
        if data.attributes.get_symbol(kind_attr) != Some(declaration.kind) {
            push_op_error(
                ctx,
                op,
                errors,
                format!("{kind_attr} does not match the resolved declaration"),
            );
        }
        check_types_equal(
            ctx,
            op,
            &value_types(ctx, args),
            &declaration.parameter_types,
            "operation argument",
            errors,
        );
        if result_ty != Some(declaration.result_type) {
            push_op_error(
                ctx,
                op,
                errors,
                "operation result type does not match the resolved declaration",
            );
        }
    });
}

fn op_is_within_region(ctx: &IrContext, op: OpRef, target: RegionRef) -> bool {
    let mut region = parent_region(ctx, op);
    while let Some(current) = region {
        if current == target {
            return true;
        }
        region = ctx
            .region(current)
            .parent_op
            .and_then(|owner| parent_region(ctx, owner));
    }
    false
}

fn nearest_enclosing_handler(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let mut owner = parent_op(ctx, op);
    while let Some(current) = owner {
        if is_control_op(ctx, current, "handler") {
            return Some(current);
        }
        owner = parent_op(ctx, current);
    }
    None
}

fn direct_call_reenters_enclosing_func(ctx: &IrContext, op: OpRef) -> bool {
    if !is_control_op(ctx, op, "call") {
        return false;
    }
    let Some(callee) = ctx.op(op).attributes.get_symbol("callee") else {
        return false;
    };
    let mut owner = parent_op(ctx, op);
    while let Some(current) = owner {
        if is_control_op(ctx, current, "func") {
            return ctx.op(current).attributes.get_symbol("sym_name") == Some(callee);
        }
        owner = parent_op(ctx, current);
    }
    false
}

fn validate_affine_lambda_carrier(
    ctx: &IrContext,
    handler: OpRef,
    carrier: ValueRef,
    visited: &mut HashSet<ValueRef>,
    errors: &mut Vec<ValidationError>,
) {
    if !visited.insert(carrier) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier path contains a cycle",
        );
        return;
    }

    let mut captures = Vec::new();
    let mut terminals = Vec::new();
    for use_ in ctx.uses(carrier) {
        if nearest_enclosing_handler(ctx, use_.user) != Some(handler) {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume-token carrier crosses into a different handler at {}",
                    use_.user
                ),
            );
            continue;
        }
        if is_control_op(ctx, use_.user, "lambda") {
            captures.push(use_.user);
        } else if (is_control_op(ctx, use_.user, "call_indirect") && use_.operand_index == 0)
            // A captured resumption may be passed to the enclosing source
            // function's recursive logical call (for example
            // `run_state(fn() { resume value }, value)`). Other direct calls
            // are rejected: their parameter ownership is not yet proven.
            || direct_call_reenters_enclosing_func(ctx, use_.user)
        {
            terminals.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume-token carrier escapes through {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }

    if has_nonexclusive_pair(ctx, &captures) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier branches into multiple lambda captures",
        );
    }
    if has_nonexclusive_pair(ctx, &terminals) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume-token carrier has multiple static terminal uses",
        );
    }

    if !captures.is_empty() {
        for terminal in &terminals {
            if is_control_op(ctx, *terminal, "call_indirect")
                && !captures.iter().any(|capture| {
                    ctx.op(*capture)
                        .regions
                        .first()
                        .is_some_and(|region| op_is_within_region(ctx, *terminal, *region))
                })
            {
                push_op_error(
                    ctx,
                    handler,
                    errors,
                    "resume-token carrier call is outside its capture path",
                );
            }
        }
    }
    for capture in captures {
        let Some(&next_carrier) = ctx.op_results(capture).first() else {
            push_op_error(
                ctx,
                handler,
                errors,
                "lambda capture carrying a resume token must produce one callable value",
            );
            return;
        };
        validate_affine_lambda_carrier(ctx, handler, next_carrier, visited, errors);
    }
}

fn validate_token_path(
    ctx: &IrContext,
    handler: OpRef,
    token_value: ValueRef,
    errors: &mut Vec<ValidationError>,
) {
    let mut resumes = Vec::new();
    let mut captures = Vec::new();
    for use_ in ctx.uses(token_value) {
        if nearest_enclosing_handler(ctx, use_.user) != Some(handler) {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume token crosses into a different handler at {}",
                    use_.user
                ),
            );
            continue;
        }
        if is_control_op(ctx, use_.user, "resume") && use_.operand_index == 0 {
            resumes.push(use_.user);
        } else if is_control_op(ctx, use_.user, "lambda") {
            captures.push(use_.user);
        } else {
            push_op_error(
                ctx,
                handler,
                errors,
                format!(
                    "resume token has forbidden use by {} at operand {}",
                    use_.user, use_.operand_index
                ),
            );
        }
    }
    if has_nonexclusive_pair(ctx, &resumes) {
        push_op_error(
            ctx,
            handler,
            errors,
            "resume token reaches more than one tribute_control.resume",
        );
    }
    for (left, right) in captures.iter().copied().tuple_combinations() {
        let left_region = ctx.op(left).regions.first().copied();
        let right_region = ctx.op(right).regions.first().copied();
        let comparable = left_region.is_some_and(|region| op_is_within_region(ctx, right, region))
            || right_region.is_some_and(|region| op_is_within_region(ctx, left, region));
        if !comparable && !ops_are_mutually_exclusive(ctx, left, right) {
            push_op_error(
                ctx,
                handler,
                errors,
                "resume token is copied into multiple capture paths",
            );
        }
    }
    for capture in &captures {
        let Some(region) = ctx.op(*capture).regions.first().copied() else {
            continue;
        };
        if !resumes
            .iter()
            .any(|resume| op_is_within_region(ctx, *resume, region))
        {
            push_op_error(
                ctx,
                handler,
                errors,
                "resume token capture does not form a single path to resume",
            );
        }
    }
    let mut visited = HashSet::new();
    for capture in captures {
        if let Some(&closure_value) = ctx.op_results(capture).first() {
            validate_affine_lambda_carrier(ctx, handler, closure_value, &mut visited, errors);
        }
    }
}

/// A handler token may have one affine resume on each branch of an ordinary
/// structured conditional: exactly one branch executes at runtime.  Static
/// use counting alone would reject a source `case` whose exhaustive arms each
/// resume the same handler-owned token.
fn has_nonexclusive_pair(ctx: &IrContext, ops: &[OpRef]) -> bool {
    ops.iter()
        .copied()
        .tuple_combinations()
        .any(|(left, right)| !ops_are_mutually_exclusive(ctx, left, right))
}

fn ops_are_mutually_exclusive(ctx: &IrContext, left: OpRef, right: OpRef) -> bool {
    let mut region = parent_region(ctx, left);
    while let Some(current) = region {
        let owner = ctx.region(current).parent_op;
        if let Some(owner) = owner
            && let Some(interface) = RegionBranchOps::get(ctx, owner)
            && let Ok(successors) = interface.successors(ctx, owner, RegionBranchPoint::Parent)
        {
            let containing_region = |candidate: OpRef| {
                successors.as_slice().iter().find_map(|successor| {
                    let RegionSuccessor::Region(region) = successor else {
                        return None;
                    };
                    op_is_within_region(ctx, candidate, *region).then_some(*region)
                })
            };
            if let (Some(left_region), Some(right_region)) =
                (containing_region(left), containing_region(right))
                && left_region != right_region
            {
                return true;
            }
        }
        region = owner.and_then(|owner| parent_region(ctx, owner));
    }
    false
}

fn validate_resume_ownership(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        if !is_control_op(ctx, op, "handler") {
            return;
        }
        let Some(&region) = ctx.op(op).regions.first() else {
            return;
        };
        let Some(&block) = ctx.region(region).blocks.first() else {
            return;
        };
        let Some(&token) = ctx.block_args(block).last() else {
            return;
        };
        if ResumeToken::matches(ctx, ctx.value_ty(token)) {
            validate_token_path(ctx, op, token, errors);
        }
    });
}

fn validate_token_placements(ctx: &IrContext, body: RegionRef, errors: &mut Vec<ValidationError>) {
    walk_region_ops(ctx, body, &mut |op| {
        for ty in ctx.op_result_types(op) {
            if ResumeToken::matches(ctx, *ty) {
                push_op_error(
                    ctx,
                    op,
                    errors,
                    "resume_token must not be produced as an operation result",
                );
            }
        }
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for (index, arg) in ctx.block_args(block).iter().copied().enumerate() {
                    if !ResumeToken::matches(ctx, ctx.value_ty(arg)) {
                        continue;
                    }
                    let allowed = is_control_op(ctx, op, "handler")
                        && index + 1 == ctx.block_args(block).len();
                    if !allowed {
                        push_op_error(
                            ctx,
                            op,
                            errors,
                            "resume_token block arguments are allowed only as the final handler argument",
                        );
                    }
                }
            }
        }
    });
}

/// Validate graph-wide `tribute_control` invariants.
pub fn validate_whole_ir(
    ctx: &IrContext,
    module: Module,
    declarations: &[OperationDeclaration],
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
) -> ValidationResult {
    let mut errors = Vec::new();
    let Some(body) = module.body(ctx) else {
        return ValidationResult { errors };
    };
    validate_module_symbol_scopes(ctx, module.op(), &mut errors);
    validate_lambda_captures(ctx, body, &mut errors);
    let declarations = declaration_map(declarations, &mut errors);
    let reachable_types = collect_reachable_ir_types(ctx, module.op());
    let nominal_layouts = canonical_nominal_layouts(ctx, &reachable_types, &mut errors);
    validate_managed_reference_boundaries(
        ctx,
        body,
        &reachable_types,
        &nominal_layouts,
        &mut errors,
    );
    validate_callable_origins(
        ctx,
        body,
        compiler_intrinsics,
        &declarations,
        &nominal_layouts,
        &mut errors,
    );
    validate_return_contracts(ctx, body, &mut errors);
    validate_declaration_uses(ctx, body, &declarations, &mut errors);
    validate_resume_ownership(ctx, body, &mut errors);
    validate_token_placements(ctx, body, &mut errors);
    ValidationResult { errors }
}

/// Run the complete local and whole-IR `tribute_control` validation.
///
/// The caller supplies resolved source operation declarations because those
/// declarations are frontend semantic metadata rather than TrunkIR operations.
pub fn validate(
    ctx: &IrContext,
    module: Module,
    declarations: &[OperationDeclaration],
    compiler_intrinsics: &[CompilerIntrinsicDeclaration],
) -> ValidationResult {
    let mut local = validate_local(ctx, module);
    local
        .errors
        .extend(validate_whole_ir(ctx, module, declarations, compiler_intrinsics).errors);
    local
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::context::{BlockArgData, BlockData, RegionData};
    use trunk_ir::dialect::core;
    use trunk_ir::parser::parse_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::AttributeMap;

    fn location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("tribute-control-test.trb".to_owned());
        Location::new(path, Span::new(7, 19))
    }

    fn simple_type(ctx: &mut IrContext, dialect: &str, name: &str) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::from_dynamic(dialect), Symbol::from_dynamic(name)).build(),
        )
    }

    fn ability_type(ctx: &mut IrContext, name: &str) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::from_dynamic(name)))
                .build(),
        )
    }

    fn block(ctx: &mut IrContext, loc: Location, args: &[TypeRef]) -> BlockRef {
        ctx.create_block(BlockData {
            location: loc,
            args: args
                .iter()
                .copied()
                .map(|ty| BlockArgData {
                    ty,
                    attrs: AttributeMap::new(),
                })
                .collect(),
            ops: Default::default(),
            parent_region: None,
        })
    }

    fn region(ctx: &mut IrContext, loc: Location, block: BlockRef) -> RegionRef {
        ctx.create_region(RegionData {
            location: loc,
            blocks: vec![block].into(),
            parent_op: None,
        })
    }

    fn module(ctx: &mut IrContext, loc: Location, ops: &[OpRef]) -> Module {
        let body_block = block(ctx, loc, &[]);
        for op in ops {
            ctx.push_op(body_block, *op);
        }
        let body = region(ctx, loc, body_block);
        let module = core::module(ctx, loc, Symbol::new("test"), body);
        Module::new(ctx, module.op_ref()).expect("core.module")
    }

    fn identity_func(
        ctx: &mut IrContext,
        loc: Location,
        symbol: &str,
        ty: TypeRef,
        value_ty: TypeRef,
    ) -> Func {
        let entry = block(ctx, loc, &[value_ty]);
        let value = ctx.block_arg(entry, 0);
        let ret = r#return(ctx, loc, value);
        ctx.push_op(entry, ret.op_ref());
        let body = region(ctx, loc, entry);
        func(ctx, loc, Symbol::from_dynamic(symbol), ty, body)
    }

    struct ValidFixture {
        ctx: IrContext,
        module: Module,
        declarations: Vec<OperationDeclaration>,
        handle: OpRef,
        handler: OpRef,
    }

    fn builder_fixture() -> ValidFixture {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let i32_ty = simple_type(&mut ctx, "core", "i32");
        let ability = ability_type(&mut ctx, "State");
        let direct = callable(&mut ctx, i32_ty, [i32_ty], CallingConvention::Direct).as_type_ref();

        let id = identity_func(&mut ctx, loc, "id", direct, i32_ty);

        let entry = block(&mut ctx, loc, &[i32_ty]);
        let x = ctx.block_arg(entry, 0);

        let function_ref = func_ref(&mut ctx, loc, direct, Symbol::new("id"));
        ctx.push_op(entry, function_ref.op_ref());
        let function_ref_value = function_ref.result(&ctx);
        let direct_call = call(&mut ctx, loc, [x], i32_ty, Symbol::new("id"));
        ctx.push_op(entry, direct_call.op_ref());
        let indirect_call = call_indirect(&mut ctx, loc, function_ref_value, [x], i32_ty);
        ctx.push_op(entry, indirect_call.op_ref());

        let lambda_ty = callable(&mut ctx, i32_ty, [], CallingConvention::Direct).as_type_ref();
        let lambda_block = block(&mut ctx, loc, &[]);
        let lambda_return = r#return(&mut ctx, loc, x);
        ctx.push_op(lambda_block, lambda_return.op_ref());
        let lambda_body = region(&mut ctx, loc, lambda_block);
        let lambda = lambda(&mut ctx, loc, [x], lambda_ty, lambda_body);
        ctx.push_op(entry, lambda.op_ref());

        let body_block = block(&mut ctx, loc, &[]);
        let perform = perform(
            &mut ctx,
            loc,
            [x],
            i32_ty,
            ability,
            Symbol::new("get"),
            Symbol::new("op"),
        );
        ctx.push_op(body_block, perform.op_ref());
        let perform_result = perform.result(&ctx);
        let body_yield = r#yield(&mut ctx, loc, perform_result);
        ctx.push_op(body_block, body_yield.op_ref());
        let handle_body = region(&mut ctx, loc, body_block);

        let completion_block = block(&mut ctx, loc, &[i32_ty]);
        let completed = ctx.block_arg(completion_block, 0);
        let completion_yield = r#yield(&mut ctx, loc, completed);
        ctx.push_op(completion_block, completion_yield.op_ref());
        let completion = region(&mut ctx, loc, completion_block);

        let token_ty = resume_token(&mut ctx, i32_ty, i32_ty).as_type_ref();
        let handler_block = block(&mut ctx, loc, &[i32_ty, token_ty]);
        let operation_arg = ctx.block_arg(handler_block, 0);
        let token = ctx.block_arg(handler_block, 1);
        let resume_op = resume(&mut ctx, loc, token, operation_arg, i32_ty);
        ctx.push_op(handler_block, resume_op.op_ref());
        let resumed_value = resume_op.result(&ctx);
        let handler_yield = r#yield(&mut ctx, loc, resumed_value);
        ctx.push_op(handler_block, handler_yield.op_ref());
        let handler_body = region(&mut ctx, loc, handler_block);
        let handler = handler(
            &mut ctx,
            loc,
            ability,
            Symbol::new("get"),
            Symbol::new("op"),
            i32_ty,
            handler_body,
        );

        let handlers_block = block(&mut ctx, loc, &[]);
        ctx.push_op(handlers_block, handler.op_ref());
        let handlers = region(&mut ctx, loc, handlers_block);
        let handle = handle(&mut ctx, loc, i32_ty, handle_body, completion, handlers);
        ctx.push_op(entry, handle.op_ref());
        let handle_result = handle.result(&ctx);
        let ret = r#return(&mut ctx, loc, handle_result);
        ctx.push_op(entry, ret.op_ref());
        let control_body = region(&mut ctx, loc, entry);
        let control = func(&mut ctx, loc, Symbol::new("control"), direct, control_body);

        let module = module(&mut ctx, loc, &[id.op_ref(), control.op_ref()]);
        let declarations = vec![OperationDeclaration::new(
            ability,
            Symbol::new("get"),
            Symbol::new("op"),
            [i32_ty],
            i32_ty,
        )];

        ValidFixture {
            ctx,
            module,
            declarations,
            handle: handle.op_ref(),
            handler: handler.op_ref(),
        }
    }

    const VALID_CONTROL_MODULE: &str = r#"core.module @test {
  !callable = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}

  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @control(%value: core.i32) -> core.i32 convention(direct) {
    %function = tribute_control.func_ref {func_ref = @id} : !callable
    %direct = tribute_control.call %value {callee = @id} : core.i32
    %indirect = tribute_control.call_indirect %function, %value : core.i32
    %closure = tribute_control.lambda() -> core.i32 convention(direct) captures [%value] {
      tribute_control.return %value
    }
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %value {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%completed: core.i32):
        tribute_control.yield %completed
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
        ^handler(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %handled
  }
}"#;

    fn parse_fixture(input: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let root = parse_module(&mut ctx, input)
            .unwrap_or_else(|error| panic!("failed to parse fixture:\n{error}\n\n{input}"));
        let module = Module::new(&ctx, root).expect("core.module fixture");
        (ctx, module)
    }

    fn control_ops(ctx: &IrContext, module: Module, name: &str) -> Vec<OpRef> {
        let mut matches = Vec::new();
        let body = module.body(ctx).expect("module body");
        walk_region_ops(ctx, body, &mut |op| {
            if is_control_op(ctx, op, name) {
                matches.push(op);
            }
        });
        matches
    }

    fn control_op(ctx: &IrContext, module: Module, name: &str) -> OpRef {
        let matches = control_ops(ctx, module, name);
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one tribute_control.{name}"
        );
        matches[0]
    }

    fn valid_fixture() -> ValidFixture {
        let (ctx, module) = parse_fixture(VALID_CONTROL_MODULE);
        let handle = control_op(&ctx, module, "handle");
        let handler = control_op(&ctx, module, "handler");
        let handler_body = ctx.op(handler).regions[0];
        let handler_block = ctx.region(handler_body).blocks[0];
        let handler_data = ctx.op(handler);
        let ability_ref = handler_data
            .attributes
            .get_type("ability_ref")
            .expect("handler ability");
        let op_name = handler_data
            .attributes
            .get_symbol("op_name")
            .expect("handler operation name");
        let kind = handler_data
            .attributes
            .get_symbol("kind")
            .expect("handler kind");
        let result_type = handler_data
            .attributes
            .get_type("operation_result_type")
            .expect("handler operation result");
        let parameter_types = block_arg_types(&ctx, handler_block);
        let declarations = vec![OperationDeclaration::new(
            ability_ref,
            op_name,
            kind,
            parameter_types[..parameter_types.len() - 1].iter().copied(),
            result_type,
        )];

        ValidFixture {
            ctx,
            module,
            declarations,
            handle,
            handler,
        }
    }

    fn assert_round_trip(ctx: &IrContext, module: Module) -> String {
        let printed = print_module(ctx, module.op());
        let mut reparsed_ctx = IrContext::new();
        let reparsed = parse_module(&mut reparsed_ctx, &printed)
            .unwrap_or_else(|error| panic!("failed to reparse:\n{error}\n\n{printed}"));
        let reprinted = print_module(&reparsed_ctx, reparsed);
        assert_eq!(printed, reprinted);
        printed
    }

    fn messages(result: &ValidationResult) -> String {
        result
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn append_module_op(ctx: &mut IrContext, module: Module, op: OpRef) {
        let body = module.body(ctx).expect("module body");
        let body_block = ctx.region(body).blocks[0];
        ctx.push_op(body_block, op);
    }

    fn assert_diagnostics(result: &ValidationResult, expected: &[&str]) {
        let messages = messages(result);
        for expected in expected {
            assert!(
                messages.contains(expected),
                "missing diagnostic `{expected}`:\n{result}"
            );
        }
    }

    fn assert_op_diagnostics(result: &ValidationResult, op: OpRef, expected: &[&str]) {
        let messages = result
            .errors
            .iter()
            .filter(|error| error.op == Some(op))
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for expected in expected {
            assert!(
                messages.contains(expected),
                "operation {op} is missing diagnostic `{expected}`:\n{result}"
            );
        }
    }

    #[test]
    fn callable_codes_and_resume_token_accessors_are_exact() {
        let mut ctx = IrContext::new();
        let i32_ty = simple_type(&mut ctx, "core", "i32");

        for (convention, code) in [
            (CallingConvention::Direct, 0),
            (CallingConvention::EvidenceDirect, 1),
            (CallingConvention::Cps, 2),
        ] {
            let ty = callable(&mut ctx, i32_ty, [i32_ty], convention);
            assert_eq!(ty.result(&ctx), i32_ty);
            assert_eq!(ty.params(&ctx), &[i32_ty]);
            assert_eq!(
                ctx.types
                    .get(ty.as_type_ref())
                    .attrs
                    .get_i128(CALLING_CONVENTION_ATTR),
                Some(code)
            );
            assert_eq!(
                callable_convention(&ctx, ty.as_type_ref()),
                Some(convention)
            );
        }

        let token = resume_token(&mut ctx, i32_ty, i32_ty);
        assert_eq!(token.input(&ctx), i32_ty);
        assert_eq!(token.answer(&ctx), i32_ty);
    }

    #[test]
    fn callable_wrapper_rejects_missing_result_component() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !malformed = tribute_control.callable() {tribute.calling_convention = 0}
}"#,
        );
        let malformed = ctx
            .types
            .iter()
            .find_map(|(ty, _)| Callable::matches(&ctx, ty).then_some(ty))
            .expect("malformed callable type");
        assert!(Callable::from_type_ref(&ctx, malformed).is_none());

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("requires one result type"));
    }

    #[test]
    fn all_operations_construct_with_typed_accessors_and_preserve_locations() {
        // This is intentionally builder-based: it verifies every public
        // constructor/accessor and preservation of caller-provided locations.
        let fixture = builder_fixture();
        let ctx = &fixture.ctx;
        let mut names = HashSet::new();
        let body = fixture.module.body(ctx).expect("module body");
        walk_region_ops(ctx, body, &mut |op| {
            let data = ctx.op(op);
            if data.dialect == Symbol::new("tribute_control") {
                names.insert(data.name);
                assert_eq!(data.location.span, Span::new(7, 19));
            }
        });
        for expected in [
            "func",
            "lambda",
            "func_ref",
            "call",
            "call_indirect",
            "return",
            "perform",
            "handle",
            "handler",
            "resume",
            "yield",
        ] {
            assert!(
                names.contains(&Symbol::from_dynamic(expected)),
                "{expected}"
            );
        }

        let handler = Handler::from_op(ctx, fixture.handler).expect("typed handler accessor");
        assert_eq!(handler.kind(ctx), Symbol::new("op"));
        let handle = Handle::from_op(ctx, fixture.handle).expect("typed handle accessor");
        assert_eq!(handle.result_ty(ctx), ctx.value_ty(handle.result(ctx)));
    }

    #[test]
    fn custom_assembly_round_trips_declarations_definitions_and_lambdas() {
        let input = r#"core.module @test {
  tribute_control.func @decl(%left: core.i32, %right: core.i32) -> core.i32 convention(direct) attributes {visibility = @private}
  tribute_control.func @identity(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_control.func @outer(%first: core.i32, %second: core.i32) -> core.i32 convention(direct) {
    %captured = tribute_control.lambda(%left: core.i32, %right: core.i32) -> core.i32 convention(direct) captures [%first, %second] attributes {debug_name = "apply", inline_hint = true} {
      tribute_control.return %first
    }
    %empty = tribute_control.lambda() -> core.i32 convention(cps) captures [] {
      %constant = arith.const {value = 1} : core.i32
      tribute_control.return %constant
    }
    tribute_control.return %first
  }
}"#;
        let (ctx, module) = parse_fixture(input);
        let printed = assert_round_trip(&ctx, module);
        assert!(printed.contains(
            "tribute_control.func @decl(%arg0: core.i32, %arg1: core.i32) -> core.i32 convention(direct) attributes {visibility = @private}"
        ));
        assert!(printed.contains("convention(evidence_direct)"));
        assert!(printed.contains("convention(cps) captures []"));
        assert!(printed.contains("convention(direct) captures [%0, %1] attributes {"));
        assert!(printed.contains("debug_name = \"apply\""));
        assert!(printed.contains("inline_hint = true"));
        assert!(!printed.contains("^bb"));
    }

    #[test]
    fn generic_control_operations_round_trip_and_validate() {
        let fixture = valid_fixture();
        let result = validate(&fixture.ctx, fixture.module, &fixture.declarations, &[]);
        assert!(result.is_ok(), "{result}");

        let printed = assert_round_trip(&fixture.ctx, fixture.module);
        for expected in [
            "tribute_control.func_ref",
            "tribute_control.call ",
            "tribute_control.call_indirect",
            "tribute_control.perform",
            "tribute_control.handle",
            "tribute_control.handler",
            "tribute_control.resume",
            "tribute_control.yield",
        ] {
            assert!(printed.contains(expected), "missing {expected}\n{printed}");
        }
        assert!(printed.contains("tribute.calling_convention = 0"));
    }

    #[test]
    fn parser_rejects_bad_or_duplicate_convention() {
        for input in [
            r#"core.module @test {
  tribute_control.func @bad() -> core.i32 convention(bogus)
}"#,
            r#"core.module @test {
  tribute_control.func @bad() -> core.i32 convention(direct) attributes {tribute.calling_convention = 0}
}"#,
            r#"core.module @test {
  %0 = tribute_control.lambda() -> core.i32 convention(cps) captures [] attributes {tribute.calling_convention = 2} {
    %1 = arith.const {value = 1} : core.i32
    tribute_control.return %1
  }
}"#,
        ] {
            let mut ctx = IrContext::new();
            assert!(parse_module(&mut ctx, input).is_err(), "{input}");
        }
    }

    #[test]
    fn parser_preserves_underscore_named_extra_attribute() {
        let input = r#"core.module @test {
  tribute_control.func @ok() -> core.i32 convention(direct) attributes {tribute_calling_convention = 7}
}"#;
        let mut ctx = IrContext::new();
        let root = parse_module(&mut ctx, input).expect("ordinary extra attribute should parse");
        let module = Module::new(&ctx, root).expect("core.module");
        let printed = assert_round_trip(&ctx, module);
        assert!(printed.contains("tribute_calling_convention = 7"));
    }

    #[test]
    fn validator_rejects_invalid_convention_and_duplicate_op_metadata() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !bad = tribute_control.callable(core.i32) {tribute.calling_convention = 3}
  %call = tribute_control.call {callee = @missing, tribute.calling_convention = 0} : core.i32
}"#,
        );
        let result = validate_local(&ctx, module);
        let messages = messages(&result);
        assert!(messages.contains("unsupported tribute.calling_convention code 3"));
        assert!(messages.contains("belongs only on tribute_control.callable"));
    }

    #[test]
    fn validator_rejects_wrong_required_attribute_types() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  !callable = tribute_control.callable(core.i32) {tribute.calling_convention = 0}
  tribute_control.func @malformed() -> core.i32 convention(direct)
  %ref = tribute_control.func_ref {func_ref = 1} : !callable
  %call = tribute_control.call {callee = 1} : core.i32
  %perform = tribute_control.perform {ability_ref = 1, op_name = 2, operation_kind = 3} : core.i32
  tribute_control.handler {ability_ref = 1, op_name = 2, kind = 3, operation_result_type = 4} {
    %constant = arith.const {value = 0} : core.i32
    tribute_control.yield %constant
  }
}"#,
        );

        // The custom func parser derives and hides sym_name/type, so malformed
        // values for those two required attributes are not text-representable.
        let malformed_func = control_op(&ctx, module, "func");
        ctx.op_mut(malformed_func)
            .attributes
            .insert(Symbol::new("sym_name"), Attribute::Int(1));
        ctx.op_mut(malformed_func)
            .attributes
            .insert(Symbol::new("type"), Attribute::Int(2));

        let result = validate_local(&ctx, module);
        assert_op_diagnostics(
            &result,
            malformed_func,
            &[
                "attribute 'sym_name' must be Symbol",
                "attribute 'type' must be Type",
            ],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, "func_ref"),
            &["attribute 'func_ref' must be Symbol"],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, "call"),
            &["attribute 'callee' must be Symbol"],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, "perform"),
            &[
                "attribute 'ability_ref' must be Type",
                "attribute 'op_name' must be Symbol",
                "attribute 'operation_kind' must be Symbol",
            ],
        );
        assert_op_diagnostics(
            &result,
            control_op(&ctx, module, "handler"),
            &[
                "attribute 'ability_ref' must be Type",
                "attribute 'op_name' must be Symbol",
                "attribute 'kind' must be Symbol",
                "attribute 'operation_result_type' must be Type",
            ],
        );
    }

    #[test]
    fn validator_rejects_nested_unresolved_call_and_perform_types() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !wrapper = core.tuple(type.var)
  !never = core.never
  %value = arith.const {value = 0} : !wrapper
  %call = tribute_control.call %value {callee = @id} : !wrapper
  %perform = tribute_control.perform %value {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : !wrapper
}"#,
        );
        let never = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("never"))
                    .then_some(ty)
            })
            .expect("core.never");
        assert!(!contains_unresolved_type(&ctx, never, &mut HashSet::new()));

        let result = validate_local(&ctx, module);
        for op in [
            control_op(&ctx, module, "call"),
            control_op(&ctx, module, "perform"),
        ] {
            assert_eq!(
                result
                    .errors
                    .iter()
                    .filter(|error| {
                        error.op == Some(op)
                            && error
                                .message
                                .contains("operands and result must be resolved")
                    })
                    .count(),
                1,
                "{result}"
            );
        }
    }

    #[test]
    fn validator_rejects_handlers_table_block_arguments() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    ^handlers(%unexpected: core.i32):
  }
}"#,
        );

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("handlers table block must not have arguments"));
    }

    #[test]
    fn validator_rejects_structure_arity_type_and_terminator_errors() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
  } {
  }
  tribute_control.func @unterminated(%value: core.i32) -> core.i32 convention(direct) {
    %unused = arith.const {value = 0} : core.i32
  }
  %indirect = tribute_control.call_indirect : core.i32
}"#,
        );

        let result = validate_local(&ctx, module);
        let messages = messages(&result);
        assert!(messages.contains("expects 3 region(s), found 2"));
        assert!(messages.contains("must terminate with tribute_control.return"));
        assert!(messages.contains("requires a callable callee operand"));
    }

    #[test]
    fn local_validator_reports_distinct_malformed_operation_contracts() {
        let (mut ctx, module) = parse_fixture(
            r#"core.module @test {
  !direct = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !token = tribute_control.resume_token(core.i32, core.i32)
  %integer = arith.const {value = 1} : core.i32
  %boolean = arith.const {value = true} : core.bool

  tribute_control.func @bad_func() -> core.i32 convention(direct) {
    tribute_control.return %integer
  }
  %bad_lambda = tribute_control.lambda() -> core.i32 convention(direct) captures [%integer] {
    tribute_control.return %integer
  }
  %bad_ref0, %bad_ref1 = tribute_control.func_ref %integer {func_ref = @bad_func, unexpected = 1} : core.i32, core.bool
  %bad_call0, %bad_call1 = tribute_control.call {unexpected = 1} : core.i32, core.bool {
  }
  %non_callable = tribute_control.call_indirect %integer, %boolean : core.i32
  %callable = tribute_control.func_ref {func_ref = @bad_func} : !direct
  %mismatched = tribute_control.call_indirect %callable, %boolean : core.bool
  %bad_perform = tribute_control.perform %integer {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @bogus} : core.i32
  tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, op_name = @get, kind = @bogus, operation_result_type = core.i32}
  %bad_resume = tribute_control.resume %integer, %boolean : core.bool
  %token = test.token : !token
  %mismatched_resume = tribute_control.resume %token, %boolean : core.bool
  tribute_control.unknown
}"#,
        );
        let loc = location(&mut ctx);
        let i32_ty = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .unwrap();
        let i32_value = module
            .ops(&ctx)
            .iter()
            .copied()
            .find(|op| {
                ctx.op(*op).dialect == Symbol::new("arith")
                    && ctx.op(*op).name == Symbol::new("const")
                    && ctx.op_result_types(*op) == [i32_ty]
            })
            .map(|op| ctx.op_result(op, 0))
            .unwrap();

        // Custom func assembly requires its symbol and callable type, so the
        // parser cannot represent their simultaneous absence.
        let missing_func = trunk_ir::OperationDataBuilder::new(
            loc,
            Symbol::new("tribute_control"),
            Symbol::new("func"),
        )
        .operand(i32_value)
        .build(&mut ctx);
        let missing_func = ctx.create_op(missing_func);
        append_module_op(&mut ctx, module, missing_func);

        let bad_func = control_ops(&ctx, module, "func")
            .into_iter()
            .find(|op| {
                ctx.op(*op).attributes.get_symbol("sym_name") == Some(Symbol::new("bad_func"))
            })
            .unwrap();
        // Custom assembly always derives a callable type and permits at most one
        // body. Mutate only those parser-enforced fields.
        ctx.op_mut(bad_func)
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(i32_ty));
        let extra_block = block(&mut ctx, loc, &[]);
        let extra_region = region(&mut ctx, loc, extra_block);
        ctx.op_mut(bad_func).regions.push(extra_region);
        ctx.region_mut(extra_region).parent_op = Some(bad_func);

        let bad_lambda = control_op(&ctx, module, "lambda");
        ctx.set_op_result_type(bad_lambda, 0, i32_ty);

        let bad_func_ref = control_ops(&ctx, module, "func_ref")
            .into_iter()
            .find(|op| {
                ctx.op(*op)
                    .attributes
                    .contains_key(Symbol::new("unexpected"))
            })
            .unwrap();
        let bad_call = control_op(&ctx, module, "call");
        let indirects = control_ops(&ctx, module, "call_indirect");
        let non_callable_indirect = indirects[0];
        let mismatched_indirect = indirects[1];
        let bad_perform = control_op(&ctx, module, "perform");
        let bad_handler = control_op(&ctx, module, "handler");
        let resumes = control_ops(&ctx, module, "resume");
        let bad_resume = resumes[0];
        let mismatched_resume = resumes[1];
        let unknown = control_op(&ctx, module, "unknown");

        let result = validate_local(&ctx, module);
        assert_op_diagnostics(
            &result,
            missing_func,
            &[
                "expects 0 operand(s), found 1",
                "requires 'sym_name' attribute",
                "requires 'type' attribute",
            ],
        );
        assert_op_diagnostics(
            &result,
            bad_func,
            &[
                "type attribute must be tribute_control.callable",
                "expects zero or one body region, found 2",
            ],
        );
        assert_op_diagnostics(
            &result,
            bad_lambda,
            &["result must have tribute_control.callable type"],
        );
        assert_op_diagnostics(
            &result,
            bad_func_ref,
            &[
                "expects 0 operand(s), found 1",
                "expects 1 result(s), found 2",
                "has unsupported attribute 'unexpected'",
                "result must have tribute_control.callable type",
            ],
        );
        assert_op_diagnostics(
            &result,
            bad_call,
            &[
                "expects 1 result(s), found 2",
                "expects 0 region(s), found 1",
                "requires 'callee' attribute",
                "has unsupported attribute 'unexpected'",
            ],
        );
        assert_op_diagnostics(
            &result,
            non_callable_indirect,
            &["callee operand must have tribute_control.callable type"],
        );
        assert_op_diagnostics(
            &result,
            mismatched_indirect,
            &[
                "argument types do not match the logical signature",
                "result types do not match the logical signature",
            ],
        );
        assert_op_diagnostics(
            &result,
            bad_perform,
            &["operation_kind must be @fn or @op, found @bogus"],
        );
        assert_op_diagnostics(
            &result,
            bad_handler,
            &[
                "kind must be @fn or @op, found @bogus",
                "expects 1 region(s), found 0",
            ],
        );
        assert_op_diagnostics(
            &result,
            bad_resume,
            &["first operand must have tribute_control.resume_token type"],
        );
        assert_op_diagnostics(
            &result,
            mismatched_resume,
            &["resume input/result types do not match the token input/answer types"],
        );
        assert_op_diagnostics(
            &result,
            unknown,
            &["unsupported tribute_control operation 'unknown'"],
        );

        let unattached_return = r#return(&mut ctx, loc, i32_value);
        let mut return_errors = Vec::new();
        validate_local_operation(&ctx, unattached_return.op_ref(), &mut return_errors);
        let return_result = ValidationResult {
            errors: return_errors,
        };
        assert_op_diagnostics(
            &return_result,
            unattached_return.op_ref(),
            &[
                "must be attached to a block",
                "must terminate a tribute_control.func or lambda body",
            ],
        );

        let unattached_yield = r#yield(&mut ctx, loc, i32_value);
        let mut yield_errors = Vec::new();
        validate_local_operation(&ctx, unattached_yield.op_ref(), &mut yield_errors);
        let yield_result = ValidationResult {
            errors: yield_errors,
        };
        assert_op_diagnostics(
            &yield_result,
            unattached_yield.op_ref(),
            &[
                "must be attached to a block",
                "must terminate a handle body/completion or handler body",
            ],
        );
    }

    #[test]
    fn local_validator_checks_handle_and_handler_region_contracts() {
        let (ctx, handle_module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    ^body(%unexpected: core.i32):
      %body_value = arith.const {value = 0} : core.i32
      tribute_control.yield %body_value
  } {
    ^completion(%value: core.bool, %extra: core.i32):
      tribute_control.yield %value
  } {
    %not_a_handler = arith.const {value = 0} : core.i32
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.bool, core.bool)):
        %wrong = arith.const {value = false} : core.bool
        tribute_control.yield %wrong
    }
  }
}"#,
        );
        let local = validate_local(&ctx, handle_module);
        assert_diagnostics(
            &local,
            &[
                "body block must not have arguments",
                "completion block must have exactly one argument",
                "completion argument type must match the body yield type",
                "completion yield type must match handle result type",
                "handlers block may contain only tribute_control.handler",
                "general handler yield type must match enclosing handle result type",
                "resume token input/answer types do not match operation and handle results",
                "general handler yield type must match handle result type",
            ],
        );

        let (ctx, fn_module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body_value = arith.const {value = 0} : core.i32
    tribute_control.yield %body_value
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @fn, op_name = @get, operation_result_type = core.bool} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        tribute_control.yield %argument
    }
  }
}"#,
        );
        let local = validate_local(&ctx, fn_module);
        assert_diagnostics(
            &local,
            &[
                "fn and op -> core.never handlers must not receive a resume token",
                "handler yield type does not match its kind-specific result",
            ],
        );

        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let i32_ty = simple_type(&mut ctx, "core", "i32");
        let module = module(&mut ctx, loc, &[]);
        // Textual regions always receive an implicit entry block, so zero-block
        // regions require direct in-memory construction.
        for malformed in 0..3 {
            let mut regions = Vec::new();
            for index in 0..3 {
                if index == malformed {
                    regions.push(ctx.create_region(RegionData {
                        location: loc,
                        blocks: Default::default(),
                        parent_op: None,
                    }));
                } else {
                    let region_block = block(&mut ctx, loc, &[]);
                    regions.push(region(&mut ctx, loc, region_block));
                }
            }
            let malformed_handle = trunk_ir::OperationDataBuilder::new(
                loc,
                Symbol::new("tribute_control"),
                Symbol::new("handle"),
            )
            .result(i32_ty)
            .region(regions[0])
            .region(regions[1])
            .region(regions[2])
            .build(&mut ctx);
            let malformed_handle = ctx.create_op(malformed_handle);
            append_module_op(&mut ctx, module, malformed_handle);
        }

        let local = validate_local(&ctx, module);
        assert_diagnostics(
            &local,
            &[
                "body region expects exactly one block, found 0",
                "completion region expects exactly one block, found 0",
                "handlers region expects exactly one block, found 0",
            ],
        );
    }

    #[test]
    fn whole_ir_rejects_missing_and_excess_lambda_captures() {
        let (missing_ctx, missing_module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @outer(%external: core.i32) -> core.i32 convention(direct) {
    %closure = tribute_control.lambda() -> core.i32 convention(direct) captures [] {
      tribute_control.return %external
    }
    tribute_control.return %external
  }
}"#,
        );
        let missing = validate_whole_ir(&missing_ctx, missing_module, &[], &[]);
        assert!(messages(&missing).contains("capture list is missing external value"));

        let (excess_ctx, excess_module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @outer(%external: core.i32) -> core.i32 convention(direct) {
    %closure = tribute_control.lambda() -> core.i32 convention(direct) captures [%external] {
      %constant = arith.const {value = 0} : core.i32
      tribute_control.return %constant
    }
    tribute_control.return %external
  }
}"#,
        );
        let excess = validate_whole_ir(&excess_ctx, excess_module, &[], &[]);
        assert!(messages(&excess).contains("capture list contains unused external value"));
    }

    #[test]
    fn whole_ir_resolves_same_named_functions_per_nested_module() {
        let (ctx, module) = parse_fixture(
            r#"core.module @outer {
  core.module @integers {
    !callable = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
    tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
      tribute_control.return %value
    }
    tribute_control.func @use(%value: core.i32) -> core.i32 convention(direct) {
      %reference = tribute_control.func_ref {func_ref = @id} : !callable
      %direct = tribute_control.call %value {callee = @id} : core.i32
      %indirect = tribute_control.call_indirect %reference, %direct : core.i32
      tribute_control.return %indirect
    }
  }
  core.module @booleans {
    !callable = tribute_control.callable(core.bool, core.bool) {tribute.calling_convention = 0}
    tribute_control.func @id(%value: core.bool) -> core.bool convention(direct) {
      tribute_control.return %value
    }
    tribute_control.func @use(%value: core.bool) -> core.bool convention(direct) {
      %reference = tribute_control.func_ref {func_ref = @id} : !callable
      %direct = tribute_control.call %value {callee = @id} : core.bool
      %indirect = tribute_control.call_indirect %reference, %direct : core.bool
      tribute_control.return %indirect
    }
  }
}"#,
        );

        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert!(result.is_ok(), "{:?}", result.errors);
    }

    #[test]
    fn whole_ir_reports_symbol_declaration_capture_and_token_contracts() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !direct = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !cps = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 2}
  !different = tribute_control.callable(core.bool) {tribute.calling_convention = 0}
  !token = tribute_control.resume_token(core.i32, core.i32)

  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct)
  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct)
  tribute_control.func @cps_target(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  %weak = tribute_control.func_ref {func_ref = @cps_target} : !direct
  %wrong_signature = tribute_control.func_ref {func_ref = @id} : !different
  %missing_ref = tribute_control.func_ref {func_ref = @missing} : !direct
  %missing_call = tribute_control.call {callee = @missing} : core.i32
  %false = arith.const {value = false} : core.bool
  %bad_call = tribute_control.call %false {callee = @id} : core.bool
  %unknown_perform = tribute_control.perform {ability_ref = core.ability_ref() {name = @State}, op_name = @missing, operation_kind = @op} : core.i32
  %bad_perform = tribute_control.perform %false {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @fn} : core.bool
  %duplicate_capture = tribute_control.lambda() -> core.bool convention(direct) captures [%false, %false] {
    tribute_control.return %false
  }
  %token_result = test.token_result : !token
  tribute_control.func @token_param(%token_arg: !token) -> core.i32 convention(direct) {
    %zero = arith.const {value = 0} : core.i32
    tribute_control.return %zero
  }
}"#,
        );
        let i32_ty = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core") && data.name == Symbol::new("i32"))
                    .then_some(ty)
            })
            .expect("core.i32");
        let ability_ref = ctx
            .types
            .iter()
            .find_map(|(ty, data)| {
                (data.dialect == Symbol::new("core")
                    && data.name == Symbol::new("ability_ref")
                    && data.attrs.get(Symbol::new("name"))
                        == Some(&Attribute::Symbol(Symbol::new("State"))))
                .then_some(ty)
            })
            .expect("State ability reference");
        // Operation declarations are semantic validation inputs rather than IR
        // operations, so they have no textual TrunkIR representation.
        let declaration = OperationDeclaration {
            ability_ref,
            op_name: Symbol::new("get"),
            kind: Symbol::new("op"),
            parameter_types: vec![i32_ty],
            result_type: i32_ty,
        };
        let declarations = [declaration.clone(), declaration];
        let result = validate_whole_ir(&ctx, module, &declarations, &[]);
        assert_diagnostics(
            &result,
            &[
                "duplicate function symbol @id",
                "func_ref result convention must be at least as strong as its target",
                "func_ref result source signature does not match its target",
                "unresolved function symbol @missing",
                "unresolved callee @missing",
                "call argument types do not match the logical signature",
                "call result types do not match the logical signature",
                "capture list contains duplicate values",
                "duplicate operation declaration",
                "no resolved operation declaration",
                "operation_kind does not match the resolved declaration",
                "operation argument types do not match the logical signature",
                "operation result type does not match the resolved declaration",
                "resume_token must not be produced as an operation result",
                "resume_token block arguments are allowed only as the final handler argument",
            ],
        );
    }

    #[test]
    fn whole_ir_rejects_duplicate_and_declaration_mismatched_handlers() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %input = arith.const {value = 0} : core.i32
    %performed = tribute_control.perform %input {ability_ref = core.ability_ref() {name = @State}, op_name = @get, operation_kind = @op} : core.i32
    tribute_control.yield %performed
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^first(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^duplicate(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }
  }
}"#,
        );
        let handler = control_ops(&ctx, module, "handler")[0];
        let handler_data = ctx.op(handler);
        // The declaration table is semantic verifier input, not textual IR.
        let declarations = [OperationDeclaration::new(
            handler_data.attributes.get_type("ability_ref").unwrap(),
            handler_data.attributes.get_symbol("op_name").unwrap(),
            Symbol::new("fn"),
            [handler_data
                .attributes
                .get_type("operation_result_type")
                .unwrap()],
            handler_data
                .attributes
                .get_type("operation_result_type")
                .unwrap(),
        )];
        let result = validate(&ctx, module, &declarations, &[]);
        let messages = messages(&result);
        assert!(messages.contains("duplicate handler clause"));
        assert!(messages.contains("kind does not match the resolved declaration"));
        assert!(messages.contains("operation_kind does not match the resolved declaration"));
    }

    #[test]
    fn whole_ir_rejects_multiple_resume_token_paths() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %first = tribute_control.resume %token, %argument : core.i32
        %second = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert!(messages(&result).contains("more than one tribute_control.resume"));
    }

    #[test]
    fn whole_ir_rejects_branching_and_escape_at_each_affine_carrier() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %inner = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %wrapper = tribute_control.lambda() -> core.i32 convention(direct) captures [%inner] {
          %called = tribute_control.call_indirect %inner : core.i32
          tribute_control.return %called
        }
        %escape = tribute_control.call %wrapper {callee = @id} : core.i32
        %first_capture = tribute_control.lambda() -> core.i32 convention(direct) captures [%wrapper] {
          %called = tribute_control.call_indirect %wrapper : core.i32
          tribute_control.return %called
        }
        %second_capture = tribute_control.lambda() -> core.i32 convention(direct) captures [%wrapper] {
          %called = tribute_control.call_indirect %wrapper : core.i32
          tribute_control.return %called
        }
        %first = tribute_control.call_indirect %wrapper : core.i32
        %second = tribute_control.call_indirect %wrapper : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert_diagnostics(
            &result,
            &[
                "resume-token carrier escapes through",
                "resume-token carrier branches into multiple lambda captures",
                "resume-token carrier call is outside its capture path",
            ],
        );
    }

    #[test]
    fn whole_ir_rejects_sibling_token_capture_paths() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %first = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %second = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        tribute_control.yield %argument
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert_diagnostics(
            &result,
            &["resume token is copied into multiple capture paths"],
        );
    }

    #[test]
    fn whole_ir_allows_mutually_exclusive_lambda_capture_and_terminal_paths() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %flag = arith.const {value = true} : core.i1
        %value = scf.if %flag : core.i32 {
          %left = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.return %resumed
          }
          %called = tribute_control.call_indirect %left : core.i32
          scf.yield %called
        } {
          %right = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.return %resumed
          }
          %called = tribute_control.call_indirect %right : core.i32
          scf.yield %called
        }
        tribute_control.yield %value
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        let diagnostics = messages(&result);
        for forbidden in [
            "resume token is copied into multiple capture paths",
            "resume token capture does not form a single path to resume",
            "resume-token carrier branches into multiple lambda captures",
            "resume-token carrier has multiple static terminal uses",
        ] {
            assert!(!diagnostics.contains(forbidden), "{result}");
        }
    }

    #[test]
    fn whole_ir_allows_mutually_exclusive_switch_capture_paths_without_default() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %choice = arith.const {value = 0} : core.i32
        scf.switch %choice {
          scf.case {value = 0} {
            %left = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
              %left_resumed = tribute_control.resume %token, %argument : core.i32
              tribute_control.return %left_resumed
            }
            %left_called = tribute_control.call_indirect %left : core.i32
            scf.yield
          }
          scf.case {value = 1} {
            %right = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
              %right_resumed = tribute_control.resume %token, %argument : core.i32
              tribute_control.return %right_resumed
            }
            %right_called = tribute_control.call_indirect %right : core.i32
            scf.yield
          }
        }
        tribute_control.yield %argument
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        let diagnostics = messages(&result);
        for forbidden in [
            "resume token is copied into multiple capture paths",
            "resume token capture does not form a single path to resume",
            "resume-token carrier branches into multiple lambda captures",
            "resume-token carrier has multiple static terminal uses",
        ] {
            assert!(!diagnostics.contains(forbidden), "{result}");
        }

        let mut branch_lambda = None;
        let mut switch = None;
        walk_region_ops(&ctx, module.body(&ctx).unwrap(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == Symbol::new("tribute_control") && data.name == Symbol::new("lambda")
            {
                branch_lambda.get_or_insert(op);
            } else if data.dialect == Symbol::new("scf") && data.name == Symbol::new("switch") {
                switch = Some(op);
            }
        });
        assert!(!ops_are_mutually_exclusive(
            &ctx,
            branch_lambda.unwrap(),
            switch.unwrap(),
        ));
    }

    #[test]
    fn whole_ir_rejects_same_path_lambda_carrier_terminals() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %continuation = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %first = tribute_control.call_indirect %continuation : core.i32
        %second = tribute_control.call_indirect %continuation : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("multiple static terminal uses"),
            "{result}"
        );
    }
    #[test]
    fn validators_reject_resume_token_escape_and_never_clause_capability() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        tribute_control.yield %token
    }
  }
}"#,
        );
        let local = validate_local(&ctx, module);
        assert!(messages(&local).contains("must not yield a resume token"));
        let whole = validate_whole_ir(&ctx, module, &[], &[]);
        assert!(messages(&whole).contains("forbidden use"));

        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.never} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %resumed = tribute_control.resume %token, %argument : core.i32
        tribute_control.yield %resumed
    }
  }
}"#,
        );
        let result = validate_local(&ctx, module);
        let messages = messages(&result);
        assert!(messages.contains("must not receive a resume token"));
        assert!(messages.contains("must not contain tribute_control.resume"));
    }

    #[test]
    fn resume_token_components_are_checked_recursively() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !physical = func.func_sig<() -> core.i32>
  !wrapper = core.tuple(!physical)
  !token = tribute_control.resume_token(!wrapper, core.i32)
}"#,
        );

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("is not a resolved logical value type"));
    }

    #[test]
    fn validator_reports_malformed_resume_token_uses_without_panicking() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !malformed = tribute_control.resume_token(core.i32)
  tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
    ^clause(%value: core.i32, %token: !malformed):
      %resumed = tribute_control.resume %token, %value : core.i32
      tribute_control.yield %resumed
  }
}"#,
        );
        let handler = control_op(&ctx, module, "handler");
        let resume = control_op(&ctx, module, "resume");
        let result = validate_local(&ctx, module);
        assert!(
            result.errors.iter().any(|error| {
                error.op.is_none()
                    && error
                        .message
                        .contains("resume_token requires exactly input and answer types")
            }),
            "{result}"
        );
        assert!(
            result.errors.iter().any(|error| {
                error.op == Some(handler)
                    && error
                        .message
                        .contains("requires a final resume_token block argument")
            }),
            "{result}"
        );
        assert!(
            result.errors.iter().any(|error| {
                error.op == Some(resume)
                    && error
                        .message
                        .contains("first operand must have tribute_control.resume_token type")
            }),
            "{result}"
        );
    }

    #[test]
    fn whole_ir_rejects_branch_after_transitive_lambda_transfer() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %handled = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^clause(%argument: core.i32, %token: tribute_control.resume_token(core.i32, core.i32)):
        %inner = tribute_control.lambda() -> core.i32 convention(direct) captures [%token, %argument] {
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.return %resumed
        }
        %wrapper = tribute_control.lambda() -> core.i32 convention(direct) captures [%inner] {
          %called = tribute_control.call_indirect %inner : core.i32
          tribute_control.return %called
        }
        %first = tribute_control.call_indirect %wrapper : core.i32
        %second = tribute_control.call_indirect %wrapper : core.i32
        tribute_control.yield %first
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("multiple static terminal uses"),
            "{result}"
        );
    }

    #[test]
    fn whole_ir_rejects_outer_token_resumed_in_nested_handler() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %outer = tribute_control.handle : core.i32 {
    %body = arith.const {value = 0} : core.i32
    tribute_control.yield %body
  } {
    ^completion(%value: core.i32):
      tribute_control.yield %value
  } {
    tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @op, op_name = @get, operation_result_type = core.i32} {
      ^outer_clause(%argument: core.i32, %outer_token: tribute_control.resume_token(core.i32, core.i32)):
        %nested = tribute_control.handle : core.i32 {
          tribute_control.yield %argument
        } {
          ^nested_completion(%value: core.i32):
            tribute_control.yield %value
        } {
          tribute_control.handler {ability_ref = core.ability_ref() {name = @State}, kind = @fn, op_name = @get, operation_result_type = core.i32} {
            ^inner_clause(%inner_argument: core.i32):
              %crossed = tribute_control.resume %outer_token, %inner_argument : core.i32
              tribute_control.yield %crossed
          }
        }
        tribute_control.yield %nested
    }
  }
}"#,
        );
        let result = validate_whole_ir(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("crosses into a different handler"),
            "{result}"
        );
    }

    #[test]
    fn local_validator_rejects_external_reference_in_isolated_func() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  %external = arith.const {value = 1} : core.i32
  tribute_control.func @invalid() -> core.i32 convention(direct) {
    tribute_control.return %external
  }
}"#,
        );

        let result = validate_local(&ctx, module);
        assert!(messages(&result).contains("references external value"));
    }

    #[test]
    fn exact_registered_intrinsic_is_accepted_but_spelling_and_abi_are_not() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"}
}"#,
        );
        let function = control_op(&ctx, module, "func");
        let callable_type = ctx.op(function).attributes.get_type("type").unwrap();
        let exact = CompilerIntrinsicDeclaration::new(
            Symbol::new("Nat::+"),
            Symbol::new("Nat::+"),
            callable_type,
        );
        assert!(validate(&ctx, module, &[], std::slice::from_ref(&exact)).is_ok());

        let wrong_signature = CompilerIntrinsicDeclaration::new(
            exact.symbol,
            exact.identity,
            ctx.types.get(callable_type).params[0],
        );
        let result = validate(&ctx, module, &[], &[wrong_signature]);
        assert!(messages(&result).contains("complete signature"), "{result}");

        let result = validate(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("unregistered declaration"),
            "{result}"
        );
    }

    #[test]
    fn registered_intrinsic_requires_direct_convention() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @read(%value: core.i32) -> core.i32 convention(evidence_direct)
    attributes {tribute.compiler_intrinsic = @read}
}"#,
        );
        let function = control_op(&ctx, module, "func");
        let callable_type = ctx.op(function).attributes.get_type("type").unwrap();
        let declaration = CompilerIntrinsicDeclaration::new(
            Symbol::new("read"),
            Symbol::new("read"),
            callable_type,
        );

        let result = validate(&ctx, module, &[], &[declaration]);
        assert!(
            messages(&result).contains("must use Direct calling convention"),
            "{result}"
        );
    }

    #[test]
    fn exact_registered_intrinsic_has_indirect_callable_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32, core.i32) {tribute.calling_convention = 0}
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"}
  tribute_control.func @caller(%left: core.i32, %right: core.i32) -> core.i32 convention(direct) {
    %callee = tribute_control.func_ref {func_ref = @"Nat::+"} : !F
    %result = tribute_control.call_indirect %callee, %left, %right : core.i32
    tribute_control.return %result
  }
}"#,
        );
        let intrinsic = control_ops(&ctx, module, "func")
            .into_iter()
            .find(|op| ctx.op(*op).attributes.get_symbol("sym_name") == Some(Symbol::new("Nat::+")))
            .unwrap();
        let callable_type = ctx.op(intrinsic).attributes.get_type("type").unwrap();
        let declaration = CompilerIntrinsicDeclaration::new(
            Symbol::new("Nat::+"),
            Symbol::new("Nat::+"),
            callable_type,
        );

        let result = validate(&ctx, module, &[], &[declaration]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn intrinsic_registry_order_duplicates_and_defined_claims_fail_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @"Int::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Int::+"}
  tribute_control.func @"Nat::+"(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
    attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"Nat::+"} {
    tribute_control.return %left
  }
}"#,
        );
        let functions = control_ops(&ctx, module, "func");
        let declaration = |symbol: Symbol| {
            let function = functions
                .iter()
                .copied()
                .find(|op| ctx.op(*op).attributes.get_symbol("sym_name") == Some(symbol))
                .unwrap();
            CompilerIntrinsicDeclaration::new(
                symbol,
                symbol,
                ctx.op(function).attributes.get_type("type").unwrap(),
            )
        };
        let nat = declaration(Symbol::new("Nat::+"));
        let int = declaration(Symbol::new("Int::+"));

        let result = validate(&ctx, module, &[], &[nat.clone(), int, nat]);
        let diagnostics = messages(&result);
        assert!(
            diagnostics.contains("not deterministically ordered"),
            "{result}"
        );
        assert!(
            diagnostics.contains("duplicate compiler intrinsic"),
            "{result}"
        );
        assert!(
            diagnostics.contains("defined Tribute function cannot claim compiler intrinsic"),
            "{result}"
        );
    }

    #[test]
    fn managed_reference_metadata_and_return_contracts_fail_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct() {name = @S, fields = []}
  !Unnamed = adt.typeref()
  !Parameterized = adt.typeref(core.i32) {name = @S}
  !Missing = adt.typeref() {name = @Missing}
  tribute_control.func @unnamed(%value: !Unnamed) -> !Unnamed convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @parameterized(%value: !Parameterized) -> !Parameterized convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @missing(%value: !Missing) -> !Missing convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @wrong_return(%integer: core.i32, %flag: core.bool) -> core.i32 convention(direct) {
    tribute_control.return %flag
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        let diagnostics = messages(&result);
        assert!(
            diagnostics.contains("requires nominal name metadata"),
            "{result}"
        );
        assert!(
            diagnostics.contains("cannot have type parameters"),
            "{result}"
        );
        assert!(
            diagnostics.contains("has no verified nominal declaration"),
            "{result}"
        );
        assert!(
            diagnostics.contains("return operand type does not match containing semantic callable"),
            "{result}"
        );
    }

    #[test]
    fn managed_defined_boundaries_null_and_compatible_cast_are_valid() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct() {name = @S, fields = []}
  !R = adt.typeref() {name = @S}
  tribute_control.func @managed(%value: !R) -> !R convention(direct) {
    %null = adt.ref_null {type = !R} : !R
    %cast = adt.ref_cast %null {type = !R} : !R
    tribute_control.return %cast
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn managed_bodyless_external_read_line_shape_and_raw_pointer_cast_chain_fail_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !ReadLineResult = adt.enum() {name = @ReadLineResult, variants = [[@ReadLine, [core.bytes]], [@ReadEndOfFile, []], [@ReadInvalidEncoding, []], [@ReadSystem, [core.i32, core.bytes]]]}
  !ReadLineResultRef = adt.typeref() {name = @ReadLineResult}
  tribute_control.func @user_read_line() -> !ReadLineResultRef convention(direct)
    attributes {abi = "intrinsic"}
  tribute_control.func @masquerade(%raw: core.ptr) -> !ReadLineResultRef convention(direct) {
    %middle = arith.cast %raw : core.i64
    %managed = core.unrealized_conversion_cast %middle : !ReadLineResultRef
    tribute_control.return %managed
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        let diagnostics = messages(&result);
        assert!(diagnostics.contains("bodyless external"), "{result}");
        assert!(diagnostics.contains("core.ptr cast chain"), "{result}");
    }

    #[test]
    fn managed_nested_aggregate_in_bodyless_external_fails_closed() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct() {name = @S, fields = []}
  !R = adt.typeref() {name = @S}
  !Container = adt.struct() {name = @Container, fields = [[@managed, !R]]}
  tribute_control.func @private_helper(%value: !Container) -> core.i32 convention(direct)
    attributes {abi = "C"}
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(messages(&result).contains("bodyless external"), "{result}");
    }

    #[test]
    fn primitive_external_remains_a_conservative_boundary() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @primitive(%buffer: core.ptr, %length: core.i32) -> core.ptr convention(direct)
    attributes {abi = "C"}
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn ref_cast_rejects_distinct_nominal_managed_references() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !S = adt.struct() {name = @S, fields = []}
  !T = adt.struct() {name = @T, fields = []}
  !RS = adt.typeref() {name = @S}
  !RT = adt.typeref() {name = @T}
  tribute_control.func @incompatible(%value: !RS) -> !RT convention(direct) {
    %cast = adt.ref_cast %value {type = !RT} : !RT
    tribute_control.return %cast
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("compatible managed nominal reference types"),
            "{result}"
        );
    }

    #[test]
    fn indirect_call_rejects_typed_callable_cast_without_semantic_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  tribute_control.func @caller(%raw: core.ptr, %value: core.i32) -> core.i32 convention(direct) {
    %callee = core.unrealized_conversion_cast %raw : tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    #[test]
    fn indirect_call_rejects_callable_from_uncontracted_structured_block_argument() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  tribute_control.func @caller(%condition: core.i1, %value: core.i32) -> core.i32 convention(direct) {
    %result = scf.if %condition : core.i32 {
      ^then(%callee: !F):
        %called = tribute_control.call_indirect %callee, %value : core.i32
        scf.yield %called
    } {
      ^else:
        scf.yield %value
    }
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    #[test]
    fn callable_handle_completion_argument_traces_the_body_yield() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  tribute_control.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_control.func @caller(%value: core.i32) -> !F convention(direct) {
    %callback = tribute_control.handle : !F {
      %source = tribute_control.func_ref {func_ref = @id} : !F
      tribute_control.yield %source
    } {
      ^completion(%callback: !F):
        %ignored = tribute_control.call_indirect %callback, %value : core.i32
        tribute_control.yield %callback
    } {
      ^handlers:
    }
    tribute_control.return %callback
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn callable_handler_parameter_requires_an_exact_operation_declaration() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  tribute_control.func @caller(%value: core.i32) -> core.i32 convention(direct) {
    %handled = tribute_control.handle : core.i32 {
      tribute_control.yield %value
    } {
      ^completion(%completed: core.i32):
        tribute_control.yield %completed
    } {
      tribute_control.handler {ability_ref = core.ability_ref() {name = @Callback}, kind = @fn, op_name = @apply, operation_result_type = core.i32} {
        ^handler(%callback: !F):
          %called = tribute_control.call_indirect %callback, %value : core.i32
          tribute_control.yield %called
      }
    }
    tribute_control.return %handled
  }
}"#,
        );
        let handler = control_op(&ctx, module, "handler");
        let handler_region = ctx.op(handler).regions[0];
        let handler_block = ctx.region(handler_region).blocks[0];
        let callback_type = ctx.value_ty(ctx.block_args(handler_block)[0]);
        let ability = ctx.op(handler).attributes.get_type("ability_ref").unwrap();
        let operation_result = ctx
            .op(handler)
            .attributes
            .get_type("operation_result_type")
            .unwrap();
        let declaration = OperationDeclaration::new(
            ability,
            Symbol::new("apply"),
            Symbol::new("fn"),
            [callback_type],
            operation_result,
        );

        let result = validate(&ctx, module, &[declaration], &[]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn exact_adt_callable_projections_have_semantic_provenance() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !Tuple = adt.struct() {name = @Tuple, fields = [[@callee, !F]]}
  !TupleRef = adt.typeref() {name = @Tuple}
  !Choice = adt.enum() {name = @Choice, variants = [[@Some, [!F]]]}
  !ChoiceRef = adt.typeref() {name = @Choice}
  tribute_control.func @caller(%tuple: !TupleRef, %choice: !ChoiceRef, %value: core.i32) -> core.i32 convention(direct) {
    %tuple_callee = adt.struct_get %tuple {type = !Tuple, field = 0} : !F
    %tuple_result = tribute_control.call_indirect %tuple_callee, %value : core.i32
    %some = adt.variant_cast %choice {type = !Choice, tag = @Some} : !Choice
    %choice_callee = adt.variant_get %some {type = !Choice, tag = @Some, field = 0} : !F
    %choice_result = tribute_control.call_indirect %choice_callee, %tuple_result : core.i32
    tribute_control.return %choice_result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn adt_callable_projection_requires_matching_declared_field_type() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !Tuple = adt.struct() {name = @Tuple, fields = [[@not_callable, core.i32]]}
  !TupleRef = adt.typeref() {name = @Tuple}
  !Choice = adt.enum() {name = @Choice, variants = [[@Some, [core.i32]]]}
  !ChoiceRef = adt.typeref() {name = @Choice}
  tribute_control.func @caller(%tuple: !TupleRef, %choice: !ChoiceRef, %value: core.i32) -> core.i32 convention(direct) {
    %tuple_callee = adt.struct_get %tuple {type = !Tuple, field = 0} : !F
    %tuple_result = tribute_control.call_indirect %tuple_callee, %value : core.i32
    %some = adt.variant_cast %choice {type = !Choice, tag = @Some} : !Choice
    %choice_callee = adt.variant_get %some {type = !Choice, tag = @Some, field = 0} : !F
    %choice_result = tribute_control.call_indirect %choice_callee, %tuple_result : core.i32
    tribute_control.return %choice_result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert_eq!(
            messages(&result).matches("callable provenance").count(),
            2,
            "{result}"
        );
    }

    #[test]
    fn duplicate_nominal_layout_rejects_callable_projection_spoofing() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  !Canonical = adt.struct() {name = @Tuple, fields = [[@value, core.i32]]}
  !Spoofed = adt.struct() {name = @Tuple, fields = [[@callee, !F]]}
  !TupleRef = adt.typeref() {name = @Tuple}
  tribute_control.func @caller(%tuple: !TupleRef, %value: core.i32) -> core.i32 convention(direct) {
    %canonical = adt.struct_new %value {type = !Canonical} : !TupleRef
    %callee = adt.struct_get %tuple {type = !Spoofed, field = 0} : !F
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        let diagnostics = messages(&result);
        assert!(diagnostics.contains("nominal layout @Tuple is declared more than once"));
        assert!(diagnostics.contains("callable provenance"), "{result}");
    }

    #[test]
    fn unreachable_nominal_layout_collision_does_not_affect_validation() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !UnusedStruct = adt.struct() {name = @Unused, fields = [[@value, core.i32]]}
  !UnusedEnum = adt.enum() {name = @Unused, variants = [[@Value, [core.i32]]]}
  tribute_control.func @caller(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn indirect_call_rejects_callable_from_unclassified_external() {
        let (ctx, module) = parse_fixture(
            r#"core.module @test {
  !F = tribute_control.callable(core.i32, core.i32) {tribute.calling_convention = 0}
  tribute_control.func @factory() -> !F convention(direct) attributes {abi = "C"}
  tribute_control.func @caller(%value: core.i32) -> core.i32 convention(direct) {
    %callee = tribute_control.call {callee = @factory} : !F
    %result = tribute_control.call_indirect %callee, %value : core.i32
    tribute_control.return %result
  }
}"#,
        );

        let result = validate(&ctx, module, &[], &[]);
        assert!(
            messages(&result).contains("callable provenance"),
            "{result}"
        );
    }

    #[test]
    fn only_func_is_isolated_and_only_safe_value_ops_are_pure() {
        let fixture = valid_fixture();
        let ctx = &fixture.ctx;
        let body = fixture.module.body(ctx).expect("module body");
        let mut func = None;
        let mut lambda = None;
        let mut func_ref = None;
        let mut call = None;
        walk_region_ops(ctx, body, &mut |op| {
            func = func.or_else(|| is_control_op(ctx, op, "func").then_some(op));
            lambda = lambda.or_else(|| is_control_op(ctx, op, "lambda").then_some(op));
            func_ref = func_ref.or_else(|| is_control_op(ctx, op, "func_ref").then_some(op));
            call = call.or_else(|| is_control_op(ctx, op, "call").then_some(op));
        });
        let func = func.expect("func");
        assert!(trunk_ir::op_interface::IsolatedFromAboveOps::is_isolated(
            ctx, func
        ));
        if let Some(lambda) = lambda {
            assert!(!trunk_ir::op_interface::IsolatedFromAboveOps::is_isolated(
                ctx, lambda
            ));
        }
        assert!(trunk_ir::op_interface::PureOps::is_pure(
            ctx,
            func_ref.expect("func_ref")
        ));
        assert!(!trunk_ir::op_interface::PureOps::is_pure(
            ctx,
            call.expect("call")
        ));
    }
}
