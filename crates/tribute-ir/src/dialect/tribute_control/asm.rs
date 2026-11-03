//! Custom assembly of `tribute_control.func` and `tribute_control.lambda`.

use super::*;

// === Custom assembly: tribute_control.func ===

pub(super) fn print_row(
    h: &mut dyn trunk_ir::asm_format::SignatureWriter,
    row: &Row,
) -> fmt::Result {
    if row.is_empty() {
        return Ok(());
    }
    write!(h, " row(")?;
    for (index, instance) in row.instances.iter().enumerate() {
        if index > 0 {
            write!(h, ", ")?;
        }
        h.write_attribute(&Attribute::SymbolRef(instance.clone()))?;
    }
    if let Some(tail) = row.tail {
        if !row.instances.is_empty() {
            write!(h, " ")?;
        }
        write!(h, "| 'a{tail}")?;
    }
    write!(h, ")")
}

/// A definition's row binders as its header declares them, or `None` when
/// the stored attributes cannot be written that way.
struct BinderHeader {
    binders: Vec<RowBinder>,
    unions: Vec<RowUnion>,
    inferred: bool,
}

impl BinderHeader {
    fn of(ctx: &IrContext, op: OpRef) -> Option<Self> {
        let binders = row_binders(ctx, op).ok()?;
        let unions = row_unions(ctx, op).ok()?;
        let in_range = |index: u32| usize::try_from(index).is_ok_and(|index| index < binders.len());
        let mut results = HashSet::default();
        let well_formed = unions.iter().all(|union| {
            in_range(union.result)
                && union.sources.iter().copied().all(in_range)
                && results.insert(union.result)
        });
        let inferred = match ctx.op_attrs(op).get(INFERRED_ROW_ATTR) {
            None | Some(Attribute::Bool(false)) => false,
            Some(Attribute::Bool(true)) => true,
            Some(_) => return None,
        };
        // An empty list written out has no header form.
        let attrs = ctx.op_attrs(op);
        let written_empty = (attrs.contains_key(ROW_BINDERS_ATTR) && binders.is_empty())
            || (attrs.contains_key(ROW_UNIONS_ATTR) && unions.is_empty());
        (well_formed && !written_empty).then_some(Self {
            binders,
            unions,
            inferred,
        })
    }

    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> fmt::Result {
        use fmt::Write;
        if self.binders.is_empty() {
            return Ok(());
        }
        write!(h, "[")?;
        for (index, binder) in self.binders.iter().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            if *binder != RowBinder::Param {
                write!(h, "{} ", binder.keyword())?;
            }
            write!(h, "'a{index}")?;
            let union = self
                .unions
                .iter()
                .find(|union| usize::try_from(union.result).ok() == Some(index));
            if let Some(union) = union {
                write!(h, " = ")?;
                for (position, source) in union.sources.iter().enumerate() {
                    if position > 0 {
                        write!(h, " | ")?;
                    }
                    write!(h, "'a{source}")?;
                }
            }
        }
        write!(h, "]")
    }
}

/// `'a<index>`, a row binder named by its position.
fn parse_binder_name(input: &mut &str) -> winnow::ModalResult<u32> {
    use winnow::ascii::digit1;
    use winnow::prelude::*;

    "'a".parse_next(input)?;
    digit1.try_map(str::parse::<u32>).parse_next(input)
}

/// A definition header's `[...]` binder list, as the raw `row_binders` and
/// `row_unions` attributes; nothing when the header has none.
fn parse_binders<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<
    Vec<(
        std::borrow::Cow<'a, str>,
        trunk_ir::parser::raw::RawAttribute<'a>,
    )>,
> {
    use trunk_ir::parser::raw::{RawAttribute, ident, ws};
    use winnow::combinator::{opt, separated};
    use winnow::prelude::*;

    let reject = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt('[').parse_next(input)?.is_none() {
        *input = checkpoint;
        return Ok(Vec::new());
    }
    let mut kinds = Vec::new();
    let mut unions = Vec::new();
    loop {
        ws.parse_next(input)?;
        let kind = match opt(ident).parse_next(input)?.map(RowBinder::from_keyword) {
            Some(Some(kind)) if kind != RowBinder::Param => {
                ws.parse_next(input)?;
                kind
            }
            None => RowBinder::Param,
            Some(_) => return Err(reject()),
        };
        let index = parse_binder_name(input)?;
        if usize::try_from(index).ok() != Some(kinds.len()) {
            return Err(reject());
        }
        kinds.push(RawAttribute::String(kind.keyword().to_owned()));
        ws.parse_next(input)?;
        if opt('=').parse_next(input)?.is_some() {
            let sources: Vec<u32> = separated(
                1..,
                (ws, parse_binder_name, ws).map(|(_, source, _)| source),
                '|',
            )
            .parse_next(input)?;
            unions.push(RawAttribute::Dict(vec![
                ("result".into(), RawAttribute::Int(i128::from(index))),
                (
                    "sources".into(),
                    RawAttribute::List(
                        sources
                            .into_iter()
                            .map(|source| RawAttribute::Int(i128::from(source)))
                            .collect(),
                    ),
                ),
            ]));
        }
        ws.parse_next(input)?;
        if opt(',').parse_next(input)?.is_none() {
            break;
        }
    }
    ']'.parse_next(input)?;
    let mut attributes = vec![(ROW_BINDERS_ATTR.into(), RawAttribute::List(kinds))];
    if !unions.is_empty() {
        attributes.push((ROW_UNIONS_ATTR.into(), RawAttribute::List(unions)));
    }
    Ok(attributes)
}

/// The `inferred` keyword before a definition's convention.
fn parse_inferred(input: &mut &str) -> winnow::ModalResult<bool> {
    use trunk_ir::parser::raw::ws;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt((
        "inferred",
        winnow::combinator::peek(winnow::ascii::multispace1),
    ))
    .parse_next(input)?
    .is_some()
    {
        return Ok(true);
    }
    *input = checkpoint;
    Ok(false)
}

pub(super) fn func_sig_parts(ctx: &IrContext, ty: TypeRef) -> Option<(TypeRef, Vec<TypeRef>)> {
    let signature = FuncSig::from_type_ref(ctx, ty)?;
    Some((signature.result(ctx), signature.inputs(ctx).to_vec()))
}

/// Whether custom source assembly can represent the complete signature.
///
/// Invalid and metadata-bearing signatures must use generic assembly so
/// printing never discards their storage.
fn has_concise_func_sig(ctx: &IrContext, ty: Option<TypeRef>) -> bool {
    ty.and_then(|ty| FuncSig::from_type_ref(ctx, ty))
        .is_some_and(|signature| !signature.has_nonreserved_attrs(ctx))
}

fn print_extra_attributes(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    excluded: &[&str],
) -> fmt::Result {
    use fmt::Write;
    let ctx = h.ctx();
    let attrs: Vec<_> = ctx
        .op_attrs(op)
        .iter()
        .filter(|(key, _)| !excluded.contains(key))
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

/// Print a source function as `name`, with the convention the operation
/// declares, if any.
pub(crate) fn print_func(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    name: &str,
    convention: Option<CallingConvention>,
) -> fmt::Result {
    use fmt::Write;

    let symbol = h.ctx().op_attrs(op).get_str("sym_name");
    let callable_ty = h.ctx().op_attrs(op).get_type("type");
    let region = h.ctx().op_region(op, 0);
    let header = BinderHeader::of(h.ctx(), op);
    let Some(header) = header.filter(|_| has_concise_func_sig(h.ctx(), callable_ty)) else {
        return h.print_generic(op);
    };
    let parts = callable_ty.and_then(|ty| func_sig_parts(h.ctx(), ty));

    write!(h, "{name} ")?;
    if let Some(symbol) = symbol {
        h.write_symbol_text(symbol)?;
    } else {
        write!(h, "@<missing>")?;
    }
    header.print(h)?;
    h.reset_numbering();

    let param_types = parts
        .as_ref()
        .map(|(_, params)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(h, region, param_types)?;

    if let Some((result, _)) = parts {
        write!(h, " -> ")?;
        h.write_type(result)?;
        if let Some(ty) = callable_ty {
            let row = FuncSig(ty).row(h.ctx());
            print_row(h, &row)?;
        }
        if header.inferred {
            write!(h, " inferred")?;
        }
        if let Some(convention) = convention {
            write!(h, " convention({convention})")?;
        }
    }
    let mut excluded = vec![
        "sym_name",
        "type",
        CALLING_CONVENTION_ATTR,
        ROW_BINDERS_ATTR,
        ROW_UNIONS_ATTR,
        INFERRED_ROW_ATTR,
    ];
    if convention.is_some() {
        excluded.push(tribute_cps::CONVENTION_ATTR);
    }
    print_extra_attributes(h, op, &excluded)?;

    match region {
        Some(region) => h.print_body(region),
        None => Ok(()),
    }
}

/// The concise ` convention(...)`; `None` when the signature has none yet.
fn parse_convention(input: &mut &str) -> winnow::ModalResult<Option<CallingConvention>> {
    use trunk_ir::parser::raw::{ident, ws};
    use winnow::combinator::opt;
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt("convention").parse_next(input)?.is_none() {
        *input = checkpoint;
        return Ok(None);
    }
    ws.parse_next(input)?;
    '('.parse_next(input)?;
    ws.parse_next(input)?;
    let keyword = ident.parse_next(input)?;
    let Ok(convention) = keyword.parse::<CallingConvention>() else {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    };
    ws.parse_next(input)?;
    ')'.parse_next(input)?;
    Ok(Some(convention))
}

/// The concise row of a source signature, `row(@A, @B | 'a1)`, as the raw
/// `ability_row` attribute; `None` when the signature writes no row.
pub(super) fn parse_row<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<Option<trunk_ir::parser::raw::RawAttribute<'a>>> {
    use trunk_ir::parser::raw::{RawAttribute, symbol_path, ws};
    use winnow::combinator::{opt, separated};
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if opt("row(").parse_next(input)?.is_none() {
        *input = checkpoint;
        return Ok(None);
    }
    ws.parse_next(input)?;
    let instances: Vec<Vec<String>> =
        separated(0.., (ws, symbol_path, ws).map(|(_, path, _)| path), ',').parse_next(input)?;
    ws.parse_next(input)?;
    let tail = opt(('|', ws, parse_binder_name, ws))
        .parse_next(input)?
        .map(|(_, _, tail, _)| tail);
    ')'.parse_next(input)?;
    if instances.is_empty() && tail.is_none() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }
    let mut entries = Vec::new();
    if !instances.is_empty() {
        entries.push((
            "instances".into(),
            RawAttribute::List(instances.into_iter().map(RawAttribute::SymbolRef).collect()),
        ));
    }
    if let Some(tail) = tail {
        entries.push(("tail".into(), RawAttribute::Int(i128::from(tail))));
    }
    Ok(Some(RawAttribute::Dict(entries)))
}

fn func_sig_raw_type<'a>(
    result: trunk_ir::parser::raw::RawType<'a>,
    params: &[(&'a str, trunk_ir::parser::raw::RawType<'a>)],
    row: Option<trunk_ir::parser::raw::RawAttribute<'a>>,
) -> trunk_ir::parser::raw::RawType<'a> {
    let attrs = row
        .map(|row| (ABILITY_ROW_ATTR.into(), row))
        .into_iter()
        .collect();
    trunk_ir::dialect::func::signature_raw_type(
        "tribute_control.func_sig",
        params.iter().map(|(_, ty)| ty.clone().into()).collect(),
        vec![result.into()],
        attrs,
    )
}

/// Whether an operation's attributes hold the row its signature type owns, or
/// the physical convention attribute.
fn has_duplicate_convention_attr(
    attrs: &[(
        std::borrow::Cow<'_, str>,
        trunk_ir::parser::raw::RawAttribute<'_>,
    )],
) -> bool {
    attrs
        .iter()
        .any(|(key, _)| key == CALLING_CONVENTION_ATTR || key == ABILITY_ROW_ATTR)
}

/// Parse a source function named `name`. With `on_operation`, the operation
/// declares the convention, which it then requires; otherwise it has none.
pub(crate) fn parse_func<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
    name: &'static str,
    on_operation: bool,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    if !results.is_empty() || sym_name.is_none() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    let binders = parse_binders(input)?;
    ws.parse_next(input)?;
    let params = func_params.parse_next(input)?;
    let result = return_type.parse_next(input)?;
    let row = parse_row(input)?;
    let inferred = parse_inferred(input)?;
    let convention = parse_convention(input)?;
    if on_operation != convention.is_some() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();
    if has_duplicate_convention_attr(&attributes)
        || attributes.iter().any(|(key, _)| {
            matches!(
                key.as_ref(),
                "sym_name" | "type" | ROW_BINDERS_ATTR | ROW_UNIONS_ATTR | INFERRED_ROW_ATTR
            ) || (on_operation && key == tribute_cps::CONVENTION_ATTR)
        })
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

    let signature = func_sig_raw_type(result, &params, row);
    let mut attributes = attributes;
    attributes.extend(binders);
    if let Some(convention) = convention {
        attributes.push((
            tribute_cps::CONVENTION_ATTR.into(),
            RawAttribute::String(convention.keyword().to_owned()),
        ));
    }
    if inferred {
        attributes.push((INFERRED_ROW_ATTR.into(), RawAttribute::Bool(true)));
    }
    attributes.push(("type".into(), RawAttribute::Type(signature)));

    Ok(RawOperation {
        results,
        name,
        sym_name,
        func_params: vec![],
        func_param_regions: vec![],
        operands: vec![],
        attributes,
        result_types: vec![],
        regions,
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Func {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        print_func(h, self.op_ref(), "tribute_control.func", None)
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        parse_func(input, results, sym_name, "tribute_control.func", false)
    }
}

// === Custom assembly: tribute_control.lambda ===

/// Print a source lambda as `name`. With the conventions of the bodies the
/// operation declares, each body is written under its convention in
/// `entries`; without them, the one body follows the captures.
pub(crate) fn print_lambda(
    h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>,
    op: OpRef,
    name: &str,
    entries: Option<&[CallingConvention]>,
) -> fmt::Result {
    use fmt::Write;

    let result_types = h.ctx().op_result_types(op);
    let callable_ty = result_types.first().copied();
    // An empty list written out has no header form.
    let quantified = quantified_rows(h.ctx(), op).ok().filter(|quantified| {
        !quantified.is_empty() || !h.ctx().op_attrs(op).contains_key(QUANTIFIED_ROWS_ATTR)
    });
    let Some(quantified) = quantified
        .filter(|_| result_types.len() == 1 && has_concise_func_sig(h.ctx(), callable_ty))
    else {
        return h.print_generic(op);
    };

    let result = h.ctx().op_results(op).first().copied();
    let result_name = result.map(|value| h.assign_value_name(value));
    let parts = callable_ty.and_then(|ty| func_sig_parts(h.ctx(), ty));
    let region = h.ctx().op_region(op, 0);

    if let Some(name) = result_name {
        write!(h, "{name} = ")?;
    }
    write!(h, "{name}")?;
    if !quantified.is_empty() {
        write!(h, "[")?;
        for (index, binder) in quantified.iter().enumerate() {
            if index > 0 {
                write!(h, ", ")?;
            }
            write!(h, "'a{binder}")?;
        }
        write!(h, "]")?;
    }

    let param_types = parts
        .as_ref()
        .map(|(_, params)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(h, region, param_types)?;
    if let Some((result, _)) = parts {
        write!(h, " -> ")?;
        h.write_type(result)?;
        if let Some(ty) = callable_ty {
            let row = FuncSig(ty).row(h.ctx());
            print_row(h, &row)?;
        }
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
    let mut excluded = vec![CALLING_CONVENTION_ATTR, QUANTIFIED_ROWS_ATTR];
    if entries.is_some() {
        excluded.push(tribute_cps::CONVENTIONS_ATTR);
    }
    print_extra_attributes(h, op, &excluded)?;

    match (region, entries) {
        (Some(first), Some(entries)) => {
            // Every body takes the parameters the header names.
            let ctx = h.ctx();
            let entry_args = |region: RegionRef| {
                ctx.region(region)
                    .blocks
                    .first()
                    .map_or(&[][..], |&block| ctx.block_args(block))
            };
            let bodies: Vec<_> = ctx.op_regions(op).collect();
            for &body in &bodies[1..] {
                for (&arg, &like) in entry_args(body).iter().zip(entry_args(first)) {
                    h.assign_value_name_like(arg, like);
                }
            }
            write!(h, " entries")?;
            h.print_labelled_bodies(
                entries
                    .iter()
                    .map(|convention| convention.keyword())
                    .zip(bodies),
            )
        }
        (Some(region), None) => h.print_body(region),
        (None, _) => Ok(()),
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

/// `entries { <convention> { .. } ... }`, the bodies of a lambda that
/// declares their conventions, in order.
fn parse_entries<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<Vec<(CallingConvention, trunk_ir::parser::raw::RawRegion<'a>)>> {
    use trunk_ir::parser::raw::{ident, raw_region, ws};
    use winnow::combinator::repeat;
    use winnow::prelude::*;

    "entries".parse_next(input)?;
    ws.parse_next(input)?;
    '{'.parse_next(input)?;
    ws.parse_next(input)?;
    let entries = repeat(
        1..,
        (
            ident.try_map(str::parse::<CallingConvention>),
            ws,
            raw_region,
            ws,
        )
            .map(|(convention, _, region, _)| (convention, region)),
    )
    .parse_next(input)?;
    '}'.parse_next(input)?;
    Ok(entries)
}

/// Parse a source lambda named `name`. With `entries`, its bodies are
/// written under the conventions the operation declares; otherwise its one
/// body follows the captures.
pub(crate) fn parse_lambda<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
    name: &'static str,
    entries: bool,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
    use trunk_ir::parser::raw::*;
    use winnow::combinator::{opt, separated};
    use winnow::prelude::*;

    if results.len() != 1 || sym_name.is_some() {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }

    ws.parse_next(input)?;
    let quantified: Option<Vec<u32>> = opt((
        '[',
        separated(
            1..,
            (ws, parse_binder_name, ws).map(|(_, binder, _)| binder),
            ',',
        ),
        ']',
    ))
    .parse_next(input)?
    .map(|(_, binders, _)| binders);
    ws.parse_next(input)?;
    let params = func_params.parse_next(input)?;
    let result = return_type.parse_next(input)?;
    let row = parse_row(input)?;
    let captures = parse_captures(input)?;
    let mut attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();
    if has_duplicate_convention_attr(&attributes)
        || attributes.iter().any(|(key, _)| {
            key == QUANTIFIED_ROWS_ATTR || (entries && key == tribute_cps::CONVENTIONS_ATTR)
        })
    {
        return Err(winnow::error::ErrMode::Backtrack(
            winnow::error::ContextError::new(),
        ));
    }
    if let Some(quantified) = quantified {
        attributes.push((
            QUANTIFIED_ROWS_ATTR.into(),
            RawAttribute::List(
                quantified
                    .into_iter()
                    .map(|binder| RawAttribute::Int(i128::from(binder)))
                    .collect(),
            ),
        ));
    }

    ws.parse_next(input)?;
    let mut regions = if entries {
        let (conventions, regions): (Vec<_>, Vec<_>) = parse_entries(input)?.into_iter().unzip();
        attributes.push((
            tribute_cps::CONVENTIONS_ATTR.into(),
            RawAttribute::List(
                conventions
                    .into_iter()
                    .map(|convention| RawAttribute::String(convention.keyword().to_owned()))
                    .collect(),
            ),
        ));
        regions
    } else {
        vec![raw_region.parse_next(input)?]
    };
    for region in &mut regions {
        if !region.blocks.is_empty() {
            let mut merged = params.clone();
            merged.append(&mut region.blocks[0].args);
            region.blocks[0].args = merged;
        }
    }

    Ok(RawOperation {
        results,
        name,
        sym_name,
        func_params: vec![],
        func_param_regions: vec![],
        operands: captures,
        attributes,
        result_types: vec![func_sig_raw_type(result, &params, row)],
        regions,
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Lambda {
    fn print(&self, h: &mut trunk_ir::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        print_lambda(h, self.op_ref(), "tribute_control.lambda", None)
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        parse_lambda(input, results, sym_name, "tribute_control.lambda", false)
    }
}
