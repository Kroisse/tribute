//! Custom assembly of `tribute_control.func` and `tribute_control.lambda`.

use super::*;
use fmt::Write as _;
use trunk_ir::printer::{
    AsmNamer, OpAsmPrinter, display_symbol_ref, display_symbol_text, entry_args,
};

// === Custom assembly: tribute_control.func ===

/// ` row(@A, @B | 'a0)`, or nothing for an empty row.
pub(super) fn display_row(row: &Row) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| {
        if row.is_empty() {
            return Ok(());
        }
        let instances = row.instances.iter().map(display_symbol_ref).format(", ");
        write!(f, " row({instances}")?;
        if let Some(tail) = row.tail {
            if !row.instances.is_empty() {
                f.write_str(" ")?;
            }
            write!(f, "| 'a{tail}")?;
        }
        f.write_str(")")
    })
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
}

/// The binder list `['a0, effect 'a1 = 'a0 | 'a2]`, or nothing without
/// binders.
impl fmt::Display for BinderHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.binders.is_empty() {
            return Ok(());
        }
        let binders = self
            .binders
            .iter()
            .enumerate()
            .format_with(", ", |(index, binder), f| {
                if *binder != RowBinder::Param {
                    f(&format_args!("{} ", binder.keyword()))?;
                }
                f(&format_args!("'a{index}"))?;
                let union = self
                    .unions
                    .iter()
                    .find(|union| usize::try_from(union.result).ok() == Some(index));
                if let Some(union) = union {
                    let sources = union
                        .sources
                        .iter()
                        .format_with(" | ", |source, f| f(&format_args!("'a{source}")));
                    f(&format_args!(" = {sources}"))?;
                }
                Ok(())
            });
        write!(f, "[{binders}]")
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

/// ` attributes {..}` holding the attributes other than `excluded`, or
/// nothing.
fn print_extra_attributes(p: &mut OpAsmPrinter<'_>, op: OpRef, excluded: &[&str]) -> fmt::Result {
    let s = p.state();
    let mut attrs = s
        .ctx()
        .op_attrs(op)
        .iter()
        .filter(|(key, _)| !excluded.contains(key))
        .peekable();
    if attrs.peek().is_none() {
        return Ok(());
    }
    let attrs = attrs.format_with(", ", |(key, value), f| {
        f(&format_args!("{key} = {}", s.attr(value)))
    });
    write!(p, " attributes {{{attrs}}}")
}

/// The parameters of a header, `(%a: A, ..)`: the entry block arguments of
/// `region`, or without one, `param_types` named by position.
fn print_signature_params(
    p: &mut OpAsmPrinter<'_>,
    region: Option<RegionRef>,
    param_types: &[TypeRef],
) -> fmt::Result {
    let s = p.state();
    match region {
        Some(region) => write!(p, "{}", s.entry_args(region)),
        None => {
            let params = param_types
                .iter()
                .enumerate()
                .format_with(", ", |(index, &ty), f| {
                    f(&format_args!("%arg{index}: {}", s.ty(ty)))
                });
            write!(p, "({params})")
        }
    }
}

/// Print a source function as `name`, with the convention the operation
/// declares, if any.
pub(crate) fn print_func(
    p: &mut OpAsmPrinter<'_>,
    op: OpRef,
    name: &str,
    convention: Option<CallingConvention>,
) -> fmt::Result {
    let s = p.state();
    let ctx = s.ctx();
    let symbol = ctx.op_attrs(op).get_str("sym_name");
    let callable_ty = ctx.op_attrs(op).get_type("type");
    let region = ctx.op_region(op, 0);
    let header = BinderHeader::of(ctx, op);
    let Some(header) = header.filter(|_| has_concise_func_sig(ctx, callable_ty)) else {
        return p.print_generic(op);
    };
    let parts = callable_ty.and_then(|ty| func_sig_parts(ctx, ty));

    match symbol {
        Some(symbol) => write!(p, "{name} {}{header}", display_symbol_text(symbol))?,
        None => write!(p, "{name} @<missing>{header}")?,
    }

    let param_types = parts
        .as_ref()
        .map(|(_, params)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(p, region, param_types)?;

    if let Some((result, _)) = parts {
        write!(p, " -> {}", s.ty(result))?;
        if let Some(ty) = callable_ty {
            write!(p, "{}", display_row(&FuncSig(ty).row(ctx)))?;
        }
        if header.inferred {
            write!(p, " inferred")?;
        }
        if let Some(convention) = convention {
            write!(p, " convention({convention})")?;
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
    print_extra_attributes(p, op, &excluded)?;

    match region {
        Some(region) => p.print_body(region),
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
        operand_types: Vec::new(),
        attributes,
        result_types: vec![],
        fixed_results: false,
        regions,
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Func {
    fn print(&self, p: &mut trunk_ir::printer::OpAsmPrinter<'_>) -> std::fmt::Result {
        print_func(p, self.op_ref(), "tribute_control.func", None)
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
    p: &mut OpAsmPrinter<'_>,
    op: OpRef,
    name: &str,
    entries: Option<&[CallingConvention]>,
) -> fmt::Result {
    let s = p.state();
    let ctx = s.ctx();
    let callable_ty = ctx.op_result_types(op).first().copied();
    let Some(quantified) = concise_lambda_rows(ctx, op) else {
        return p.print_generic(op);
    };

    let results = ctx.op_results(op);
    if !results.is_empty() {
        write!(p, "{} = ", s.values(results))?;
    }
    p.write_str(name)?;
    if !quantified.is_empty() {
        let binders = quantified
            .iter()
            .format_with(", ", |binder, f| f(&format_args!("'a{binder}")));
        write!(p, "[{binders}]")?;
    }

    let parts = callable_ty.and_then(|ty| func_sig_parts(ctx, ty));
    let region = ctx.op_region(op, 0);
    let param_types = parts
        .as_ref()
        .map(|(_, params)| params.as_slice())
        .unwrap_or(&[]);
    print_signature_params(p, region, param_types)?;
    if let Some((result, _)) = parts {
        write!(p, " -> {}", s.ty(result))?;
        if let Some(ty) = callable_ty {
            write!(p, "{}", display_row(&FuncSig(ty).row(ctx)))?;
        }
    }
    write!(p, " captures [{}]", s.values(ctx.op_operands(op)))?;
    let mut excluded = vec![CALLING_CONVENTION_ATTR, QUANTIFIED_ROWS_ATTR];
    if entries.is_some() {
        excluded.push(tribute_cps::CONVENTIONS_ATTR);
    }
    print_extra_attributes(p, op, &excluded)?;

    match (region, entries) {
        (Some(_), Some(entries)) => {
            // Every body takes the parameters the header names, which
            // `assign_lambda_names` gives them.
            write!(p, " entries")?;
            p.print_labelled_bodies(
                entries
                    .iter()
                    .map(|convention| convention.keyword())
                    .zip(ctx.op_regions(op)),
            )
        }
        (Some(region), None) => p.print_body(region),
        (None, _) => Ok(()),
    }
}

/// The quantified rows of a lambda its concise form writes, or `None` when
/// it prints generically.
fn concise_lambda_rows(ctx: &IrContext, op: OpRef) -> Option<Vec<u32>> {
    let result_types = ctx.op_result_types(op);
    // An empty list written out has no header form.
    let quantified = quantified_rows(ctx, op).ok().filter(|quantified| {
        !quantified.is_empty() || !ctx.op_attrs(op).contains_key(QUANTIFIED_ROWS_ATTR)
    });
    quantified.filter(|_| {
        result_types.len() == 1 && has_concise_func_sig(ctx, result_types.first().copied())
    })
}

/// Name a lambda as [`print_lambda`] writes it: with the conventions of its
/// bodies, every body shares the parameters the header names.
pub(crate) fn assign_lambda_names(namer: &mut AsmNamer<'_>, op: OpRef, entries: bool) {
    let ctx = namer.ctx();
    if !entries || concise_lambda_rows(ctx, op).is_none() {
        return namer.default_op(op);
    }
    namer.results(op);
    let bodies: Vec<_> = ctx.op_regions(op).collect();
    let Some((&first, rest)) = bodies.split_first() else {
        return;
    };
    let params = entry_args(ctx, first);
    namer.values(params);
    for &body in rest {
        namer.share(entry_args(ctx, body), params);
    }
    for body in bodies {
        namer.region_without_entry(body);
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
        operand_types: Vec::new(),
        attributes,
        result_types: vec![func_sig_raw_type(result, &params, row)],
        fixed_results: false,
        regions,
        successors: vec![],
    })
}

impl trunk_ir::asm_format::OpAsmFormat for Lambda {
    fn print(&self, p: &mut trunk_ir::printer::OpAsmPrinter<'_>) -> std::fmt::Result {
        print_lambda(p, self.op_ref(), "tribute_control.lambda", None)
    }

    fn assign_names(&self, namer: &mut AsmNamer<'_>) {
        assign_lambda_names(namer, self.op_ref(), false);
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<trunk_ir::parser::raw::RawOperation<'a>> {
        parse_lambda(input, results, sym_name, "tribute_control.lambda", false)
    }
}
