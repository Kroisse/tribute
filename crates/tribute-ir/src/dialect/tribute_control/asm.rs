//! Custom assembly of `tribute_control.func` and `tribute_control.lambda`.

use super::*;
use fmt::Write as _;
use trunk_ir::parser::{Argument, AsmParser, OpAsmParser, OperationState, ParseResult};
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
        write!(f, " {}", display_row_argument(row))
    })
}

/// `row(@A, @B | 'a0)`, written in full even for the empty row `row()`, as
/// a nominal row argument is.
pub(crate) fn display_row_argument(row: &Row) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| {
        let instances = row.instances.iter().map(display_symbol_ref).format(", ");
        write!(f, "row({instances}")?;
        if let Some(tail) = row.tail {
            if !row.instances.is_empty() {
                f.write_str(" ")?;
            }
            write!(f, "| 'a{tail}")?;
        }
        f.write_str(")")
    })
}

/// A list of row binders `['a0, 'a1]` with no kinds or unions, as a nominal
/// layout declares them; nothing for no binders.
pub(crate) fn display_layout_binders(count: u32) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        if count == 0 {
            return Ok(());
        }
        let binders = (0..count).format_with(", ", |index, f| f(&format_args!("'a{index}")));
        write!(f, "[{binders}]")
    })
}

/// The binder list of [`display_layout_binders`], as its count; zero when
/// the text has no list.
pub(crate) fn parse_layout_binders(p: &mut AsmParser<'_>) -> ParseResult<u32> {
    if !p.parse_optional_punct("[") {
        return Ok(0);
    }
    let mut count = 0;
    loop {
        p.peek();
        let offset = p.offset();
        let index = p.parse_token("a row binder `'aN`", parse_binder_name)?;
        if index != count {
            return Err(p.error_at(offset, "expected row binders numbered in order"));
        }
        count += 1;
        if !p.parse_optional_punct(",") {
            break;
        }
    }
    p.parse_punct("]")?;
    Ok(count)
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

/// A definition header's `[...]` binder list, as the `row_binders` and
/// `row_unions` attributes; nothing when the header has none.
fn parse_binders(p: &mut AsmParser<'_>) -> ParseResult<Vec<(&'static str, Attribute)>> {
    if !p.parse_optional_punct("[") {
        return Ok(Vec::new());
    }
    let mut kinds = Vec::new();
    let mut unions = Vec::new();
    loop {
        p.peek();
        let offset = p.offset();
        let kind = match p.parse_optional_identifier() {
            None => RowBinder::Param,
            Some(word) => match RowBinder::from_keyword(word) {
                Some(kind) if kind != RowBinder::Param => kind,
                _ => return Err(p.error_at(offset, "expected a row binder kind")),
            },
        };
        p.peek();
        let offset = p.offset();
        let index = p.parse_token("a row binder `'aN`", parse_binder_name)?;
        if usize::try_from(index).ok() != Some(kinds.len()) {
            return Err(p.error_at(offset, "expected row binders numbered in order"));
        }
        kinds.push(p.ctx_mut().string_attr(kind.keyword()));
        if p.parse_optional_punct("=") {
            let mut sources = vec![p.parse_token("a row binder `'aN`", parse_binder_name)?];
            while p.parse_optional_punct("|") {
                sources.push(p.parse_token("a row binder `'aN`", parse_binder_name)?);
            }
            let sources = sources
                .into_iter()
                .map(|source| Attribute::Int(i128::from(source)))
                .collect();
            unions.push(Attribute::Dict(AttributeMap::from_entries(
                p.ctx_mut(),
                [
                    ("result", Attribute::Int(i128::from(index))),
                    ("sources", Attribute::List(sources)),
                ],
            )));
        }
        if !p.parse_optional_punct(",") {
            break;
        }
    }
    p.parse_punct("]")?;
    let mut attributes = vec![(ROW_BINDERS_ATTR, Attribute::List(kinds))];
    if !unions.is_empty() {
        attributes.push((ROW_UNIONS_ATTR, Attribute::List(unions)));
    }
    Ok(attributes)
}

/// The `inferred` keyword before a definition's convention.
fn parse_inferred(p: &mut AsmParser<'_>) -> ParseResult<bool> {
    use winnow::prelude::*;

    let inferred = p.parse_optional_token("`inferred`", |input| {
        (
            "inferred",
            winnow::combinator::peek(winnow::ascii::multispace1),
        )
            .parse_next(input)
    })?;
    Ok(inferred.is_some())
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
fn parse_convention(p: &mut AsmParser<'_>) -> ParseResult<Option<CallingConvention>> {
    if !p.parse_optional_keyword("convention") {
        return Ok(None);
    }
    p.parse_punct("(")?;
    p.peek();
    let offset = p.offset();
    let keyword = p.parse_identifier()?;
    let convention = keyword
        .parse::<CallingConvention>()
        .map_err(|_| p.error_at(offset, "expected a calling convention"))?;
    p.parse_punct(")")?;
    Ok(Some(convention))
}

/// The concise row of a source signature, `row(@A, @B | 'a1)`, as the
/// `ability_row` attribute; `None` when the signature writes no row.
pub(super) fn parse_row(p: &mut AsmParser<'_>) -> ParseResult<Option<Attribute>> {
    if !p.parse_optional_punct("row(") {
        return Ok(None);
    }
    let row = parse_row_entries(p)?;
    if row.is_empty() {
        return Err(p.expected("a row with abilities or a tail"));
    }
    Ok(Some(row.to_attribute(p.ctx_mut())))
}

/// A nominal row argument, `row(@A | 'a0)` or the empty `row()`.
pub(crate) fn parse_row_argument(p: &mut AsmParser<'_>) -> ParseResult<Row> {
    p.parse_punct("row(")?;
    parse_row_entries(p)
}

/// The entries of a row after its `row(`, through its `)`.
fn parse_row_entries(p: &mut AsmParser<'_>) -> ParseResult<Row> {
    let mut instances = Vec::new();
    if p.peek().starts_with('@') {
        loop {
            instances.push(p.parse_symbol_ref()?);
            if !p.parse_optional_punct(",") {
                break;
            }
        }
    }
    let tail = if p.parse_optional_punct("|") {
        Some(p.parse_token("a row variable `'aN`", parse_binder_name)?)
    } else {
        None
    };
    p.parse_punct(")")?;
    // Kept in written order, so the verifier rejects an unsorted row.
    Ok(Row { instances, tail })
}

/// The source signature `(params) -> result` with `row`.
fn func_sig_type(
    p: &mut AsmParser<'_>,
    result: TypeRef,
    params: &[Argument],
    row: Option<Attribute>,
) -> ParseResult<TypeRef> {
    let mut attrs = AttributeMap::new();
    if let Some(row) = row {
        attrs.view_mut(p.ctx_mut()).insert(ABILITY_ROW_ATTR, row);
    }
    trunk_ir::dialect::func::build_signature_type(
        p,
        "tribute_control.func_sig",
        params
            .iter()
            .map(|param| (param.ty, AttributeMap::new()))
            .collect(),
        vec![(result, AttributeMap::new())],
        attrs,
    )
}

/// Whether an operation's attributes hold the row its signature type owns,
/// or the physical convention attribute.
fn has_duplicate_convention_attr(ctx: &IrContext, attrs: &AttributeMap) -> bool {
    let attrs = attrs.view(ctx);
    attrs.contains_key(CALLING_CONVENTION_ATTR) || attrs.contains_key(ABILITY_ROW_ATTR)
}

/// ` attributes {..}`, or nothing.
fn parse_extra_attributes(p: &mut AsmParser<'_>) -> ParseResult<AttributeMap> {
    if p.parse_optional_keyword("attributes") {
        p.parse_attr_dict()
    } else {
        Ok(AttributeMap::new())
    }
}

/// Parse a source function. With `on_operation`, the operation declares the
/// convention, which it then requires; otherwise it has none.
pub(crate) fn parse_func(
    p: &mut OpAsmParser<'_>,
    state: &mut OperationState,
    on_operation: bool,
) -> ParseResult<()> {
    if p.num_results() != 0 {
        return Err(p.expected("a function without results"));
    }
    let sym_name = p.parse_symbol_name()?;
    let binders = parse_binders(p)?;
    let params = p.parse_entry_args()?;
    p.parse_punct("->")?;
    let result = p.parse_type()?;
    let row = parse_row(p)?;
    let inferred = parse_inferred(p)?;
    let convention = parse_convention(p)?;
    if on_operation != convention.is_some() {
        return Err(p.expected(if on_operation {
            "`convention(..)`"
        } else {
            "no `convention(..)`"
        }));
    }
    let attributes = parse_extra_attributes(p)?;
    let view = attributes.view(p.ctx());
    if has_duplicate_convention_attr(p.ctx(), &attributes)
        || [
            "sym_name",
            "type",
            ROW_BINDERS_ATTR,
            ROW_UNIONS_ATTR,
            INFERRED_ROW_ATTR,
        ]
        .into_iter()
        .any(|key| view.contains_key(key))
        || (on_operation && view.contains_key(tribute_cps::CONVENTION_ATTR))
    {
        return Err(p.expected("attributes without the ones the header writes"));
    }

    if p.peek().starts_with('{') {
        state.regions.push(p.parse_region_with_args(&params)?);
    }

    let signature = func_sig_type(p, result, &params, row)?;
    let ctx = p.ctx_mut();
    state.add_attributes(ctx, attributes);
    for (key, value) in binders {
        state.add_attribute(ctx, key, value);
    }
    if let Some(convention) = convention {
        let convention = ctx.string_attr(convention.keyword());
        state.add_attribute(ctx, tribute_cps::CONVENTION_ATTR, convention);
    }
    if inferred {
        state.add_attribute(ctx, INFERRED_ROW_ATTR, Attribute::Bool(true));
    }
    state.add_attribute(ctx, "type", Attribute::Type(signature));
    let sym_name = ctx.string_attr(&sym_name);
    state.add_attribute(ctx, "sym_name", sym_name);
    Ok(())
}

impl trunk_ir::asm_format::OpAsmFormat for Func {
    fn print(&self, p: &mut trunk_ir::printer::OpAsmPrinter<'_>) -> std::fmt::Result {
        print_func(p, self.op_ref(), "tribute_control.func", None)
    }

    fn parse(p: &mut OpAsmParser<'_>, state: &mut OperationState) -> ParseResult<()> {
        parse_func(p, state, false)
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

/// `entries { <convention> { .. } ... }`, the bodies of a lambda that
/// declares their conventions, in order, each taking `params`.
fn parse_entries(
    p: &mut OpAsmParser<'_>,
    params: &[Argument],
) -> ParseResult<Vec<(CallingConvention, RegionRef)>> {
    use trunk_ir::parser::lexer::ident;
    use winnow::prelude::*;

    p.parse_keyword("entries")?;
    p.parse_punct("{")?;
    let mut entries = Vec::new();
    while let Some(convention) = p.parse_optional_token("a calling convention", |input| {
        ident
            .try_map(str::parse::<CallingConvention>)
            .parse_next(input)
    })? {
        entries.push((convention, p.parse_region_with_args(params)?));
    }
    if entries.is_empty() {
        return Err(p.expected("an entry `<convention> { .. }`"));
    }
    p.parse_punct("}")?;
    Ok(entries)
}

/// Parse a source lambda. With `entries`, its bodies are written under the
/// conventions the operation declares; otherwise its one body follows the
/// captures.
pub(crate) fn parse_lambda(
    p: &mut OpAsmParser<'_>,
    state: &mut OperationState,
    entries: bool,
) -> ParseResult<()> {
    if p.num_results() != 1 {
        return Err(p.expected("a lambda with one result"));
    }
    let quantified = if p.parse_optional_punct("[") {
        let mut binders = vec![p.parse_token("a row binder `'aN`", parse_binder_name)?];
        while p.parse_optional_punct(",") {
            binders.push(p.parse_token("a row binder `'aN`", parse_binder_name)?);
        }
        p.parse_punct("]")?;
        Some(binders)
    } else {
        None
    };
    let params = p.parse_entry_args()?;
    p.parse_punct("->")?;
    let result = p.parse_type()?;
    let row = parse_row(p)?;
    p.parse_keyword("captures")?;
    let captures = p.parse_delimited("[", "]", |p| p.parse_operand())?;
    let attributes = parse_extra_attributes(p)?;
    let view = attributes.view(p.ctx());
    if has_duplicate_convention_attr(p.ctx(), &attributes)
        || view.contains_key(QUANTIFIED_ROWS_ATTR)
        || (entries && view.contains_key(tribute_cps::CONVENTIONS_ATTR))
    {
        return Err(p.expected("attributes without the ones the header writes"));
    }

    if entries {
        let (conventions, regions): (Vec<_>, Vec<_>) =
            parse_entries(p, &params)?.into_iter().unzip();
        let conventions = conventions
            .into_iter()
            .map(|convention| p.ctx_mut().string_attr(convention.keyword()))
            .collect();
        state.add_attribute(
            p.ctx_mut(),
            tribute_cps::CONVENTIONS_ATTR,
            Attribute::List(conventions),
        );
        state.regions = regions;
    } else {
        state.regions = vec![p.parse_region_with_args(&params)?];
    }

    let signature = func_sig_type(p, result, &params, row)?;
    state.add_attributes(p.ctx_mut(), attributes);
    if let Some(quantified) = quantified {
        let quantified = quantified
            .into_iter()
            .map(|binder| Attribute::Int(i128::from(binder)))
            .collect();
        state.add_attribute(
            p.ctx_mut(),
            QUANTIFIED_ROWS_ATTR,
            Attribute::List(quantified),
        );
    }
    state.operands = captures;
    state.result_types = vec![signature];
    Ok(())
}

impl trunk_ir::asm_format::OpAsmFormat for Lambda {
    fn print(&self, p: &mut trunk_ir::printer::OpAsmPrinter<'_>) -> std::fmt::Result {
        print_lambda(p, self.op_ref(), "tribute_control.lambda", None)
    }

    fn assign_names(&self, namer: &mut AsmNamer<'_>) {
        assign_lambda_names(namer, self.op_ref(), false);
    }

    fn parse(p: &mut OpAsmParser<'_>, state: &mut OperationState) -> ParseResult<()> {
        parse_lambda(p, state, false)
    }
}
