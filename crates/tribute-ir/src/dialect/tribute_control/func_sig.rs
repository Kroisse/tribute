//! The source callable type `tribute_control.func_sig`.

use super::*;

/// Why a name-matching source signature does not satisfy its storage contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum FuncSigTypeError {
    #[display("missing required `{_0}` u32 attribute")]
    MissingCount(#[error(not(source))] &'static str),
    #[display("`{_0}` must be a u32 attribute")]
    InvalidCount(#[error(not(source))] &'static str),
    #[display("input and result counts overflow u32")]
    CountOverflow,
    #[display("input and result counts must equal the parameter count")]
    LengthMismatch,
    #[display("a source signature has exactly one result, found {_0}")]
    ResultCount(#[error(not(source))] u32),
    /// A callable type has no convention; the operations that call or make
    /// a callable declare it (`design/tribute-dialects.md`, "Convention entry").
    #[display("a source callable type has no calling convention")]
    Convention,
    /// The ability row is malformed, or written out although closed and empty.
    #[display("malformed ability row")]
    Row,
}

/// Typed wrapper for an input-first, exactly-one-result source signature.
///
/// ```text
/// tribute_control.func_sig<(T) -> R>
/// tribute_control.func_sig<(T) -> R row(@"<instance A>", @"<instance B>" | 'a1)>
/// ```
///
/// The type stores its inputs and results flat, as `[inputs..., results...]`,
/// split by the required `u32` delimiters `num_inputs` and `num_results`; the
/// source-owned validated API requires exactly one result. `Params` and
/// `Result` are source-logical types. The type carries no calling convention
/// (`design/tribute-dialects.md`, "Convention entry").
///
/// Its ability row is the `ability_row` type attribute, stored as
/// `{ability_row = {instances = [@"<instance A>", @"<instance B>"], tail = 1}}`:
/// `instances` lists symbol references to `ability.decl` definitions, sorted
/// by symbol path with each instance once, and `tail` is the index of a row
/// binder of the enclosing definition. Either may be absent; with both absent
/// the row is closed and empty, and the attribute itself is omitted. The
/// function form writes the row as a `row(..)` clause after the result:
/// the instances, then `|` and the tail binder, so `row(| 'a1)` without
/// instances and `row(@A)` without a tail. The closed empty row is never
/// written. A tail inside a type alias has the same meaning as one written
/// inline. What the row means and where it may appear is in
/// `design/tribute-dialects.md`, "Ability row".
///
/// The verifier checks that both delimiters are present `u32`s whose sum is
/// the parameter count, that there is exactly one result, that no
/// `tribute.calling_convention` attribute is present, and that the stored row
/// is well formed and not written out when closed and empty. It does not
/// inspect the component types, since a type is interned once and shared by
/// every phase. The `tribute_control` validator checks them at the
/// source-logical boundary: no component contains an unresolved type or a
/// type implementing `PhysicalCallableType`
/// (`design/tribute-dialects.md`, "직접형 호출 대상과 제어").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSig(pub(super) TypeRef);

static FUNC_SIG_DEF: trunk_ir::TypeDef = trunk_ir::TypeDef::new("tribute_control.func_sig")
    .with_verifier::<FuncSig>()
    .with_asm_format::<FuncSig>();

trunk_ir::register_type_def!(&FUNC_SIG_DEF);

/// The function syntax `func.func_sig` also uses, with exactly one result
/// and the row as a clause after it, `tribute_control.func_sig<(A) -> R
/// row(@A | 'a0)>`. A signature with any other type attribute writes them
/// all after a comma, as `func.func_sig` does.
impl trunk_ir::asm_format::TypeAsmFormat for FuncSig {
    fn print(&self, p: &mut trunk_ir::printer::AsmPrinter<'_>) -> fmt::Result {
        use fmt::Write;

        let s = p.state();
        let ctx = s.ctx();
        if self.has_nonreserved_attrs(ctx) {
            return trunk_ir::dialect::func::print_signature_type(p, self.0);
        }
        let inputs = self
            .inputs(ctx)
            .iter()
            .map(|&input| s.ty(input))
            .format(", ");
        write!(
            p,
            "tribute_control.func_sig<({inputs}) -> {}{}>",
            s.ty(self.result(ctx)),
            display_row(&self.row(ctx))
        )
    }

    fn parse(
        p: &mut trunk_ir::parser::AsmParser<'_>,
        name: &str,
    ) -> trunk_ir::parser::ParseResult<Option<TypeRef>> {
        use trunk_ir::asm_format::{parse_dict_tail, parse_signature_results};

        if !p.peek().starts_with('(') {
            return Ok(None);
        }
        let inputs = p.parse_param_list()?;
        p.parse_punct("->")?;
        let results = parse_signature_results(p)?;
        let mut tail = parse_dict_tail(p)?;
        if let Some(row) = parse_row(p)? {
            if tail.view(p.ctx()).contains_key(ABILITY_ROW_ATTR) {
                return Err(p.expected("the row once, as either `row(..)` or `ability_row`"));
            }
            tail.view_mut(p.ctx_mut()).insert(ABILITY_ROW_ATTR, row);
        }
        p.parse_punct(">")?;
        trunk_ir::dialect::func::signature_from_parts(p, name, inputs, results, tail, Some(1))
            .map(Some)
    }
}

/// The function form `(A, B) -> R row(@State | 'a0)`: the type's own text
/// without its `tribute_control.func_sig<..>` wrapper. The row, when present,
/// is the whole tail; a signature with any other type attribute has no
/// function form.
impl trunk_ir::asm_format::SignatureCodec for FuncSig {
    fn decompose(ctx: &IrContext, ty: TypeRef) -> Option<trunk_ir::asm_format::SignatureParts<'_>> {
        Self::from_type_ref(ctx, ty)?;
        let parts = trunk_ir::dialect::func::signature_codec_parts(ctx, ty)?;
        parts
            .tail
            .iter()
            .all(|&(key, _)| ctx.str(key) == ABILITY_ROW_ATTR)
            .then_some(parts)
    }

    fn write_tail(
        p: &mut trunk_ir::printer::AsmPrinter<'_>,
        tail: &[(StringRef, &Attribute)],
    ) -> fmt::Result {
        let Some(&(_, Attribute::Dict(entries))) = tail.first() else {
            return Ok(());
        };
        let mut row = Row::default();
        for value in entries.values() {
            match value {
                Attribute::List(instances) => {
                    row.instances = instances
                        .iter()
                        .filter_map(|instance| match instance {
                            Attribute::SymbolRef(path) => Some(path.clone()),
                            _ => None,
                        })
                        .collect();
                }
                Attribute::Int(index) => row.tail = u32::try_from(*index).ok(),
                _ => {}
            }
        }
        fmt::Write::write_fmt(p, format_args!("{}", display_row(&row)))
    }

    fn parse_tail(
        p: &mut trunk_ir::parser::AsmParser<'_>,
    ) -> trunk_ir::parser::ParseResult<AttributeMap> {
        let mut tail = AttributeMap::new();
        if let Some(row) = parse_row(p)? {
            tail.view_mut(p.ctx_mut()).insert(ABILITY_ROW_ATTR, row);
        }
        Ok(tail)
    }

    fn build(
        p: &mut trunk_ir::parser::AsmParser<'_>,
        inputs: Vec<(TypeRef, AttributeMap)>,
        results: Vec<(TypeRef, AttributeMap)>,
        tail: AttributeMap,
    ) -> trunk_ir::parser::ParseResult<TypeRef> {
        if results.len() != 1 {
            return Err(p.expected("exactly one result"));
        }
        trunk_ir::dialect::func::signature_from_parts(
            p,
            "tribute_control.func_sig",
            inputs,
            results,
            tail,
            Some(1),
        )
    }
}

impl DialectType for FuncSig {
    const DEF: &'static trunk_ir::TypeDef = &FUNC_SIG_DEF;

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        Self::matches(ctx, ty)
            .then(|| <Self as trunk_ir::type_verifier::VerifyType>::verify(ctx, ty).ok())
            .flatten()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl trunk_ir::type_constraint::TypeConstraint for FuncSig {
    const DESC: &'static trunk_ir::type_constraint::ConstraintDesc =
        &trunk_ir::type_constraint::ConstraintDesc {
            name: "tribute_control.func_sig",
            exact: true,
            projections: &[
                trunk_ir::type_constraint::ProjectionDesc {
                    name: "Inputs",
                    kind: trunk_ir::type_constraint::ProjectionKind::List,
                },
                trunk_ir::type_constraint::ProjectionDesc {
                    name: "Result",
                    kind: trunk_ir::type_constraint::ProjectionKind::One,
                },
            ],
            matches: |ctx, ty| FuncSig::from_type_ref(ctx, ty).is_some(),
            project: |ctx, ty, index| {
                let signature = FuncSig::from_type_ref(ctx, ty)?;
                match index {
                    0 => Some(trunk_ir::type_constraint::Projected::List(
                        signature.inputs(ctx),
                    )),
                    1 => Some(trunk_ir::type_constraint::Projected::One(
                        signature.result(ctx),
                    )),
                    _ => None,
                }
            },
            fixed: None,
        };
}

impl From<FuncSig> for TypeRef {
    fn from(value: FuncSig) -> Self {
        value.0
    }
}

impl trunk_ir::type_verifier::VerifyType for FuncSig {
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, trunk_ir::ops::VerificationError> {
        let data = ctx.get_type(ty);
        let inputs = read_count(ctx, &data.attrs, trunk_ir::dialect::func::NUM_INPUTS_ATTR)?;
        let results = read_count(ctx, &data.attrs, trunk_ir::dialect::func::NUM_RESULTS_ATTR)?;
        if results != 1 {
            return Err(FuncSigTypeError::ResultCount(results).into());
        }
        let total = inputs
            .checked_add(results)
            .ok_or(FuncSigTypeError::CountOverflow)?;
        if usize::try_from(total).ok() != Some(data.params.len()) {
            return Err(FuncSigTypeError::LengthMismatch.into());
        }
        if data.attrs.view(ctx).contains_key(CALLING_CONVENTION_ATTR) {
            return Err(FuncSigTypeError::Convention.into());
        }
        if let Some(row) = data.attrs.view(ctx).get(ABILITY_ROW_ATTR)
            && Row::from_attribute(ctx, row).is_ok_and(|row| row.is_empty())
        {
            return Err(FuncSigTypeError::Row.into());
        }
        row::stored_row(ctx, data)
            .expect("a func_sig stores a row")
            .map_err(|_| FuncSigTypeError::Row)?;
        Ok(Self(ty))
    }
}

impl FuncSig {
    pub fn as_type_ref(self) -> TypeRef {
        self.0
    }

    pub fn result(self, ctx: &IrContext) -> TypeRef {
        self.results(ctx)[0]
    }

    pub fn inputs(self, ctx: &IrContext) -> &[TypeRef] {
        let count = read_count(
            ctx,
            &ctx.get_type(self.0).attrs,
            trunk_ir::dialect::func::NUM_INPUTS_ATTR,
        )
        .expect("validated source func_sig retains input count") as usize;
        &ctx.get_type(self.0).params[..count]
    }

    pub fn results(self, ctx: &IrContext) -> &[TypeRef] {
        let inputs = self.inputs(ctx).len();
        &ctx.get_type(self.0).params[inputs..]
    }

    /// Each input paired with its own attributes.
    pub fn inputs_with_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (TypeRef, &AttributeMap)> {
        let count = self.inputs(ctx).len();
        ctx.get_type(self.0).params_with_attrs(ctx).take(count)
    }

    /// The result paired with its own attributes.
    pub fn result_with_attrs(self, ctx: &IrContext) -> (TypeRef, &AttributeMap) {
        let index = self.inputs(ctx).len();
        let data = ctx.get_type(self.0);
        (data.params[index], data.param_attrs(ctx, index))
    }

    /// The callable's ability row; closed and empty when the type omits it.
    pub fn row(self, ctx: &IrContext) -> Row {
        row::stored_row(ctx, ctx.get_type(self.0))
            .expect("a func_sig stores a row")
            .expect("validated source func_sig retains a well-formed row")
    }

    /// Remove the storage delimiters and the source-only row.
    ///
    /// Consumers rebuilding an equivalent signature in another dialect retain
    /// every other attribute through their owning typed constructor.
    pub fn remove_reserved_attrs(ctx: &mut IrContext, attrs: &mut AttributeMap) {
        let mut attrs = attrs.view_mut(ctx);
        for key in RESERVED_ATTRS {
            attrs.remove(key);
        }
    }

    /// Whether this signature carries type-owned metadata beyond storage and
    /// row fields.
    pub(crate) fn has_nonreserved_attrs(self, ctx: &IrContext) -> bool {
        ctx.type_attrs(self.0)
            .iter()
            .any(|(key, _)| !RESERVED_ATTRS.contains(&key))
    }
}

/// The storage delimiters and source-only row metadata.
pub(super) const RESERVED_ATTRS: [&str; 3] = [
    trunk_ir::dialect::func::NUM_INPUTS_ATTR,
    trunk_ir::dialect::func::NUM_RESULTS_ATTR,
    ABILITY_ROW_ATTR,
];

fn read_count(
    ctx: &IrContext,
    attrs: &AttributeMap,
    key: &'static str,
) -> Result<u32, FuncSigTypeError> {
    match attrs.view(ctx).get(key) {
        Some(Attribute::Int(value)) => {
            u32::try_from(*value).map_err(|_| FuncSigTypeError::InvalidCount(key))
        }
        Some(_) => Err(FuncSigTypeError::InvalidCount(key)),
        None => Err(FuncSigTypeError::MissingCount(key)),
    }
}

impl FuncSig {
    /// Start a source signature with `inputs` and `result`: no row and no
    /// other attributes until the builder sets them.
    pub fn params(inputs: impl IntoIterator<Item = TypeRef>, result: TypeRef) -> FuncSigBuilder {
        FuncSigBuilder {
            inputs: inputs.into_iter().collect(),
            result,
            row: Row::default(),
            attrs: AttributeMap::new(),
        }
    }
}

/// Builds a [`FuncSig`]; start one with [`FuncSig::params`].
#[must_use]
pub struct FuncSigBuilder {
    inputs: Vec<TypeRef>,
    result: TypeRef,
    row: Row,
    attrs: AttributeMap,
}

impl FuncSigBuilder {
    /// The ability row; an empty row is the closed pure row and is not stored.
    pub fn row(mut self, row: Row) -> Self {
        self.row = row;
        self
    }

    /// Metadata other than the reserved delimiters and row.
    pub fn attrs(mut self, attrs: AttributeMap) -> Self {
        self.attrs = attrs;
        self
    }

    pub fn intern(self, ctx: &mut IrContext) -> FuncSig {
        assert!(
            self.attrs
                .iter()
                .all(|(key, _)| !RESERVED_ATTRS.contains(&ctx.str(key))),
            "source func_sig reserves delimiters and row metadata"
        );
        let count =
            u32::try_from(self.inputs.len()).expect("source func_sig input count exceeds u32");
        let mut builder = TypeDataBuilder::new("tribute_control.func_sig")
            .params(self.inputs)
            .param(self.result);
        for (key, value) in self.attrs {
            builder = builder.attr(key, value);
        }
        builder = builder
            .attr(
                trunk_ir::dialect::func::NUM_INPUTS_ATTR,
                Attribute::from(count),
            )
            .attr(
                trunk_ir::dialect::func::NUM_RESULTS_ATTR,
                Attribute::from(1_u32),
            );
        if !self.row.is_empty() {
            let row = self.row.to_attribute(ctx);
            builder = builder.attr(ABILITY_ROW_ATTR, row);
        }
        let ty = builder.intern(ctx);
        FuncSig::from_type_ref(ctx, ty)
            .expect("source func_sig constructor must produce a valid type")
    }
}

#[cfg(test)]
mod signature_codec_tests {
    use super::*;
    use trunk_ir::asm_format::{
        SignatureCodec, display_signature_param, display_signature_results, parse_signature_results,
    };
    use trunk_ir::parser::AsmParser;
    use trunk_ir::printer::{AsmPrinter, AsmState};

    fn function_form(ctx: &IrContext, ty: TypeRef) -> Option<String> {
        use fmt::Write;
        let parts = FuncSig::decompose(ctx, ty)?;
        let state = AsmState::without_aliases(ctx);
        let mut out = String::new();
        let mut p = AsmPrinter::new(&state, &mut out);
        let inputs = parts
            .inputs
            .iter()
            .map(|&(input, attrs)| display_signature_param(&state, input, attrs))
            .format(", ");
        write!(
            p,
            "({inputs}) -> {}",
            display_signature_results(&state, &parts.results)
        )
        .ok()?;
        FuncSig::write_tail(&mut p, &parts.tail).ok()?;
        Some(out)
    }

    fn parse_function_form(ctx: &mut IrContext, text: &str) -> TypeRef {
        let mut p = AsmParser::new(ctx, text);
        let inputs = p.parse_param_list().unwrap();
        p.parse_punct("->").unwrap();
        let results = parse_signature_results(&mut p).unwrap();
        let tail = FuncSig::parse_tail(&mut p).unwrap();
        assert!(p.at_end(), "unparsed rest of `{text}`");
        FuncSig::build(&mut p, inputs, results, tail).unwrap()
    }

    #[test]
    fn function_form_round_trips() {
        let mut ctx = IrContext::new();
        let i32_ty = core::I32::type_ref(&mut ctx);
        let frame = TypeDataBuilder::new("test.f").intern(&mut ctx);
        let state = SymbolPath::new(["State"]);
        let cases = [
            (
                FuncSig::params([i32_ty], i32_ty).intern(&mut ctx),
                "(core.i32) -> core.i32",
            ),
            (
                FuncSig::params([], i32_ty)
                    .row(Row::new([], Some(0)))
                    .intern(&mut ctx),
                "() -> core.i32 row(| 'a0)",
            ),
            (
                FuncSig::params([frame, i32_ty], i32_ty)
                    .row(Row::new([state.clone()], Some(1)))
                    .intern(&mut ctx),
                "(test.f, core.i32) -> core.i32 row(@State | 'a1)",
            ),
            (
                FuncSig::params([i32_ty], frame)
                    .row(Row::new([state, SymbolPath::new(["Console"])], None))
                    .intern(&mut ctx),
                "(core.i32) -> test.f row(@Console, @State)",
            ),
        ];
        for (sig, expected) in cases {
            let ty = sig.as_type_ref();
            let text = function_form(&ctx, ty).expect("a function form");
            assert_eq!(text, expected);
            assert_eq!(parse_function_form(&mut ctx, &text), ty, "{text}");
        }
    }

    #[test]
    fn other_attributes_and_result_counts_have_no_function_form() {
        let mut ctx = IrContext::new();
        let i32_ty = core::I32::type_ref(&mut ctx);
        let mut attrs = AttributeMap::new();
        attrs
            .view_mut(&mut ctx)
            .insert("metadata", Attribute::Int(1));
        let sig = FuncSig::params([i32_ty], i32_ty)
            .attrs(attrs)
            .intern(&mut ctx);
        assert!(FuncSig::decompose(&ctx, sig.as_type_ref()).is_none());

        let mut p = AsmParser::new(&mut ctx, "(core.i32, core.i32)");
        let params = p.parse_param_list().unwrap();
        assert!(FuncSig::build(&mut p, Vec::new(), Vec::new(), AttributeMap::new()).is_err());
        assert!(FuncSig::build(&mut p, Vec::new(), params, AttributeMap::new()).is_err());
    }
}
