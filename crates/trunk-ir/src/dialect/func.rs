//! Arena-based func dialect.

use crate::attr_kind::SymbolRef;
use crate::attr_kind::Type;
use crate::op_flag::{IsolatedFromAbove, Pure};
use crate::op_interface::{
    self, CallableExit, CallableOwner, ControlFlowInterfaceError, IndirectCallLike,
};
use crate::ops::{DialectOp, DialectType};
use crate::parser::raw::{RawAttrDict, RawAttribute, RawParam, RawType};
use crate::{Attribute, AttributeMap, IrContext, SymbolPath, TypeDataBuilder, TypeRef};

/// The exact callable signature retained after a typed indirect callee becomes
/// a runtime function or table index.
const INDIRECT_CALL_SIGNATURE_ATTR: &str = "signature";

/// Shared accessors for function calls with ordinary results or arguments.
///
/// This is a static Rust interface over the four `func.*call*` operations;
/// it does not participate in dynamic operation dispatch.
pub trait CallLike: crate::ops::DialectOp {
    /// Arguments supplied to the callable, excluding an indirect callee.
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef];

    /// Result types produced by this call.
    fn call_result_types<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::TypeRef] {
        ctx.op_result_types(self.op_ref())
    }
}

/// Marker and resultless query for proper-tail transfers.
///
/// Exact callable signatures belong only to `IndirectCallLike`, keeping
/// CPS/tail legality independent from indirect-call metadata.
pub trait TailCallLike: CallLike {
    /// Proper tail transfers cannot produce SSA results.
    fn is_resultless(&self, ctx: &crate::IrContext) -> bool {
        self.call_result_types(ctx).is_empty()
    }
}

#[trunk_ir::dialect]
mod func {
    #[asm_format]
    #[implements(CallableOwner, op_interface::SymbolDefinition)]
    #[flags(IsolatedFromAbove)]
    fn func(sym_name: Attr<String>, r#type: Attr<Type>) {
        #[region(body?)]
        {}
    }

    #[implements(op_interface::CallLike)]
    fn call(callee: Attr<SymbolRef>, args: Variadic<_>) -> Variadic<_> {}

    #[verify]
    #[implements(IndirectCallLike)]
    fn call_indirect<S: FuncSig>(
        signature: Attr<TypeOf<S>>,
        callee: Value<_>,
        args: Values<S::Inputs>,
    ) -> Values<S::Results> {
    }

    #[implements(CallableExit, op_interface::CallLike)]
    fn tail_call(callee: Attr<SymbolRef>, args: Variadic<_>) {}

    #[verify]
    #[implements(CallableExit, IndirectCallLike)]
    fn tail_call_indirect<S: FuncSig>(
        signature: Attr<TypeOf<S>>,
        callee: Value<_>,
        args: Values<S::Inputs>,
    ) {
    }

    #[implements(CallableExit)]
    fn r#return(values: Variadic<_>) {}

    #[flags(Pure)]
    fn constant(func_ref: Attr<SymbolRef>) -> Value<_> {}

    #[implements(CallableExit)]
    fn unreachable() {}
}

/// Signature attribute recording the machine calling convention.
pub const CALL_CONV_ATTR: &str = "call_conv";

/// Machine calling convention of a physical callable signature.
///
/// The convention belongs to the signature, so a definition, the direct
/// callees it declares, and every indirect call read it from the same type.
/// Targets without distinct machine conventions ignore it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CallConv {
    /// The platform's default convention; the attribute is absent.
    #[default]
    Platform,
    /// The convention that supports guaranteed proper tail transfers.
    Tail,
}

impl CallConv {
    fn name(self) -> Option<&'static str> {
        match self {
            Self::Platform => None,
            Self::Tail => Some("tail"),
        }
    }

    /// Read the convention from signature attributes.
    ///
    /// Returns `None` for a malformed `call_conv` value.
    pub fn from_attrs(ctx: &IrContext, attrs: &AttributeMap) -> Option<Self> {
        match attrs.view(ctx).get(CALL_CONV_ATTR) {
            None => Some(Self::Platform),
            Some(Attribute::String(name)) if ctx.str(*name) == "tail" => Some(Self::Tail),
            Some(_) => None,
        }
    }

    /// Write the convention into signature attributes.
    pub fn set_in(self, ctx: &mut IrContext, attrs: &mut AttributeMap) {
        match self.name() {
            Some(name) => {
                let name = ctx.string_attr(name);
                attrs.view_mut(ctx).insert(CALL_CONV_ATTR, name);
            }
            None => {
                attrs.view_mut(ctx).remove(CALL_CONV_ATTR);
            }
        }
    }
}

/// Reserved delimiter attribute for the number of function inputs.
pub const NUM_INPUTS_ATTR: &str = "num_inputs";

/// Reserved delimiter attribute for the number of function results.
pub const NUM_RESULTS_ATTR: &str = "num_results";

/// Return the interned name of the `func.func_sig` type.
/// Why a name-matching `func.func_sig` does not satisfy its storage invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuncSigTypeError {
    MissingCount(&'static str),
    InvalidCount(&'static str),
    CountOverflow,
    LengthMismatch {
        num_inputs: u32,
        num_results: u32,
        params: usize,
    },
    UnsupportedResultCount(u32),
}

impl std::fmt::Display for FuncSigTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCount(name) => write!(f, "missing required `{name}` u32 attribute"),
            Self::InvalidCount(name) => write!(f, "`{name}` must be a u32 attribute"),
            Self::CountOverflow => write!(f, "input and result counts overflow u32"),
            Self::LengthMismatch {
                num_inputs,
                num_results,
                params,
            } => write!(
                f,
                "num_inputs ({num_inputs}) + num_results ({num_results}) must equal params length ({params})"
            ),
            Self::UnsupportedResultCount(count) => {
                write!(f, "currently supports at most one result, found {count}")
            }
        }
    }
}

impl std::error::Error for FuncSigTypeError {}

/// Validated wrapper for an input-first, zero-or-one-result `func.func_sig` type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSig(TypeRef);

crate::impl_func_sig_constraint!(FuncSig, "func.func_sig");

impl crate::type_verifier::VerifyType for FuncSig {
    /// Validate a name-matching `func.func_sig`, including both delimiter counts.
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, crate::ops::VerificationError> {
        let data = ctx.get_type(ty);
        debug_assert!(data.name == <Self as DialectType>::NAME);

        let num_inputs = read_count(ctx, &data.attrs, NUM_INPUTS_ATTR)?;
        let num_results = read_count(ctx, &data.attrs, NUM_RESULTS_ATTR)?;
        if num_results > 1 {
            return Err(FuncSigTypeError::UnsupportedResultCount(num_results).into());
        }
        let total = num_inputs
            .checked_add(num_results)
            .ok_or(FuncSigTypeError::CountOverflow)?;
        if usize::try_from(total).ok() != Some(data.params.len()) {
            return Err(FuncSigTypeError::LengthMismatch {
                num_inputs,
                num_results,
                params: data.params.len(),
            }
            .into());
        }
        Ok(Self(ty))
    }
}

impl FuncSig {
    fn counts(self, ctx: &IrContext) -> (usize, usize) {
        let data = ctx.get_type(self.0);
        let num_inputs = read_count(ctx, &data.attrs, NUM_INPUTS_ATTR)
            .expect("validated func.func_sig must retain a valid num_inputs attribute");
        let num_results = read_count(ctx, &data.attrs, NUM_RESULTS_ATTR)
            .expect("validated func.func_sig must retain a valid num_results attribute");
        (
            usize::try_from(num_inputs).expect("u32 must fit usize"),
            usize::try_from(num_results).expect("u32 must fit usize"),
        )
    }

    pub fn as_type_ref(&self) -> TypeRef {
        self.0
    }

    pub fn inputs<'a>(&self, ctx: &'a IrContext) -> &'a [TypeRef] {
        let (num_inputs, _) = self.counts(ctx);
        &ctx.get_type(self.0).params[..num_inputs]
    }

    pub fn results<'a>(&self, ctx: &'a IrContext) -> &'a [TypeRef] {
        let (num_inputs, num_results) = self.counts(ctx);
        &ctx.get_type(self.0).params[num_inputs..num_inputs + num_results]
    }

    /// The attributes of each input, in order; empty for an input without any.
    pub fn input_attrs<'a>(&self, ctx: &'a IrContext) -> impl Iterator<Item = &'a AttributeMap> {
        let (num_inputs, _) = self.counts(ctx);
        let data = ctx.get_type(self.0);
        (0..num_inputs).map(move |index| data.param_attrs(ctx, index))
    }

    /// The attributes of each result, in order; empty for a result without any.
    pub fn result_attrs<'a>(&self, ctx: &'a IrContext) -> impl Iterator<Item = &'a AttributeMap> {
        let (num_inputs, num_results) = self.counts(ctx);
        let data = ctx.get_type(self.0);
        (num_inputs..num_inputs + num_results).map(move |index| data.param_attrs(ctx, index))
    }

    /// Each input paired with its own attributes, for rebuilding a signature
    /// with [`func_sig_with_param_attrs`].
    pub fn inputs_with_attrs<'a>(
        &self,
        ctx: &'a IrContext,
    ) -> impl Iterator<Item = (TypeRef, &'a AttributeMap)> {
        self.inputs(ctx).iter().copied().zip(self.input_attrs(ctx))
    }

    /// Each result paired with its own attributes, for rebuilding a signature
    /// with [`func_sig_with_param_attrs`].
    pub fn results_with_attrs<'a>(
        &self,
        ctx: &'a IrContext,
    ) -> impl Iterator<Item = (TypeRef, &'a AttributeMap)> {
        self.results(ctx)
            .iter()
            .copied()
            .zip(self.result_attrs(ctx))
    }

    /// Rebuild this signature after `edit` inserts, removes, or replaces
    /// parameter/attribute pairs, so each attribute stays with its parameter.
    /// Non-reserved attributes are kept.
    pub fn rebuild(
        self,
        ctx: &mut IrContext,
        edit: impl FnOnce(&mut Vec<(TypeRef, AttributeMap)>, &mut Vec<(TypeRef, AttributeMap)>),
    ) -> FuncSig {
        let owned = |(ty, attrs): (TypeRef, &AttributeMap)| (ty, attrs.clone());
        let mut inputs: Vec<_> = self.inputs_with_attrs(ctx).map(owned).collect();
        let mut results: Vec<_> = self.results_with_attrs(ctx).map(owned).collect();
        let mut attrs = ctx.get_type(self.0).attrs.clone();
        Self::remove_reserved_attrs(ctx, &mut attrs);
        edit(&mut inputs, &mut results);
        func_sig_with_param_attrs(ctx, inputs, results, attrs)
    }

    pub fn single_result(&self, ctx: &IrContext) -> Option<TypeRef> {
        self.results(ctx).first().copied()
    }

    /// Iterate function metadata without the input/result count delimiters.
    pub fn non_reserved_attrs<'a>(
        &self,
        ctx: &'a IrContext,
    ) -> impl Iterator<Item = (crate::StringRef, &'a Attribute)> {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|&(key, _)| ![NUM_INPUTS_ATTR, NUM_RESULTS_ATTR].contains(&ctx.str(key)))
    }

    /// Remove input/result count delimiters from owned function metadata before rebuilding it.
    pub fn remove_reserved_attrs(ctx: &mut crate::IrContext, attrs: &mut AttributeMap) {
        let mut attrs = attrs.view_mut(ctx);
        attrs.remove(NUM_INPUTS_ATTR);
        attrs.remove(NUM_RESULTS_ATTR);
    }

    pub fn is_resultless(&self, ctx: &IrContext) -> bool {
        self.results(ctx).is_empty()
    }

    /// The machine calling convention, or `None` if `call_conv` is malformed.
    pub fn call_conv(&self, ctx: &IrContext) -> Option<CallConv> {
        CallConv::from_attrs(ctx, &ctx.get_type(self.0).attrs)
    }

    /// Return this signature with its machine calling convention replaced.
    pub fn with_call_conv(self, ctx: &mut IrContext, call_conv: CallConv) -> Self {
        let inputs = self.inputs(ctx).to_vec();
        let results = self.results(ctx).to_vec();
        let mut attrs = ctx.get_type(self.0).attrs.clone();
        Self::remove_reserved_attrs(ctx, &mut attrs);
        call_conv.set_in(ctx, &mut attrs);
        func_sig_with_attrs(ctx, inputs, results, attrs)
    }
}

static FUNC_SIG_DEF: crate::type_def::TypeDef = crate::type_def::TypeDef::new("func.func_sig")
    .with_verifier::<FuncSig>()
    .with_asm_format::<FuncSig>();

crate::register_type_def!(&FUNC_SIG_DEF);

impl crate::asm_format::TypeAsmFormat for FuncSig {
    fn print(&self, h: &mut crate::printer::TypePrintHelper<'_, '_>) -> std::fmt::Result {
        print_signature_type(h, self.0)
    }

    fn parse<'a>(input: &mut &'a str, name: &'a str) -> winnow::ModalResult<RawType<'a>> {
        parse_signature_type(input, name, Some(1))
    }
}

/// Print a validated signature type as `name<(inputs) -> results, {attrs}>`.
///
/// The type stores its inputs then its results as parameters, delimited by
/// [`NUM_INPUTS_ATTR`] and [`NUM_RESULTS_ATTR`], which the syntax implies and
/// does not print. One result is written bare, any other count in
/// parentheses. Parameter attributes follow their parameter. Signature types
/// of other dialects that share this layout reuse it with
/// [`parse_signature_type`] in their
/// [`TypeAsmFormat`](crate::asm_format::TypeAsmFormat).
pub fn print_signature_type(
    h: &mut crate::printer::TypePrintHelper<'_, '_>,
    ty: TypeRef,
) -> std::fmt::Result {
    use std::fmt::Write;

    let ctx = h.ctx();
    let data = ctx.get_type(ty);
    let num_inputs = read_count(ctx, &data.attrs, NUM_INPUTS_ATTR)
        .expect("a validated signature type stores its input count") as usize;
    let (inputs, results) = data.params.split_at(num_inputs);
    let inline_param_attrs = data.validate_param_attrs(ctx).is_ok();
    let write_param = |h: &mut crate::printer::TypePrintHelper<'_, '_>, index: usize| {
        h.write_type(data.params[index])?;
        let attrs = data.param_attrs(ctx, index);
        if inline_param_attrs && !attrs.is_empty() {
            h.write_char(' ')?;
            h.write_attr_dict(attrs.iter())?;
        }
        Ok(())
    };

    write!(h, "{}<(", data.name.display(ctx))?;
    for index in 0..inputs.len() {
        if index > 0 {
            h.write_str(", ")?;
        }
        write_param(h, index)?;
    }
    h.write_str(") -> ")?;
    if results.len() == 1 {
        write_param(h, num_inputs)?;
    } else {
        h.write_char('(')?;
        for index in 0..results.len() {
            if index > 0 {
                h.write_str(", ")?;
            }
            write_param(h, num_inputs + index)?;
        }
        h.write_char(')')?;
    }

    let mut attrs = data
        .attrs
        .iter()
        .filter(|&(key, _)| {
            let key = ctx.str(key);
            key != NUM_INPUTS_ATTR
                && key != NUM_RESULTS_ATTR
                && !(inline_param_attrs && key == crate::types::PARAM_ATTRS_ATTR)
        })
        .peekable();
    if attrs.peek().is_some() {
        h.write_str(", ")?;
        h.write_attr_dict(attrs)?;
    }
    h.write_char('>')
}

/// Parse the rest of `name<(inputs) -> results, {attrs}>` after its opening
/// bracket into the generic form [`signature_raw_type`] builds.
///
/// A list that does not start with `(` backtracks to generic parsing. The
/// syntax owns the delimiter counts, so an attribute dictionary that writes
/// them is rejected, as is a result count above `max_results`.
pub fn parse_signature_type<'a>(
    input: &mut &'a str,
    name: &'a str,
    max_results: Option<usize>,
) -> winnow::ModalResult<RawType<'a>> {
    use crate::parser::raw::{raw_attr_dict, raw_param, raw_param_list, ws};
    use winnow::combinator::{opt, preceded};
    use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
    use winnow::prelude::*;

    let cut = |description: &'static str| {
        let mut error = ContextError::new();
        error.push(StrContext::Expected(StrContextValue::Description(
            description,
        )));
        ErrMode::Cut(error)
    };
    if !input.starts_with('(') {
        return Err(ErrMode::Backtrack(ContextError::new()));
    }
    let inputs = raw_param_list.parse_next(input)?;
    ws.parse_next(input)?;
    "->".parse_next(input)?;
    ws.parse_next(input)?;
    let results = if input.starts_with('(') {
        raw_param_list.parse_next(input)?
    } else {
        vec![raw_param.parse_next(input)?]
    };
    ws.parse_next(input)?;
    let attrs = opt(preceded((',', ws), raw_attr_dict))
        .parse_next(input)?
        .unwrap_or_default();
    ws.parse_next(input)?;
    '>'.parse_next(input)?;

    if max_results.is_some_and(|max| results.len() > max) {
        return Err(cut("no more results than the signature type allows"));
    }
    if attrs
        .iter()
        .any(|(key, _)| key == NUM_INPUTS_ATTR || key == NUM_RESULTS_ATTR)
    {
        return Err(cut(
            "signature attributes without `num_inputs` or `num_results`, which the syntax implies",
        ));
    }
    Ok(signature_raw_type(name, inputs, results, attrs))
}

/// The generic raw form of a signature type: its inputs then its results as
/// parameters, with [`NUM_INPUTS_ATTR`] and [`NUM_RESULTS_ATTR`] added to
/// `attrs`.
pub fn signature_raw_type<'a>(
    name: &'a str,
    inputs: Vec<RawParam<'a>>,
    results: Vec<RawParam<'a>>,
    mut attrs: RawAttrDict<'a>,
) -> RawType<'a> {
    attrs.push((
        NUM_INPUTS_ATTR.into(),
        RawAttribute::Int(inputs.len() as i128),
    ));
    attrs.push((
        NUM_RESULTS_ATTR.into(),
        RawAttribute::Int(results.len() as i128),
    ));
    let mut params = inputs;
    params.extend(results);
    RawType::Concrete {
        name,
        params,
        attrs,
    }
}

impl DialectType for FuncSig {
    const DEF: &'static crate::type_def::TypeDef = &FUNC_SIG_DEF;

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        if !Self::matches(ctx, ty) {
            return None;
        }
        <Self as crate::type_verifier::VerifyType>::verify(ctx, ty).ok()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl From<FuncSig> for TypeRef {
    fn from(ty: FuncSig) -> Self {
        ty.0
    }
}

fn read_count(
    ctx: &crate::IrContext,
    attrs: &AttributeMap,
    name: &'static str,
) -> Result<u32, FuncSigTypeError> {
    match attrs.view(ctx).get(name) {
        None => Err(FuncSigTypeError::MissingCount(name)),
        Some(Attribute::Int(value)) => {
            u32::try_from(*value).map_err(|_| FuncSigTypeError::InvalidCount(name))
        }
        Some(_) => Err(FuncSigTypeError::InvalidCount(name)),
    }
}

/// Construct a canonical `func.func_sig` with zero or one result.
pub fn func_sig(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
) -> FuncSig {
    func_sig_with_attrs(ctx, inputs, results, AttributeMap::new())
}

/// Construct a canonical `func.func_sig` whose inputs and results carry their
/// own attributes, stored as the type's per-parameter attributes.
///
/// A per-parameter attribute value already in `attrs` is replaced, so a caller
/// that adds, removes, or replaces parameters may pass the source signature's
/// remaining attributes unchanged.
pub fn func_sig_with_param_attrs(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = (TypeRef, AttributeMap)>,
    results: impl IntoIterator<Item = (TypeRef, AttributeMap)>,
    mut attrs: AttributeMap,
) -> FuncSig {
    let (inputs, input_attrs): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();
    let (results, result_attrs): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    match crate::types::param_attrs_attribute(input_attrs.into_iter().chain(result_attrs)) {
        Some(value) => {
            attrs
                .view_mut(ctx)
                .insert(crate::types::PARAM_ATTRS_ATTR, value);
        }
        None => {
            attrs.view_mut(ctx).remove(crate::types::PARAM_ATTRS_ATTR);
        }
    }
    func_sig_with_attrs(ctx, inputs, results, attrs)
}

/// Construct a canonical `func.func_sig` while preserving non-reserved attributes.
///
/// Per-parameter attributes in `attrs` must have one entry per input and result.
pub fn func_sig_with_attrs(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
    mut attrs: AttributeMap,
) -> FuncSig {
    assert!(
        !attrs.view(ctx).contains_key(NUM_INPUTS_ATTR)
            && !attrs.view(ctx).contains_key(NUM_RESULTS_ATTR),
        "func.func_sig count attributes are reserved"
    );

    let inputs: Vec<_> = inputs.into_iter().collect();
    let results: Vec<_> = results.into_iter().collect();
    assert!(
        results.len() <= 1,
        "func.func_sig currently supports at most one result"
    );
    let num_inputs = u32::try_from(inputs.len()).expect("func.func_sig input count exceeds u32");
    let num_results = u32::try_from(results.len()).expect("func.func_sig result count exceeds u32");
    if let Err(error) =
        crate::types::normalize_param_attrs(ctx, &mut attrs, inputs.len() + results.len())
    {
        panic!("func.func_sig: {error}");
    }

    let mut builder = TypeDataBuilder::new(<FuncSig as DialectType>::NAME)
        .params(inputs)
        .params(results);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let ty = builder
        .attr(NUM_INPUTS_ATTR, Attribute::from(num_inputs))
        .attr(NUM_RESULTS_ATTR, Attribute::from(num_results))
        .intern(ctx);
    <FuncSig as crate::type_verifier::VerifyType>::verify(ctx, ty)
        .expect("func.func_sig constructor must produce a valid type")
}

// One-result source producers use these explicit convenience accessors. Generic
// consumers use `results` / `single_result` and validate cardinality first.
macro_rules! single_call_result {
    ($name:ident) => {
        impl $name {
            pub fn single_result(&self, ctx: &crate::IrContext) -> Option<crate::ValueRef> {
                match self.results(ctx) {
                    [result] => Some(*result),
                    _ => None,
                }
            }
            pub fn result(&self, ctx: &crate::IrContext) -> crate::ValueRef {
                self.single_result(ctx)
                    .expect("one-result call required by producer")
            }
            pub fn result_ty(&self, ctx: &crate::IrContext) -> crate::TypeRef {
                ctx.value_ty(self.result(ctx))
            }
        }
    };
}
single_call_result!(Call);
single_call_result!(CallIndirect);

impl CallLike for Call {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl CallLike for CallIndirect {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl IndirectCallLike for CallIndirect {
    fn exact_signature(&self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        indirect_call_signature(ctx, self.op_ref())
    }

    fn set_exact_signature(&self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl crate::ops::Verify for CallIndirect {
    fn verify(&self, ctx: &crate::IrContext) -> Result<(), crate::ops::VerificationError> {
        Ok(verify_typed_callee(
            ctx,
            self.callee(ctx),
            self.signature(ctx),
        )?)
    }
}

impl crate::ops::Verify for TailCallIndirect {
    fn verify(&self, ctx: &crate::IrContext) -> Result<(), crate::ops::VerificationError> {
        Ok(verify_typed_callee(
            ctx,
            self.callee(ctx),
            self.signature(ctx),
        )?)
    }
}

/// A callee whose type is a `func.func_sig` must carry that exact signature.
/// Any other callee, such as a closure whose type names a callable identity
/// rather than an entry ABI, a pointer, or a table index, is not checked.
fn verify_typed_callee(
    ctx: &crate::IrContext,
    callee: crate::ValueRef,
    signature: crate::TypeRef,
) -> Result<(), String> {
    let callee_ty = ctx.value_ty(callee);
    if FuncSig::from_type_ref(ctx, callee_ty).is_some() && callee_ty != signature {
        return Err("exact indirect signature differs from typed callee".into());
    }
    Ok(())
}

impl CallLike for TailCall {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl TailCallLike for TailCall {}

impl CallLike for TailCallIndirect {
    fn call_args<'a>(&self, ctx: &'a crate::IrContext) -> &'a [crate::ValueRef] {
        self.args(ctx)
    }
}

impl IndirectCallLike for TailCallIndirect {
    fn exact_signature(&self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        indirect_call_signature(ctx, self.op_ref())
    }

    fn set_exact_signature(&self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl TailCallLike for TailCallIndirect {}

impl CallableExit for Return {}

impl CallableExit for TailCall {
    fn verify_callable_exit(
        &self,
        ctx: &crate::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        if ctx
            .op_attrs(self.op_ref())
            .get_symbol_ref("callee")
            .is_some()
        {
            Ok(())
        } else {
            Err(ControlFlowInterfaceError::new(
                "func.tail_call CallableExit requires a callee symbol",
            ))
        }
    }
}

impl CallableExit for TailCallIndirect {
    fn verify_callable_exit(
        &self,
        ctx: &crate::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        if ctx.op_operands(self.op_ref()).is_empty() {
            return Err(ControlFlowInterfaceError::new(
                "func.tail_call_indirect CallableExit requires a callee operand",
            ));
        }
        match ctx.op_attrs(self.op_ref()).get("signature") {
            None => Ok(()),
            Some(Attribute::Type(signature))
                if FuncSig::from_type_ref(ctx, *signature).is_some() =>
            {
                Ok(())
            }
            Some(_) => Err(ControlFlowInterfaceError::new(
                "func.tail_call_indirect CallableExit has an invalid exact signature",
            )),
        }
    }
}

impl CallableExit for Unreachable {
    fn verify_callable_exit(
        &self,
        ctx: &crate::IrContext,
    ) -> Result<(), ControlFlowInterfaceError> {
        if ctx.op_operands(self.op_ref()).is_empty() {
            Ok(())
        } else {
            Err(ControlFlowInterfaceError::new(
                "func.unreachable CallableExit must not have operands",
            ))
        }
    }
}

impl op_interface::CallLike for Call {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}
impl op_interface::CallLike for TailCall {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}

/// Attach an exact callable contract to a `func` indirect transfer.
///
/// Returns `false` without mutation when the operation is not a `func`
/// indirect call or the supplied type is not a `func.func_sig` contract.
pub fn set_indirect_call_signature(
    ctx: &mut crate::IrContext,
    op: crate::OpRef,
    signature: crate::TypeRef,
) -> bool {
    if crate::dialect::func::FuncSig::from_type_ref(ctx, signature).is_none()
        || (CallIndirect::from_op(ctx, op).is_err() && TailCallIndirect::from_op(ctx, op).is_err())
    {
        return false;
    }
    ctx.op_attrs_mut(op).insert(
        INDIRECT_CALL_SIGNATURE_ATTR,
        crate::Attribute::Type(signature),
    );
    true
}

/// Read the exact signature without assuming the operation passed its schema;
/// interface queries must fail closed on malformed IR.
fn indirect_call_signature(ctx: &crate::IrContext, op: crate::OpRef) -> Option<crate::TypeRef> {
    ctx.op_attrs(op)
        .get_type(INDIRECT_CALL_SIGNATURE_ATTR)
        .filter(|&ty| FuncSig::from_type_ref(ctx, ty).is_some())
}

/// Remove the `func`-owned exact-signature attribute from copied metadata.
pub fn remove_indirect_call_signature(
    ctx: &mut crate::IrContext,
    attributes: &mut crate::AttributeMap,
) {
    attributes
        .view_mut(ctx)
        .remove(INDIRECT_CALL_SIGNATURE_ATTR);
}

impl Func {
    /// Return the function body when this declaration has one.
    pub fn body_if_present(&self, ctx: &crate::IrContext) -> Option<crate::RegionRef> {
        ctx.op_region(self.op_ref(), 0)
    }
}

// === Custom assembly format for func.func ===

/// Print func.func with decomposed signature:
/// `func.func @name(%arg: type, ...) -> return_type effects eff_type { body }`
fn print_func(h: &mut crate::printer::OpPrintHelper<'_, '_>, op: crate::OpRef) -> std::fmt::Result {
    use std::fmt::Write;

    let sym_name = h.ctx().op_attrs(op).get_str("sym_name");

    write!(h, "func.func")?;

    // Function name
    if let Some(name) = sym_name {
        write!(h, " ")?;
        h.write_symbol_text(name)?;
    }

    // Reset numbering for function body
    h.reset_numbering();

    // Extract region ref and func type info before mutable operations
    let region = {
        let regions = h.ctx().op_region_count(op);
        assert!(
            regions <= 1,
            "print_func: expected at most one region, found {regions}",
        );
        h.ctx().op_region(op, 0)
    };

    // Extract the validated input/result lists from the func.func_sig type attribute.
    let type_info = {
        h.ctx()
            .op_attrs(op)
            .get_type("type")
            .and_then(|ty| crate::dialect::func::FuncSig::from_type_ref(h.ctx(), ty))
            .map(|func| (func.single_result(h.ctx()), func.inputs(h.ctx()).to_vec()))
    };

    if let Some(region) = region {
        // Print entry block args as function signature
        let entry_args: Vec<_> = {
            let blocks = &h.ctx().region(region).blocks;
            blocks
                .first()
                .map(|&b| h.ctx().block_args(b).to_vec())
                .unwrap_or_default()
        };

        write!(h, "(")?;
        for (i, &arg) in entry_args.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            let name = h.assign_value_name(arg);
            let ty = h.ctx().value_ty(arg);
            write!(h, "{name}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
    } else if let Some((_, ref param_tys)) = type_info {
        // Body-less declaration: synthesize params from type
        write!(h, "(")?;
        for (i, &ty) in param_tys.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "%arg{i}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
    } else {
        write!(h, "()")?;
    }

    if let Some((Some(result_ty), _)) = type_info {
        write!(h, " -> ")?;
        h.write_type(result_ty)?;
    }

    // Extra attributes (everything except sym_name and type, which are
    // already encoded in the signature).  Clone to avoid borrow conflicts
    // with the mutable write helpers.
    let preserve_type = h.ctx().op_attrs(op).get_type("type").is_some_and(|ty| {
        h.ctx()
            .get_type(ty)
            .attrs
            .keys()
            .any(|key| ![NUM_INPUTS_ATTR, NUM_RESULTS_ATTR].contains(&h.ctx().str(key)))
    });
    let extra_attrs: Vec<_> = h
        .ctx()
        .op_attrs(op)
        .iter()
        .filter(|&(k, _)| k != "sym_name" && (preserve_type || k != "type"))
        .collect();
    if !extra_attrs.is_empty() {
        write!(h, " attributes {{")?;
        for (i, (key, val)) in extra_attrs.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "{key} = ")?;
            h.write_attribute(val)?;
        }
        write!(h, "}}")?;
    }

    if let Some(region) = region {
        h.print_body(region)?;
    }

    Ok(())
}

/// Parse func.func custom format back into a RawOperation.
fn parse_func<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<crate::parser::raw::RawOperation<'a>> {
    use crate::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    // "(%arg: type, ...)" or "()"
    ws.parse_next(input)?;
    let params = if input.starts_with('(') {
        func_params.parse_next(input)?
    } else {
        vec![]
    };

    // "-> return_type" (optional)
    let ret_ty = opt(return_type).parse_next(input)?;

    // "attributes { key = value, ... }" (optional extra attributes)
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();

    // "{ body }"
    ws.parse_next(input)?;
    let mut regions = Vec::new();
    if input.starts_with('{') {
        let region = raw_region.parse_next(input)?;
        regions.push(region);
    }

    Ok(RawOperation {
        results,
        name: "func.func",
        sym_name,
        has_func_signature: true,
        func_params: params,
        return_type: ret_ty,
        operands: vec![],
        attributes,
        result_types: vec![],
        regions,
        successors: vec![],
    })
}

impl crate::asm_format::OpAsmFormat for Func {
    fn print(&self, h: &mut crate::printer::OpPrintHelper<'_, '_>) -> std::fmt::Result {
        print_func(h, self.op_ref())
    }

    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<crate::parser::raw::RawOperation<'a>> {
        parse_func(input, results, sym_name)
    }
}

impl crate::op_interface::CallableOwner for Func {
    fn callable_signature(
        &self,
        ctx: &crate::IrContext,
        _: crate::refs::RegionRef,
    ) -> Option<crate::dialect::func::FuncSig> {
        ctx.op_attrs(self.op_ref())
            .get_type("type")
            .and_then(|ty| crate::dialect::func::FuncSig::from_type_ref(ctx, ty))
    }
}
/// A function with a body and an `abi` attribute is callable from outside
/// the module, so it is kept. A bodyless `abi` declaration is an import and
/// goes once nothing uses it.
impl op_interface::SymbolDefinition for Func {
    fn is_discardable(&self, ctx: &IrContext) -> bool {
        !(ctx.op_attrs(self.op_ref()).contains_key("abi") && ctx.op_has_regions(self.op_ref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolPath;
    use crate::op_interface::{DynCallableExit, DynIndirectCallLike};
    use crate::ops::DialectOp;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

    fn parse_raw_signature(text: &str) -> RawType<'_> {
        use winnow::prelude::*;
        let mut input = text;
        let raw = crate::parser::raw::raw_type
            .parse_next(&mut input)
            .expect("signature should parse");
        assert!(input.is_empty(), "unparsed: {input}");
        raw
    }

    fn raw_count(attrs: &RawAttrDict<'_>, key: &str) -> i128 {
        match attrs.iter().find(|(k, _)| k == key) {
            Some((_, RawAttribute::Int(count))) => *count,
            other => panic!("expected integer `{key}`, got {other:?}"),
        }
    }

    #[test]
    fn signature_syntax_parses_into_the_generic_form() {
        let raw = parse_raw_signature(
            "func.func_sig<(core.i32 {a = @x}, core.ptr) -> core.i64 {b = @y}, {call_conv = \"tail\"}>",
        );
        let RawType::Concrete {
            name,
            params,
            attrs,
        } = raw
        else {
            panic!("expected the generic form")
        };
        assert_eq!(name, "func.func_sig");
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].attrs.len(), 1);
        assert!(params[1].attrs.is_empty());
        assert_eq!(params[2].attrs.len(), 1);
        assert_eq!(raw_count(&attrs, NUM_INPUTS_ATTR), 2);
        assert_eq!(raw_count(&attrs, NUM_RESULTS_ATTR), 1);
        assert_eq!(attrs.len(), 3);
    }

    #[test]
    fn signature_syntax_counts_empty_and_multiple_results() {
        let RawType::Concrete { params, attrs, .. } =
            parse_raw_signature("func.func_sig<(core.i32) -> ()>")
        else {
            panic!("expected the generic form")
        };
        assert_eq!(params.len(), 1);
        assert_eq!(raw_count(&attrs, NUM_RESULTS_ATTR), 0);

        let RawType::Concrete { params, attrs, .. } =
            parse_raw_signature("wasm.func_sig<() -> (core.i32, core.i64)>")
        else {
            panic!("expected the generic form")
        };
        assert_eq!(params.len(), 2);
        assert_eq!(raw_count(&attrs, NUM_INPUTS_ATTR), 0);
        assert_eq!(raw_count(&attrs, NUM_RESULTS_ATTR), 2);
    }

    #[test]
    fn signature_generic_spelling_round_trips_to_the_custom_syntax() {
        let input = r#"core.module @test {
  !generic = func.func_sig<core.i32 {a = 1}, core.i64, {num_inputs = 1, num_results = 1}>
}"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("!generic = func.func_sig<(core.i32 {a = 1}) -> core.i64>"),
            "{printed}"
        );
    }

    #[test]
    fn call_conv_is_a_signature_attribute_that_round_trips() {
        let input = r#"core.module @test {
  func.func @transfer(%callee: func.func_sig<(core.i32) -> (), {call_conv = "tail"}>, %value: core.i32) attributes {type = func.func_sig<(func.func_sig<(core.i32) -> (), {call_conv = "tail"}>, core.i32) -> (), {call_conv = "tail"}>} {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> (), {call_conv = "tail"}>}
  }
}"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let function = Func::from_op(&ctx, module.ops(&ctx)[0]).expect("function");
        let signature = FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).expect("signature");
        assert_eq!(signature.call_conv(&ctx), Some(CallConv::Tail));

        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("call_conv = \"tail\""), "{printed}");
        let mut reparsed_ctx = crate::IrContext::new();
        let reparsed = parse_test_module(&mut reparsed_ctx, &printed);
        assert_eq!(print_module(&reparsed_ctx, reparsed.op()), printed);
    }

    #[test]
    fn with_call_conv_changes_type_identity_and_preserves_the_rest() {
        let mut ctx = crate::IrContext::new();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let mut attrs = AttributeMap::new();
        attrs
            .view_mut(&mut ctx)
            .insert("note", Attribute::SymbolRef(SymbolPath::new(["kept"])));
        let platform = func_sig_with_attrs(&mut ctx, [i32_ty], [], attrs);
        assert_eq!(platform.call_conv(&ctx), Some(CallConv::Platform));

        let tail = platform.with_call_conv(&mut ctx, CallConv::Tail);
        assert_ne!(tail, platform);
        assert_eq!(tail.call_conv(&ctx), Some(CallConv::Tail));
        assert_eq!(tail.inputs(&ctx), [i32_ty]);
        assert!(tail.results(&ctx).is_empty());
        assert!(
            tail.non_reserved_attrs(&ctx)
                .any(|(key, _)| ctx.str(key) == "note")
        );
        assert_eq!(tail.with_call_conv(&mut ctx, CallConv::Platform), platform);
    }

    #[test]
    fn inserting_a_parameter_moves_parameter_attributes_with_their_parameters() {
        let mut ctx = crate::IrContext::new();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let ptr_ty = crate::dialect::core::Ptr::params()
            .intern(&mut ctx)
            .as_type_ref();
        let marked = AttributeMap::from_entries(
            &mut ctx,
            [("k", Attribute::SymbolRef(SymbolPath::new(["v"])))],
        );
        let source = func_sig_with_param_attrs(
            &mut ctx,
            [(i32_ty, marked.clone())],
            [(i32_ty, AttributeMap::new())],
            AttributeMap::new(),
        );

        let rebuilt = source.rebuild(&mut ctx, |inputs, _| {
            inputs.insert(0, (ptr_ty, AttributeMap::new()));
        });

        assert_eq!(rebuilt.inputs(&ctx), [ptr_ty, i32_ty]);
        assert!(
            rebuilt
                .inputs_with_attrs(&ctx)
                .eq([(ptr_ty, &AttributeMap::new()), (i32_ty, &marked)])
        );
        assert!(
            rebuilt
                .results_with_attrs(&ctx)
                .eq([(i32_ty, &AttributeMap::new())])
        );
    }

    #[test]
    fn malformed_call_conv_is_reported_rather_than_defaulted() {
        let mut ctx = crate::IrContext::new();
        let mut attrs = AttributeMap::new();
        let fast = ctx.string_attr("fast");
        attrs.view_mut(&mut ctx).insert(CALL_CONV_ATTR, fast);
        let signature = func_sig_with_attrs(&mut ctx, [], [], attrs);
        assert_eq!(signature.call_conv(&ctx), None);
    }

    #[test]
    fn indirect_signature_is_declared_and_round_trips() {
        let input = r#"core.module @test {
  func.func @run(%callee: func.func_sig<(core.i32) -> core.i32>, %value: core.i32) -> core.i32 {
    %result = func.call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.i32>} : core.i32
    func.return %result
  }
}"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let function = Func::from_op(&ctx, ctx.block(module.first_block(&ctx).unwrap()).ops[0])
            .expect("function");
        let body = function.body_if_present(&ctx).expect("function body");
        let call = CallIndirect::from_op(&ctx, ctx.block(ctx.region(body).blocks[0]).ops[0])
            .expect("indirect call");
        assert!(
            FuncSig::from_type_ref(&ctx, call.signature(&ctx)).is_some(),
            "declared signature"
        );

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("!t0 = func.func_sig<(core.i32) -> core.i32>"),
            "{printed}"
        );
        assert!(printed.contains("signature = !t0"), "{printed}");
        assert!(!printed.contains("func.indirect_call_signature"));
    }

    #[test]
    fn indirect_call_signature_query_rejects_non_signature_types() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%callee: core.ptr, %value: core.i32) {
    func.call_indirect %callee, %value {signature = core.i32}
    clif.call_indirect %callee, %value {sig = core.i32}
    clif.return_call_indirect %callee, %value {sig = core.i32}
  }
}"#,
        );
        let function = Func::from_op(&ctx, module.ops(&ctx)[0]).expect("function");
        let ops = &ctx
            .block(ctx.region(function.body_if_present(&ctx).unwrap()).blocks[0])
            .ops;
        for &op in ops {
            let call = DynIndirectCallLike::get(&ctx, op).expect("indirect call");
            assert_eq!(call.exact_signature(&ctx), None);
        }
    }

    #[test]
    fn indirect_call_signature_recognizes_only_indirect_calls() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @ordinary(%callee: func.func_sig<(core.i32) -> core.i32>, %value: core.i32) -> core.i32 {
    %result = func.call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.i32>} : core.i32
    func.return %result
  }
  func.func @tail(%callee: func.func_sig<(core.i32) -> core.nil>, %value: core.i32) -> core.nil {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> core.nil>}
  }
  func.func @direct() -> core.nil {
    func.return
  }
}"#,
        );

        let functions = module.ops(&ctx);
        let ordinary = Func::from_op(&ctx, functions[0]).expect("ordinary function");
        let tail = Func::from_op(&ctx, functions[1]).expect("tail function");
        let direct = Func::from_op(&ctx, functions[2]).expect("direct function");
        let ordinary_op = ctx
            .block(ctx.region(ordinary.body_if_present(&ctx).unwrap()).blocks[0])
            .ops[0];
        let tail_op = ctx
            .block(ctx.region(tail.body_if_present(&ctx).unwrap()).blocks[0])
            .ops[0];
        let direct_op = ctx
            .block(ctx.region(direct.body_if_present(&ctx).unwrap()).blocks[0])
            .ops[0];

        for op in [ordinary_op, tail_op] {
            let call = DynIndirectCallLike::get(&ctx, op).expect("indirect call");
            let operands = ctx.op_operands(op);
            assert!(call.exact_signature(&ctx).is_some());
            assert_eq!(call.indirect_callee(&ctx), Some(operands[0]));
            assert_eq!(call.indirect_arguments(&ctx), Some(&operands[1..]));
        }
        assert!(DynIndirectCallLike::get(&ctx, direct_op).is_none());
    }

    #[test]
    fn callable_exit_rejects_wrong_kind_indirect_tail_signature() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @tail(%callee: func.func_sig<() -> ()>) {
    func.tail_call_indirect %callee
  }
}"#,
        );
        let function = Func::from_op(&ctx, module.ops(&ctx)[0]).expect("tail function");
        let tail = ctx.block(ctx.region(function.body(&ctx)).blocks[0]).ops[0];
        ctx.op_attrs_mut(tail)
            .insert("signature", Attribute::Int(0));

        assert!(DynCallableExit::prove(&ctx, tail).is_err());
    }
}

#[cfg(test)]
mod result_list_tests {
    use super::*;
    use crate::IrContext;
    use crate::SymbolPath;
    use crate::dialect::core;
    use crate::parser::parse_module;
    use crate::printer::print_module;
    use crate::rewrite::Module;
    use crate::validation::validate_all;

    #[test]
    fn function_assembly_preserves_arity_and_type_attributes() {
        for inputs in ["", "%x: core.i32, %y: core.i64"] {
            for result in ["", " -> core.nil"] {
                for body in ["", " { func.unreachable }"] {
                    let input =
                        format!("core.module @m {{ func.func @f({inputs}){result}{body} }}");
                    let mut ctx = IrContext::new();
                    let op = parse_module(&mut ctx, &input).unwrap();
                    let printed = print_module(&ctx, op);
                    let copy = parse_module(&mut ctx, &printed).unwrap();
                    let a = Module::new(&ctx, op).unwrap().ops(&ctx)[0];
                    let b = Module::new(&ctx, copy).unwrap().ops(&ctx)[0];
                    assert_eq!(
                        ctx.op_attrs(a).get_type("type"),
                        ctx.op_attrs(b).get_type("type")
                    );
                    assert_eq!(print_module(&ctx, copy), printed);
                }
            }
        }
        for input in [
            "core.module @m { func.func {sym_name = \"f\", type = func.func_sig<(core.i32) -> (), {tag = @kept, nested = [core.i64]}>} }",
            "core.module @m { func.func @f(%x: core.i32) attributes {type = func.func_sig<(core.i32) -> (), {tag = @kept, nested = [core.i64]}>} { func.return } }",
        ] {
            let mut ctx = IrContext::new();
            let op = parse_module(&mut ctx, input).unwrap();
            let printed = print_module(&ctx, op);
            assert!(printed.contains("tag = @kept"), "{printed}");
            let copy = parse_module(&mut ctx, &printed).unwrap();
            let a = Module::new(&ctx, op).unwrap().ops(&ctx)[0];
            let b = Module::new(&ctx, copy).unwrap().ops(&ctx)[0];
            assert_eq!(
                ctx.op_attrs(a).get_type("type"),
                ctx.op_attrs(b).get_type("type")
            );
        }
        let mut ctx = IrContext::new();
        assert!(
            parse_module(
                &mut ctx,
                "core.module @m { func.func @f() attributes {type = func.func_sig<() -> core.nil>} }"
            )
            .is_err()
        );
    }

    #[test]
    fn call_constructors_support_zero_and_one_results() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, "core.module @m {}");
        let loc = ctx.op(module.op()).location;
        let nil = core::Nil::params().intern(&mut ctx).as_type_ref();
        for results in [vec![], vec![nil]] {
            let signature = func_sig(&mut ctx, [], results.clone()).as_type_ref();
            let callee = Constant::operands()
                .func_ref(SymbolPath::new(["f"]))
                .results(signature)
                .build(&mut ctx, loc)
                .result(&ctx);
            let direct = Call::operands([])
                .callee(SymbolPath::new(["f"]))
                .results(results.clone())
                .build(&mut ctx, loc);
            let indirect = CallIndirect::operands(callee, [])
                .signature(signature)
                .build(&mut ctx, loc);
            assert_eq!(direct.call_result_types(&ctx), results);
            assert_eq!(indirect.call_result_types(&ctx), results);
            assert_eq!(direct.single_result(&ctx).is_some(), results.len() == 1);
        }
    }

    fn verify(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        validate_all(&ctx, module, &mut Default::default()).to_string()
    }

    #[test]
    fn complete_call_return_and_tail_contracts() {
        let valid = "core.module @m {
          func.func @sink(%x: core.i32)
          func.func @value(%x: core.i32) -> core.i32
          func.func @run(%k: func.func_sig<(core.i32) -> ()>, %x: core.i32) {
            func.call %x {callee = @sink}
            func.call_indirect %k, %x {signature = func.func_sig<(core.i32) -> ()>}
            %r = func.call %x {callee = @value} : core.i32
            func.return
          }
          func.func @tail(%x: core.i32) { func.tail_call %x {callee = @sink} }
          func.func @logical(%k: func.func_sig<(core.i32) -> core.never>, %x: core.i32) -> core.never {
            func.tail_call_indirect %k, %x {signature = func.func_sig<(core.i32) -> core.never>}
          }
          func.func @one(%k: func.func_sig<(core.i32) -> core.i32>, %x: core.i32) -> core.i32 {
            %r = func.call_indirect %k, %x {signature = func.func_sig<(core.i32) -> core.i32>} : core.i32
            func.return %r
          }
        }";
        assert_eq!(verify(valid), "validation passed");
        for (old, new, expected) in [
            (
                "func.call %x {callee = @sink}",
                "%bad = func.call %x {callee = @sink} : core.nil",
                "call result list mismatch",
            ),
            (
                "func.call %x {callee = @sink}",
                "func.call {callee = @sink}",
                "call argument count mismatch",
            ),
            ("func.return %r", "func.return", "return count mismatch"),
            (
                "func.return %r",
                "func.return %k",
                "return #0 type mismatch",
            ),
            (
                "@tail(%x: core.i32)",
                "@tail(%x: core.i32) -> core.nil",
                "tail caller/callee result lists differ",
            ),
            (
                "signature = func.func_sig<(core.i32) -> ()>",
                "signature = func.func_sig<(core.i64) -> ()>",
                "expected S::Inputs = (core.i64), found (core.i32)",
            ),
            (
                "callee = @sink",
                "callee = 1",
                "attribute `callee` must be a SymbolRef attribute",
            ),
        ] {
            let text = verify(&valid.replace(old, new));
            assert!(text.contains(expected), "{expected}: {text}");
        }
        let order = "core.module @m { func.func @f(%a: core.i32, %b: core.i64) func.func @g(%a: core.i32, %b: core.i64) { func.call %b, %a {callee = @f} func.return } }";
        let errors = verify(order);
        assert!(
            errors.contains("call argument #0 type mismatch"),
            "{errors}"
        );
        assert!(
            errors.contains("call argument #1 type mismatch"),
            "{errors}"
        );
    }
}

#[cfg(test)]
mod owner_identity_tests {
    #[test]
    fn operation_name_and_type_attribute_do_not_establish_ownership() {
        let mut ctx = crate::IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            "core.module @m {
          func.func @f(%k: func.func_sig<() -> core.never>) -> core.never {
            test.lambda {type = func.func_sig<() -> core.nil>} { func.tail_call_indirect %k {signature = func.func_sig<() -> core.never>} }
            func.unreachable
          }
        }",
        );
        let result = crate::validation::validate_all(&ctx, module, &mut Default::default());
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn missing_runtime_symbols_do_not_hide_known_invalid_declarations() {
        for declaration in [
            "func.func @f() func.func @f()",
            "test.object {sym_name = \"f\"}",
        ] {
            let mut ctx = crate::IrContext::new();
            let input = format!(
                "core.module @m {{ {declaration} func.func @run() {{ func.call {{callee = @runtime}} func.call {{callee = @f}} func.return }} }}"
            );
            let module = crate::parser::parse_test_module(&mut ctx, &input);
            let result = crate::validation::validate_all(&ctx, module, &mut Default::default());
            assert!(
                result
                    .to_string()
                    .contains("uniquely resolved valid callable"),
                "{result}"
            );
        }
    }
}

#[cfg(test)]
mod normal_validation_regressions {
    #[test]
    fn normal_validation_rejects_known_contract_mismatches() {
        for (body, expected) in [
            (
                "%a, %b = func.call {callee = @runtime} : core.i32, core.i32\nfunc.return",
                "multiple call results",
            ),
            (
                "%a, %b = func.call_indirect %k {signature = func.func_sig<() -> ()>} : core.i32, core.i32\nfunc.return",
                "expected S::Results = (), found (core.i32, core.i32)",
            ),
            (
                "%a = func.call {callee = @sink} : core.i32\nfunc.return",
                "call result list mismatch",
            ),
            (
                "%a = func.call_indirect %k {signature = func.func_sig<() -> ()>} : core.i32\nfunc.return",
                "expected S::Results = (), found (core.i32)",
            ),
            ("func.return %k", "return count mismatch"),
        ] {
            let mut ctx = crate::IrContext::new();
            let input = format!(
                "core.module @m {{ func.func @sink() func.func @run(%k: func.func_sig<() -> ()>) {{ func.call {{callee = @runtime}} {body} }} }}"
            );
            let module = crate::parser::parse_test_module(&mut ctx, &input);
            let result = crate::validation::validate_all(&ctx, module, &mut Default::default());
            assert!(
                result.to_string().contains(expected),
                "{expected}: {result}"
            );
        }
    }

    #[test]
    fn malformed_explicit_custom_function_type_is_rejected() {
        let mut ctx = crate::IrContext::new();
        let error = crate::parser::parse_module(
            &mut ctx,
            "core.module @m { func.func @f() attributes {type = 1} }",
        )
        .unwrap_err();
        assert!(error.message.contains("type attribute"), "{error}");
    }
}
