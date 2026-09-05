//! Arena-based func dialect.

use crate::op_interface::{IndirectCallLikeModel, IndirectCallLikeOps};
use crate::ops::{DialectOp, DialectType};
use crate::{Attribute, AttributeMap, IrContext, Symbol, TypeDataBuilder, TypeRef};

/// The optional exact callable signature retained after a typed indirect callee
/// becomes a runtime function or table index.
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

// === Operation registrations ===
crate::register_pure_op!(func.constant);
crate::register_isolated_op!(func.func);

#[trunk_ir::dialect]
mod func {
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)]
        {}
    }

    #[attr(callee: Symbol)]
    #[rest_results]
    fn call(#[rest] args: ()) -> results {}

    #[attr(signature?: Type)]
    #[rest_results]
    fn call_indirect(callee: (), #[rest] args: ()) -> results {}

    #[attr(callee: Symbol)]
    fn tail_call(#[rest] args: ()) {}

    #[attr(signature?: Type)]
    fn tail_call_indirect(callee: (), #[rest] args: ()) {}

    fn r#return(#[rest] values: ()) {}

    #[attr(func_ref: Symbol)]
    fn constant() -> result {}

    fn unreachable() {}
}

/// Reserved delimiter attribute for the number of function inputs.
pub const NUM_INPUTS_ATTR: &str = "num_inputs";

/// Reserved delimiter attribute for the number of function results.
pub const NUM_RESULTS_ATTR: &str = "num_results";

/// Return the interned name of the `func.func_sig` type.
#[allow(non_snake_case)]
#[inline]
pub fn FUNC_SIG() -> Symbol {
    Symbol::new("func_sig")
}

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

impl FuncSig {
    /// Validate a name-matching `func.func_sig`, including both delimiter counts.
    pub(crate) fn validate(ctx: &IrContext, ty: TypeRef) -> Result<Self, FuncSigTypeError> {
        let data = ctx.types.get(ty);
        debug_assert!(data.dialect == DIALECT_NAME() && data.name == FUNC_SIG());

        let num_inputs = read_count(&data.attrs, NUM_INPUTS_ATTR)?;
        let num_results = read_count(&data.attrs, NUM_RESULTS_ATTR)?;
        if num_results > 1 {
            return Err(FuncSigTypeError::UnsupportedResultCount(num_results));
        }
        let total = num_inputs
            .checked_add(num_results)
            .ok_or(FuncSigTypeError::CountOverflow)?;
        if usize::try_from(total).ok() != Some(data.params.len()) {
            return Err(FuncSigTypeError::LengthMismatch {
                num_inputs,
                num_results,
                params: data.params.len(),
            });
        }
        Ok(Self(ty))
    }

    fn counts(self, ctx: &IrContext) -> (usize, usize) {
        let data = ctx.types.get(self.0);
        let num_inputs = read_count(&data.attrs, NUM_INPUTS_ATTR)
            .expect("validated func.func_sig must retain a valid num_inputs attribute");
        let num_results = read_count(&data.attrs, NUM_RESULTS_ATTR)
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
        &ctx.types.get(self.0).params[..num_inputs]
    }

    pub fn results<'a>(&self, ctx: &'a IrContext) -> &'a [TypeRef] {
        let (num_inputs, num_results) = self.counts(ctx);
        &ctx.types.get(self.0).params[num_inputs..num_inputs + num_results]
    }

    pub fn single_result(&self, ctx: &IrContext) -> Option<TypeRef> {
        self.results(ctx).first().copied()
    }

    /// Iterate function metadata without the input/result count delimiters.
    pub fn non_reserved_attrs<'a>(
        &self,
        ctx: &'a IrContext,
    ) -> impl Iterator<Item = (&'a Symbol, &'a Attribute)> {
        ctx.types.get(self.0).attrs.iter().filter(|(key, _)| {
            **key != Symbol::new(NUM_INPUTS_ATTR) && **key != Symbol::new(NUM_RESULTS_ATTR)
        })
    }

    /// Remove input/result count delimiters from owned function metadata before rebuilding it.
    pub fn remove_reserved_attrs(attrs: &mut AttributeMap) {
        attrs.remove(NUM_INPUTS_ATTR);
        attrs.remove(NUM_RESULTS_ATTR);
    }

    pub fn is_resultless(&self, ctx: &IrContext) -> bool {
        self.results(ctx).is_empty()
    }
}

impl DialectType for FuncSig {
    const DIALECT_NAME: &'static str = "func";
    const TYPE_NAME: &'static str = "func_sig";

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        if !Self::matches(ctx, ty) {
            return None;
        }
        Self::validate(ctx, ty).ok()
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

fn read_count(attrs: &AttributeMap, name: &'static str) -> Result<u32, FuncSigTypeError> {
    match attrs.get(name) {
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

/// Construct a canonical `func.func_sig` while preserving non-reserved attributes.
pub fn func_sig_with_attrs(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
    attrs: AttributeMap,
) -> FuncSig {
    assert!(
        !attrs.contains_key(NUM_INPUTS_ATTR) && !attrs.contains_key(NUM_RESULTS_ATTR),
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

    let mut builder = TypeDataBuilder::new(DIALECT_NAME(), FUNC_SIG())
        .params(inputs)
        .params(results);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let ty = ctx.types.intern(
        builder
            .attr(NUM_INPUTS_ATTR, Attribute::from(num_inputs))
            .attr(NUM_RESULTS_ATTR, Attribute::from(num_results))
            .build(),
    );
    FuncSig::validate(ctx, ty).expect("func.func_sig constructor must produce a valid type")
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

impl IndirectCallLikeModel for CallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        CallIndirect::signature(&self, ctx)
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
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

impl IndirectCallLikeModel for TailCallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        TailCallIndirect::signature(&self, ctx)
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl TailCallLike for TailCallIndirect {}

inventory::submit! {
    IndirectCallLikeOps::register::<CallIndirect>()
}

inventory::submit! {
    IndirectCallLikeOps::register::<TailCallIndirect>()
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
    set_indirect_call_signature_attribute(&mut ctx.op_mut(op).attributes, signature);
    true
}

fn set_indirect_call_signature_attribute(
    attributes: &mut crate::AttributeMap,
    signature: crate::TypeRef,
) {
    attributes.insert(
        crate::Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR),
        crate::Attribute::Type(signature),
    );
}

/// Remove the `func`-owned exact-signature attribute from copied metadata.
pub fn remove_indirect_call_signature(attributes: &mut crate::AttributeMap) {
    attributes.remove(INDIRECT_CALL_SIGNATURE_ATTR);
}

impl Func {
    /// Return the function body when this declaration has one.
    pub fn body_if_present(&self, ctx: &crate::IrContext) -> Option<crate::RegionRef> {
        ctx.op(self.op_ref()).regions.first().copied()
    }
}

// === Custom assembly format for func.func ===

/// Print func.func with decomposed signature:
/// `func.func @name(%arg: type, ...) -> return_type effects eff_type { body }`
fn print_func(
    h: &mut crate::printer::OpPrintHelper<'_, '_>,
    op: crate::OpRef,
    indent: usize,
) -> std::fmt::Result {
    use std::fmt::Write;

    let indent_str = " ".repeat(indent);

    // Extract sym_name before mutable operations
    let sym_name = {
        let data = h.ctx().op(op);
        data.attributes.get_symbol("sym_name")
    };

    write!(h, "{indent_str}func.func")?;

    // Function name
    if let Some(name) = sym_name {
        write!(h, " @")?;
        name.with_str(|s| {
            let needs_quoting =
                s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if needs_quoting {
                write!(h, "\"")?;
                crate::printer::write_escaped_string(h, s)?;
                write!(h, "\"")
            } else {
                write!(h, "{s}")
            }
        })?;
    }

    // Reset numbering for function body
    h.reset_numbering();

    // Extract region ref and func type info before mutable operations
    let region = {
        let data = h.ctx().op(op);
        assert!(
            data.regions.len() <= 1,
            "print_func: expected at most one region, found {}",
            data.regions.len(),
        );
        data.regions.first().copied()
    };

    // Extract the validated input/result lists from the func.func_sig type attribute.
    let type_info = {
        let data = h.ctx().op(op);
        data.attributes
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
    let preserve_type = h
        .ctx()
        .op(op)
        .attributes
        .get_type("type")
        .is_some_and(|ty| {
            h.ctx().types.get(ty).attrs.keys().any(|key| {
                *key != crate::Symbol::new(crate::dialect::func::NUM_INPUTS_ATTR)
                    && *key != crate::Symbol::new(crate::dialect::func::NUM_RESULTS_ATTR)
            })
        });
    let extra_attrs: Vec<_> = {
        let data = h.ctx().op(op);
        data.attributes
            .iter()
            .filter(|(k, _)| {
                **k != crate::Symbol::new("sym_name")
                    && (preserve_type || **k != crate::Symbol::new("type"))
            })
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    };
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
        // Body
        writeln!(h, " {{")?;
        h.print_region_eliding_entry(region, indent + 2)?;
        writeln!(h, "{indent_str}}}")?;
    } else {
        writeln!(h)?;
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
        dialect: "func",
        op_name: "func",
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

inventory::submit! {
    crate::op_interface::OpAsmFormat {
        dialect: "func",
        op_name: "func",
        print_fn: print_func,
        parse_fn: parse_func,
    }
}

impl crate::op_interface::CallableOwnerModel for Func {
    fn callable_signature(self, ctx: &crate::IrContext) -> Option<crate::dialect::func::FuncSig> {
        ctx.op(self.op_ref())
            .attributes
            .get_type("type")
            .and_then(|ty| crate::dialect::func::FuncSig::from_type_ref(ctx, ty))
    }
}
inventory::submit! { crate::op_interface::CallableOwnerOps::register::<Func>() }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_interface::IndirectCallLikeOps;
    use crate::ops::DialectOp;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

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
        assert!(call.signature(&ctx).is_some(), "declared signature");

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("!t0 = func.func_sig<(core.i32) -> core.i32>"),
            "{printed}"
        );
        assert!(printed.contains("signature = !t0"), "{printed}");
        assert!(!printed.contains("func.indirect_call_signature"));
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

        assert!(IndirectCallLikeOps::get(&ctx, ordinary_op).is_some());
        assert!(IndirectCallLikeOps::get(&ctx, tail_op).is_some());
        assert!(IndirectCallLikeOps::exact_signature(&ctx, ordinary_op).is_some());
        assert!(IndirectCallLikeOps::exact_signature(&ctx, tail_op).is_some());
        for op in [ordinary_op, tail_op] {
            let operands = ctx.op_operands(op);
            assert_eq!(IndirectCallLikeOps::callee(&ctx, op), Some(operands[0]));
            assert_eq!(
                IndirectCallLikeOps::arguments(&ctx, op),
                Some(&operands[1..])
            );
        }
        assert!(IndirectCallLikeOps::get(&ctx, direct_op).is_none());
        assert_eq!(IndirectCallLikeOps::callee(&ctx, direct_op), None);
        assert_eq!(IndirectCallLikeOps::arguments(&ctx, direct_op), None);
    }
}

#[cfg(test)]
mod result_list_tests {
    use super::*;
    use crate::dialect::core;
    use crate::parser::parse_module;
    use crate::printer::print_module;
    use crate::rewrite::Module;
    use crate::validation::validate_all;
    use crate::{IrContext, Symbol};

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
                        ctx.op(a).attributes.get_type("type"),
                        ctx.op(b).attributes.get_type("type")
                    );
                    assert_eq!(print_module(&ctx, copy), printed);
                }
            }
        }
        for input in [
            "core.module @m { func.func {sym_name = @f, type = func.func_sig<(core.i32) -> ()> {tag = @kept, nested = [core.i64]}} }",
            "core.module @m { func.func @f(%x: core.i32) attributes {type = func.func_sig<(core.i32) -> ()> {tag = @kept, nested = [core.i64]}} { func.return } }",
        ] {
            let mut ctx = IrContext::new();
            let op = parse_module(&mut ctx, input).unwrap();
            let printed = print_module(&ctx, op);
            assert!(printed.contains("tag = @kept"), "{printed}");
            let copy = parse_module(&mut ctx, &printed).unwrap();
            let a = Module::new(&ctx, op).unwrap().ops(&ctx)[0];
            let b = Module::new(&ctx, copy).unwrap().ops(&ctx)[0];
            assert_eq!(
                ctx.op(a).attributes.get_type("type"),
                ctx.op(b).attributes.get_type("type")
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
        let nil = core::nil(&mut ctx).as_type_ref();
        let signature = func_sig(&mut ctx, [], []).as_type_ref();
        let callee = constant(&mut ctx, loc, signature, Symbol::new("f")).result(&ctx);
        for results in [vec![], vec![nil]] {
            let direct = call(&mut ctx, loc, [], results.clone(), Symbol::new("f"));
            let indirect = call_indirect(&mut ctx, loc, callee, [], results.clone(), None);
            assert_eq!(direct.call_result_types(&ctx), results);
            assert_eq!(indirect.call_result_types(&ctx), results);
            assert_eq!(direct.single_result(&ctx).is_some(), results.len() == 1);
        }
    }

    fn verify(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        validate_all(&ctx, module).to_string()
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
            func.tail_call_indirect %k, %x
          }
          func.func @one(%k: func.func_sig<(core.i32) -> core.i32>, %x: core.i32) -> core.i32 {
            %r = func.call_indirect %k, %x : core.i32
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
                "exact indirect signature differs",
            ),
            (
                "callee = @sink",
                "callee = 1",
                "requires symbol callee attribute",
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
            test.lambda {type = func.func_sig<() -> core.nil>} { func.tail_call_indirect %k }
            func.unreachable
          }
        }",
        );
        let result = crate::validation::validate_all(&ctx, module);
        assert!(result.is_ok(), "{result}");
    }

    #[test]
    fn missing_runtime_symbols_do_not_hide_known_invalid_declarations() {
        for declaration in [
            "func.func @f() func.func @f()",
            "test.object {sym_name = @f}",
        ] {
            let mut ctx = crate::IrContext::new();
            let input = format!(
                "core.module @m {{ {declaration} func.func @run() {{ func.call {{callee = @runtime}} func.call {{callee = @f}} func.return }} }}"
            );
            let module = crate::parser::parse_test_module(&mut ctx, &input);
            let result = crate::validation::validate_all(&ctx, module);
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
                "%a, %b = func.call_indirect %k : core.i32, core.i32\nfunc.return",
                "multiple call results",
            ),
            (
                "%a = func.call {callee = @sink} : core.i32\nfunc.return",
                "call result list mismatch",
            ),
            (
                "%a = func.call_indirect %k : core.i32\nfunc.return",
                "call result list mismatch",
            ),
            ("func.return %k", "return count mismatch"),
        ] {
            let mut ctx = crate::IrContext::new();
            let input = format!(
                "core.module @m {{ func.func @sink() func.func @run(%k: func.func_sig<() -> ()>) {{ func.call {{callee = @runtime}} {body} }} }}"
            );
            let module = crate::parser::parse_test_module(&mut ctx, &input);
            let result = crate::validation::validate_all(&ctx, module);
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
