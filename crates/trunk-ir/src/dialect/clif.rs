//! Arena-based clif dialect.

use crate::op_interface::{IndirectCallLikeModel, IndirectCallLikeOps};
use crate::ops::{DialectOp, DialectType};
use crate::types::{Attribute, AttributeMap, TypeDataBuilder};
use itertools::Itertools;

#[trunk_ir::dialect]
mod clif {
    // Module
    fn func<S: FuncSig>(sym_name: Attr<Symbol>, r#type: Attr<S::Type>) {
        #[region(body?)]
        {}
    }

    fn call(callee: Attr<Symbol>, args: Variadic<_>) -> Variadic<_> {}

    #[verify]
    fn call_indirect<S: FuncSig>(
        sig: Attr<S::Type>,
        callee: Value<_>,
        args: Values<S::Inputs>,
    ) -> Variadic<_> {
    }

    fn r#return(values: Variadic<_>) {}

    // Constants
    fn iconst(value: Attr<i64>) -> Value<_> {}

    fn f32const(value: Attr<f32>) -> Value<_> {}

    fn f64const(value: Attr<f64>) -> Value<_> {}

    // Integer arithmetic
    fn iadd(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn isub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn imul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn sdiv(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn udiv(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn srem(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn urem(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn ineg(operand: Value<_>) -> Value<_> {}

    // Float arithmetic
    fn fadd(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn fsub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn fmul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn fdiv(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn fneg(operand: Value<_>) -> Value<_> {}

    // Comparisons
    fn icmp(cond: Attr<Symbol>, lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    fn fcmp(cond: Attr<Symbol>, lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // Bitwise
    fn band(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn bor(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn bxor(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn ishl(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn sshr(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn ushr(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // Control flow
    fn brif(cond: Value<_>) {
        #[successor(then_dest)]
        {}
        #[successor(else_dest)]
        {}
    }

    fn jump(args: Variadic<_>) {
        #[successor(dest)]
        {}
    }

    fn br_table(table: Attr<_>, index: Value<_>) {}

    fn trap(code: Attr<Symbol>) {}

    fn return_call(callee: Attr<Symbol>, args: Variadic<_>) {}

    fn return_call_indirect<S: FuncSig>(
        sig: Attr<S::Type>,
        callee: Value<_>,
        args: Values<S::Inputs>,
    ) {
    }

    // Memory
    fn load(offset: Attr<i32>, addr: Value<_>) -> Value<_> {}

    fn store(offset: Attr<i32>, value: Value<_>, addr: Value<_>) {}

    fn atomic_rmw(
        op: Attr<Symbol>,
        offset: Attr<i32>,
        addr: Value<_>,
        value: Value<_>,
    ) -> Value<_> {
    }

    fn stack_slot(size: Attr<u32>, align: Attr<u32>) -> Value<_> {}

    fn stack_addr(slot: Value<_>) -> Value<_> {}

    fn symbol_addr(sym: Attr<Symbol>) -> Value<_> {}

    // Type conversions
    fn ireduce(operand: Value<_>) -> Value<_> {}
    fn uextend(operand: Value<_>) -> Value<_> {}
    fn sextend(operand: Value<_>) -> Value<_> {}
    fn fpromote(operand: Value<_>) -> Value<_> {}
    fn fdemote(operand: Value<_>) -> Value<_> {}
    fn fcvt_to_sint(operand: Value<_>) -> Value<_> {}
    fn fcvt_from_sint(operand: Value<_>) -> Value<_> {}
    fn fcvt_to_uint(operand: Value<_>) -> Value<_> {}
    fn fcvt_from_uint(operand: Value<_>) -> Value<_> {}
}

const INDIRECT_CALL_SIGNATURE_ATTR: &str = "sig";

pub const NUM_INPUTS_ATTR: &str = "num_inputs";
pub const NUM_RESULTS_ATTR: &str = "num_results";

#[allow(non_snake_case)]
#[inline]
pub fn FUNC_SIG() -> crate::Symbol {
    crate::Symbol::new("func_sig")
}

/// Why a name-matching `clif.func_sig` does not satisfy its storage invariant.
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
                "num_inputs ({num_inputs}) + num_results ({num_results}) must equal params length ({params})",
            ),
        }
    }
}

impl std::error::Error for FuncSigTypeError {}

/// Validated wrapper for an input-first, zero-or-more-result `clif.func_sig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSig(crate::TypeRef);

crate::impl_func_sig_constraint!(FuncSig, "clif.func_sig");

impl FuncSig {
    pub(crate) fn validate(
        ctx: &crate::IrContext,
        ty: crate::TypeRef,
    ) -> Result<Self, FuncSigTypeError> {
        let data = ctx.get_type(ty);
        debug_assert!(data.dialect == DIALECT_NAME() && data.name == FUNC_SIG());
        let num_inputs = read_count(&data.attrs, NUM_INPUTS_ATTR)?;
        let num_results = read_count(&data.attrs, NUM_RESULTS_ATTR)?;
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

    fn counts(self, ctx: &crate::IrContext) -> (usize, usize) {
        let data = ctx.get_type(self.0);
        let inputs = usize::try_from(
            read_count(&data.attrs, NUM_INPUTS_ATTR)
                .expect("validated clif.func_sig must retain num_inputs"),
        )
        .expect("u32 must fit usize");
        let results = usize::try_from(
            read_count(&data.attrs, NUM_RESULTS_ATTR)
                .expect("validated clif.func_sig must retain num_results"),
        )
        .expect("u32 must fit usize");
        (inputs, results)
    }

    pub fn inputs(self, ctx: &crate::IrContext) -> &[crate::TypeRef] {
        let (inputs, _) = self.counts(ctx);
        &ctx.get_type(self.0).params[..inputs]
    }

    pub fn results(self, ctx: &crate::IrContext) -> &[crate::TypeRef] {
        let (inputs, results) = self.counts(ctx);
        &ctx.get_type(self.0).params[inputs..inputs + results]
    }

    pub fn is_resultless(self, ctx: &crate::IrContext) -> bool {
        self.results(ctx).is_empty()
    }

    pub fn single_result(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        let results = self.results(ctx);
        (results.len() == 1).then(|| results[0])
    }

    pub fn non_reserved_attrs(
        self,
        ctx: &crate::IrContext,
    ) -> impl Iterator<Item = (&crate::Symbol, &Attribute)> {
        ctx.get_type(self.0).attrs.iter().filter(|(key, _)| {
            **key != crate::Symbol::new(NUM_INPUTS_ATTR)
                && **key != crate::Symbol::new(NUM_RESULTS_ATTR)
        })
    }

    pub fn remove_reserved_attrs(attrs: &mut AttributeMap) {
        attrs.remove(NUM_INPUTS_ATTR);
        attrs.remove(NUM_RESULTS_ATTR);
    }
}

impl DialectType for FuncSig {
    const DIALECT_NAME: &'static str = "clif";
    const TYPE_NAME: &'static str = "func_sig";

    fn from_type_ref(ctx: &crate::IrContext, ty: crate::TypeRef) -> Option<Self> {
        Self::matches(ctx, ty)
            .then(|| Self::validate(ctx, ty).ok())
            .flatten()
    }

    fn as_type_ref(&self) -> crate::TypeRef {
        self.0
    }
}

impl From<FuncSig> for crate::TypeRef {
    fn from(value: FuncSig) -> Self {
        value.0
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

/// Construct a canonical target-owned `clif.func_sig`.
pub fn func_sig(
    ctx: &mut crate::IrContext,
    inputs: impl IntoIterator<Item = crate::TypeRef>,
    results: impl IntoIterator<Item = crate::TypeRef>,
) -> FuncSig {
    func_sig_with_attrs(ctx, inputs, results, AttributeMap::new())
}

/// Construct a canonical target signature while preserving non-reserved attributes.
pub fn func_sig_with_attrs(
    ctx: &mut crate::IrContext,
    inputs: impl IntoIterator<Item = crate::TypeRef>,
    results: impl IntoIterator<Item = crate::TypeRef>,
    attrs: AttributeMap,
) -> FuncSig {
    assert!(
        !attrs.contains_key(NUM_INPUTS_ATTR) && !attrs.contains_key(NUM_RESULTS_ATTR),
        "clif.func_sig count attributes are reserved"
    );
    let inputs: Vec<_> = inputs.into_iter().collect();
    let results: Vec<_> = results.into_iter().collect();
    let num_inputs = u32::try_from(inputs.len()).expect("clif.func_sig input count exceeds u32");
    let num_results = u32::try_from(results.len()).expect("clif.func_sig result count exceeds u32");
    let mut builder = TypeDataBuilder::new(DIALECT_NAME(), FUNC_SIG())
        .params(inputs)
        .params(results);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let ty = ctx.intern_type(
        builder
            .attr(NUM_INPUTS_ATTR, Attribute::from(num_inputs))
            .attr(NUM_RESULTS_ATTR, Attribute::from(num_results))
            .build(),
    );
    FuncSig(ty)
}

impl CallIndirect {
    /// Results match the signature's result list, or its projection without
    /// zero-width `core.nil` slots, which the emitter does not materialize.
    fn verify(self, ctx: &crate::IrContext) -> Result<(), String> {
        let signature =
            FuncSig::from_type_ref(ctx, self.sig(ctx)).expect("schema-verified clif.func_sig");
        let expected = signature.results(ctx);
        let actual = ctx.op_result_types(self.op_ref());
        let runtime = expected
            .iter()
            .copied()
            .filter(|&ty| !crate::dialect::core::Nil::matches(ctx, ty));
        if actual == expected || actual.iter().copied().eq(runtime) {
            return Ok(());
        }
        Err(format!(
            "results ({}) match neither the signature results ({}) nor their runtime projection",
            actual
                .iter()
                .map(|&ty| crate::printer::print_type(ctx, ty))
                .format(", "),
            expected
                .iter()
                .map(|&ty| crate::printer::print_type(ctx, ty))
                .format(", "),
        ))
    }
}

/// Read the exact signature without assuming the operation passed its schema;
/// interface queries must fail closed on malformed IR.
fn exact_indirect_signature(ctx: &crate::IrContext, op: crate::OpRef) -> Option<crate::TypeRef> {
    ctx.op(op)
        .attributes
        .get_type("sig")
        .filter(|&ty| FuncSig::from_type_ref(ctx, ty).is_some())
}

impl IndirectCallLikeModel for CallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        exact_indirect_signature(ctx, self.op_ref())
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl IndirectCallLikeModel for ReturnCallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        exact_indirect_signature(ctx, self.op_ref())
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

inventory::submit! {
    IndirectCallLikeOps::register::<CallIndirect>()
}

inventory::submit! {
    IndirectCallLikeOps::register::<ReturnCallIndirect>()
}

/// Attach the required exact callable contract to a `clif` indirect transfer.
///
/// Returns `false` without mutation when the operation is not a `clif`
/// indirect call or the supplied type is not a `clif.func_sig` contract.
pub fn set_indirect_call_signature(
    ctx: &mut crate::IrContext,
    op: crate::OpRef,
    signature: crate::TypeRef,
) -> bool {
    if FuncSig::from_type_ref(ctx, signature).is_none()
        || (CallIndirect::from_op(ctx, op).is_err()
            && ReturnCallIndirect::from_op(ctx, op).is_err())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::func;
    use crate::op_interface::IndirectCallLikeOps;
    use crate::ops::DialectType;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

    #[test]
    fn indirect_call_interface_uses_clif_owned_sig() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @ordinary(%callee: core.ptr, %value: core.i32) -> core.i32 {
    %result = clif.call_indirect %callee, %value {sig = clif.func_sig<(core.i32) -> core.i32>} : core.i32
    clif.return %result
  }
  clif.func @tail(%callee: core.ptr, %value: core.i32) -> core.nil {
    clif.return_call_indirect %callee, %value {sig = clif.func_sig<(core.i32) -> core.nil>}
  }
  clif.func @direct() -> core.nil {
    clif.return
  }
}"#,
        );
        let functions = module.ops(&ctx);
        let body_op = |index| {
            let body = ctx.op(functions[index]).regions[0];
            ctx.block(ctx.region(body).blocks[0]).ops[0]
        };

        let ordinary = body_op(0);
        let tail = body_op(1);
        let direct = body_op(2);
        assert!(IndirectCallLikeOps::exact_signature(&ctx, ordinary).is_some());
        assert!(IndirectCallLikeOps::exact_signature(&ctx, tail).is_some());
        let nil = crate::dialect::core::nil(&mut ctx).as_type_ref();
        let shared_sig = func::func_sig(&mut ctx, [], [nil]).as_type_ref();
        assert!(
            !set_indirect_call_signature(&mut ctx, ordinary, shared_sig),
            "a shared signature must not replace the target-owned contract"
        );
        for op in [ordinary, tail] {
            let operands = ctx.op_operands(op);
            assert_eq!(IndirectCallLikeOps::callee(&ctx, op), Some(operands[0]));
            assert_eq!(
                IndirectCallLikeOps::arguments(&ctx, op),
                Some(&operands[1..])
            );
        }
        assert!(IndirectCallLikeOps::get(&ctx, direct).is_none());
        assert_eq!(IndirectCallLikeOps::callee(&ctx, direct), None);
        assert_eq!(IndirectCallLikeOps::arguments(&ctx, direct), None);

        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("clif.call_indirect"));
        assert!(printed.contains("sig = clif.func_sig<(core.i32) -> core.i32>"));
    }

    #[test]
    fn func_sig_owns_zero_and_multiple_result_lists_and_metadata() {
        let mut ctx = crate::IrContext::new();
        let i32 = ctx.intern_type(
            TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i32")).build(),
        );
        let i64 = ctx.intern_type(
            TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i64")).build(),
        );
        let mut attrs = AttributeMap::new();
        attrs.insert(crate::Symbol::new("kept"), Attribute::Type(i64));
        let zero = func_sig(&mut ctx, [i32], []).as_type_ref();
        let one = func_sig(&mut ctx, [i32], [i64]).as_type_ref();
        let many = func_sig_with_attrs(&mut ctx, [i32], [i32, i64], attrs).as_type_ref();
        assert!(
            FuncSig::from_type_ref(&ctx, zero)
                .unwrap()
                .is_resultless(&ctx)
        );
        assert_eq!(
            FuncSig::from_type_ref(&ctx, one).unwrap().inputs(&ctx),
            [i32]
        );
        assert_eq!(
            FuncSig::from_type_ref(&ctx, one).unwrap().results(&ctx),
            [i64]
        );
        assert_ne!(one, func::func_sig(&mut ctx, [i32], [i64]).as_type_ref());
        let many_sig = FuncSig::from_type_ref(&ctx, many).unwrap();
        assert_eq!(many_sig.inputs(&ctx), [i32]);
        assert_eq!(many_sig.results(&ctx), [i32, i64]);
        assert_eq!(many_sig.non_reserved_attrs(&ctx).count(), 1);
        let mut same_attrs = AttributeMap::new();
        same_attrs.insert(crate::Symbol::new("kept"), Attribute::Type(i64));
        assert_eq!(
            many,
            func_sig_with_attrs(&mut ctx, [i32], [i32, i64], same_attrs).as_type_ref()
        );
    }

    #[test]
    fn func_sig_round_trips_through_type_aliases() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !scalar = core.i32
  !callable = clif.func_sig<(!scalar) -> (!scalar, !scalar)> {nested = [!scalar]}
}"#,
        );
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains(
                "!callable = clif.func_sig<(!scalar) -> (!scalar, !scalar)> {nested = [!scalar]}"
            ),
            "{printed}"
        );
        let mut reparsed = crate::IrContext::new();
        let copy = parse_test_module(&mut reparsed, &printed);
        assert_eq!(print_module(&reparsed, copy.op()), printed);
    }

    #[test]
    fn func_sig_rejects_malformed_delimiters_without_slicing() {
        let mut ctx = crate::IrContext::new();
        let i32 = ctx.intern_type(
            TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i32")).build(),
        );
        let malformed = ctx.intern_type(
            TypeDataBuilder::new(DIALECT_NAME(), FUNC_SIG())
                .param(i32)
                .attr(NUM_INPUTS_ATTR, Attribute::Int(2))
                .attr(NUM_RESULTS_ATTR, Attribute::Int(1))
                .build(),
        );
        assert!(FuncSig::from_type_ref(&ctx, malformed).is_none());
        let missing = ctx.intern_type(
            TypeDataBuilder::new(DIALECT_NAME(), FUNC_SIG())
                .param(i32)
                .attr(NUM_INPUTS_ATTR, Attribute::Int(1))
                .build(),
        );
        assert_eq!(
            FuncSig::validate(&ctx, missing),
            Err(FuncSigTypeError::MissingCount(NUM_RESULTS_ATTR))
        );

        let wrong_kind = ctx.intern_type(
            TypeDataBuilder::new(DIALECT_NAME(), FUNC_SIG())
                .param(i32)
                .attr(
                    NUM_INPUTS_ATTR,
                    Attribute::Symbol(crate::Symbol::new("one")),
                )
                .attr(NUM_RESULTS_ATTR, Attribute::Int(0))
                .build(),
        );
        assert_eq!(
            FuncSig::validate(&ctx, wrong_kind),
            Err(FuncSigTypeError::InvalidCount(NUM_INPUTS_ATTR))
        );

        let overflow = ctx.intern_type(
            TypeDataBuilder::new(DIALECT_NAME(), FUNC_SIG())
                .attr(NUM_INPUTS_ATTR, Attribute::Int(i128::from(u32::MAX)))
                .attr(NUM_RESULTS_ATTR, Attribute::Int(1))
                .build(),
        );
        assert_eq!(
            FuncSig::validate(&ctx, overflow),
            Err(FuncSigTypeError::CountOverflow)
        );
    }

    #[test]
    #[should_panic(expected = "count attributes are reserved")]
    fn func_sig_constructor_rejects_reserved_count_attributes() {
        let mut ctx = crate::IrContext::new();
        let mut attrs = AttributeMap::new();
        attrs.insert(NUM_INPUTS_ATTR.into(), Attribute::Int(0));
        let _ = func_sig_with_attrs(&mut ctx, [], [], attrs);
    }
}
