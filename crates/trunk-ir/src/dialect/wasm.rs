//! Arena-based wasm dialect.

use crate::op_interface::{IndirectCallLikeModel, IndirectCallLikeOps};
use crate::ops::{DialectOp, DialectType};
use crate::{Attribute, AttributeMap, IrContext, Symbol, TypeDataBuilder, TypeRef};

crate::register_isolated_op!(wasm.func);

#[trunk_ir::dialect]
mod wasm {
    // Types (abstract heap types)
    struct Anyref;
    struct Eqref;
    struct I31ref;
    struct Structref;
    struct Arrayref;
    struct Funcref;
    struct Externref;

    // Control flow
    fn block() -> result {
        #[region(body)]
        {}
    }

    fn r#loop(#[rest] init: ()) -> result {
        #[region(body)]
        {}
    }

    fn r#if(cond: ()) -> result {
        #[region(then_region)]
        {}
        #[region(else_region)]
        {}
    }

    #[attr(target: u32)]
    fn br() {}

    #[attr(target: u32)]
    fn br_if(cond: ()) {}

    fn r#return(#[rest] values: ()) {}
    fn r#yield(value: ()) {}
    fn drop(value: ()) {}

    // Functions
    #[attr(callee: Symbol)]
    #[rest_results]
    fn call(#[rest] args: ()) -> results {}

    #[attr(type_idx: u32, table: u32, signature?: Type)]
    #[rest_results]
    fn call_indirect(#[rest] args: ()) -> results {}

    #[attr(callee: Symbol)]
    fn return_call(#[rest] args: ()) {}

    #[attr(type_idx: u32, table: u32, signature?: Type)]
    fn return_call_indirect(#[rest] args: ()) {}

    fn unreachable() {}
    fn nop() -> result {}

    // Module
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)]
        {}
    }

    #[attr(module: Symbol, name: Symbol, sym_name: Symbol, r#type: Type)]
    fn import_func() {}

    #[attr(name: String, func: Symbol)]
    fn export_func() {}

    #[attr(name: String, index: u32)]
    fn export_memory() {}

    #[attr(min: u32, max: u32, shared: bool, memory64: bool)]
    fn memory() {}

    #[attr(offset: u32, bytes: any, passive: bool)]
    fn data() {}

    #[attr(reftype: Symbol, min: u32, max?: u32)]
    fn table() {}

    #[attr(table?: u32, offset?: u32)]
    fn elem() {
        #[region(funcs)]
        {}
    }

    #[attr(valtype: Symbol, mutable: bool, init: any)]
    fn global() {}

    #[attr(index: u32)]
    fn global_get() -> result {}

    #[attr(index: u32)]
    fn global_set(value: ()) {}

    // i32
    #[attr(value: i32)]
    fn i32_const() -> result {}

    fn i32_add(lhs: (), rhs: ()) -> result {}
    fn i32_sub(lhs: (), rhs: ()) -> result {}
    fn i32_mul(lhs: (), rhs: ()) -> result {}
    fn i32_div_s(lhs: (), rhs: ()) -> result {}
    fn i32_div_u(lhs: (), rhs: ()) -> result {}
    fn i32_rem_s(lhs: (), rhs: ()) -> result {}
    fn i32_rem_u(lhs: (), rhs: ()) -> result {}

    fn i32_eq(lhs: (), rhs: ()) -> result {}
    fn i32_ne(lhs: (), rhs: ()) -> result {}
    fn i32_lt_s(lhs: (), rhs: ()) -> result {}
    fn i32_lt_u(lhs: (), rhs: ()) -> result {}
    fn i32_le_s(lhs: (), rhs: ()) -> result {}
    fn i32_le_u(lhs: (), rhs: ()) -> result {}
    fn i32_gt_s(lhs: (), rhs: ()) -> result {}
    fn i32_gt_u(lhs: (), rhs: ()) -> result {}
    fn i32_ge_s(lhs: (), rhs: ()) -> result {}
    fn i32_ge_u(lhs: (), rhs: ()) -> result {}

    fn i32_and(lhs: (), rhs: ()) -> result {}
    fn i32_or(lhs: (), rhs: ()) -> result {}
    fn i32_xor(lhs: (), rhs: ()) -> result {}
    fn i32_shl(lhs: (), rhs: ()) -> result {}
    fn i32_shr_s(lhs: (), rhs: ()) -> result {}
    fn i32_shr_u(lhs: (), rhs: ()) -> result {}

    // i64
    #[attr(value: i64)]
    fn i64_const() -> result {}

    fn i64_add(lhs: (), rhs: ()) -> result {}
    fn i64_sub(lhs: (), rhs: ()) -> result {}
    fn i64_mul(lhs: (), rhs: ()) -> result {}
    fn i64_div_s(lhs: (), rhs: ()) -> result {}
    fn i64_div_u(lhs: (), rhs: ()) -> result {}
    fn i64_rem_s(lhs: (), rhs: ()) -> result {}
    fn i64_rem_u(lhs: (), rhs: ()) -> result {}

    fn i64_eq(lhs: (), rhs: ()) -> result {}
    fn i64_ne(lhs: (), rhs: ()) -> result {}
    fn i64_lt_s(lhs: (), rhs: ()) -> result {}
    fn i64_lt_u(lhs: (), rhs: ()) -> result {}
    fn i64_le_s(lhs: (), rhs: ()) -> result {}
    fn i64_le_u(lhs: (), rhs: ()) -> result {}
    fn i64_gt_s(lhs: (), rhs: ()) -> result {}
    fn i64_gt_u(lhs: (), rhs: ()) -> result {}
    fn i64_ge_s(lhs: (), rhs: ()) -> result {}
    fn i64_ge_u(lhs: (), rhs: ()) -> result {}

    fn i64_and(lhs: (), rhs: ()) -> result {}
    fn i64_or(lhs: (), rhs: ()) -> result {}
    fn i64_xor(lhs: (), rhs: ()) -> result {}
    fn i64_shl(lhs: (), rhs: ()) -> result {}
    fn i64_shr_s(lhs: (), rhs: ()) -> result {}
    fn i64_shr_u(lhs: (), rhs: ()) -> result {}

    // f32
    #[attr(value: f32)]
    fn f32_const() -> result {}

    fn f32_add(lhs: (), rhs: ()) -> result {}
    fn f32_sub(lhs: (), rhs: ()) -> result {}
    fn f32_mul(lhs: (), rhs: ()) -> result {}
    fn f32_div(lhs: (), rhs: ()) -> result {}
    fn f32_neg(operand: ()) -> result {}

    fn f32_eq(lhs: (), rhs: ()) -> result {}
    fn f32_ne(lhs: (), rhs: ()) -> result {}
    fn f32_lt(lhs: (), rhs: ()) -> result {}
    fn f32_le(lhs: (), rhs: ()) -> result {}
    fn f32_gt(lhs: (), rhs: ()) -> result {}
    fn f32_ge(lhs: (), rhs: ()) -> result {}

    // f64
    #[attr(value: f64)]
    fn f64_const() -> result {}

    fn f64_add(lhs: (), rhs: ()) -> result {}
    fn f64_sub(lhs: (), rhs: ()) -> result {}
    fn f64_mul(lhs: (), rhs: ()) -> result {}
    fn f64_div(lhs: (), rhs: ()) -> result {}
    fn f64_neg(operand: ()) -> result {}

    fn f64_eq(lhs: (), rhs: ()) -> result {}
    fn f64_ne(lhs: (), rhs: ()) -> result {}
    fn f64_lt(lhs: (), rhs: ()) -> result {}
    fn f64_le(lhs: (), rhs: ()) -> result {}
    fn f64_gt(lhs: (), rhs: ()) -> result {}
    fn f64_ge(lhs: (), rhs: ()) -> result {}

    // Local variables
    #[attr(index: u32)]
    fn local_get() -> result {}

    #[attr(index: u32)]
    fn local_set(value: ()) {}

    #[attr(index: u32)]
    fn local_tee(value: ()) -> result {}

    // GC structs
    #[attr(type_idx: u32)]
    fn struct_new(#[rest] fields: ()) -> result {}

    #[attr(type_idx: u32, field_idx: u32)]
    fn struct_get(r#ref: ()) -> result {}

    #[attr(type_idx: u32, field_idx: u32)]
    fn struct_set(r#ref: (), value: ()) {}

    // GC arrays
    #[attr(type_idx: u32)]
    fn array_new(size: (), init: ()) -> result {}

    #[attr(type_idx: u32)]
    fn array_new_default(size: ()) -> result {}

    #[attr(type_idx: u32, data_idx: u32)]
    fn array_new_data(offset: (), size: ()) -> result {}

    #[attr(data_idx: u32, offset: u32, len: u32)]
    fn bytes_from_data() -> result {}

    #[attr(type_idx: u32)]
    fn array_get(r#ref: (), index: ()) -> result {}

    #[attr(type_idx: u32)]
    fn array_get_s(r#ref: (), index: ()) -> result {}

    #[attr(type_idx: u32)]
    fn array_get_u(r#ref: (), index: ()) -> result {}

    #[attr(type_idx: u32)]
    fn array_set(r#ref: (), index: (), value: ()) {}

    fn array_len(r#ref: ()) -> result {}

    #[attr(dst_type_idx: u32, src_type_idx: u32)]
    fn array_copy(dst: (), dst_offset: (), src: (), src_offset: (), len: ()) {}

    // References
    #[attr(heap_type: Symbol, type_idx?: u32)]
    fn ref_null() -> result {}

    #[attr(func_name: Symbol)]
    fn ref_func() -> result {}

    fn ref_is_null(r#ref: ()) -> result {}

    #[attr(target_type: Type, type_idx?: u32)]
    fn ref_cast(r#ref: ()) -> result {}

    #[attr(target_type: Type, type_idx?: u32)]
    fn ref_test(r#ref: ()) -> result {}

    // i31ref
    fn ref_i31(value: ()) -> result {}
    fn i31_get_s(r#ref: ()) -> result {}
    fn i31_get_u(r#ref: ()) -> result {}

    // Type conversions (integer)
    fn i32_wrap_i64(operand: ()) -> result {}
    fn i64_extend_i32_s(operand: ()) -> result {}
    fn i64_extend_i32_u(operand: ()) -> result {}

    // Type conversions (float to int)
    fn i32_trunc_f32_s(operand: ()) -> result {}
    fn i32_trunc_f32_u(operand: ()) -> result {}
    fn i32_trunc_f64_s(operand: ()) -> result {}
    fn i32_trunc_f64_u(operand: ()) -> result {}
    fn i64_trunc_f32_s(operand: ()) -> result {}
    fn i64_trunc_f32_u(operand: ()) -> result {}
    fn i64_trunc_f64_s(operand: ()) -> result {}
    fn i64_trunc_f64_u(operand: ()) -> result {}

    // Type conversions (int to float)
    fn f32_convert_i32_s(operand: ()) -> result {}
    fn f32_convert_i32_u(operand: ()) -> result {}
    fn f32_convert_i64_s(operand: ()) -> result {}
    fn f32_convert_i64_u(operand: ()) -> result {}
    fn f64_convert_i32_s(operand: ()) -> result {}
    fn f64_convert_i32_u(operand: ()) -> result {}
    fn f64_convert_i64_s(operand: ()) -> result {}
    fn f64_convert_i64_u(operand: ()) -> result {}

    // Float conversions
    fn f32_demote_f64(operand: ()) -> result {}
    fn f64_promote_f32(operand: ()) -> result {}

    // Bitcast
    fn i32_reinterpret_f32(operand: ()) -> result {}
    fn i64_reinterpret_f64(operand: ()) -> result {}
    fn f32_reinterpret_i32(operand: ()) -> result {}
    fn f64_reinterpret_i64(operand: ()) -> result {}

    // Linear memory
    #[attr(memory: u32)]
    fn memory_size() -> result {}

    #[attr(memory: u32)]
    fn memory_grow(delta: ()) -> result {}

    // Memory loads (full width)
    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_load(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn f32_load(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn f64_load(addr: ()) -> result {}

    // Memory loads (partial width i32)
    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_load8_s(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_load8_u(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_load16_s(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_load16_u(addr: ()) -> result {}

    // Memory loads (partial width i64)
    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load8_s(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load8_u(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load16_s(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load16_u(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load32_s(addr: ()) -> result {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_load32_u(addr: ()) -> result {}

    // Memory stores (full width)
    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_store(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_store(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn f32_store(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn f64_store(addr: (), value: ()) {}

    // Memory stores (partial width)
    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_store8(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i32_store16(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_store8(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_store16(addr: (), value: ()) {}

    #[attr(offset: u32, align: u32, memory: u32)]
    fn i64_store32(addr: (), value: ()) {}
}

/// Reserved delimiter attribute for the number of Wasm signature inputs.
pub const NUM_INPUTS_ATTR: &str = "num_inputs";
/// Reserved delimiter attribute for the number of Wasm signature results.
pub const NUM_RESULTS_ATTR: &str = "num_results";

#[allow(non_snake_case)]
#[inline]
pub fn FUNC_SIG() -> Symbol {
    Symbol::new("func_sig")
}

/// Why a name-matching `wasm.func_sig` does not satisfy its storage invariant.
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

/// Validated wrapper for an input-first, zero-or-more-result `wasm.func_sig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSig(TypeRef);

impl FuncSig {
    pub(crate) fn validate(ctx: &IrContext, ty: TypeRef) -> Result<Self, FuncSigTypeError> {
        let data = ctx.types.get(ty);
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

    fn counts(self, ctx: &IrContext) -> (usize, usize) {
        let data = ctx.types.get(self.0);
        let inputs = usize::try_from(
            read_count(&data.attrs, NUM_INPUTS_ATTR)
                .expect("validated wasm.func_sig must retain num_inputs"),
        )
        .expect("u32 must fit usize");
        let results = usize::try_from(
            read_count(&data.attrs, NUM_RESULTS_ATTR)
                .expect("validated wasm.func_sig must retain num_results"),
        )
        .expect("u32 must fit usize");
        (inputs, results)
    }

    pub fn inputs(self, ctx: &IrContext) -> &[TypeRef] {
        let (inputs, _) = self.counts(ctx);
        &ctx.types.get(self.0).params[..inputs]
    }

    pub fn results(self, ctx: &IrContext) -> &[TypeRef] {
        let (inputs, results) = self.counts(ctx);
        &ctx.types.get(self.0).params[inputs..inputs + results]
    }

    pub fn is_resultless(self, ctx: &IrContext) -> bool {
        self.results(ctx).is_empty()
    }

    pub fn single_result(self, ctx: &IrContext) -> Option<TypeRef> {
        let results = self.results(ctx);
        (results.len() == 1).then(|| results[0])
    }

    pub fn non_reserved_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (&Symbol, &Attribute)> {
        ctx.types.get(self.0).attrs.iter().filter(|(key, _)| {
            **key != Symbol::new(NUM_INPUTS_ATTR) && **key != Symbol::new(NUM_RESULTS_ATTR)
        })
    }

    pub fn remove_reserved_attrs(attrs: &mut AttributeMap) {
        attrs.remove(NUM_INPUTS_ATTR);
        attrs.remove(NUM_RESULTS_ATTR);
    }
}

impl DialectType for FuncSig {
    const DIALECT_NAME: &'static str = "wasm";
    const TYPE_NAME: &'static str = "func_sig";

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        Self::matches(ctx, ty)
            .then(|| Self::validate(ctx, ty).ok())
            .flatten()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl From<FuncSig> for TypeRef {
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

/// Construct a canonical target-owned `wasm.func_sig`.
pub fn func_sig(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
) -> FuncSig {
    func_sig_with_attrs(ctx, inputs, results, AttributeMap::new())
}

/// Construct a canonical target signature while preserving non-reserved attributes.
pub fn func_sig_with_attrs(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
    attrs: AttributeMap,
) -> FuncSig {
    assert!(
        !attrs.contains_key(NUM_INPUTS_ATTR) && !attrs.contains_key(NUM_RESULTS_ATTR),
        "wasm.func_sig count attributes are reserved"
    );
    let inputs: Vec<_> = inputs.into_iter().collect();
    let results: Vec<_> = results.into_iter().collect();
    let num_inputs = u32::try_from(inputs.len()).expect("wasm.func_sig input count exceeds u32");
    let num_results = u32::try_from(results.len()).expect("wasm.func_sig result count exceeds u32");
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
    FuncSig(ty)
}

const INDIRECT_CALL_SIGNATURE_ATTR: &str = "signature";

impl IndirectCallLikeModel for CallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        self.signature(ctx)
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl IndirectCallLikeModel for ReturnCallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        self.signature(ctx)
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

/// Attach an exact callable contract to a `wasm` indirect transfer.
///
/// Returns `false` without mutation when the operation is not a `wasm`
/// indirect call or the supplied type is not a `wasm.func_sig` contract.
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
    use crate::op_interface::IndirectCallLikeOps;
    use crate::ops::DialectType;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

    #[test]
    fn indirect_call_interface_uses_wasm_owned_signature() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @ordinary(%table: core.i32, %value: core.i32) -> core.i32 {
    %result = wasm.call_indirect %table, %value {signature = wasm.func_sig<(core.i32) -> core.i32>, table = 0, type_idx = 0} : core.i32
    wasm.return %result
  }
  wasm.func @plain(%table: core.i32, %value: core.i32) -> core.i32 {
    %result = wasm.call_indirect %table, %value {table = 0, type_idx = 0} : core.i32
    wasm.return %result
  }
  wasm.func @tail(%table: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table, %value {signature = wasm.func_sig<(core.i32) -> core.nil>, table = 0, type_idx = 0}
  }
  wasm.func @direct() -> core.nil {
    wasm.return
  }
}"#,
        );
        let functions = module.ops(&ctx);
        let body_op = |index| {
            let body = ctx.op(functions[index]).regions[0];
            ctx.block(ctx.region(body).blocks[0]).ops[0]
        };
        let ordinary = body_op(0);
        let plain = body_op(1);
        let tail = body_op(2);
        let direct = body_op(3);

        assert!(IndirectCallLikeOps::exact_signature(&ctx, ordinary).is_some());
        assert!(IndirectCallLikeOps::get(&ctx, plain).is_some());
        assert_eq!(IndirectCallLikeOps::exact_signature(&ctx, plain), None);
        assert!(IndirectCallLikeOps::exact_signature(&ctx, tail).is_some());
        for op in [ordinary, tail] {
            let operands = ctx.op_operands(op);
            assert_eq!(IndirectCallLikeOps::callee(&ctx, op), Some(operands[0]));
            assert_eq!(
                IndirectCallLikeOps::arguments(&ctx, op),
                Some(&operands[1..])
            );
        }
        assert!(IndirectCallLikeOps::callee(&ctx, plain).is_some());
        assert!(IndirectCallLikeOps::arguments(&ctx, plain).is_some());
        assert!(IndirectCallLikeOps::get(&ctx, direct).is_none());
        assert_eq!(IndirectCallLikeOps::callee(&ctx, direct), None);
        assert_eq!(IndirectCallLikeOps::arguments(&ctx, direct), None);

        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("wasm.call_indirect"));
        assert!(printed.contains("signature = wasm.func_sig<(core.i32) -> core.i32>"));
    }

    #[test]
    fn func_sig_owns_zero_and_multiple_result_lists_and_metadata() {
        let mut ctx = crate::IrContext::new();
        let i32 = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i32"))
                .build(),
        );
        let i64 = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i64"))
                .build(),
        );
        let mut attrs = crate::AttributeMap::new();
        attrs.insert(crate::Symbol::new("kept"), crate::Attribute::Type(i64));
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
        let many_sig = FuncSig::from_type_ref(&ctx, many).unwrap();
        assert_eq!(many_sig.inputs(&ctx), [i32]);
        assert_eq!(many_sig.results(&ctx), [i32, i64]);
        assert_eq!(many_sig.non_reserved_attrs(&ctx).count(), 1);
        let mut same_attrs = crate::AttributeMap::new();
        same_attrs.insert(crate::Symbol::new("kept"), crate::Attribute::Type(i64));
        assert_eq!(
            many,
            func_sig_with_attrs(&mut ctx, [i32], [i32, i64], same_attrs).as_type_ref()
        );
    }

    #[test]
    fn func_sig_rejects_malformed_delimiters_without_slicing() {
        let mut ctx = crate::IrContext::new();
        let i32 = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i32"))
                .build(),
        );
        let malformed = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("wasm"), FUNC_SIG())
                .param(i32)
                .attr(NUM_INPUTS_ATTR, crate::Attribute::Int(2))
                .attr(NUM_RESULTS_ATTR, crate::Attribute::Int(1))
                .build(),
        );
        assert!(FuncSig::from_type_ref(&ctx, malformed).is_none());

        let missing = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("wasm"), FUNC_SIG())
                .param(i32)
                .attr(NUM_INPUTS_ATTR, crate::Attribute::Int(1))
                .build(),
        );
        assert_eq!(
            FuncSig::validate(&ctx, missing),
            Err(FuncSigTypeError::MissingCount(NUM_RESULTS_ATTR))
        );

        let wrong_kind = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("wasm"), FUNC_SIG())
                .param(i32)
                .attr(
                    NUM_INPUTS_ATTR,
                    crate::Attribute::Symbol(crate::Symbol::new("one")),
                )
                .attr(NUM_RESULTS_ATTR, crate::Attribute::Int(0))
                .build(),
        );
        assert_eq!(
            FuncSig::validate(&ctx, wrong_kind),
            Err(FuncSigTypeError::InvalidCount(NUM_INPUTS_ATTR))
        );

        let overflow = ctx.types.intern(
            crate::TypeDataBuilder::new(crate::Symbol::new("wasm"), FUNC_SIG())
                .attr(NUM_INPUTS_ATTR, crate::Attribute::Int(i128::from(u32::MAX)))
                .attr(NUM_RESULTS_ATTR, crate::Attribute::Int(1))
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
        let mut attrs = crate::AttributeMap::new();
        attrs.insert(NUM_INPUTS_ATTR.into(), crate::Attribute::Int(0));
        let _ = func_sig_with_attrs(&mut ctx, [], [], attrs);
    }
}
