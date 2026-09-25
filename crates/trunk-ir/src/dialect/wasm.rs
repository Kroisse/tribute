//! Arena-based wasm dialect.

use crate::dialect::core::I32;
use crate::op_interface::{IndirectCallLikeModel, IndirectCallLikeOps};
use crate::ops::{DialectOp, DialectType};
use crate::{Attribute, AttributeMap, IrContext, Symbol, TypeDataBuilder, TypeRef};

crate::register_isolated_op!(Func);

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
    fn block() -> Variadic<_> {
        #[region(body)]
        {}
    }

    fn r#loop(init: Variadic<_>) -> Variadic<_> {
        #[region(body)]
        {}
    }

    fn r#if(cond: Value<_>) -> Variadic<_> {
        #[region(then_region)]
        {}
        #[region(else_region)]
        {}
    }

    fn br(target: Attr<u32>) {}

    fn br_if(target: Attr<u32>, cond: Value<_>) {}

    fn r#return(values: Variadic<_>) {}
    fn r#yield(value: Value<_>) {}
    fn drop(value: Value<_>) {}

    // Functions
    fn call(callee: Attr<Symbol>, args: Variadic<_>) -> Variadic<_> {}

    fn call_indirect(
        type_idx: Attr<u32>,
        table: Attr<u32>,
        signature: Option<Attr<Type>>,
        args: Variadic<_>,
    ) -> Variadic<_> {
    }

    fn return_call(callee: Attr<Symbol>, args: Variadic<_>) {}

    fn return_call_indirect(
        type_idx: Attr<u32>,
        table: Attr<u32>,
        signature: Option<Attr<Type>>,
        args: Variadic<_>,
    ) {
    }

    fn unreachable() {}
    fn nop() -> Value<_> {}

    // Module
    fn func(sym_name: Attr<Symbol>, r#type: Attr<Type>) {
        #[region(body)]
        {}
    }

    fn import_func(
        module: Attr<Symbol>,
        name: Attr<Symbol>,
        sym_name: Attr<Symbol>,
        r#type: Attr<Type>,
    ) {
    }

    fn export_func(name: Attr<String>, func: Attr<Symbol>) {}

    fn export_memory(name: Attr<String>, index: Attr<u32>) {}

    fn memory(min: Attr<u32>, max: Attr<u32>, shared: Attr<bool>, memory64: Attr<bool>) {}

    fn data(offset: Attr<u32>, bytes: Attr<_>, passive: Attr<bool>) {}

    fn table(reftype: Attr<Symbol>, min: Attr<u32>, max: Option<Attr<u32>>) {}

    fn elem(table: Option<Attr<u32>>, offset: Option<Attr<u32>>) {
        #[region(funcs)]
        {}
    }

    fn global(valtype: Attr<Symbol>, mutable: Attr<bool>, init: Attr<_>) {}

    fn global_get(index: Attr<u32>) -> Value<_> {}

    fn global_set(index: Attr<u32>, value: Value<_>) {}

    // i32
    fn i32_const(value: Attr<i32>) -> Value<_> {}

    fn i32_add(lhs: Value<I32>, rhs: Value<I32>) -> Value<I32> {}
    fn i32_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_div_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_div_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_rem_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_rem_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    fn i32_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_lt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_lt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_le_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_le_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_gt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_gt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_ge_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_ge_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    fn i32_and(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_or(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_xor(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_shl(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_shr_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i32_shr_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // i64
    fn i64_const(value: Attr<i64>) -> Value<_> {}

    fn i64_add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_div_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_div_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_rem_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_rem_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    fn i64_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_lt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_lt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_le_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_le_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_gt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_gt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_ge_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_ge_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    fn i64_and(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_or(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_xor(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_shl(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_shr_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn i64_shr_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // f32
    fn f32_const(value: Attr<f32>) -> Value<_> {}

    fn f32_add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_div(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_neg(operand: Value<_>) -> Value<_> {}

    fn f32_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_lt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_le(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_gt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f32_ge(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // f64
    fn f64_const(value: Attr<f64>) -> Value<_> {}

    fn f64_add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_div(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_neg(operand: Value<_>) -> Value<_> {}

    fn f64_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_lt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_le(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_gt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    fn f64_ge(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // Local variables
    fn local_get(index: Attr<u32>) -> Value<_> {}

    fn local_set(index: Attr<u32>, value: Value<_>) {}

    fn local_tee(index: Attr<u32>, value: Value<_>) -> Value<_> {}

    // GC structs
    fn struct_new(type_idx: Attr<u32>, fields: Variadic<_>) -> Value<_> {}

    fn struct_get(type_idx: Attr<u32>, field_idx: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(type_idx: Attr<u32>, field_idx: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    // GC arrays
    fn array_new(type_idx: Attr<u32>, size: Value<_>, init: Value<_>) -> Value<_> {}

    fn array_new_default(type_idx: Attr<u32>, size: Value<_>) -> Value<_> {}

    fn array_new_data(
        type_idx: Attr<u32>,
        data_idx: Attr<u32>,
        offset: Value<_>,
        size: Value<_>,
    ) -> Value<_> {
    }

    fn bytes_from_data(data_idx: Attr<u32>, offset: Attr<u32>, len: Attr<u32>) -> Value<_> {}

    fn array_get(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_get_s(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_get_u(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_set(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<_>, value: Value<_>) {}

    fn array_len(r#ref: Value<_>) -> Value<_> {}

    fn array_copy(
        dst_type_idx: Attr<u32>,
        src_type_idx: Attr<u32>,
        dst: Value<_>,
        dst_offset: Value<_>,
        src: Value<_>,
        src_offset: Value<_>,
        len: Value<_>,
    ) {
    }

    // References
    fn ref_null(heap_type: Attr<Symbol>, type_idx: Option<Attr<u32>>) -> Value<_> {}

    fn ref_func(func_name: Attr<Symbol>) -> Value<_> {}

    fn ref_is_null(r#ref: Value<_>) -> Value<_> {}

    fn ref_cast(target_type: Attr<Type>, type_idx: Option<Attr<u32>>, r#ref: Value<_>) -> Value<_> {
    }

    fn ref_test(target_type: Attr<Type>, type_idx: Option<Attr<u32>>, r#ref: Value<_>) -> Value<_> {
    }

    // i31ref
    fn ref_i31(value: Value<_>) -> Value<_> {}
    fn i31_get_s(r#ref: Value<_>) -> Value<_> {}
    fn i31_get_u(r#ref: Value<_>) -> Value<_> {}

    // Type conversions (integer)
    fn i32_wrap_i64(operand: Value<_>) -> Value<_> {}
    fn i64_extend_i32_s(operand: Value<_>) -> Value<_> {}
    fn i64_extend_i32_u(operand: Value<_>) -> Value<_> {}

    // Type conversions (float to int)
    fn i32_trunc_f32_s(operand: Value<_>) -> Value<_> {}
    fn i32_trunc_f32_u(operand: Value<_>) -> Value<_> {}
    fn i32_trunc_f64_s(operand: Value<_>) -> Value<_> {}
    fn i32_trunc_f64_u(operand: Value<_>) -> Value<_> {}
    fn i64_trunc_f32_s(operand: Value<_>) -> Value<_> {}
    fn i64_trunc_f32_u(operand: Value<_>) -> Value<_> {}
    fn i64_trunc_f64_s(operand: Value<_>) -> Value<_> {}
    fn i64_trunc_f64_u(operand: Value<_>) -> Value<_> {}

    // Type conversions (int to float)
    fn f32_convert_i32_s(operand: Value<_>) -> Value<_> {}
    fn f32_convert_i32_u(operand: Value<_>) -> Value<_> {}
    fn f32_convert_i64_s(operand: Value<_>) -> Value<_> {}
    fn f32_convert_i64_u(operand: Value<_>) -> Value<_> {}
    fn f64_convert_i32_s(operand: Value<_>) -> Value<_> {}
    fn f64_convert_i32_u(operand: Value<_>) -> Value<_> {}
    fn f64_convert_i64_s(operand: Value<_>) -> Value<_> {}
    fn f64_convert_i64_u(operand: Value<_>) -> Value<_> {}

    // Float conversions
    fn f32_demote_f64(operand: Value<_>) -> Value<_> {}
    fn f64_promote_f32(operand: Value<_>) -> Value<_> {}

    // Bitcast
    fn i32_reinterpret_f32(operand: Value<_>) -> Value<_> {}
    fn i64_reinterpret_f64(operand: Value<_>) -> Value<_> {}
    fn f32_reinterpret_i32(operand: Value<_>) -> Value<_> {}
    fn f64_reinterpret_i64(operand: Value<_>) -> Value<_> {}

    // Linear memory
    fn memory_size(memory: Attr<u32>) -> Value<_> {}

    fn memory_grow(memory: Attr<u32>, delta: Value<_>) -> Value<_> {}

    // Memory loads (full width)
    fn i32_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i64_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn f32_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn f64_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    // Memory loads (partial width i32)
    fn i32_load8_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i32_load8_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i32_load16_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i32_load16_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    // Memory loads (partial width i64)
    fn i64_load8_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i64_load8_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i64_load16_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i64_load16_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i64_load32_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    fn i64_load32_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    // Memory stores (full width)
    fn i32_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn i64_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn f32_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn f64_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    // Memory stores (partial width)
    fn i32_store8(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn i32_store16(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn i64_store8(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn i64_store16(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    fn i64_store32(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }
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

crate::impl_func_sig_constraint!(FuncSig, "wasm.func_sig");

impl FuncSig {
    pub(crate) fn validate(ctx: &IrContext, ty: TypeRef) -> Result<Self, FuncSigTypeError> {
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

    fn counts(self, ctx: &IrContext) -> (usize, usize) {
        let data = ctx.get_type(self.0);
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
        &ctx.get_type(self.0).params[..inputs]
    }

    pub fn results(self, ctx: &IrContext) -> &[TypeRef] {
        let (inputs, results) = self.counts(ctx);
        &ctx.get_type(self.0).params[inputs..inputs + results]
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
        ctx.get_type(self.0).attrs.iter().filter(|(key, _)| {
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
    let ty = ctx.intern_type(
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
        let i32 = ctx.intern_type(
            crate::TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i32"))
                .build(),
        );
        let i64 = ctx.intern_type(
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
        let i32 = ctx.intern_type(
            crate::TypeDataBuilder::new(crate::Symbol::new("core"), crate::Symbol::new("i32"))
                .build(),
        );
        let malformed = ctx.intern_type(
            crate::TypeDataBuilder::new(crate::Symbol::new("wasm"), FUNC_SIG())
                .param(i32)
                .attr(NUM_INPUTS_ATTR, crate::Attribute::Int(2))
                .attr(NUM_RESULTS_ATTR, crate::Attribute::Int(1))
                .build(),
        );
        assert!(FuncSig::from_type_ref(&ctx, malformed).is_none());

        let missing = ctx.intern_type(
            crate::TypeDataBuilder::new(crate::Symbol::new("wasm"), FUNC_SIG())
                .param(i32)
                .attr(NUM_INPUTS_ATTR, crate::Attribute::Int(1))
                .build(),
        );
        assert_eq!(
            FuncSig::validate(&ctx, missing),
            Err(FuncSigTypeError::MissingCount(NUM_RESULTS_ATTR))
        );

        let wrong_kind = ctx.intern_type(
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

        let overflow = ctx.intern_type(
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
