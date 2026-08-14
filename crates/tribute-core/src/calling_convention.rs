//! Compiler-wide calling-convention requirements.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::{Attribute, TypeDataBuilder};

pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";
/// Result type carried by a private immutable CPS parent frame.
pub const CPS_PARENT_RESULT_ATTR: &str = "tribute.cps_parent_result";
pub const INDIRECT_CALL_SIGNATURE_ATTR: &str =
    trunk_ir::dialect::func::INDIRECT_CALL_SIGNATURE_ATTR;
pub const CLOSURE_CALLABLE_TYPE_ATTR: &str = "tribute.closure_callable_type";
pub const CLOSURE_ENVIRONMENT_INDEX_ATTR: &str = "tribute.closure_environment_index";

/// The ABI strength required to call a function.
///
/// Ordering is significant: composing requirements selects the stronger
/// convention with [`CallingConvention::join`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, salsa::Update)]
#[repr(u8)]
pub enum CallingConvention {
    /// Pure function: source parameters and source result only.
    #[default]
    Direct = 0,
    /// Tail-resumptive effect: evidence parameter, direct source result.
    EvidenceDirect = 1,
    /// General control effect: evidence and done continuation.
    Cps = 2,
}

const CALLING_CONVENTIONS_BY_CODE: &[CallingConvention] = &[
    CallingConvention::Direct,
    CallingConvention::EvidenceDirect,
    CallingConvention::Cps,
];

impl CallingConvention {
    /// Compose two requirements by selecting the stronger convention.
    pub fn join(self, other: Self) -> Self {
        self.max(other)
    }

    /// Whether the convention carries an evidence parameter.
    pub fn needs_evidence(self) -> bool {
        self >= Self::EvidenceDirect
    }

    /// Whether the convention carries a done continuation.
    pub fn needs_done_k(self) -> bool {
        self == Self::Cps
    }

    /// Physical closure environment slot for this convention.
    pub fn closure_environment_index(self) -> usize {
        usize::from(self.needs_evidence())
    }
}

impl TryFrom<u8> for CallingConvention {
    type Error = u8;

    fn try_from(code: u8) -> Result<Self, Self::Error> {
        CALLING_CONVENTIONS_BY_CODE
            .get(usize::from(code))
            .copied()
            .ok_or(code)
    }
}

/// Attach the logical calling convention to a high-level IR operation.
pub fn set_calling_convention(ctx: &mut IrContext, op: OpRef, convention: CallingConvention) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(CALLING_CONVENTION_ATTR),
        Attribute::Int(convention as i128),
    );
}

/// Read explicitly attached calling-convention metadata.
pub fn get_calling_convention(ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
    let code = ctx
        .op(op)
        .attributes
        .get_u8(CALLING_CONVENTION_ATTR)
        .ok()??;
    code.try_into().ok()
}

/// Result-indexed CPS completion target `Done<R>`.
pub fn cps_done_type(ctx: &mut IrContext, result: TypeRef) -> TypeRef {
    let never = core::never(ctx).as_type_ref();
    let function = core::func(ctx, never, [result]).as_type_ref();
    generated_cps_closure_type(ctx, function)
}

/// Make the nominal reference for one private immutable `Parent<R>` frame.
///
/// Its paired layout may recursively use this reference.
pub fn cps_parent_ref_type(ctx: &mut IrContext, name: Symbol, result: TypeRef) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
            .attr("name", Attribute::Symbol(name))
            .attr(CPS_PARENT_RESULT_ATTR, Attribute::Type(result))
            .build(),
    )
}

/// Read result-index metadata only from an explicit parent-frame type.
pub fn cps_parent_result_type(ctx: &IrContext, parent: TypeRef) -> Option<TypeRef> {
    let data = ctx.types.get(parent);
    (data.dialect == Symbol::new("adt")
        && matches!(data.name, name if name == Symbol::new("typeref") || name == Symbol::new("struct")))
    .then(|| data.attrs.get_type(CPS_PARENT_RESULT_ATTR))
    .flatten()
}

/// Make the exact immutable layout for [`cps_parent_ref_type`].
pub fn cps_parent_layout_type(
    ctx: &mut IrContext,
    name: Symbol,
    result: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(name))
            .attr(CPS_PARENT_RESULT_ATTR, Attribute::Type(result))
            .attr(
                "fields",
                Attribute::List(vec![
                    Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("done")),
                        Attribute::Type(done),
                    ]),
                    Attribute::List(vec![
                        Attribute::Symbol(Symbol::new("dispatch")),
                        Attribute::Type(dispatch),
                    ]),
                ]),
            )
            .build(),
    )
}

/// Strict suffix continuation `Completion<X, R> = (Evidence, Parent<R>, X) -> never`.
pub fn cps_completion_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    value: TypeRef,
    parent: TypeRef,
) -> TypeRef {
    let never = core::never(ctx).as_type_ref();
    let function = core::func(ctx, never, [evidence, parent, value]).as_type_ref();
    generated_cps_closure_type(ctx, function)
}

/// Exact operation resumption `ResumeExact<I, R> = (Evidence, Parent<R>, I) -> never`.
pub fn cps_resume_exact_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    input: TypeRef,
    parent: TypeRef,
) -> TypeRef {
    cps_completion_type(ctx, evidence, input, parent)
}

/// Erased-input resumption `Resume<R> = (Evidence, Parent<R>, anyref) -> never`.
pub fn cps_resume_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    parent: TypeRef,
    anyref: TypeRef,
) -> TypeRef {
    cps_completion_type(ctx, evidence, anyref, parent)
}

/// Result-indexed general-operation dispatcher.
pub fn cps_dispatch_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    parent: TypeRef,
    anyref: TypeRef,
    i32_type: TypeRef,
) -> TypeRef {
    let never = core::never(ctx).as_type_ref();
    let resume = cps_resume_type(ctx, evidence, parent, anyref);
    let function = core::func(
        ctx,
        never,
        [evidence, resume, i32_type, i32_type, i32_type, anyref],
    )
    .as_type_ref();
    physical_closure_type(ctx, function, CallingConvention::Cps)
}

/// Return the underlying exact `core.func` only for a provenance-bearing CPS closure.
///
/// Callers must reject absent or malformed outer closure provenance rather than infer a
/// callable contract from a structurally similar type.
pub fn cps_closure_function_type(ctx: &IrContext, closure: TypeRef) -> Option<TypeRef> {
    let environment_index = get_physical_closure_environment_index(ctx, closure)?;
    if get_physical_closure_convention(ctx, closure) != Some(CallingConvention::Cps) {
        return None;
    }
    let [function] = ctx.types.get(closure).params.as_slice() else {
        return None;
    };
    let data = ctx.types.get(*function);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return None;
    }
    let argument_count = data.params.len().checked_sub(1)?;
    (environment_index <= argument_count).then_some(*function)
}

fn generated_cps_closure_type(ctx: &mut IrContext, function: TypeRef) -> TypeRef {
    physical_closure_type_with_environment_index(ctx, function, CallingConvention::Cps, 0)
}

/// Attach the exact callable signature to an indirect transfer.
pub fn set_indirect_call_signature(ctx: &mut IrContext, op: OpRef, signature: TypeRef) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR),
        Attribute::Type(signature),
    );
}

/// Read the exact callable signature retained on an indirect transfer.
pub fn get_indirect_call_signature(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    ctx.op(op).attributes.get_type(INDIRECT_CALL_SIGNATURE_ATTR)
}

/// Retain a typed closure contract on its canonical runtime pair.
pub fn set_closure_callable_type(ctx: &mut IrContext, op: OpRef, closure: TypeRef) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(CLOSURE_CALLABLE_TYPE_ATTR),
        Attribute::Type(closure),
    );
}

/// Read typed closure provenance from a canonical runtime pair.
pub fn get_closure_callable_type(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    ctx.op(op).attributes.get_type(CLOSURE_CALLABLE_TYPE_ATTR)
}

/// Build a closure type whose outer occurrence carries exact convention
/// provenance for physical callable producers and consumers.
pub fn physical_closure_type(
    ctx: &mut IrContext,
    function: TypeRef,
    convention: CallingConvention,
) -> TypeRef {
    physical_closure_type_with_environment_index(
        ctx,
        function,
        convention,
        convention.closure_environment_index(),
    )
}

/// Build a convention-proven closure type with its exact environment slot.
///
/// Most callables use the convention's standard slot. Generated continuation
/// closures may have a narrower physical entry shape, so their producer must
/// record the slot explicitly instead of asking a later consumer to infer it.
pub fn physical_closure_type_with_environment_index(
    ctx: &mut IrContext,
    function: TypeRef,
    convention: CallingConvention,
    environment_index: usize,
) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
            .param(function)
            .attr(CALLING_CONVENTION_ATTR, Attribute::Int(convention as i128))
            .attr(
                CLOSURE_ENVIRONMENT_INDEX_ATTR,
                Attribute::Int(environment_index as i128),
            )
            .build(),
    )
}

/// Read exact convention provenance from an outer physical closure type.
pub fn get_physical_closure_convention(
    ctx: &IrContext,
    closure: TypeRef,
) -> Option<CallingConvention> {
    let data = ctx.types.get(closure);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    data.attrs
        .get_u8(CALLING_CONVENTION_ATTR)
        .ok()??
        .try_into()
        .ok()
}

/// Read the exact environment slot from a convention-proven closure type.
pub fn get_physical_closure_environment_index(ctx: &IrContext, closure: TypeRef) -> Option<usize> {
    let data = ctx.types.get(closure);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    data.attrs
        .get_u32(CLOSURE_ENVIRONMENT_INDEX_ATTR)
        .ok()??
        .try_into()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_codes_round_trip() {
        for convention in CALLING_CONVENTIONS_BY_CODE {
            let code = *convention as u8;
            assert_eq!(CallingConvention::try_from(code), Ok(*convention));
        }

        assert_eq!(CallingConvention::try_from(3), Err(3));
    }

    #[test]
    fn physical_closure_convention_is_exact_type_identity() {
        let mut ctx = IrContext::new();
        let never = trunk_ir::dialect::core::never(&mut ctx).as_type_ref();
        let function = trunk_ir::dialect::core::func(&mut ctx, never, []).as_type_ref();
        let direct = physical_closure_type(&mut ctx, function, CallingConvention::Direct);
        let cps = physical_closure_type(&mut ctx, function, CallingConvention::Cps);

        assert_ne!(direct, cps);
        assert_eq!(
            get_physical_closure_convention(&ctx, direct),
            Some(CallingConvention::Direct)
        );
        assert_eq!(
            get_physical_closure_convention(&ctx, cps),
            Some(CallingConvention::Cps)
        );
        assert_eq!(get_physical_closure_environment_index(&ctx, cps), Some(1));

        let continuation = physical_closure_type_with_environment_index(
            &mut ctx,
            function,
            CallingConvention::Cps,
            0,
        );
        assert_eq!(
            get_physical_closure_environment_index(&ctx, continuation),
            Some(0)
        );
    }

    #[test]
    fn result_indexed_parent_builders_preserve_exact_types_and_provenance() {
        let mut ctx = IrContext::new();
        let evidence = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("ability"), Symbol::new("evidence")).build());
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let anyref = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build());
        let parent = cps_parent_ref_type(&mut ctx, Symbol::new("ParentI32"), i32_ty);
        let done = cps_done_type(&mut ctx, i32_ty);
        let dispatch = cps_dispatch_type(&mut ctx, evidence, parent, anyref, i32_ty);
        let layout =
            cps_parent_layout_type(&mut ctx, Symbol::new("ParentI32"), i32_ty, done, dispatch);
        let completion = cps_completion_type(&mut ctx, evidence, i32_ty, parent);
        let resume_exact = cps_resume_exact_type(&mut ctx, evidence, i32_ty, parent);
        let resume = cps_resume_type(&mut ctx, evidence, parent, anyref);
        let never = core::never(&mut ctx).as_type_ref();

        assert_eq!(cps_parent_result_type(&ctx, parent), Some(i32_ty));
        assert_eq!(cps_parent_result_type(&ctx, layout), Some(i32_ty));
        assert_eq!(
            ctx.types.get(layout).attrs.get("fields"),
            Some(&Attribute::List(vec![
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new("done")),
                    Attribute::Type(done),
                ]),
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new("dispatch")),
                    Attribute::Type(dispatch),
                ]),
            ]))
        );

        for (closure, environment_index, expected) in [
            (done, 0, vec![never, i32_ty]),
            (completion, 0, vec![never, evidence, parent, i32_ty]),
            (resume_exact, 0, vec![never, evidence, parent, i32_ty]),
            (resume, 0, vec![never, evidence, parent, anyref]),
            (
                dispatch,
                1,
                vec![never, evidence, resume, i32_ty, i32_ty, i32_ty, anyref],
            ),
        ] {
            let function = cps_closure_function_type(&ctx, closure).expect("exact CPS closure");
            assert_eq!(
                ctx.types.get(function).params.as_slice(),
                expected.as_slice()
            );
            assert_eq!(
                get_physical_closure_environment_index(&ctx, closure),
                Some(environment_index)
            );
        }
    }

    #[test]
    fn parent_and_cps_closure_readers_fail_closed_on_malformed_metadata() {
        let mut ctx = IrContext::new();
        let never = core::never(&mut ctx).as_type_ref();
        let function = core::func(&mut ctx, never, []).as_type_ref();
        let missing_environment = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
                .param(function)
                .attr(
                    CALLING_CONVENTION_ATTR,
                    Attribute::Int(CallingConvention::Cps as i128),
                )
                .build(),
        );
        let extra_outer_parameter = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
                .param(function)
                .param(never)
                .attr(
                    CALLING_CONVENTION_ATTR,
                    Attribute::Int(CallingConvention::Cps as i128),
                )
                .attr(CLOSURE_ENVIRONMENT_INDEX_ATTR, Attribute::Int(0))
                .build(),
        );
        let out_of_range_environment = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
                .param(function)
                .attr(
                    CALLING_CONVENTION_ATTR,
                    Attribute::Int(CallingConvention::Cps as i128),
                )
                .attr(CLOSURE_ENVIRONMENT_INDEX_ATTR, Attribute::Int(1))
                .build(),
        );
        let unmarked_parent = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", Attribute::Symbol(Symbol::new("Parent")))
                .build(),
        );

        assert_eq!(cps_closure_function_type(&ctx, missing_environment), None);
        assert_eq!(cps_closure_function_type(&ctx, extra_outer_parameter), None);
        assert_eq!(
            cps_closure_function_type(&ctx, out_of_range_environment),
            None
        );
        assert_eq!(cps_parent_result_type(&ctx, unmarked_parent), None);
        assert_eq!(cps_parent_result_type(&ctx, missing_environment), None);
    }
}
