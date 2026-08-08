//! Compiler-wide calling-convention requirements.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::{Attribute, TypeDataBuilder};

pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";
/// Exact physical callable contract carried across closure lowering when an
/// indirect callee becomes an untyped table/function pointer.
pub const INDIRECT_CALL_SIGNATURE_ATTR: &str =
    trunk_ir::dialect::func::INDIRECT_CALL_SIGNATURE_ATTR;
/// Exact logical closure type retained on a lowered runtime closure value.
/// This is value provenance, not a target ABI signature: consumers must use
/// it only when the value is the result of the annotated closure packing op.
pub const CLOSURE_CALLABLE_TYPE_ATTR: &str = "tribute.closure_callable_type";
pub const ROOT_EXPORT_CONVENTION_ATTR: &str = "tribute.root_export_convention";
pub const ROOT_SOURCE_RESULT_ATTR: &str = "tribute.root_source_result";
pub const CPS_PARENT_RESULT_ATTR: &str = "tribute.cps_parent_result";

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

/// Attach the exact callable type for an indirect call or proper-tail transfer.
/// The type includes the physical closure environment parameter when present.
pub fn set_indirect_call_signature(ctx: &mut IrContext, op: OpRef, signature: TypeRef) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR),
        Attribute::Type(signature),
    );
}

/// Read the exact callable type preserved for an indirect transfer.
pub fn get_indirect_call_signature(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    ctx.op(op).attributes.get_type(INDIRECT_CALL_SIGNATURE_ATTR)
}

/// Preserve the typed closure occurrence while lowering it to `_closure`.
pub fn set_closure_callable_type(ctx: &mut IrContext, op: OpRef, closure: TypeRef) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(CLOSURE_CALLABLE_TYPE_ATTR),
        Attribute::Type(closure),
    );
}

/// Read the exact typed closure provenance from a lowered closure value.
pub fn get_closure_callable_type(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    ctx.op(op).attributes.get_type(CLOSURE_CALLABLE_TYPE_ATTR)
}

/// Result-indexed CPS completion target `Done<R>`.
pub fn cps_done_type(ctx: &mut IrContext, result: TypeRef) -> TypeRef {
    let never = core::never(ctx).as_type_ref();
    let function = core::func(ctx, never, [result]).as_type_ref();
    physical_closure_type(ctx, function, CallingConvention::Cps)
}

/// Make the nominal reference for one private immutable `Flow<R>` pack.
/// The paired struct layout may recursively use this reference.
pub fn cps_parent_ref_type(ctx: &mut IrContext, name: Symbol, result: TypeRef) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
            .attr("name", Attribute::Symbol(name))
            .attr(CPS_PARENT_RESULT_ATTR, Attribute::Type(result))
            .build(),
    )
}

/// Make the exact layout for [`cps_parent_ref_type`].
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

pub fn cps_parent_result_type(ctx: &IrContext, parent: TypeRef) -> Option<TypeRef> {
    let data = ctx.types.get(parent);
    (data.dialect == Symbol::new("adt") && data.name == Symbol::new("typeref"))
        .then(|| data.attrs.get_type(CPS_PARENT_RESULT_ATTR))
        .flatten()
}

/// Whether `parent` has one exact nominal immutable layout for `dispatch`.
///
/// The result tag on a typeref alone is insufficient provenance: its matching
/// struct layout must own the same nominal identity and carry the canonical
/// `done` and `dispatch` fields.
pub fn has_canonical_cps_parent_layout(
    ctx: &IrContext,
    parent: TypeRef,
    dispatch: TypeRef,
) -> bool {
    let Some(result) = cps_parent_result_type(ctx, parent) else {
        return false;
    };
    let Some(name) = ctx.types.get(parent).attrs.get_symbol("name") else {
        return false;
    };

    let is_never = |ty| {
        let data = ctx.types.get(ty);
        data.dialect == Symbol::new("core") && data.name == Symbol::new("never")
    };
    let is_done = |ty| {
        let Some(function) = cps_closure_function_type(ctx, ty) else {
            return false;
        };
        let params = &ctx.types.get(function).params;
        params.len() == 2 && is_never(params[0]) && params[1] == result
    };
    let mut matching_name = 0;
    let mut canonical = 0;
    for (_, data) in ctx.types.iter() {
        if data.dialect != Symbol::new("adt")
            || data.name != Symbol::new("struct")
            || data.attrs.get_symbol("name") != Some(name)
        {
            continue;
        }
        matching_name += 1;
        let Some(Attribute::List(fields)) = data.attrs.get("fields") else {
            continue;
        };
        let [Attribute::List(done), Attribute::List(actual_dispatch)] = fields.as_slice() else {
            continue;
        };
        let [Attribute::Symbol(done_name), Attribute::Type(done_type)] = done.as_slice() else {
            continue;
        };
        let [
            Attribute::Symbol(dispatch_name),
            Attribute::Type(dispatch_type),
        ] = actual_dispatch.as_slice()
        else {
            continue;
        };
        if data.attrs.get_type(CPS_PARENT_RESULT_ATTR) == Some(result)
            && *done_name == Symbol::new("done")
            && is_done(*done_type)
            && *dispatch_name == Symbol::new("dispatch")
            && *dispatch_type == dispatch
        {
            canonical += 1;
        }
    }
    matching_name == 1 && canonical == 1
}

/// Strict suffix continuation
/// `Completion<X, R> = (Evidence, Parent<R>, X) -> never`.
pub fn cps_completion_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    value: TypeRef,
    parent: TypeRef,
) -> TypeRef {
    let never = core::never(ctx).as_type_ref();
    let function = core::func(ctx, never, [evidence, parent, value]).as_type_ref();
    physical_closure_type(ctx, function, CallingConvention::Cps)
}

/// Exact operation resumption
/// `ResumeExact<I, R> = (Evidence, Parent<R>, I) -> never`.
pub fn cps_resume_exact_type(
    ctx: &mut IrContext,
    evidence: TypeRef,
    input: TypeRef,
    parent: TypeRef,
) -> TypeRef {
    cps_completion_type(ctx, evidence, input, parent)
}

/// Erased-input resumption
/// `Resume<R> = (Evidence, Parent<R>, anyref) -> never`.
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

/// Return the underlying exact `core.func` only for a convention-proven CPS
/// closure. Callers must reject absent provenance rather than infer it from
/// structural shape.
pub fn cps_closure_function_type(ctx: &IrContext, closure: TypeRef) -> Option<TypeRef> {
    (get_physical_closure_convention(ctx, closure) == Some(CallingConvention::Cps))
        .then(|| ctx.types.get(closure).params.first().copied())
        .flatten()
        .filter(|function| {
            let data = ctx.types.get(*function);
            data.dialect == Symbol::new("core") && data.name == Symbol::new("func")
        })
}

/// Build a physical closure type that preserves its exact callable
/// convention on the outer `closure.closure` occurrence.
pub fn physical_closure_type(
    ctx: &mut IrContext,
    function: TypeRef,
    convention: CallingConvention,
) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure"))
            .param(function)
            .attr(CALLING_CONVENTION_ATTR, Attribute::Int(convention as i128))
            .build(),
    )
}

/// Read exact convention provenance from a physical outer closure type.
pub fn get_physical_closure_convention(
    ctx: &IrContext,
    closure: TypeRef,
) -> Option<CallingConvention> {
    let data = ctx.types.get(closure);
    if data.dialect != Symbol::new("closure") || data.name != Symbol::new("closure") {
        return None;
    }
    let code = data.attrs.get_u8(CALLING_CONVENTION_ATTR).ok()??;
    code.try_into().ok()
}

/// Preserve the checker-selected Direct/EvidenceDirect convention for the
/// exact root entry while its physical worker is promoted to CPS.
pub fn set_root_export_convention(ctx: &mut IrContext, op: OpRef, convention: CallingConvention) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(ROOT_EXPORT_CONVENTION_ATTR),
        Attribute::Int(convention as i128),
    );
}

/// Read the checker-selected convention for the exact root export.
pub fn get_root_export_convention(ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
    let code = ctx
        .op(op)
        .attributes
        .get_u8(ROOT_EXPORT_CONVENTION_ATTR)
        .ok()??;
    code.try_into().ok()
}

/// Preserve the exact source result type of the root export while its worker
/// is promoted to the physical CPS ABI.
pub fn set_root_source_result(ctx: &mut IrContext, op: OpRef, result: trunk_ir::TypeRef) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(ROOT_SOURCE_RESULT_ATTR),
        Attribute::Type(result),
    );
}

/// Read the preserved source result type of the exact root export.
pub fn get_root_source_result(ctx: &IrContext, op: OpRef) -> Option<trunk_ir::TypeRef> {
    ctx.op(op).attributes.get_type(ROOT_SOURCE_RESULT_ATTR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::types::TypeDataBuilder;

    #[test]
    fn integer_codes_round_trip() {
        for convention in CALLING_CONVENTIONS_BY_CODE {
            let code = *convention as u8;
            assert_eq!(CallingConvention::try_from(code), Ok(*convention));
        }

        assert_eq!(CallingConvention::try_from(3), Err(3));
    }

    #[test]
    fn physical_closure_convention_is_type_identity() {
        let mut ctx = IrContext::new();
        let never = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("never")).build());
        let function = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(never)
                .build(),
        );
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
    }

    #[test]
    fn result_indexed_cps_types_preserve_all_boundary_types() {
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
        assert_eq!(
            ctx.types.get(layout).attrs.get_type(CPS_PARENT_RESULT_ATTR),
            Some(i32_ty)
        );
        assert_eq!(
            ctx.types.get(layout).attrs.get("fields"),
            Some(&Attribute::List(vec![
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new("done")),
                    Attribute::Type(done)
                ]),
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new("dispatch")),
                    Attribute::Type(dispatch),
                ]),
            ]))
        );

        for (closure, expected) in [
            (completion, vec![never, evidence, parent, i32_ty]),
            (resume_exact, vec![never, evidence, parent, i32_ty]),
            (resume, vec![never, evidence, parent, anyref]),
            (
                dispatch,
                vec![never, evidence, resume, i32_ty, i32_ty, i32_ty, anyref],
            ),
        ] {
            let function = cps_closure_function_type(&ctx, closure).unwrap();
            assert_eq!(
                ctx.types.get(function).params.as_slice(),
                expected.as_slice()
            );
        }
        assert_ne!(parent, anyref);
        assert_ne!(
            completion,
            cps_completion_type(&mut ctx, evidence, anyref, parent)
        );
    }
}
