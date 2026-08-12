//! Compiler-wide calling-convention requirements.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::types::{Attribute, TypeDataBuilder};

pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";
pub const INDIRECT_CALL_SIGNATURE_ATTR: &str =
    trunk_ir::dialect::func::INDIRECT_CALL_SIGNATURE_ATTR;
pub const CLOSURE_CALLABLE_TYPE_ATTR: &str = "tribute.closure_callable_type";

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
/// provenance. This is dormant until a producer selects the physical CPS path.
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
            get_physical_closure_convention(&ctx, cps),
            Some(CallingConvention::Cps)
        );
    }
}
