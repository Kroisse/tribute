//! Compiler-wide calling-convention requirements.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::OpRef;
use trunk_ir::types::Attribute;

pub const CALLING_CONVENTION_ATTR: &str = "tribute.calling_convention";

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
    let Attribute::Int(code) = ctx.op(op).attributes.get(CALLING_CONVENTION_ATTR)? else {
        return None;
    };
    let code = u8::try_from(*code).ok()?;
    code.try_into().ok()
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
}
