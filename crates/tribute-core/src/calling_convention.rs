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
pub enum CallingConvention {
    /// Pure function: source parameters and source result only.
    #[default]
    Direct,
    /// Tail-resumptive effect: evidence parameter, direct source result.
    EvidenceDirect,
    /// General control effect: evidence and done continuation.
    Cps,
}

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

    fn code(self) -> i128 {
        match self {
            Self::Direct => 0,
            Self::EvidenceDirect => 1,
            Self::Cps => 2,
        }
    }

    fn from_code(code: i128) -> Option<Self> {
        match code {
            0 => Some(Self::Direct),
            1 => Some(Self::EvidenceDirect),
            2 => Some(Self::Cps),
            _ => None,
        }
    }
}

/// Attach the logical calling convention to a high-level IR operation.
pub fn set_calling_convention(ctx: &mut IrContext, op: OpRef, convention: CallingConvention) {
    ctx.op_mut(op).attributes.insert(
        Symbol::new(CALLING_CONVENTION_ATTR),
        Attribute::Int(convention.code()),
    );
}

/// Read explicitly attached calling-convention metadata.
pub fn get_calling_convention(ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
    let Attribute::Int(code) = ctx
        .op(op)
        .attributes
        .get(&Symbol::new(CALLING_CONVENTION_ATTR))?
    else {
        return None;
    };
    CallingConvention::from_code(*code)
}
