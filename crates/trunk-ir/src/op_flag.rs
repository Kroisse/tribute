//! Operation flags: properties an operation's definition declares, with no
//! behavior of their own.
//!
//! An operation declares its flags with `#[flags(..)]` in `#[dialect]`, and
//! its [`OpDef`] holds them. Unlike an operation interface, a flag has no
//! implementation and cannot be added to an operation after its definition.

use crate::op_def::OpDef;
use crate::{IrContext, OpRef};

/// The identity of one operation flag, compared by address.
#[derive(Debug)]
pub struct OpFlagDef {
    pub name: &'static str,
}

/// A type naming an operation flag, as `#[flags(..)]` lists it.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an operation flag",
    label = "listed in `#[flags(..)]`",
    note = "an operation interface is listed in `#[implements(..)]`"
)]
pub trait OpFlag {
    const DEF: &'static OpFlagDef;

    /// Whether `op`'s operation declares this flag.
    fn is_set(ctx: &IrContext, op: OpRef) -> bool {
        OpDef::of(ctx, op).is_some_and(OpDef::has_flag::<Self>)
    }
}

/// Pure operations: no side effects, safe to remove if unused.
///
/// DCE may eliminate an operation with this flag if its results are unused.
pub struct Pure;

static PURE: OpFlagDef = OpFlagDef { name: "Pure" };

impl OpFlag for Pure {
    const DEF: &'static OpFlagDef = &PURE;
}

/// Operations whose regions cannot reference values from above.
///
/// The regions of an operation with this flag are "isolated" from the
/// enclosing scope - they cannot directly reference SSA values defined outside
/// the region. This is important for:
///
/// 1. **Verification**: Check that isolated regions don't have stale references
/// 2. **Rewrite passes**: Different handling for isolated vs non-isolated ops
/// 3. **Code generation**: Isolated regions can be compiled independently
///
/// Examples of isolated operations:
/// - `func.func` - function bodies (must receive values via parameters)
/// - `core.module` - module bodies
///
/// Examples of non-isolated operations (can capture outer values):
/// - `scf.if`, `scf.for` - control flow
/// - `closure.new` - closure creation
pub struct IsolatedFromAbove;

static ISOLATED_FROM_ABOVE: OpFlagDef = OpFlagDef {
    name: "IsolatedFromAbove",
};

impl OpFlag for IsolatedFromAbove {
    const DEF: &'static OpFlagDef = &ISOLATED_FROM_ABOVE;
}
