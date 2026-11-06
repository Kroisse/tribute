//! Operation flags: properties an operation's definition declares, with no
//! behavior of their own.
//!
//! A flag is a unit struct declared with [`#[op_flag]`](macro@crate::op_flag).
//! An operation declares its flags with `#[flags(..)]` in `#[dialect]`, and
//! its [`OpDef`] holds them. Unlike an operation interface, a flag has no
//! implementation and cannot be added to an operation after its definition.

use crate::op_def::OpDef;
use crate::{IrContext, OpRef};
use trunk_ir_macros::op_flag;

/// The identity of one operation flag, compared by address.
#[derive(Debug)]
pub struct OpFlagDef {
    pub name: &'static str,
}

/// A type naming an operation flag, as `#[flags(..)]` lists it.
///
/// Implemented by `#[op_flag]`.
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
#[op_flag]
pub struct Pure;

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
#[op_flag]
pub struct IsolatedFromAbove;
