//! Source dialect operations.
//!
//! The `src` dialect represents unresolved AST constructs before name resolution
//! and type inference. All `src.*` operations should be eliminated after resolution.

use crate::dialect;

dialect! {
    src {
        /// `src.call` operation: unresolved function call.
        /// The callee name will be resolved to a concrete function reference.
        op call[name](..args) -> result {};

        /// `src.var` operation: unresolved variable reference.
        /// The name will be resolved to a concrete value (parameter, local, etc.).
        op var[name]() -> result {};
    }
}
