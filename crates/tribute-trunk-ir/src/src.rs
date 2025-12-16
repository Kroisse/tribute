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

        /// `src.binop` operation: unresolved binary operation.
        /// Used for operators that need type-directed resolution (e.g., `<>` concat).
        /// The `op` attribute holds the operator name.
        op binop[op](lhs, rhs) -> result {};

        /// `src.block` operation: block expression.
        /// Preserves block structure for source mapping and analysis.
        /// The body region contains the statements, and the result is the block's value.
        op block() -> result { body };

        /// `src.yield` operation: yields a value from a block.
        /// Used to specify the result value of a `src.block`.
        op r#yield(value) {};
    }
}
