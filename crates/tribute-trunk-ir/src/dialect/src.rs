//! Source dialect operations.
//!
//! The `src` dialect represents unresolved AST constructs before name resolution
//! and type inference. All `src.*` operations should be eliminated after resolution.

use crate::dialect;

dialect! {
    mod src {
        /// `src.call` operation: unresolved function call.
        /// The callee name will be resolved to a concrete function reference.
        #[attr(name)]
        fn call(#[rest] args) -> result;

        /// `src.var` operation: unresolved variable reference.
        /// The name will be resolved to a concrete value (parameter, local, etc.).
        #[attr(name)]
        fn var() -> result;

        /// `src.binop` operation: unresolved binary operation.
        /// Used for operators that need type-directed resolution (e.g., `<>` concat).
        /// The `op` attribute holds the operator name.
        #[attr(op)]
        fn binop(lhs, rhs) -> result;

        /// `src.block` operation: block expression.
        /// Preserves block structure for source mapping and analysis.
        /// The body region contains the statements, and the result is the block's value.
        fn block() -> result {
            #[region(body)] {}
        };

        /// `src.yield` operation: yields a value from a block.
        /// Used to specify the result value of a `src.block`.
        fn r#yield(value);

        /// `src.lambda` operation: lambda expression.
        /// Represents an anonymous function before capture analysis.
        /// The `type` attribute holds the function type (params -> result).
        /// The body region contains the lambda body, ending with `src.yield`.
        #[attr(r#type)]
        fn lambda() -> result {
            #[region(body)] {}
        };

        /// `src.tuple` operation: tuple construction.
        /// Takes variadic operands (tuple elements) and produces a tuple value.
        fn tuple(#[rest] elements) -> result;
    }
}
