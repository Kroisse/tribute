//! Ability dialect operations.
//!
//! This dialect represents language-level ability (algebraic effect) operations.
//! These are high-level operations that get lowered to the `cont` dialect.
//!
//! ## Design
//!
//! Handler pattern matching is done via `case.case`, not in this dialect.
//! The `ability.prompt` operation runs the body in a delimited context and
//! returns a `Request` value, which is then pattern-matched by `case.case`
//! with handler patterns (`case.handler_done`, `case.handler_suspend`).
//!
//! ```text
//! // Source: case handle expr { ... }
//! // Lowers to:
//! %request = ability.prompt { expr }
//! case.case(%request) {
//!     case.arm("{result}") { ... }
//!     case.arm("{State::get() -> k}") { ... }
//! }
//! ```

use crate::dialect;

dialect! {
    mod ability {
        /// `ability.perform` operation: performs an ability operation.
        ///
        /// Invokes an operation from an ability, capturing the current continuation
        /// until a handler is found. Returns when resumed by a handler.
        #[attr(ability_ref: QualifiedName, op: Symbol)]
        fn perform(#[rest] args) -> result;

        /// `ability.prompt` operation: runs body in a delimited context.
        ///
        /// Executes the body region until it either:
        /// - Completes with a value → returns `Request::Done(value)`
        /// - Performs an ability operation → returns `Request::Suspend(op, args, continuation)`
        ///
        /// The returned `Request` is typically pattern-matched using `case.case`
        /// with handler patterns.
        fn prompt() -> request {
            #[region(body)] {}
        };

        /// `ability.resume` operation: resumes a captured continuation.
        ///
        /// Continues execution from where `perform` was called, providing a value.
        /// The continuation is consumed (linear type).
        fn resume(continuation, value) -> result;

        /// `ability.abort` operation: discards a continuation without resuming.
        ///
        /// Satisfies the linear type requirement for continuations by explicitly
        /// dropping them. Used when a handler doesn't want to continue execution
        /// (e.g., `Fail::fail` handler returning `None`).
        fn abort(continuation);
    }
}
