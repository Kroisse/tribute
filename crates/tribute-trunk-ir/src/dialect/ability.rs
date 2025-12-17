//! Ability dialect operations.
//!
//! This dialect represents language-level ability (algebraic effect) operations.
//! These are high-level operations that get lowered to the `cont` dialect.

use crate::dialect;

dialect! {
    mod ability {
        /// `ability.perform` operation: performs an ability operation.
        ///
        /// Invokes an operation from an ability, capturing the current continuation
        /// until a handler is found.
        #[attr(ability_ref: SymbolRef, op: Symbol)]
        fn perform(#[rest] args) -> result;

        /// `ability.handle` operation: installs a handler and executes the body.
        ///
        /// The handler catches ability operations performed within the body region.
        #[attr(clauses: any)]
        fn handle() -> result {
            #[region(body)] {}
        };

        /// `ability.resume` operation: resumes a captured continuation.
        ///
        /// Continues execution from where `perform` was called, providing a value.
        fn resume(continuation, value) -> result;

        /// `ability.abort` operation: discards a continuation without resuming.
        ///
        /// Satisfies the linear type requirement for continuations by explicitly
        /// dropping them.
        fn abort(continuation);
    }
}
