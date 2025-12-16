//! Ability dialect operations.
//!
//! This dialect represents language-level ability (algebraic effect) operations.
//! These are high-level operations that get lowered to the `cont` dialect.

use crate::dialect;

dialect! {
    ability {
        /// `ability.perform` operation: performs an ability operation.
        ///
        /// Invokes an operation from an ability, capturing the current continuation
        /// until a handler is found.
        ///
        /// Attributes:
        /// - `ability`: Reference to the ability type
        /// - `op`: The operation name within the ability
        op perform[ability_ref, op](..args) -> result {};

        /// `ability.handle` operation: installs a handler and executes the body.
        ///
        /// The handler catches ability operations performed within the body region.
        ///
        /// Attributes:
        /// - `clauses`: Handler clauses for each operation
        op handle[clauses]() -> result { body };

        /// `ability.resume` operation: resumes a captured continuation.
        ///
        /// Continues execution from where `perform` was called, providing a value.
        op resume(continuation, value) -> result {};

        /// `ability.abort` operation: discards a continuation without resuming.
        ///
        /// Satisfies the linear type requirement for continuations by explicitly
        /// dropping them.
        op abort(continuation) {};
    }
}
