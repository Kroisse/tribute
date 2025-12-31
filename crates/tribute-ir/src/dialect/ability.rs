//! Ability dialect operations and types.
//!
//! This dialect represents ability (algebraic effect) operations that get
//! lowered to the `cont` dialect for continuation-based control flow.
//!
//! ## Design
//!
//! Ability declarations (`tribute.ability_def`, `tribute.op`) are in the tribute dialect.
//! This dialect contains the runtime operations:
//! - `ability.perform`: invoke an ability operation
//! - `ability.resume`: resume a captured continuation
//! - `ability.abort`: discard a continuation
//!
//! Handler pattern matching is done via `tribute.case` with `tribute.handle`:
//!
//! ```text
//! // Source: case handle expr { ... }
//! // Lowers to:
//! %request = tribute.handle { expr }
//! tribute.case(%request) {
//!     tribute.arm("{result}") { ... }
//!     tribute.arm("{State::get() -> k}") { ... }
//! }
//! ```

use trunk_ir::dialect;

dialect! {
    mod ability {
        // === Operations ===

        /// `ability.perform` operation: performs an ability operation.
        ///
        /// Invokes an operation from an ability, capturing the current continuation
        /// until a handler is found. Returns when resumed by a handler.
        ///
        /// The `ability_ref` attribute is a `Type` (specifically `core.ability_ref`)
        /// to support parameterized abilities like `State(Int)`.
        #[attr(ability_ref: Type, op: Symbol)]
        fn perform(#[rest] args) -> result;

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

        // === Types ===

        /// `ability.evidence_ptr` type: pointer to evidence struct.
        ///
        /// Evidence is a runtime structure containing ability markers for
        /// dynamic handler dispatch. Passed as first argument to effectful functions.
        ///
        /// See `new-plans/implementation.md` for the evidence passing design.
        type evidence_ptr;
    }
}

// === Printable interface registrations ===

use trunk_ir::type_interface::Printable;

// evidence_ptr -> "Evidence"
inventory::submit! { Printable::implement("ability", "evidence_ptr", |_, _, f: &mut std::fmt::Formatter<'_>| f.write_str("Evidence")) }
