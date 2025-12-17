//! Structured Control Flow dialect operations.
//!
//! Low-level control flow primitives that map directly to target instructions.
//! This dialect provides the building blocks for wasm control flow:
//! - `if` → wasm `if` or `select`
//! - `switch` → wasm `br_table`
//!
//! Note: Tribute has no loop construct in the source language; only recursion.
//! The `loop`, `continue`, and `break` operations are produced by tail call inlining optimization.

use crate::dialect;

dialect! {
    mod scf {
        // === Conditional ===

        /// `scf.if` operation: conditional branch with then/else bodies.
        /// Both regions must yield the same type.
        /// Maps to wasm `if` instruction (or `select` for simple cases).
        fn r#if(cond) -> result {
            #[region(then)] {}
            #[region(r#else)] {}
        };

        // === Multi-way Branch ===

        /// `scf.switch` operation: multi-way branch on integer discriminant.
        /// The body region contains `scf.case` operations followed by `scf.default`.
        /// Maps to wasm `br_table` instruction.
        fn switch(discriminant) {
            #[region(body)] {}
        };

        /// `scf.case` operation: a single case in a switch.
        /// The `value` attribute is the integer to match.
        /// The body region contains the case's code.
        #[attr(value: any)]
        fn r#case() {
            #[region(body)] {}
        };

        /// `scf.default` operation: default case in a switch.
        fn default() {
            #[region(body)] {}
        };

        // === Region Termination ===

        /// `scf.yield` operation: returns values from a region.
        fn r#yield(#[rest] values);

        // === Tail Call Optimization Results ===

        /// `scf.loop` operation: loop produced by tail recursion optimization.
        /// The body region receives loop-carried values and must end with
        /// either `scf.continue` (loop back) or `scf.break` (exit).
        fn r#loop(#[rest] init) -> result {
            #[region(body)] {}
        };

        /// `scf.continue` operation: jump to loop start with new arguments.
        fn r#continue(#[rest] values);

        /// `scf.break` operation: exit loop with result value.
        fn r#break(value);
    }
}
