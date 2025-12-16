//! Structured Control Flow dialect operations.
//!
//! Tribute uses structured control flow rather than arbitrary CFG.
//! Note: Tribute has no loop construct in the source language; only recursion.
//! The `loop`, `continue`, and `break` operations are produced by tail call inlining optimization.

use crate::dialect;

dialect! {
    scf {
        /// `scf.case` operation: pattern matching with branches.
        /// All branch regions must yield the same type.
        op case(scrutinee) { branches };

        /// `scf.yield` operation: returns values from a region.
        op r#yield(..values) {};

        // === Tail Call Optimization Results ===

        /// `scf.loop` operation: loop produced by tail recursion optimization.
        op r#loop(..init) -> result { body };

        /// `scf.continue` operation: jump to loop start with new arguments.
        op r#continue(..values) {};

        /// `scf.break` operation: exit loop with result value.
        op r#break(value) {};
    }
}
