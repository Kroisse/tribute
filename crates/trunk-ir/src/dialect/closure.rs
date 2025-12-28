//! Closure dialect operations.
//!
//! High-level operations for creating and destructuring closures.
//! Closures combine a function reference with captured environment values.
//! This dialect is lowered to target-specific representations (wasm/clif).

use crate::dialect;

dialect! {
    mod closure {
        /// `closure.new` operation: creates a closure from a function reference and environment.
        ///
        /// The func_ref points to a lifted lambda function that takes env as first arg.
        /// The env operand is typically an `adt.struct` containing captured values.
        #[attr(func_ref: QualifiedName)]
        fn new(env) -> result;

        /// `closure.func` operation: extracts funcref from closure.
        /// Returns a function reference that can be used with `func.call_indirect`.
        fn func(closure) -> result;

        /// `closure.env` operation: extracts environment struct from closure.
        /// Returns the same env value that was passed to `closure.new`.
        fn env(closure) -> result;
    }
}

// === Pure operation registrations ===
// All closure operations are pure

crate::register_pure_op!(closure.new);
crate::register_pure_op!(closure.func);
crate::register_pure_op!(closure.env);
