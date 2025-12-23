//! Closure dialect operations.
//!
//! High-level operations for creating and destructuring closures.
//! Closures combine a function reference with captured environment values.
//! This dialect is lowered to target-specific representations (wasm/clif).

use crate::dialect;

dialect! {
    mod closure {
        /// `closure.new` operation: creates a closure with captured values.
        /// The func_ref points to a lifted lambda function that takes env as first arg.
        #[attr(func_ref: QualifiedName)]
        fn new(#[rest] captures) -> result;

        /// `closure.func` operation: extracts funcref from closure.
        fn func(closure) -> result;

        /// `closure.env` operation: extracts environment from closure.
        fn env(closure) -> result;
    }
}
