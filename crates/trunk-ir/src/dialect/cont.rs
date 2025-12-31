//! Delimited continuation dialect operations.
//!
//! The `ability` dialect lowers to these operations.
//! These represent the core continuation primitives.

use crate::dialect;

dialect! {
    mod cont {
        /// `cont.push_prompt` operation: installs a prompt and executes body.
        #[attr(tag: any)]
        fn push_prompt() -> result {
            #[region(body)] {}
        };

        /// `cont.shift` operation: captures continuation and jumps to handler.
        ///
        /// The optional `value` operand is passed to the handler along with
        /// the captured continuation.
        #[attr(tag: any)]
        fn shift(#[rest] values) {
            #[region(handler)] {}
        };

        /// `cont.resume` operation: resumes a captured continuation.
        fn resume(continuation, value) -> result;

        /// `cont.drop` operation: drops a continuation (satisfies linear type).
        fn drop(continuation);
    }
}
