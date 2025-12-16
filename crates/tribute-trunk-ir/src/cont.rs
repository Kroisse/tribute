//! Delimited continuation dialect operations.
//!
//! The `ability` dialect lowers to these operations.
//! These represent the core continuation primitives.

use crate::dialect;

dialect! {
    cont {
        /// `cont.push_prompt` operation: installs a prompt and executes body.
        op push_prompt[tag]() -> result { body };

        /// `cont.shift` operation: captures continuation and jumps to handler.
        op shift[tag]() { handler };

        /// `cont.resume` operation: resumes a captured continuation.
        op resume(continuation, value) -> result {};

        /// `cont.drop` operation: drops a continuation (satisfies linear type).
        op drop(continuation) {};
    }
}
