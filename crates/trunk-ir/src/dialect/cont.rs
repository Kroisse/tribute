//! Delimited continuation dialect operations.
//!
//! The `ability` dialect lowers to these operations.
//! These represent the core continuation primitives.

use crate::dialect;

dialect! {
    mod cont {
        /// `cont.push_prompt` operation: installs a prompt and executes body.
        #[attr(tag: u32)]
        fn push_prompt() -> result {
            #[region(body)] {}
        };

        /// `cont.shift` operation: captures continuation and jumps to handler.
        ///
        /// The optional `value` operands are passed to the handler along with
        /// the captured continuation. Currently only the first value is used.
        ///
        /// The result is the value passed when the continuation is resumed.
        /// This corresponds to the value returned by `ability.perform`.
        ///
        /// - `tag`: prompt tag for matching handler
        /// - `op_idx`: index of the ability operation (for multi-op abilities)
        #[attr(tag: u32, op_idx: u32)]
        fn shift(#[rest] value) -> result {
            #[region(handler)] {}
        };

        /// `cont.resume` operation: resumes a captured continuation.
        fn resume(continuation, value) -> result;

        /// `cont.drop` operation: drops a continuation (satisfies linear type).
        fn drop(continuation);

        /// `cont.handler_dispatch` operation: dispatches on handler result.
        ///
        /// This operation is used after `push_prompt` returns to dispatch
        /// between the "done" case (normal return) and "suspend" cases
        /// (effect operations).
        ///
        /// The `done_body` region is executed when the computation completed normally.
        /// The `suspend_body` region is executed when an effect was performed.
        ///
        /// In yield bubbling, this checks the global yield state:
        /// - If not yielding: execute done_body
        /// - If yielding: execute suspend_body with continuation/value bound
        fn handler_dispatch(result) -> output {
            #[region(done_body)] {}
            #[region(suspend_body)] {}
        };

        /// `cont.get_continuation` operation: gets the current continuation from yield state.
        ///
        /// This operation can only be used inside handler arm bodies (suspend_body).
        /// It retrieves the continuation that was captured by the most recent `shift`.
        /// The WASM backend converts this to global.get + ref_cast to get the continuation struct.
        fn get_continuation() -> result;

        /// `cont.get_shift_value` operation: gets the shift_value from the current continuation.
        ///
        /// This operation can only be used inside handler arm bodies (suspend_body).
        /// It retrieves the value that was passed to the effect operation (e.g., the `n` in `State::set!(n)`).
        /// The WASM backend converts this to struct.get on the continuation struct's field 3.
        fn get_shift_value() -> result;
    }
}
