//! Delimited continuation dialect operations and types.
//!
//! The `ability` dialect lowers to these operations.
//! These represent the core continuation primitives.

use crate::{Symbol, dialect};

dialect! {
    mod cont {
        /// `cont.push_prompt` operation: installs a prompt and executes body.
        ///
        /// The `handlers` region is typically empty in the current implementation.
        /// Handler dispatch logic is instead implemented using `cont.handler_dispatch`,
        /// which examines the Step result returned by push_prompt and dispatches to
        /// appropriate handler arms:
        /// - "done" handler uses `cont.get_done_value` to extract the result value
        /// - "suspend" handlers use `cont.get_continuation` and `cont.get_shift_value`
        ///   to access the captured continuation and effect arguments
        #[attr(tag: u32)]
        fn push_prompt() -> result {
            #[region(body)] {}
            #[region(handlers)] {}
        };

        /// `cont.shift` operation: captures continuation and jumps to handler.
        ///
        /// The optional `value` operands are passed to the handler along with
        /// the captured continuation. Currently only the first value is used.
        ///
        /// The result is the value passed when the continuation is resumed.
        /// This corresponds to the value returned by `ability.perform`.
        ///
        /// - `tag`: prompt tag for matching handler instance (runtime identifier)
        /// - `ability_ref`: ability reference type (semantic information)
        /// - `op_name`: operation name symbol (semantic information)
        ///
        /// Note: Operation index (op_idx) is computed deterministically from op_name
        /// during WASM lowering and is not stored in the IR.
        #[attr(tag: u32, ability_ref: Type, op_name: Symbol)]
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
        /// The `body` region contains multiple blocks:
        /// - Block 0: "done" case, executed when computation completed normally
        /// - Block 1+: "suspend" cases, one per handled operation
        ///
        /// Suspend blocks have a marker block argument (nil type) with attributes:
        /// - `ability_ref`: the ability type (for distinguishing same-named ops)
        /// - `op_name`: the operation name symbol
        ///
        /// In yield bubbling, this checks the global yield state:
        /// - If not yielding: execute block 0 (done)
        /// - If yielding: dispatch to appropriate suspend block based on ability_ref + op_name
        fn handler_dispatch(result) -> output {
            #[region(body)] {}
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

        /// `cont.get_done_value` operation: extracts the value from a Done Step.
        ///
        /// This operation is used inside handler "done" arm bodies to extract the
        /// result value from a Step struct that was returned by push_prompt.
        /// The Step layout is (tag, value, prompt, op_idx), and this extracts field 1 (value).
        /// The WASM backend converts this to struct.get on the Step's value field.
        fn get_done_value(step) -> result;

        // === Types ===

        /// `cont.continuation` type: delimited continuation.
        ///
        /// Represents a captured continuation that can be resumed with a value.
        /// - First param: argument type (value passed when resuming)
        /// - Second param: result type (what resuming returns)
        #[attr(effect: Type)]
        type continuation(arg, result);
    }
}
