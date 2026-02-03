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

        /// `cont.shift` operation: captures continuation with dynamic tag.
        ///
        /// Takes the prompt tag as an operand (runtime value) to enable
        /// evidence-based handler dispatch where the tag is looked up at runtime
        /// from the evidence structure.
        ///
        /// The first operand is the prompt tag value, followed by optional value operands.
        /// The optional `value` operands are passed to the handler along with
        /// the captured continuation. Currently only the first value is used.
        ///
        /// The result is the value passed when the continuation is resumed.
        /// This corresponds to the value returned by `ability.perform`.
        ///
        /// - `ability_ref`: ability reference type (semantic information)
        /// - `op_name`: operation name symbol (semantic information)
        /// - `op_table_index`: index into the operation table for this handler (optional)
        /// - `op_offset`: offset within the handler's operation list (optional)
        ///
        /// The op_table_index and op_offset are used for table-based dispatch.
        /// When not specified, hash-based dispatch is used as a fallback.
        ///
        /// Used for evidence-based dispatch:
        /// ```text
        /// %marker = call @__tribute_evidence_lookup(%ev, ability_id)
        /// %tag = call @__tribute_marker_prompt(%marker)
        /// %result = cont.shift(%tag, %args...) { ability_ref, op_name, op_table_index?, op_offset? }
        /// ```
        #[attr(ability_ref: Type, op_name: Symbol, op_table_index?: u32, op_offset?: u32)]
        fn shift(tag, #[rest] value) -> result {
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
        /// The `tag` attribute identifies which prompt this handler is associated with.
        /// When a shift occurs, the tag is compared to determine which handler should
        /// process the effect. If the tags don't match, the effect propagates upward.
        ///
        /// The `result_type` attribute stores the user-facing result type (the type
        /// that the done branch extracts from Step). This is needed because the
        /// operation's output type may be changed to Step during trampoline lowering.
        ///
        /// In yield bubbling, this checks the global yield state:
        /// - If not yielding: execute block 0 (done)
        /// - If yielding: dispatch to appropriate suspend block based on ability_ref + op_name
        #[attr(tag: u32, result_type: Type)]
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

        /// `cont.prompt_tag` type: prompt tag for delimited control.
        ///
        /// A unique identifier that connects `cont.push_prompt` with `cont.shift`.
        /// At runtime, this is represented as an i32 value.
        type prompt_tag;
    }
}

// === Printable interface registrations ===

use crate::type_interface::{PrintContext, Printable};
use std::fmt::Write;

// prompt_tag -> "PromptTag"
inventory::submit! { Printable::implement("cont", "prompt_tag", |_, _, f: &mut PrintContext<'_, '_>| f.write_str("PromptTag")) }
