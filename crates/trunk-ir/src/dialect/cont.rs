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
        /// - `cont.done` child op receives the result value as a block argument
        /// - `cont.suspend` child ops receive (continuation, shift_value) as block arguments
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
        /// %tag = adt.struct_get(%marker, 1)  // field 1 = prompt_tag
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
        /// The `body` region contains a single block with child operations:
        /// - `cont.done`: "done" case, executed when computation completed normally.
        ///   Its body region's entry block receives the done value as a block argument.
        /// - `cont.suspend`: "suspend" cases, one per handled operation.
        ///   Each has `ability_ref` and `op_name` attributes, and its body region's
        ///   entry block receives (continuation, shift_value) as block arguments.
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
        /// - If not yielding: execute cont.done arm
        /// - If yielding: dispatch to appropriate cont.suspend arm based on ability_ref + op_name
        #[attr(tag: u32, result_type: Type)]
        fn handler_dispatch(result) -> output {
            #[region(body)] {}
        };

        /// `cont.done` operation: handler dispatch done arm.
        ///
        /// Used as a child operation inside `cont.handler_dispatch`'s body region.
        /// The body region's entry block receives the done value (extracted from Step)
        /// as a block argument, eliminating the need for `cont.get_done_value`.
        fn done() { #[region(body)] {} };

        /// `cont.suspend` operation: handler dispatch suspend arm.
        ///
        /// Used as a child operation inside `cont.handler_dispatch`'s body region.
        /// The `ability_ref` attribute identifies the ability type and `op_name`
        /// identifies the operation name for dispatch.
        ///
        /// The body region's entry block receives (continuation, shift_value) as
        /// block arguments, eliminating the need for `cont.get_continuation` and
        /// `cont.get_shift_value`.
        #[attr(ability_ref: Type, op_name: Symbol)]
        fn suspend() { #[region(body)] {} };

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
