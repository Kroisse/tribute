//! Trampoline-based continuation implementation dialect.
//!
//! This dialect provides operations for implementing delimited continuations
//! using the yield-bubbling (trampoline) strategy. It sits between the high-level
//! `cont` dialect and the target-specific backends (wasm, cranelift).
//!
//! The trampoline strategy works by:
//! 1. Each function that may yield returns a `Step` value
//! 2. `Step::Done(value)` indicates normal completion
//! 3. `Step::Shift(continuation)` indicates a captured continuation
//! 4. Callers check the result and either continue or propagate the yield
//!
//! # Type Layout
//!
//! ## Step (trampoline return value)
//! ```text
//! (tag: i32, value: anyref, prompt: i32, op_idx: i32)
//! ```
//! - tag: 0 = Done, 1 = Shift
//! - value: the result value (Done) or unused (Shift)
//! - prompt: prompt tag for handler matching
//! - op_idx: operation index within ability
//!
//! ## Continuation (captured continuation)
//! ```text
//! (resume_fn: funcref, state: structref, tag: i32, shift_value: anyref)
//! ```
//! - resume_fn: function to call when resuming
//! - state: captured local variables
//! - tag: prompt tag
//! - shift_value: value passed to the effect operation
//!
//! ## ResumeWrapper (resume argument)
//! ```text
//! (state: anyref, resume_value: anyref)
//! ```
//! - state: the captured state to restore
//! - resume_value: the value passed to resume

use crate::dialect;

/// Tag value for Step::Done variant
pub const STEP_TAG_DONE: i32 = 0;
/// Tag value for Step::Shift variant
pub const STEP_TAG_SHIFT: i32 = 1;

dialect! {
    mod trampoline {
        // === Types ===

        /// `trampoline.step` type: result of a trampolined function.
        ///
        /// Layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
        type step;

        /// `trampoline.continuation` type: captured continuation.
        ///
        /// Layout: (resume_fn: funcref, state: structref, tag: i32, shift_value: anyref)
        type continuation;

        /// `trampoline.state` type: captured local variables at shift point.
        ///
        /// The actual layout depends on the captured variables.
        type state;

        /// `trampoline.resume_wrapper` type: argument passed when resuming.
        ///
        /// Layout: (state: anyref, resume_value: anyref)
        type resume_wrapper;

        // === Operations ===

        /// `trampoline.check_yield` operation: checks if currently in yield state.
        ///
        /// Returns a boolean (i32) indicating whether a yield is in progress.
        /// Used to implement yield bubbling after function calls.
        fn check_yield() -> result;

        /// `trampoline.build_continuation` operation: creates a continuation struct.
        ///
        /// Creates a continuation with the captured state for later resumption.
        /// - `tag`: prompt tag for handler matching (static)
        /// - `op_idx`: operation index within the ability
        #[attr(tag: u32, op_idx: u32)]
        fn build_continuation(resume_fn, state, shift_value) -> result;

        /// `trampoline.build_continuation_dynamic` operation: creates a continuation with dynamic tag.
        ///
        /// Same as build_continuation but takes tag as an operand for evidence-based dispatch.
        /// - First operand: tag (i32) - prompt tag from evidence lookup
        /// - `op_idx`: operation index within the ability
        #[attr(op_idx: u32)]
        fn build_continuation_dynamic(tag, resume_fn, state, shift_value) -> result;

        /// `trampoline.step_done` operation: creates a Done step.
        ///
        /// Indicates that a computation completed normally with the given value.
        fn step_done(value) -> result;

        /// `trampoline.step_shift` operation: creates a Shift step.
        ///
        /// Indicates that a continuation was captured and should be propagated.
        /// - `prompt`: prompt tag for handler matching (static)
        /// - `op_idx`: operation index within the ability
        #[attr(prompt: u32, op_idx: u32)]
        fn step_shift(continuation) -> result;

        /// `trampoline.step_shift_dynamic` operation: creates a Shift step with dynamic tag.
        ///
        /// Same as step_shift but takes prompt tag as an operand for evidence-based dispatch.
        /// - First operand: prompt (i32) - prompt tag from evidence lookup
        /// - `op_idx`: operation index within the ability
        #[attr(op_idx: u32)]
        fn step_shift_dynamic(prompt, continuation) -> result;

        /// `trampoline.continuation_get` operation: extracts a field from continuation.
        ///
        /// - `field`: field name ("resume_fn", "state", "tag", "shift_value")
        #[attr(field: Symbol)]
        fn continuation_get(cont) -> result;

        /// `trampoline.step_get` operation: extracts a field from step.
        ///
        /// - `field`: field name ("tag", "value", "prompt", "op_idx")
        #[attr(field: Symbol)]
        fn step_get(step) -> result;

        /// `trampoline.set_yield_state` operation: sets global yield state.
        ///
        /// Called when performing a shift to propagate yield information.
        /// - `tag`: prompt tag (static)
        /// - `op_idx`: operation index
        #[attr(tag: u32, op_idx: u32)]
        fn set_yield_state(continuation);

        /// `trampoline.set_yield_state_dynamic` operation: sets global yield state with dynamic tag.
        ///
        /// Same as set_yield_state but takes tag as an operand for evidence-based dispatch.
        /// - First operand: tag (i32) - prompt tag from evidence lookup
        /// - `op_idx`: operation index
        #[attr(op_idx: u32)]
        fn set_yield_state_dynamic(tag, continuation);

        /// `trampoline.reset_yield_state` operation: clears global yield state.
        ///
        /// Called when a handler catches the yield or when resuming a continuation.
        fn reset_yield_state();

        /// `trampoline.get_yield_continuation` operation: gets continuation from global yield state.
        ///
        /// Returns the continuation that was stored in global state by set_yield_state.
        /// Used in handler arms to access the captured continuation.
        fn get_yield_continuation() -> result;

        /// `trampoline.get_yield_shift_value` operation: gets shift_value from global yield state.
        ///
        /// Returns the shift_value from the continuation in global state.
        /// This is the value passed to the effect operation (e.g., `State::set!(n)`).
        fn get_yield_shift_value() -> result;

        /// `trampoline.get_yield_op_idx` operation: gets op_idx from global yield state.
        ///
        /// Returns the op_idx that was stored by set_yield_state.
        /// Used in handler dispatch to determine which handler arm to execute.
        fn get_yield_op_idx() -> result;

        /// `trampoline.build_state` operation: creates a state struct.
        ///
        /// Captures local variables at a shift point for later restoration.
        /// - `state_type`: the concrete state struct type
        #[attr(state_type: Type)]
        fn build_state(#[rest] locals) -> result;

        /// `trampoline.build_resume_wrapper` operation: creates a resume wrapper.
        ///
        /// Packages state and resume value for passing to the resume function.
        fn build_resume_wrapper(state, resume_value) -> result;

        /// `trampoline.resume_wrapper_get` operation: extracts a field from resume wrapper.
        ///
        /// - `field`: field name ("state", "resume_value")
        #[attr(field: Symbol)]
        fn resume_wrapper_get(wrapper) -> result;

        /// `trampoline.state_get` operation: extracts a field from state struct.
        ///
        /// Used in resume functions to restore captured local variables.
        /// - `field`: field name (e.g., "field0", "field1", ...)
        #[attr(field: Symbol)]
        fn state_get(state) -> result;
    }
}
