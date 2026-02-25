//! Arena-based trampoline dialect.

crate::arena_dialect_internal! {
    mod trampoline {
        fn check_yield() -> result;

        #[attr(op_idx: u32)]
        fn build_continuation(tag, resume_fn, state, shift_value) -> result;

        fn step_done(value) -> result;

        #[attr(op_idx: u32)]
        fn step_shift(prompt, continuation) -> result;

        #[attr(field: u32)]
        fn continuation_get(cont) -> result;

        #[attr(field: u32)]
        fn step_get(step) -> result;

        #[attr(op_idx: u32)]
        fn set_yield_state(tag, continuation);

        fn reset_yield_state();

        fn get_yield_continuation() -> result;

        fn get_yield_shift_value() -> result;

        fn get_yield_op_idx() -> result;

        #[attr(state_type: Type)]
        fn build_state(#[rest] locals) -> result;

        fn build_resume_wrapper(state, resume_value) -> result;

        #[attr(field: u32)]
        fn resume_wrapper_get(wrapper) -> result;

        #[attr(field: u32)]
        fn state_get(state) -> result;
    }
}
