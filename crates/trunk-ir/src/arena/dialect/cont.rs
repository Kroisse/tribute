//! Arena-based cont dialect.

#[crate::arena_dialect(crate = crate)]
mod cont {
    #[attr(tag: any)]
    fn push_prompt(#[rest] args: ()) -> result {
        #[region(body)]
        {}
        #[region(handlers)]
        {}
    }

    #[attr(ability_ref: Type, op_name: Symbol, op_table_index?: u32, op_offset?: u32)]
    fn shift(tag: (), #[rest] values: ()) -> result {
        #[region(handler)]
        {}
    }

    fn resume(continuation: (), value: ()) -> result {}

    fn drop(continuation: ()) {}

    #[attr(tag: u32, result_type: Type)]
    fn handler_dispatch(value: ()) -> result {
        #[region(body)]
        {}
    }

    fn done() {
        #[region(body)]
        {}
    }

    #[attr(ability_ref: Type, op_name: Symbol)]
    fn suspend() {
        #[region(body)]
        {}
    }
}
