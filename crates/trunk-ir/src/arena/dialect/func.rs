//! Arena-based func dialect.

crate::arena_dialect_internal! {
    mod func {
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() {
            #[region(body)] {}
        }

        #[attr(callee: Symbol)]
        fn call(#[rest] args: ()) -> result {}

        fn call_indirect(callee: (), #[rest] args: ()) -> result {}

        #[attr(callee: Symbol)]
        fn tail_call(#[rest] args: ()) {}

        fn r#return(#[rest] values: ()) {}

        #[attr(func_ref: Symbol)]
        fn constant() -> result {}

        fn unreachable() {}
    }
}
