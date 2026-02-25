//! Arena-based core dialect.

crate::arena_dialect_internal! {
    mod core {
        #[attr(sym_name: Symbol)]
        fn module() {
            #[region(body)] {}
        }

        fn unrealized_conversion_cast(value: ()) -> result {}
    }
}
