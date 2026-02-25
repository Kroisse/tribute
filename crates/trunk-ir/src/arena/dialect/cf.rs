//! Arena-based cf dialect.

crate::arena_dialect_internal! {
    mod cf {
        fn br(#[rest] args) {
            #[successor(dest)]
        };

        fn cond_br(cond) {
            #[successor(then_dest)]
            #[successor(else_dest)]
        };
    }
}
