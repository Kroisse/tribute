//! Arena-based cf dialect.

use crate::arena_dialect;

arena_dialect! {
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
