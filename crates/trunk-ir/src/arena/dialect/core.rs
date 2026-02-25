//! Arena-based core dialect.

use crate::arena_dialect;

arena_dialect! {
    mod core {
        #[attr(sym_name: Symbol)]
        fn module() {
            #[region(body)] {}
        };

        fn unrealized_conversion_cast(value) -> result;
    }
}
