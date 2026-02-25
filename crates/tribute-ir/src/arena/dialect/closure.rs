//! Arena-based closure dialect.

use trunk_ir::arena_dialect;

arena_dialect! {
    mod closure {
        #[attr(func_ref: Symbol)]
        fn new(env) -> result;

        fn func(closure) -> result;

        fn env(closure) -> result;
    }
}
