//! Arena-based mem dialect.

use crate::arena_dialect;

arena_dialect! {
    mod mem {
        #[attr(bytes: any)]
        fn data() -> result;

        #[attr(offset: u32)]
        fn load(ptr) -> result;

        #[attr(offset: u32)]
        fn store(ptr, value);
    }
}
