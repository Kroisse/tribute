//! Arena-based tribute_rt dialect.

use trunk_ir::arena_dialect;

arena_dialect! {
    mod tribute_rt {
        fn box_int(value) -> result;
        fn unbox_int(value) -> result;
        fn box_nat(value) -> result;
        fn unbox_nat(value) -> result;
        fn box_float(value) -> result;
        fn unbox_float(value) -> result;
        fn box_bool(value) -> result;
        fn unbox_bool(value) -> result;

        fn retain(ptr) -> result;

        #[attr(alloc_size: u64)]
        fn release(ptr);
    }
}
