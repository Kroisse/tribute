//! Arena-based mem dialect.

// === Pure operation registrations ===
crate::register_pure_op!(mem.data);
crate::register_pure_op!(mem.load);

#[crate::dialect(crate = crate)]
mod mem {
    #[attr(bytes: any)]
    fn data() -> result {}

    #[attr(offset: u32)]
    fn load(ptr: ()) -> result {}

    #[attr(offset: u32)]
    fn store(ptr: (), value: ()) {}
}
