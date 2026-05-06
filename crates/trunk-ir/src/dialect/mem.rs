//! Arena-based mem dialect.

// === Pure operation registrations ===
crate::register_pure_op!(mem.data);
// mem.load is intentionally NOT pure: loads depend on mutable memory and may trap.

#[trunk_ir::dialect]
mod mem {
    #[attr(bytes: any)]
    fn data() -> result {}

    #[attr(offset: u32)]
    fn load(ptr: ()) -> result {}

    #[attr(offset: u32)]
    fn store(ptr: (), value: ()) {}
}
