//! Arena-based mem dialect.

// === Pure operation registrations ===
crate::register_pure_op!(Data);
// mem.load is intentionally NOT pure: loads depend on mutable memory and may trap.

#[trunk_ir::dialect]
mod mem {
    fn data(bytes: Attr<_>) -> Value<_> {}

    fn load(offset: Attr<u32>, ptr: Value<_>) -> Value<_> {}

    fn store(offset: Attr<u32>, ptr: Value<_>, value: Value<_>) {}
}
