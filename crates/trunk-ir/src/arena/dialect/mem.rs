//! Arena-based mem dialect.

#[crate::arena_dialect(crate = crate)]
mod mem {
    #[attr(bytes: any)]
    fn data() -> result {}

    #[attr(offset: u32)]
    fn load(ptr: ()) -> result {}

    #[attr(offset: u32)]
    fn store(ptr: (), value: ()) {}
}
