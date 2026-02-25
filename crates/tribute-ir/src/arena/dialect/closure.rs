//! Arena-based closure dialect.

#[trunk_ir::arena_dialect]
mod closure {
    #[attr(func_ref: Symbol)]
    fn new(env: ()) -> result {}

    fn func(closure: ()) -> result {}

    fn env(closure: ()) -> result {}
}
