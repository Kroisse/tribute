//! Arena-based cf dialect.

#[trunk_ir::dialect]
mod cf {
    fn br(#[rest] args: ()) {
        #[successor(dest)]
        {}
    }

    fn cond_br(cond: ()) {
        #[successor(then_dest)]
        {}
        #[successor(else_dest)]
        {}
    }
}
