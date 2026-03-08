//! Arena-based cf dialect.

#[crate::dialect(crate = crate)]
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
