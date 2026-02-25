//! Arena-based scf dialect.

crate::arena_dialect_internal! {
    mod scf {
        fn r#if(cond: ()) -> result {
            #[region(then_region)] {}
            #[region(else_region)] {}
        }

        fn switch(discriminant: ()) {
            #[region(body)] {}
        }

        #[attr(value: any)]
        fn case() {
            #[region(body)] {}
        }

        fn default() {
            #[region(body)] {}
        }

        fn r#yield(#[rest] values: ()) {}

        fn r#loop(#[rest] init: ()) -> result {
            #[region(body)] {}
        }

        fn r#continue(#[rest] values: ()) {}

        fn r#break(value: ()) {}
    }
}
