//! Arena-based core dialect.

// === Operation registrations ===
crate::register_isolated_op!(core.module);

#[crate::dialect(crate = crate)]
mod core {
    #[attr(sym_name: Symbol)]
    fn module() {
        #[region(body)]
        {}
    }

    fn unrealized_conversion_cast(value: ()) -> result {}

    struct Nil;
    struct Never;
    struct Bytes;
    struct Ptr;
    struct Array<Element>;
    #[attr(nullable: bool)]
    struct Ref<Pointee>;
    struct Tuple<#[rest] Elements>;
    struct Func<Return, #[rest] Params>;
}
