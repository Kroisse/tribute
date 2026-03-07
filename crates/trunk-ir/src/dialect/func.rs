//! Func dialect operation registrations.
//!
//! Arena dialect definitions are in `arena/dialect/func.rs`.

crate::register_pure_op!(func.constant);
crate::register_isolated_op!(func.func);
