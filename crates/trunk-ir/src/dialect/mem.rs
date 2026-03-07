//! Mem dialect operation registrations.
//!
//! Arena dialect definitions are in `arena/dialect/mem.rs`.

crate::register_pure_op!(mem.data);
crate::register_pure_op!(mem.load);
