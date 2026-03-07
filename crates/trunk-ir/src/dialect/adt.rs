//! ADT dialect operation registrations.
//!
//! The old Salsa-based `dialect!` macro definitions and type constructors have been removed.
//! Arena dialect definitions are in `arena/dialect/adt.rs`.
//! This file retains only `register_pure_op!` entries for the `inventory` registry.

// === Pure operation registrations ===
crate::register_pure_op!(adt.struct_new);
crate::register_pure_op!(adt.struct_get);

crate::register_pure_op!(adt.variant_new);
crate::register_pure_op!(adt.variant_is);
crate::register_pure_op!(adt.variant_cast);
crate::register_pure_op!(adt.variant_get);

crate::register_pure_op!(adt.array_new);
crate::register_pure_op!(adt.array_get);
crate::register_pure_op!(adt.array_len);

crate::register_pure_op!(adt.ref_null);
crate::register_pure_op!(adt.ref_is_null);
crate::register_pure_op!(adt.ref_cast);

crate::register_pure_op!(adt.string_const);
crate::register_pure_op!(adt.bytes_const);
