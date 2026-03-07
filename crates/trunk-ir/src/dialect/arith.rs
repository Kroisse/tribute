//! Arith dialect operation registrations.
//!
//! Arena dialect definitions are in `arena/dialect/arith.rs`.

crate::register_pure_op!(arith.r#const);
crate::register_pure_op!(arith.add);
crate::register_pure_op!(arith.sub);
crate::register_pure_op!(arith.mul);
crate::register_pure_op!(arith.div);
crate::register_pure_op!(arith.rem);
crate::register_pure_op!(arith.neg);
crate::register_pure_op!(arith.cmp_eq);
crate::register_pure_op!(arith.cmp_ne);
crate::register_pure_op!(arith.cmp_lt);
crate::register_pure_op!(arith.cmp_le);
crate::register_pure_op!(arith.cmp_gt);
crate::register_pure_op!(arith.cmp_ge);
crate::register_pure_op!(arith.and);
crate::register_pure_op!(arith.or);
crate::register_pure_op!(arith.xor);
crate::register_pure_op!(arith.shl);
crate::register_pure_op!(arith.shr);
crate::register_pure_op!(arith.shru);
crate::register_pure_op!(arith.cast);
crate::register_pure_op!(arith.trunc);
crate::register_pure_op!(arith.extend);
crate::register_pure_op!(arith.convert);
