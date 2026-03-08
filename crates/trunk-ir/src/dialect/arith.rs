//! Arena-based arith dialect.

// === Pure operation registrations ===
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

#[crate::dialect(crate = crate)]
mod arith {
    #[attr(value: any)]
    fn r#const() -> result {}

    fn add(lhs: (), rhs: ()) -> result {}
    fn sub(lhs: (), rhs: ()) -> result {}
    fn mul(lhs: (), rhs: ()) -> result {}
    fn div(lhs: (), rhs: ()) -> result {}
    fn rem(lhs: (), rhs: ()) -> result {}
    fn neg(operand: ()) -> result {}

    fn cmp_eq(lhs: (), rhs: ()) -> result {}
    fn cmp_ne(lhs: (), rhs: ()) -> result {}
    fn cmp_lt(lhs: (), rhs: ()) -> result {}
    fn cmp_le(lhs: (), rhs: ()) -> result {}
    fn cmp_gt(lhs: (), rhs: ()) -> result {}
    fn cmp_ge(lhs: (), rhs: ()) -> result {}

    fn and(lhs: (), rhs: ()) -> result {}
    fn or(lhs: (), rhs: ()) -> result {}
    fn xor(lhs: (), rhs: ()) -> result {}
    fn shl(value: (), amount: ()) -> result {}
    fn shr(value: (), amount: ()) -> result {}
    fn shru(value: (), amount: ()) -> result {}

    fn cast(operand: ()) -> result {}
    fn trunc(operand: ()) -> result {}
    fn extend(operand: ()) -> result {}
    fn convert(operand: ()) -> result {}
}
