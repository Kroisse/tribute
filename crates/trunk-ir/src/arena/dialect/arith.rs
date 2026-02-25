//! Arena-based arith dialect.

use crate::arena_dialect;

arena_dialect! {
    mod arith {
        #[attr(value: any)]
        fn r#const() -> result;

        fn add(lhs, rhs) -> result;
        fn sub(lhs, rhs) -> result;
        fn mul(lhs, rhs) -> result;
        fn div(lhs, rhs) -> result;
        fn rem(lhs, rhs) -> result;
        fn neg(operand) -> result;

        fn cmp_eq(lhs, rhs) -> result;
        fn cmp_ne(lhs, rhs) -> result;
        fn cmp_lt(lhs, rhs) -> result;
        fn cmp_le(lhs, rhs) -> result;
        fn cmp_gt(lhs, rhs) -> result;
        fn cmp_ge(lhs, rhs) -> result;

        fn and(lhs, rhs) -> result;
        fn or(lhs, rhs) -> result;
        fn xor(lhs, rhs) -> result;
        fn shl(value, amount) -> result;
        fn shr(value, amount) -> result;
        fn shru(value, amount) -> result;

        fn cast(operand) -> result;
        fn trunc(operand) -> result;
        fn extend(operand) -> result;
        fn convert(operand) -> result;
    }
}
