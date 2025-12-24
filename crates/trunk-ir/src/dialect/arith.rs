//! Arithmetic dialect operations.

use super::core;
use crate::{Attribute, DialectType, Location, dialect};

dialect! {
    mod arith {
        // === Constants ===

        /// `arith.const` operation: produces a constant value.
        #[attr(value: any)]
        fn r#const() -> result;

        // === Arithmetic ===

        /// `arith.add` operation: integer/float addition.
        fn add(lhs, rhs) -> result;

        /// `arith.sub` operation: integer/float subtraction.
        fn sub(lhs, rhs) -> result;

        /// `arith.mul` operation: integer/float multiplication.
        fn mul(lhs, rhs) -> result;

        /// `arith.div` operation: integer/float division.
        fn div(lhs, rhs) -> result;

        /// `arith.rem` operation: integer/float remainder.
        fn rem(lhs, rhs) -> result;

        /// `arith.neg` operation: integer/float negation.
        fn neg(operand) -> result;

        // === Comparisons ===

        /// `arith.cmp_eq` operation: equality comparison.
        fn cmp_eq(lhs, rhs) -> result;

        /// `arith.cmp_ne` operation: inequality comparison.
        fn cmp_ne(lhs, rhs) -> result;

        /// `arith.cmp_lt` operation: less-than comparison.
        fn cmp_lt(lhs, rhs) -> result;

        /// `arith.cmp_le` operation: less-than-or-equal comparison.
        fn cmp_le(lhs, rhs) -> result;

        /// `arith.cmp_gt` operation: greater-than comparison.
        fn cmp_gt(lhs, rhs) -> result;

        /// `arith.cmp_ge` operation: greater-than-or-equal comparison.
        fn cmp_ge(lhs, rhs) -> result;

        // === Bitwise ===

        /// `arith.and` operation: bitwise AND.
        fn and(lhs, rhs) -> result;

        /// `arith.or` operation: bitwise OR.
        fn or(lhs, rhs) -> result;

        /// `arith.xor` operation: bitwise XOR.
        fn xor(lhs, rhs) -> result;

        /// `arith.shl` operation: shift left.
        fn shl(value, amount) -> result;

        /// `arith.shr` operation: arithmetic shift right (sign-extending).
        fn shr(value, amount) -> result;

        /// `arith.shru` operation: logical shift right (zero-extending).
        fn shru(value, amount) -> result;

        // === Type Conversions ===

        /// `arith.cast` operation: sign extension/truncation.
        fn cast(operand) -> result;

        /// `arith.trunc` operation: truncation to smaller type.
        fn trunc(operand) -> result;

        /// `arith.extend` operation: extension to larger type.
        fn extend(operand) -> result;

        /// `arith.convert` operation: int ↔ float conversion.
        fn convert(operand) -> result;
    }
}

impl<'db> Const<'db> {
    /// Create a new i32 constant.
    pub fn i32(db: &'db dyn salsa::Database, location: Location<'db>, value: i32) -> Self {
        r#const(
            db,
            location,
            core::I32::new(db).as_type(),
            i64::from(value).into(),
        )
    }

    /// Create a new i64 constant.
    pub fn i64(db: &'db dyn salsa::Database, location: Location<'db>, value: i64) -> Self {
        r#const(db, location, core::I64::new(db).as_type(), value.into())
    }

    /// Create a new u64 constant.
    pub fn u64(db: &'db dyn salsa::Database, location: Location<'db>, value: u64) -> Self {
        r#const(db, location, core::I64::new(db).as_type(), value.into())
    }

    /// Create a new f32 constant.
    pub fn f32(db: &'db dyn salsa::Database, location: Location<'db>, value: f32) -> Self {
        r#const(
            db,
            location,
            core::F32::new(db).as_type(),
            Attribute::FloatBits(u64::from(value.to_bits())),
        )
    }

    /// Create a new f64 constant.
    pub fn f64(db: &'db dyn salsa::Database, location: Location<'db>, value: f64) -> Self {
        r#const(
            db,
            location,
            core::F64::new(db).as_type(),
            Attribute::FloatBits(value.to_bits()),
        )
    }
}

// === Pure operation registrations ===
// All arith operations are pure (no side effects)

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
