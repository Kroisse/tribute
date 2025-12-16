//! Arithmetic dialect operations.

use crate::{Attribute, Type, dialect};
use tribute_core::Location;

dialect! {
    arith {
        // === Constants ===

        /// `arith.const` operation: produces a constant value.
        pub op r#const[value]() -> result {};

        // === Arithmetic ===

        /// `arith.add` operation: integer/float addition.
        pub op add(lhs, rhs) -> result {};

        /// `arith.sub` operation: integer/float subtraction.
        pub op sub(lhs, rhs) -> result {};

        /// `arith.mul` operation: integer/float multiplication.
        pub op mul(lhs, rhs) -> result {};

        /// `arith.div` operation: integer/float division.
        pub op div(lhs, rhs) -> result {};

        /// `arith.rem` operation: integer/float remainder.
        pub op rem(lhs, rhs) -> result {};

        /// `arith.neg` operation: integer/float negation.
        pub op neg(operand) -> result {};

        // === Comparisons ===

        /// `arith.cmp_eq` operation: equality comparison.
        pub op cmp_eq(lhs, rhs) -> result {};

        /// `arith.cmp_ne` operation: inequality comparison.
        pub op cmp_ne(lhs, rhs) -> result {};

        /// `arith.cmp_lt` operation: less-than comparison.
        pub op cmp_lt(lhs, rhs) -> result {};

        /// `arith.cmp_le` operation: less-than-or-equal comparison.
        pub op cmp_le(lhs, rhs) -> result {};

        /// `arith.cmp_gt` operation: greater-than comparison.
        pub op cmp_gt(lhs, rhs) -> result {};

        /// `arith.cmp_ge` operation: greater-than-or-equal comparison.
        pub op cmp_ge(lhs, rhs) -> result {};

        // === Bitwise ===

        /// `arith.and` operation: bitwise AND.
        pub op and(lhs, rhs) -> result {};

        /// `arith.or` operation: bitwise OR.
        pub op or(lhs, rhs) -> result {};

        /// `arith.xor` operation: bitwise XOR.
        pub op xor(lhs, rhs) -> result {};

        /// `arith.shl` operation: shift left.
        pub op shl(value, amount) -> result {};

        /// `arith.shr` operation: arithmetic shift right (sign-extending).
        pub op shr(value, amount) -> result {};

        /// `arith.shru` operation: logical shift right (zero-extending).
        pub op shru(value, amount) -> result {};

        // === Type Conversions ===

        /// `arith.cast` operation: sign extension/truncation.
        pub op cast(operand) -> result {};

        /// `arith.trunc` operation: truncation to smaller type.
        pub op trunc(operand) -> result {};

        /// `arith.extend` operation: extension to larger type.
        pub op extend(operand) -> result {};

        /// `arith.convert` operation: int ↔ float conversion.
        pub op convert(operand) -> result {};
    }
}

impl<'db> Const<'db> {
    /// Create a new i32 constant.
    pub fn i32(db: &'db dyn salsa::Database, location: Location<'db>, value: i64) -> Self {
        Self::new(db, location, Type::I { bits: 32 }, Attribute::Int(value))
    }

    /// Create a new i64 constant.
    pub fn i64(db: &'db dyn salsa::Database, location: Location<'db>, value: i64) -> Self {
        Self::new(db, location, Type::I { bits: 64 }, Attribute::Int(value))
    }

    /// Create a new f32 constant.
    pub fn f32(db: &'db dyn salsa::Database, location: Location<'db>, value: f32) -> Self {
        Self::new(
            db,
            location,
            Type::F { bits: 32 },
            Attribute::FloatBits(value.to_bits() as u64),
        )
    }

    /// Create a new f64 constant.
    pub fn f64(db: &'db dyn salsa::Database, location: Location<'db>, value: f64) -> Self {
        Self::new(
            db,
            location,
            Type::F { bits: 64 },
            Attribute::FloatBits(value.to_bits()),
        )
    }
}
