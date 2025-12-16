//! Arithmetic dialect operations.

use crate::{Attribute, Type, dialect};
use tribute_core::Location;

dialect! {
    arith {
        /// `arith.const` operation: produces a constant value.
        pub op r#const[value]() -> result {};

        /// `arith.add` operation: adds two values.
        pub op add(lhs, rhs) -> result {};
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
