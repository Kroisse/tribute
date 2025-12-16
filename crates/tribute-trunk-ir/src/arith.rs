//! Arithmetic dialect operations.

use crate::{Attribute, OpNameId, Operation, Symbol, Type, define_dialect_op};
use tribute_core::Location;

define_dialect_op! {
    /// `arith.const` operation: produces a constant value.
    pub struct Const("arith", "const") {
        single_result,
        has_attr("value"),
    }
}

impl<'db> Const<'db> {
    /// Create a new i32 constant.
    pub fn i32(db: &'db dyn salsa::Database, value: i64, location: Location<'db>) -> Self {
        Self::create(db, Type::I { bits: 32 }, Attribute::Int(value), location)
    }

    /// Create a new i64 constant.
    pub fn i64(db: &'db dyn salsa::Database, value: i64, location: Location<'db>) -> Self {
        Self::create(db, Type::I { bits: 64 }, Attribute::Int(value), location)
    }

    /// Create a new f32 constant.
    pub fn f32(db: &'db dyn salsa::Database, value: f32, location: Location<'db>) -> Self {
        Self::create(
            db,
            Type::F { bits: 32 },
            Attribute::FloatBits(value.to_bits() as u64),
            location,
        )
    }

    /// Create a new f64 constant.
    pub fn f64(db: &'db dyn salsa::Database, value: f64, location: Location<'db>) -> Self {
        Self::create(
            db,
            Type::F { bits: 64 },
            Attribute::FloatBits(value.to_bits()),
            location,
        )
    }

    fn create(
        db: &'db dyn salsa::Database,
        ty: Type,
        value: Attribute<'db>,
        location: Location<'db>,
    ) -> Self {
        let name = OpNameId::new(db, "arith", "const");
        let op = Operation::of(db, name, location)
            .result(ty)
            .attr("value", value)
            .build();
        Self::wrap_unchecked(op)
    }

    /// Get the constant value attribute.
    pub fn value(&self, db: &'db dyn salsa::Database) -> Attribute<'db> {
        let key = Symbol::new(db, "value");
        self.op.attributes(db).get(&key).cloned().unwrap()
    }
}

define_dialect_op! {
    /// `arith.add` operation: adds two values.
    pub struct Add("arith", "add") {
        single_result,
        operand_count(2),
    }
}

impl<'db> Add<'db> {
    /// Create a new add operation.
    pub fn new(
        db: &'db dyn salsa::Database,
        lhs: crate::Value<'db>,
        rhs: crate::Value<'db>,
        result_ty: Type,
        location: Location<'db>,
    ) -> Self {
        let name = OpNameId::new(db, "arith", "add");
        let op = Operation::of(db, name, location)
            .operands(vec![lhs, rhs])
            .result(result_ty)
            .build();
        Self::wrap_unchecked(op)
    }

    /// Get the left-hand side operand.
    pub fn lhs(&self, db: &'db dyn salsa::Database) -> crate::Value<'db> {
        self.op.operands(db)[0]
    }

    /// Get the right-hand side operand.
    pub fn rhs(&self, db: &'db dyn salsa::Database) -> crate::Value<'db> {
        self.op.operands(db)[1]
    }
}
