//! Control Flow dialect operations.
//!
//! CFG-based control flow primitives for lowering structured control flow (scf)
//! to explicit branch operations. This dialect is target-independent and serves
//! as an intermediate step before target-specific lowering (e.g., cf → clif).
//!
//! Operations:
//! - `cf.br` — unconditional branch with arguments
//! - `cf.cond_br` — conditional branch to one of two successor blocks

use crate::{Block, ConversionError, DialectOp, IdVec, Location, Operation, Value, idvec, symbols};

symbols! {
    DIALECT_NAME => "cf",
    BR => "br",
    COND_BR => "cond_br",
}

// ============================================================================
// cf.br — unconditional branch
// ============================================================================

/// `cf.br` operation: unconditional branch to a successor block.
///
/// Transfers control to the single successor block, passing `args` as
/// block arguments.
///
/// ```text
/// cf.br(%a, %b) -> ^target
/// ```
#[derive(Clone, Copy, PartialEq, Eq, salsa::Update)]
pub struct Br<'db> {
    op: Operation<'db>,
}

impl<'db> Br<'db> {
    /// Arguments passed to the successor block.
    pub fn args(&self, db: &'db dyn salsa::Database) -> &[Value<'db>] {
        self.op.operands(db)
    }

    /// The target successor block.
    pub fn dest(&self, db: &'db dyn salsa::Database) -> Block<'db> {
        self.op.successors(db)[0]
    }
}

impl<'db> std::ops::Deref for Br<'db> {
    type Target = Operation<'db>;
    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl<'db> DialectOp<'db> for Br<'db> {
    const DIALECT_NAME: &'static str = "cf";
    const OP_NAME: &'static str = "br";

    fn from_operation(
        db: &'db dyn salsa::Database,
        op: Operation<'db>,
    ) -> Result<Self, ConversionError> {
        if op.dialect(db) != DIALECT_NAME() || op.name(db) != BR() {
            return Err(ConversionError::WrongOperation {
                expected: "cf.br",
                actual: op.full_name(db),
            });
        }
        if op.successors(db).is_empty() {
            return Err(ConversionError::MissingRegion);
        }
        Ok(Self { op })
    }

    fn as_operation(&self) -> Operation<'db> {
        self.op
    }
}

/// Construct a `cf.br` operation.
///
/// `args` are passed to the target block as block arguments.
pub fn br<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    args: impl IntoIterator<Item = Value<'db>>,
    dest: Block<'db>,
) -> Br<'db> {
    let operands: IdVec<Value<'db>> = args.into_iter().collect();
    let op = Operation::of(db, location, DIALECT_NAME(), BR())
        .operands(operands)
        .successors(idvec![dest])
        .build();
    Br { op }
}

// ============================================================================
// cf.cond_br — conditional branch
// ============================================================================

/// `cf.cond_br` operation: conditional branch to one of two successor blocks.
///
/// If `cond` is true (nonzero), branches to `then_dest`; otherwise to `else_dest`.
/// No arguments are passed to successor blocks (use merge block arguments via cf.br).
///
/// ```text
/// cf.cond_br(%cond) -> ^then_block, ^else_block
/// ```
#[derive(Clone, Copy, PartialEq, Eq, salsa::Update)]
pub struct CondBr<'db> {
    op: Operation<'db>,
}

impl<'db> CondBr<'db> {
    /// The condition value.
    pub fn cond(&self, db: &'db dyn salsa::Database) -> Value<'db> {
        self.op.operands(db)[0]
    }

    /// The "then" successor block (taken when cond is true).
    pub fn then_dest(&self, db: &'db dyn salsa::Database) -> Block<'db> {
        self.op.successors(db)[0]
    }

    /// The "else" successor block (taken when cond is false).
    pub fn else_dest(&self, db: &'db dyn salsa::Database) -> Block<'db> {
        self.op.successors(db)[1]
    }
}

impl<'db> std::ops::Deref for CondBr<'db> {
    type Target = Operation<'db>;
    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl<'db> DialectOp<'db> for CondBr<'db> {
    const DIALECT_NAME: &'static str = "cf";
    const OP_NAME: &'static str = "cond_br";

    fn from_operation(
        db: &'db dyn salsa::Database,
        op: Operation<'db>,
    ) -> Result<Self, ConversionError> {
        if op.dialect(db) != DIALECT_NAME() || op.name(db) != COND_BR() {
            return Err(ConversionError::WrongOperation {
                expected: "cf.cond_br",
                actual: op.full_name(db),
            });
        }
        if op.operands(db).is_empty() {
            return Err(ConversionError::WrongOperandCount {
                expected: 1,
                actual: 0,
            });
        }
        if op.successors(db).len() < 2 {
            return Err(ConversionError::MissingRegion);
        }
        Ok(Self { op })
    }

    fn as_operation(&self) -> Operation<'db> {
        self.op
    }
}

/// Construct a `cf.cond_br` operation.
pub fn cond_br<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    cond: Value<'db>,
    then_dest: Block<'db>,
    else_dest: Block<'db>,
) -> CondBr<'db> {
    let op = Operation::of(db, location, DIALECT_NAME(), COND_BR())
        .operand(cond)
        .successors(idvec![then_dest, else_dest])
        .build();
    CondBr { op }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::core;
    use crate::types::DialectType;
    use crate::{BlockBuilder, PathId, Span};
    use salsa_test_macros::salsa_test;

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_br_op(db: &dyn salsa::Database) -> Br<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let target = BlockBuilder::new(db, location).arg(i32_ty).build();
        let dummy_const = crate::dialect::arith::Const::i32(db, location, 42);

        br(db, location, [dummy_const.result(db)], target)
    }

    #[salsa_test]
    fn test_cf_br_roundtrip(db: &salsa::DatabaseImpl) {
        let br_op = make_br_op(db);
        let op = br_op.as_operation();

        // Verify dialect and name
        assert_eq!(op.dialect(db), DIALECT_NAME());
        assert_eq!(op.name(db), BR());

        // Verify from_operation roundtrip
        let recovered = Br::from_operation(db, op).unwrap();
        assert_eq!(recovered.args(db).len(), 1);
        assert_eq!(recovered.as_operation().successors(db).len(), 1);
    }

    #[salsa::tracked]
    fn make_cond_br_op(db: &dyn salsa::Database) -> CondBr<'_> {
        let location = test_location(db);
        let i1_ty = core::I1::new(db).as_type();

        let cond_val =
            crate::dialect::arith::r#const(db, location, i1_ty, crate::Attribute::Bool(true));

        let then_block = BlockBuilder::new(db, location).build();
        let else_block = BlockBuilder::new(db, location).build();

        cond_br(db, location, cond_val.result(db), then_block, else_block)
    }

    #[salsa_test]
    fn test_cf_cond_br_roundtrip(db: &salsa::DatabaseImpl) {
        let cond_br_op = make_cond_br_op(db);
        let op = cond_br_op.as_operation();

        assert_eq!(op.dialect(db), DIALECT_NAME());
        assert_eq!(op.name(db), COND_BR());

        let recovered = CondBr::from_operation(db, op).unwrap();
        assert_eq!(recovered.as_operation().successors(db).len(), 2);
    }
}
