//! Control Flow dialect operations.
//!
//! CFG-based control flow primitives for lowering structured control flow (scf)
//! to explicit branch operations. This dialect is target-independent and serves
//! as an intermediate step before target-specific lowering (e.g., cf → clif).
//!
//! Operations:
//! - `cf.br` — unconditional branch with arguments
//! - `cf.cond_br` — conditional branch to one of two successor blocks

use crate::dialect;

dialect! {
    mod cf {
        /// `cf.br` operation: unconditional branch to a successor block.
        ///
        /// Transfers control to the single successor block, passing `args` as
        /// block arguments.
        fn br(#[rest] args) {
            #[successor(dest)]
        };

        /// `cf.cond_br` operation: conditional branch to one of two successor blocks.
        ///
        /// If `cond` is true (nonzero), branches to `then_dest`; otherwise to `else_dest`.
        fn cond_br(cond) {
            #[successor(then_dest)]
            #[successor(else_dest)]
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::core;
    use crate::ops::DialectOp;
    use crate::types::DialectType;
    use crate::{BlockBuilder, Location, PathId, Span};
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

    /// Manually constructing a cf.cond_br with missing successors should
    /// return ConversionError::MissingSuccessor (not MissingRegion).
    ///
    /// Returns true if from_operation correctly returns MissingSuccessor.
    #[salsa::tracked]
    fn check_cond_br_missing_successor(db: &dyn salsa::Database) -> bool {
        let location = test_location(db);
        let i1_ty = core::I1::new(db).as_type();

        let cond_val =
            crate::dialect::arith::r#const(db, location, i1_ty, crate::Attribute::Bool(true));

        // Build a cf.cond_br operation manually with only 1 successor (needs 2)
        let one_block = BlockBuilder::new(db, location).build();
        let op = crate::ir::OperationBuilder::new(db, location, DIALECT_NAME(), COND_BR())
            .operand(cond_val.result(db))
            .successors(crate::idvec![one_block])
            .build();

        matches!(
            CondBr::from_operation(db, op),
            Err(crate::ConversionError::MissingSuccessor)
        )
    }

    #[salsa_test]
    fn test_cf_cond_br_missing_successor(db: &salsa::DatabaseImpl) {
        assert!(
            check_cond_br_missing_successor(db),
            "Expected ConversionError::MissingSuccessor for cond_br with 1 successor"
        );
    }
}
