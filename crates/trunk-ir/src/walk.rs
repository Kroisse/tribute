//! Recursive operation traversal utilities.
//!
//! Provides a typed `walk` method for traversing nested operations,
//! similar to MLIR's `Operation::walk()`.
//!
//! # Example
//!
//! ```ignore
//! use std::ops::ControlFlow;
//! use trunk_ir::walk::{OperationWalk, WalkAction};
//!
//! // Find first shift operation
//! let result = region.walk::<cont::Shift, _>(db, |shift| {
//!     ControlFlow::Break(shift)
//! });
//!
//! // Check if any shift exists
//! let has_shift = region
//!     .walk::<cont::Shift, _>(db, |_| ControlFlow::Break(()))
//!     .is_break();
//!
//! // Collect all shifts
//! let mut shifts = Vec::new();
//! region.walk::<cont::Shift, _>(db, |shift| {
//!     shifts.push(shift);
//!     ControlFlow::Continue(WalkAction::Advance)
//! });
//! ```

use std::ops::ControlFlow;

use crate::{Block, DialectOp, Operation, Region};

/// Controls whether to descend into children during a walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkAction {
    /// Continue walking and descend into nested regions.
    Advance,
    /// Skip the nested regions of the current operation.
    Skip,
}

/// Trait for recursive operation traversal.
pub trait OperationWalk<'db> {
    /// Walk all operations recursively, applying `f` to each.
    ///
    /// Returns `ControlFlow::Break(b)` if the callback returns `Break(b)`.
    /// The callback can return `Continue(Skip)` to skip nested regions.
    fn walk_all<B>(
        &self,
        db: &'db dyn salsa::Database,
        f: impl FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()>;

    /// Walk operations of a specific dialect type recursively.
    ///
    /// Only calls `f` for operations that match type `T`.
    /// Non-matching operations are still traversed (their children are visited).
    fn walk<T, B>(
        &self,
        db: &'db dyn salsa::Database,
        f: impl FnMut(T) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()>
    where
        T: DialectOp<'db>;
}

// Internal helper to avoid recursion limit issues with impl FnMut
fn walk_op_internal<'db, B>(
    db: &'db dyn salsa::Database,
    op: Operation<'db>,
    f: &mut dyn FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()> {
    match f(op) {
        ControlFlow::Break(b) => return ControlFlow::Break(b),
        ControlFlow::Continue(WalkAction::Skip) => return ControlFlow::Continue(()),
        ControlFlow::Continue(WalkAction::Advance) => {}
    }
    for region in op.regions(db).iter() {
        walk_region_internal(db, *region, f)?;
    }
    ControlFlow::Continue(())
}

fn walk_region_internal<'db, B>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    f: &mut dyn FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()> {
    for block in region.blocks(db).iter() {
        walk_block_internal(db, *block, f)?;
    }
    ControlFlow::Continue(())
}

fn walk_block_internal<'db, B>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
    f: &mut dyn FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()> {
    for op in block.operations(db).iter() {
        walk_op_internal(db, *op, f)?;
    }
    ControlFlow::Continue(())
}

impl<'db> OperationWalk<'db> for Operation<'db> {
    fn walk_all<B>(
        &self,
        db: &'db dyn salsa::Database,
        mut f: impl FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()> {
        walk_op_internal(db, *self, &mut f)
    }

    fn walk<T, B>(
        &self,
        db: &'db dyn salsa::Database,
        mut f: impl FnMut(T) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()>
    where
        T: DialectOp<'db>,
    {
        self.walk_all(db, |op| {
            if let Ok(typed_op) = T::from_operation(db, op) {
                f(typed_op)
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        })
    }
}

impl<'db> OperationWalk<'db> for Region<'db> {
    fn walk_all<B>(
        &self,
        db: &'db dyn salsa::Database,
        mut f: impl FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()> {
        walk_region_internal(db, *self, &mut f)
    }

    fn walk<T, B>(
        &self,
        db: &'db dyn salsa::Database,
        mut f: impl FnMut(T) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()>
    where
        T: DialectOp<'db>,
    {
        self.walk_all(db, |op| {
            if let Ok(typed_op) = T::from_operation(db, op) {
                f(typed_op)
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        })
    }
}

impl<'db> OperationWalk<'db> for Block<'db> {
    fn walk_all<B>(
        &self,
        db: &'db dyn salsa::Database,
        mut f: impl FnMut(Operation<'db>) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()> {
        walk_block_internal(db, *self, &mut f)
    }

    fn walk<T, B>(
        &self,
        db: &'db dyn salsa::Database,
        mut f: impl FnMut(T) -> ControlFlow<B, WalkAction>,
    ) -> ControlFlow<B, ()>
    where
        T: DialectOp<'db>,
    {
        self.walk_all(db, |op| {
            if let Ok(typed_op) = T::from_operation(db, op) {
                f(typed_op)
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core, func};
    use crate::ir::BlockBuilder;
    use crate::{DialectType, IdVec, Location, PathId, Span, idvec};
    use salsa_test_macros::salsa_test;

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn build_simple_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let mut builder = BlockBuilder::new(db, location);
        builder.op(arith::Const::i32(db, location, 42));
        builder.op(arith::Const::i32(db, location, 100));
        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }

    #[salsa::tracked]
    fn build_three_const_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let mut builder = BlockBuilder::new(db, location);
        builder.op(arith::Const::i32(db, location, 1));
        builder.op(arith::Const::i32(db, location, 2));
        builder.op(arith::Const::i32(db, location, 3));
        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }

    #[salsa::tracked]
    fn build_nested_func_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create nested structure: func containing arith.const
        let func_op = func::Func::build(db, location, "test_fn", idvec![], i32_ty, |entry| {
            entry.op(arith::Const::i32(db, location, 42));
        });

        let mut outer_builder = BlockBuilder::new(db, location);
        outer_builder.op(func_op.as_operation());
        let outer_block = outer_builder.build();
        Region::new(db, location, IdVec::from(vec![outer_block]))
    }

    #[salsa_test]
    fn test_walk_finds_operation(db: &salsa::DatabaseImpl) {
        let region = build_simple_region(db);

        // Count arith.const operations
        let mut count = 0;
        let _ = region.walk::<arith::Const, ()>(db, |_| {
            count += 1;
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(count, 2);
    }

    #[salsa_test]
    fn test_walk_early_exit(db: &salsa::DatabaseImpl) {
        let region = build_three_const_region(db);

        // Find first const and break immediately
        let mut visited = 0;
        let result = region.walk::<arith::Const, _>(db, |_c| {
            visited += 1;
            ControlFlow::Break(())
        });

        assert!(result.is_break());
        assert_eq!(
            visited, 1,
            "Should only visit one operation before breaking"
        );
    }

    #[salsa_test]
    fn test_walk_skip_nested(db: &salsa::DatabaseImpl) {
        let outer_region = build_nested_func_region(db);

        // The const should not be visited when we skip func's nested regions
        let mut found_const = false;
        let _ = outer_region.walk_all::<()>(db, |op| {
            let dialect = op.dialect(db).to_string();
            let name = op.name(db).to_string();
            if dialect == "func" && name == "func" {
                ControlFlow::Continue(WalkAction::Skip)
            } else if dialect == "arith" {
                found_const = true;
                ControlFlow::Continue(WalkAction::Advance)
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        });

        assert!(
            !found_const,
            "const should not be found when skipping func body"
        );
    }

    #[salsa_test]
    fn test_walk_nested_regions(db: &salsa::DatabaseImpl) {
        let outer_region = build_nested_func_region(db);

        // Walk with Advance should find nested const
        let mut found_const = false;
        let _ = outer_region.walk::<arith::Const, ()>(db, |_| {
            found_const = true;
            ControlFlow::Continue(WalkAction::Advance)
        });

        assert!(found_const, "should find nested const with Advance");
    }
}
