//! Pattern applicator for driving IR rewrites.
//!
//! The `PatternApplicator` manages a set of rewrite patterns and
//! applies them to a module until fixpoint is reached.

use std::collections::HashMap;

use crate::dialect::core::Module;
use crate::{Block, BlockId, IdVec, Operation, Region, Type};

use super::context::RewriteContext;
use super::conversion_target::{ConversionError, ConversionTarget};
use super::op_adaptor::OpAdaptor;
use super::pattern::RewritePattern;
use super::result::RewriteResult;

/// Result of applying patterns to a module.
pub struct ApplyResult<'db> {
    /// The transformed module.
    pub module: Module<'db>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Total number of changes across all iterations.
    pub total_changes: usize,
    /// Whether fixpoint was reached (no more changes possible).
    pub reached_fixpoint: bool,
}

impl<'db> ApplyResult<'db> {
    /// Verify that the transformed module contains no illegal operations
    /// according to the given conversion target.
    ///
    /// Returns `Ok(self)` if verification passes, or `Err(ConversionError)`
    /// if illegal operations remain.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let target = ConversionTarget::new()
    ///     .legal_dialect("trampoline")
    ///     .illegal_dialect("cont");
    ///
    /// let result = applicator.apply(db, module)
    ///     .verify(db, &target)?;  // Fails if any cont.* ops remain
    /// ```
    pub fn verify(
        self,
        db: &'db dyn salsa::Database,
        target: &ConversionTarget,
    ) -> Result<Self, ConversionError> {
        target.verify(db, &self.module)?;
        Ok(self)
    }
}

/// Applies a set of rewrite patterns to IR until fixpoint.
///
/// # Example
///
/// ```
/// # use salsa::Database;
/// # use salsa::DatabaseImpl;
/// # use trunk_ir::{Attribute, Block, BlockId, DialectOp, Location, Operation, PathId, Region, Span, Symbol, idvec};
/// # use trunk_ir::dialect::{arith, core};
/// # use trunk_ir::dialect::core::Module;
/// # use trunk_ir::types::DialectType;
/// use trunk_ir::rewrite::{ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
///
/// /// Pattern that replaces `arith.const(0)` with `arith.const(1)`.
/// struct ZeroToOnePattern;
///
/// impl<'db> RewritePattern<'db> for ZeroToOnePattern {
///     fn match_and_rewrite(
///         &self,
///         db: &'db dyn salsa::Database,
///         op: &Operation<'db>,
///         _adaptor: &OpAdaptor<'db, '_>,
///     ) -> RewriteResult<'db> {
///         let Ok(const_op) = arith::Const::from_operation(db, *op) else {
///             return RewriteResult::Unchanged;
///         };
///         if const_op.value(db) != Attribute::IntBits(0) {
///             return RewriteResult::Unchanged;
///         }
///         let i32_ty = core::I32::new(db).as_type();
///         let new_op = arith::r#const(db, op.location(db), i32_ty, Attribute::IntBits(1));
///         RewriteResult::Replace(new_op.as_operation())
///     }
/// }
/// # #[salsa::tracked]
/// # fn make_module(db: &dyn salsa::Database) -> Module<'_> {
/// #     let path = PathId::new(db, "file:///test.trb".to_owned());
/// #     let location = Location::new(path, Span::new(0, 0));
/// #     let i32_ty = core::I32::new(db).as_type();
/// #     let op = arith::r#const(db, location, i32_ty, Attribute::IntBits(0)).as_operation();
/// #     let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![op]);
/// #     let region = Region::new(db, location, idvec![block]);
/// #     Module::create(db, location, Symbol::new("test"), region)
/// # }
/// # #[salsa::tracked]
/// # fn apply_pattern(db: &dyn salsa::Database, module: Module<'_>) -> bool {
/// #     use trunk_ir::rewrite::TypeConverter;
/// #     let target = ConversionTarget::new();
/// #     let applicator = PatternApplicator::new(TypeConverter::new())
/// #         .add_pattern(ZeroToOnePattern)
/// #         .with_max_iterations(50);
/// #     let result = applicator.apply_partial(db, module, target);
/// #     result.reached_fixpoint
/// # }
/// # DatabaseImpl::default().attach(|db| {
/// #     let module = make_module(db);
/// let reached = apply_pattern(db, module);
/// assert!(reached);
/// # });
/// ```
pub struct PatternApplicator<'db> {
    patterns: Vec<Box<dyn RewritePattern<'db> + 'db>>,
    max_iterations: usize,
    type_converter: super::TypeConverter,
}

impl<'db> PatternApplicator<'db> {
    /// Create a new pattern applicator with a type converter.
    ///
    /// The `OpAdaptor` will convert types using this converter,
    /// providing patterns with already-converted types via `operand_type()`.
    ///
    /// Use `TypeConverter::new()` for an empty converter if no type
    /// conversions are needed.
    pub fn new(type_converter: super::TypeConverter) -> Self {
        Self {
            patterns: Vec::new(),
            max_iterations: 100,
            type_converter,
        }
    }

    /// Set the maximum number of iterations before giving up.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Add a pattern to the applicator.
    pub fn add_pattern<P>(mut self, pattern: P) -> Self
    where
        P: RewritePattern<'db> + 'db,
    {
        self.patterns.push(Box::new(pattern));
        self
    }

    /// Apply all patterns to a module until fixpoint, then verify conversion.
    ///
    /// This method:
    /// 1. Skips pattern matching for operations that are already legal
    /// 2. Applies patterns until fixpoint is reached
    /// 3. Verifies that no illegal operations remain
    ///
    /// Returns `Err(ConversionError)` if illegal operations remain after conversion.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let target = ConversionTarget::new()
    ///     .legal_dialect("trampoline")
    ///     .illegal_dialect("cont");
    ///
    /// let result = applicator.apply(db, module, target)?;
    /// // All cont.* ops are guaranteed to be converted
    /// ```
    pub fn apply(
        &self,
        db: &'db dyn salsa::Database,
        module: Module<'db>,
        target: ConversionTarget,
    ) -> Result<ApplyResult<'db>, ConversionError> {
        let result = self.apply_internal(db, module, &target);
        target.verify(db, &result.module)?;
        Ok(result)
    }

    /// Apply all patterns to a module until fixpoint, without verification.
    ///
    /// This method skips pattern matching for legal operations (optimization)
    /// but does NOT verify that all illegal operations are converted.
    ///
    /// Use this for:
    /// - Partial conversions where some illegal ops may remain
    /// - Multi-phase lowering where verification happens at a later stage
    ///
    /// # Example
    ///
    /// ```ignore
    /// let target = ConversionTarget::new()
    ///     .legal_dialect("func")
    ///     .illegal_dialect("tribute");
    ///
    /// // Only convert some tribute.* ops, others handled by later passes
    /// let result = applicator.apply_partial(db, module, target);
    /// ```
    pub fn apply_partial(
        &self,
        db: &'db dyn salsa::Database,
        module: Module<'db>,
        target: ConversionTarget,
    ) -> ApplyResult<'db> {
        self.apply_internal(db, module, &target)
    }

    /// Internal apply implementation used by both `apply` and `apply_partial`.
    fn apply_internal(
        &self,
        db: &'db dyn salsa::Database,
        module: Module<'db>,
        target: &ConversionTarget,
    ) -> ApplyResult<'db> {
        let mut current = module;
        let mut total_changes = 0;

        for iteration in 0..self.max_iterations {
            // Collect raw block argument types for this iteration.
            // Type conversion is applied at access sites (OpAdaptor::get_value_type).
            let block_arg_types = collect_block_arg_types(db, &current);
            let mut ctx = RewriteContext::with_block_arg_types(block_arg_types);
            let new_module = self.rewrite_module(db, &current, &mut ctx, target);

            if ctx.changes_made() == 0 {
                // Fixpoint reached
                return ApplyResult {
                    module: new_module,
                    iterations: iteration + 1,
                    total_changes,
                    reached_fixpoint: true,
                };
            }

            total_changes += ctx.changes_made();
            current = new_module;
        }

        // Max iterations reached without fixpoint
        ApplyResult {
            module: current,
            iterations: self.max_iterations,
            total_changes,
            reached_fixpoint: false,
        }
    }

    /// Rewrite a module (single pass).
    fn rewrite_module(
        &self,
        db: &'db dyn salsa::Database,
        module: &Module<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Module<'db> {
        let body = module.body(db);
        let new_body = self.rewrite_region(db, &body, ctx, target);

        // Rebuild module with new body
        Module::create(db, module.location(db), module.name(db), new_body)
    }

    /// Rewrite a region.
    fn rewrite_region(
        &self,
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(db)
            .iter()
            .map(|block| self.rewrite_block(db, block, ctx, target))
            .collect();

        Region::new(db, region.location(db), new_blocks)
    }

    /// Rewrite a block.
    fn rewrite_block(
        &self,
        db: &'db dyn salsa::Database,
        block: &Block<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Block<'db> {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(db)
            .iter()
            .flat_map(|op| self.rewrite_operation(db, op, ctx, target))
            .collect();

        Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            new_ops,
        )
    }

    /// Check if an operation is legal according to the conversion target.
    ///
    /// This uses `is_legal_op()` which includes dynamic legality checks.
    fn is_op_legal(
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        target: &ConversionTarget,
    ) -> bool {
        target.is_legal_op(db, *op)
    }

    /// Rewrite a single operation.
    ///
    /// This is the core rewrite loop:
    /// 0. Skip pattern matching if operation is already legal
    /// 1. Remap operands using the current value map
    /// 2. Compute converted operand types using the type converter
    /// 3. Create OpAdaptor with remapped operands and pre-converted types
    /// 4. Try each pattern in order
    /// 5. If a pattern matches, apply it and record mappings
    /// 6. Recursively rewrite any nested regions
    /// 7. Map original operation results to final operation results
    fn rewrite_operation(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Vec<Operation<'db>> {
        // Step 0: Skip pattern matching for legal operations
        // (still need to remap operands and process nested regions)
        // Only skip if target has constraints - otherwise try all patterns
        if target.has_constraints() && Self::is_op_legal(db, op, target) {
            let remapped_op = ctx.remap_operands(db, op);
            let final_op = self.rewrite_op_regions(db, &remapped_op, ctx, target);
            if final_op != *op {
                ctx.map_results(db, op, &final_op);
            }
            return vec![final_op];
        }

        // Step 1: Remap operands from previous transformations
        let remapped_op = ctx.remap_operands(db, op);

        // Step 2: Compute converted operand types
        let remapped_operands = remapped_op.operands(db).clone();
        let operand_types: Vec<Option<Type<'db>>> = remapped_operands
            .iter()
            .map(|v| {
                ctx.get_value_type(db, *v)
                    .map(|ty| self.type_converter.convert_type(db, ty).unwrap_or(ty))
            })
            .collect();

        // Step 3: Create OpAdaptor with remapped operands and pre-converted types
        let adaptor = OpAdaptor::new(
            remapped_op,
            remapped_operands,
            operand_types,
            ctx,
            &self.type_converter,
        );

        // Step 4: Try each pattern
        for pattern in &self.patterns {
            match pattern.match_and_rewrite(db, &remapped_op, &adaptor) {
                RewriteResult::Unchanged => continue,

                RewriteResult::Replace(new_op) => {
                    ctx.record_change();
                    // Recursively rewrite regions in the new operation
                    let final_op = self.rewrite_op_regions(db, &new_op, ctx, target);
                    // Map ORIGINAL op results to FINAL op results
                    ctx.map_results(db, op, &final_op);
                    return vec![final_op];
                }

                RewriteResult::Expand(ops) => {
                    ctx.record_change();
                    // Recursively rewrite regions in all new operations
                    let final_ops: Vec<_> = ops
                        .into_iter()
                        .map(|expanded_op| self.rewrite_op_regions(db, &expanded_op, ctx, target))
                        .collect();
                    // Map ORIGINAL op results to LAST expanded op results.
                    // The pattern is: earlier ops produce intermediate values,
                    // the last op produces the final result that replaces the original.
                    if let Some(last) = final_ops.last() {
                        ctx.map_results(db, op, last);
                    }
                    return final_ops;
                }

                RewriteResult::Erase { replacement_values } => {
                    ctx.record_change();
                    // Validate replacement count matches result count
                    let result_count = op.results(db).len();
                    debug_assert_eq!(
                        replacement_values.len(),
                        result_count,
                        "RewriteResult::Erase: replacement_values count ({}) must match operation result count ({})",
                        replacement_values.len(),
                        result_count
                    );
                    // Map ORIGINAL op results to replacement values
                    for (i, val) in replacement_values.into_iter().enumerate() {
                        if i < result_count {
                            let old_val = op.result(db, i);
                            ctx.map_value(old_val, val);
                        }
                    }
                    return vec![];
                }
            }
        }

        // Step 5: No pattern matched - recursively process regions
        let final_op = self.rewrite_op_regions(db, &remapped_op, ctx, target);

        // Step 6: Map ORIGINAL op results to FINAL op results if they differ
        // This is critical when operands were remapped but no pattern matched
        if final_op != *op {
            ctx.map_results(db, op, &final_op);
        }

        vec![final_op]
    }

    /// Rewrite nested regions within an operation.
    fn rewrite_op_regions(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Operation<'db> {
        let regions = op.regions(db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.rewrite_region(db, region, ctx, target))
            .collect();

        op.modify(db).regions(new_regions).build()
    }
}

impl<'db> Default for PatternApplicator<'db> {
    fn default() -> Self {
        Self::new(super::TypeConverter::new())
    }
}

/// Collect block argument types from a module.
///
/// Traverses all blocks in the module and collects the raw (unconverted) types
/// of their arguments. This is needed because `ValueDef::BlockArg` only stores
/// the `BlockId`, not the type information.
///
/// Type conversion is applied at access sites (e.g., `OpAdaptor::get_value_type`)
/// rather than during collection to avoid double conversion.
fn collect_block_arg_types<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> HashMap<(BlockId, usize), Type<'db>> {
    let mut map = HashMap::new();
    collect_from_region(db, &module.body(db), &mut map);
    map
}

fn collect_from_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    map: &mut HashMap<(BlockId, usize), Type<'db>>,
) {
    for block in region.blocks(db).iter() {
        let block_id = block.id(db);
        for (idx, arg) in block.args(db).iter().enumerate() {
            map.insert((block_id, idx), arg.ty(db));
        }
        // Recursively collect from nested regions in operations
        for op in block.operations(db).iter() {
            for nested_region in op.regions(db).iter() {
                collect_from_region(db, nested_region, map);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core};
    use crate::rewrite::TypeConverter;
    use crate::types::DialectType;
    use crate::{Attribute, BlockId, DialectOp, Location, PathId, Span, Symbol, idvec};
    use salsa_test_macros::salsa_test;

    /// A simple test pattern that rewrites `arith.const(42)` → `arith.mul(42, 2)`.
    /// Uses Replace to avoid infinite loop.
    struct ConstToMulPattern;

    impl<'db> RewritePattern<'db> for ConstToMulPattern {
        fn match_and_rewrite(
            &self,
            db: &'db dyn salsa::Database,
            op: &Operation<'db>,
            _adaptor: &OpAdaptor<'db, '_>,
        ) -> RewriteResult<'db> {
            // Match arith.const with value 42 only
            let Ok(const_op) = arith::Const::from_operation(db, *op) else {
                return RewriteResult::Unchanged;
            };

            let value = const_op.value(db);
            let Attribute::IntBits(42) = value else {
                return RewriteResult::Unchanged;
            };

            let location = op.location(db);
            let i32_ty = core::I32::new(db).as_type();

            // Replace const(42) with mul(7, 6) for simplicity
            let lhs_const = arith::r#const(db, location, i32_ty, Attribute::IntBits(7));
            let rhs_const = arith::r#const(db, location, i32_ty, Attribute::IntBits(6));
            let mul_op = arith::mul(
                db,
                location,
                lhs_const.result(db),
                rhs_const.result(db),
                i32_ty,
            );

            // Expand to include all operations
            RewriteResult::expand(vec![
                lhs_const.as_operation(),
                rhs_const.as_operation(),
                mul_op.as_operation(),
            ])
        }
    }

    /// Create a test module with an arith.const operation.
    #[salsa::tracked]
    fn make_const_module(db: &dyn salsa::Database) -> Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42)).as_operation();
        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, Symbol::new("test"), region)
    }

    /// Create a test module with an arith.mul operation (not matched by pattern).
    #[salsa::tracked]
    fn make_other_module(db: &dyn salsa::Database) -> Module<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        // Create dummy values for mul operands
        let dummy_const = arith::r#const(db, location, i32_ty, Attribute::IntBits(5));
        let lhs = dummy_const.result(db);
        let rhs = dummy_const.result(db);

        let const_op = dummy_const.as_operation();
        let mul_op = arith::mul(db, location, lhs, rhs, i32_ty).as_operation();
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![const_op, mul_op],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, Symbol::new("test"), region)
    }

    /// Apply patterns and return results (tracked to enable IR creation during rewrite).
    #[salsa::tracked]
    fn apply_const_to_add_pattern<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, usize, usize, Module<'db>) {
        use super::ConversionTarget;

        // Use apply_partial with an empty target (no legality constraints)
        let target = ConversionTarget::new();
        let applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(ConstToMulPattern);
        let result = applicator.apply_partial(db, module, target);
        (
            result.reached_fixpoint,
            result.total_changes,
            result.iterations,
            result.module,
        )
    }

    #[salsa_test]
    fn test_pattern_applicator_basic(db: &salsa::DatabaseImpl) {
        let module = make_const_module(db);
        let (reached_fixpoint, total_changes, _iterations, result_module) =
            apply_const_to_add_pattern(db, module);

        // Should have made one change and reached fixpoint
        assert!(reached_fixpoint);
        assert_eq!(total_changes, 1);

        // Verify we have 3 operations: const(7), const(6), mul
        let ops = result_module.body(db).blocks(db)[0].operations(db);
        assert_eq!(ops.len(), 3);

        // Verify operation names
        assert_eq!(ops[0].name(db), arith::CONST());
        assert_eq!(ops[1].name(db), arith::CONST());
        assert_eq!(ops[2].name(db), arith::MUL());

        // Verify const values
        let const0 = arith::Const::from_operation(db, ops[0]).unwrap();
        let const1 = arith::Const::from_operation(db, ops[1]).unwrap();
        assert_eq!(const0.value(db), Attribute::IntBits(7));
        assert_eq!(const1.value(db), Attribute::IntBits(6));
    }

    #[salsa_test]
    fn test_pattern_applicator_no_match(db: &salsa::DatabaseImpl) {
        let module = make_other_module(db);
        let (reached_fixpoint, total_changes, iterations, _result_module) =
            apply_const_to_add_pattern(db, module);

        // Should reach fixpoint immediately with no changes
        assert!(reached_fixpoint);
        assert_eq!(total_changes, 0);
        assert_eq!(iterations, 1);
    }

    /// Apply patterns with a conversion target that makes arith.const legal.
    #[salsa::tracked]
    fn apply_with_legal_arith<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, usize, usize, Module<'db>) {
        use super::ConversionTarget;

        let target = ConversionTarget::new().legal_dialect("arith");
        let applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(ConstToMulPattern);
        let result = applicator.apply_partial(db, module, target);
        (
            result.reached_fixpoint,
            result.total_changes,
            result.iterations,
            result.module,
        )
    }

    #[salsa_test]
    fn test_conversion_target_skips_legal_ops(db: &salsa::DatabaseImpl) {
        // Same module as basic test, but arith is marked as legal
        let module = make_const_module(db);
        let (reached_fixpoint, total_changes, iterations, result_module) =
            apply_with_legal_arith(db, module);

        // Should reach fixpoint with NO changes because arith.const is legal
        assert!(reached_fixpoint);
        assert_eq!(total_changes, 0, "Legal ops should be skipped");
        assert_eq!(iterations, 1);

        // Module should be unchanged (still has arith.const(42))
        let ops = result_module.body(db).blocks(db)[0].operations(db);
        assert_eq!(ops.len(), 1, "Module should still have 1 operation");
        assert_eq!(ops[0].name(db), arith::CONST());

        let const_op = arith::Const::from_operation(db, ops[0]).unwrap();
        assert_eq!(const_op.value(db), Attribute::IntBits(42));
    }
}
