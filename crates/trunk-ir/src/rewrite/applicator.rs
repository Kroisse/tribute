//! Pattern applicator for driving IR rewrites.
//!
//! The `PatternApplicator` manages a set of rewrite patterns and
//! applies them to a module until fixpoint is reached.

use std::collections::HashMap;

use crate::dialect::core::{self, Module};
use crate::ops::DialectOp;
use crate::{Block, BlockId, IdVec, Operation, Region, Type, Value};

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
    /// 1. Skips pattern matching and cast insertion for operations that are already legal
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
    ///     .illegal_dialect("ability");
    ///
    /// // Only convert some ability.* ops, others handled by later passes
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
    ///
    /// Uses a 2-pass approach to handle successor block references:
    /// 1. Rewrite all blocks, collecting old→new block mappings
    /// 2. Remap successor references in all operations using the block map
    fn rewrite_region(
        &self,
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Region<'db> {
        // Pass 1: Rewrite all blocks and collect old→new mappings
        let pairs: Vec<(Block<'db>, Block<'db>)> = region
            .blocks(db)
            .iter()
            .map(|old| (*old, self.rewrite_block(db, old, ctx, target)))
            .collect();

        let block_map: HashMap<Block<'db>, Block<'db>> = pairs.iter().copied().collect();

        // Pass 2: Remap successor references using the block map
        let new_blocks: IdVec<Block<'db>> = pairs
            .into_iter()
            .map(|(_, new_block)| remap_block_successors(db, new_block, &block_map))
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

    /// Insert `core.unrealized_conversion_cast` operations for type mismatches.
    ///
    /// When an operand's raw type differs from its converted type, this method
    /// inserts a cast operation to bridge the type mismatch. The cast will be
    /// resolved later by a materialization pass.
    ///
    /// Returns:
    /// - The (possibly modified) operands with cast results replacing original values
    /// - A vector of cast operations to prepend before the current operation
    fn insert_conversion_casts(
        &self,
        db: &'db dyn salsa::Database,
        location: crate::Location<'db>,
        operands: &IdVec<Value<'db>>,
        ctx: &RewriteContext<'db>,
    ) -> (IdVec<Value<'db>>, Vec<Operation<'db>>) {
        let mut cast_ops = Vec::new();
        let mut new_operands = IdVec::with_capacity(operands.len());

        for operand in operands.iter() {
            let Some(raw_ty) = ctx.get_value_type(db, *operand) else {
                new_operands.push(*operand);
                continue;
            };

            let converted_ty = self.type_converter.convert_type(db, raw_ty);
            if let Some(target_ty) = converted_ty {
                // Type needs conversion - insert unrealized_conversion_cast
                let cast_op = core::unrealized_conversion_cast(db, location, *operand, target_ty);
                let cast_result = cast_op.as_operation().result(db, 0);
                cast_ops.push(cast_op.as_operation());
                new_operands.push(cast_result);
            } else {
                // Type is already legal - keep original operand
                new_operands.push(*operand);
            }
        }

        (new_operands, cast_ops)
    }

    /// Rewrite a single operation.
    ///
    /// This is the core rewrite loop:
    /// 1. Remap operands using the current value map
    /// 2. Skip if operation is already legal (remap + region recursion only, NO casts)
    /// 3. Insert unrealized_conversion_cast for type mismatches (illegal ops only)
    /// 4. Compute converted operand types using the type converter
    /// 5. Create OpAdaptor with remapped operands and pre-converted types
    /// 6. Try each pattern in order
    /// 7. If a pattern matches, apply it and record mappings
    /// 8. Recursively rewrite any nested regions
    /// 9. Map original operation results to final operation results
    fn rewrite_operation(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        ctx: &mut RewriteContext<'db>,
        target: &ConversionTarget,
    ) -> Vec<Operation<'db>> {
        // Step 1: Remap operands from previous transformations
        let remapped_op = ctx.remap_operands(db, op);

        // Step 2: Skip pattern matching and cast insertion for legal operations
        // Legal ops only need operand remapping and region recursion - no casts.
        // Only skip if target has constraints - otherwise try all patterns.
        if target.has_constraints() && Self::is_op_legal(db, &remapped_op, target) {
            let final_op = self.rewrite_op_regions(db, &remapped_op, ctx, target);
            if final_op != *op {
                ctx.map_results(db, op, &final_op);
            }
            return vec![final_op];
        }

        // Step 3: Insert conversion casts for type mismatches (illegal ops only)
        // Skip inserting casts for unrealized_conversion_cast itself to avoid infinite loops.
        // The cast op is the type conversion boundary - its operands keep the original type.
        let location = remapped_op.location(db);
        let remapped_operands = remapped_op.operands(db).clone();
        let (casted_operands, cast_ops) = if remapped_op.dialect(db) == core::DIALECT_NAME()
            && remapped_op.name(db) == core::UNREALIZED_CONVERSION_CAST()
        {
            (remapped_operands.clone(), Vec::new())
        } else {
            self.insert_conversion_casts(db, location, &remapped_operands, ctx)
        };

        // Update operation with casted operands if any casts were inserted
        let remapped_op = if !cast_ops.is_empty() {
            ctx.record_change();
            remapped_op
                .modify(db)
                .operands(casted_operands.clone())
                .build()
        } else {
            remapped_op
        };

        // Step 4: Compute converted operand types (now using casted operands)
        let operand_types: Vec<Option<Type<'db>>> = casted_operands
            .iter()
            .map(|v| {
                ctx.get_value_type(db, *v)
                    .map(|ty| self.type_converter.convert_type(db, ty).unwrap_or(ty))
            })
            .collect();

        // Step 5: Create OpAdaptor with remapped operands and pre-converted types
        let adaptor = OpAdaptor::new(
            remapped_op,
            casted_operands,
            operand_types,
            ctx,
            &self.type_converter,
        );

        // Step 6: Try each pattern
        for pattern in &self.patterns {
            match pattern.match_and_rewrite(db, &remapped_op, &adaptor) {
                RewriteResult::Unchanged => continue,

                RewriteResult::Replace(new_op) => {
                    ctx.record_change();
                    // Recursively rewrite regions in the new operation
                    let final_op = self.rewrite_op_regions(db, &new_op, ctx, target);
                    // Map ORIGINAL op results to FINAL op results
                    ctx.map_results(db, op, &final_op);
                    // Prepend cast ops before the result
                    let mut result = cast_ops;
                    result.push(final_op);
                    return result;
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
                    // Prepend cast ops before the expanded ops
                    let mut result = cast_ops;
                    result.extend(final_ops);
                    return result;
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
                    // Cast ops still need to be in the output even if op is erased
                    return cast_ops;
                }
            }
        }

        // Step 7: No pattern matched - recursively process regions
        let final_op = self.rewrite_op_regions(db, &remapped_op, ctx, target);

        // Step 8: Map ORIGINAL op results to FINAL op results if they differ
        // This is critical when operands were remapped but no pattern matched
        if final_op != *op {
            ctx.map_results(db, op, &final_op);
        }

        // Prepend cast ops before the final op
        let mut result = cast_ops;
        result.push(final_op);
        result
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

/// Remap successor block references in all operations within a block.
///
/// If any operation has successors that appear in `block_map`, the successors
/// are replaced with the mapped blocks. Returns the block unchanged if no
/// successors need remapping (avoids unnecessary rebuilds).
fn remap_block_successors<'db>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
    block_map: &HashMap<Block<'db>, Block<'db>>,
) -> Block<'db> {
    if block_map.is_empty() {
        return block;
    }

    let ops = block.operations(db);
    let mut any_changed = false;
    let new_ops: IdVec<Operation<'db>> = ops
        .iter()
        .map(|op| {
            let successors = op.successors(db);
            if successors.is_empty() {
                return *op;
            }

            let mut changed = false;
            let new_successors: IdVec<Block<'db>> = successors
                .iter()
                .map(|succ| {
                    if let Some(&mapped) = block_map.get(succ) {
                        changed = true;
                        mapped
                    } else {
                        *succ
                    }
                })
                .collect();

            if changed {
                any_changed = true;
                op.modify(db).successors(new_successors).build()
            } else {
                *op
            }
        })
        .collect();

    if any_changed {
        Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            new_ops,
        )
    } else {
        block
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
    use crate::parser::parse_test_module;
    use crate::rewrite::TypeConverter;
    use crate::types::DialectType;
    use crate::{Attribute, BlockId, DialectOp, Location, PathId, Span, Symbol, ValueDef, idvec};
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
        parse_test_module(
            db,
            r#"core.module @test {
  %0 = arith.const {value = 42} : core.i32
}"#,
        )
    }

    /// Create a test module with an arith.mul operation (not matched by pattern).
    #[salsa::tracked]
    fn make_other_module(db: &dyn salsa::Database) -> Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test {
  %0 = arith.const {value = 5} : core.i32
  %1 = arith.mul %0, %0 : core.i32
}"#,
        )
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

    /// Create a module with a mul operation that uses i32 operands.
    #[salsa::tracked]
    fn make_mul_module_i32(db: &dyn salsa::Database) -> Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test {
  %0 = arith.const {value = 3} : core.i32
  %1 = arith.const {value = 4} : core.i32
  %2 = arith.mul %0, %1 : core.i32
}"#,
        )
    }

    /// Apply with type converter that converts i32 → i64.
    #[salsa::tracked]
    fn apply_with_i32_to_i64_converter<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, usize, Module<'db>) {
        use super::ConversionTarget;

        let type_converter = TypeConverter::new().add_conversion(|db, ty| {
            core::I32::from_type(db, ty).map(|_| core::I64::new(db).as_type())
        });

        let target = ConversionTarget::new();
        let applicator = PatternApplicator::new(type_converter);
        let result = applicator.apply_partial(db, module, target);
        (result.reached_fixpoint, result.total_changes, result.module)
    }

    #[salsa_test]
    fn test_unrealized_conversion_cast_insertion(db: &salsa::DatabaseImpl) {
        let module = make_mul_module_i32(db);
        let (reached_fixpoint, total_changes, result_module) =
            apply_with_i32_to_i64_converter(db, module);

        assert!(reached_fixpoint);
        // We should have changes because casts were inserted
        assert!(total_changes > 0, "Type conversion should trigger changes");

        // The module should now contain unrealized_conversion_cast operations
        let ops = result_module.body(db).blocks(db)[0].operations(db);

        // Count unrealized_conversion_cast operations
        let cast_count = ops
            .iter()
            .filter(|op| op.name(db) == core::UNREALIZED_CONVERSION_CAST())
            .count();

        // We expect casts for the mul operands (2 operands that need i32 → i64 conversion)
        assert!(
            cast_count >= 2,
            "Expected at least 2 unrealized_conversion_cast ops, got {}",
            cast_count
        );

        // Verify cast ops have correct target type (i64)
        let i64_ty = core::I64::new(db).as_type();
        for op in ops.iter() {
            if op.name(db) == core::UNREALIZED_CONVERSION_CAST() {
                let result_ty = op.results(db)[0];
                assert_eq!(
                    result_ty, i64_ty,
                    "unrealized_conversion_cast should produce i64 type"
                );
            }
        }
    }

    // === Block Argument Reference Tests ===

    /// Create a module where an operation uses a block argument as operand.
    /// This tests that block argument references are preserved after rewriting.
    #[salsa::tracked]
    fn make_module_with_block_arg_usage(db: &dyn salsa::Database) -> Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test {
  ^bb0(%arg0: core.i32):
    %0 = arith.const {value = 42} : core.i32
    %1 = arith.mul %arg0, %0 : core.i32
}"#,
        )
    }

    /// Apply patterns to module with block arg usage.
    #[salsa::tracked]
    fn apply_to_block_arg_module<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, Module<'db>) {
        use super::ConversionTarget;

        // Pattern that rewrites const(42) -> const(100)
        struct ConstRewritePattern;

        impl<'db> RewritePattern<'db> for ConstRewritePattern {
            fn match_and_rewrite(
                &self,
                db: &'db dyn salsa::Database,
                op: &Operation<'db>,
                _adaptor: &OpAdaptor<'db, '_>,
            ) -> RewriteResult<'db> {
                let Ok(const_op) = arith::Const::from_operation(db, *op) else {
                    return RewriteResult::Unchanged;
                };
                if const_op.value(db) != Attribute::IntBits(42) {
                    return RewriteResult::Unchanged;
                }
                let location = op.location(db);
                let result_ty = op.results(db)[0];
                let new_op = arith::r#const(db, location, result_ty, Attribute::IntBits(100));
                RewriteResult::Replace(new_op.as_operation())
            }
        }

        let target = ConversionTarget::new();
        let applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(ConstRewritePattern);
        let result = applicator.apply_partial(db, module, target);
        (result.reached_fixpoint, result.module)
    }

    #[salsa_test]
    fn test_block_arg_reference_preserved_after_rewrite(db: &salsa::DatabaseImpl) {
        use crate::ValueDef;

        let module = make_module_with_block_arg_usage(db);

        // Get original block ID and verify structure
        let original_block = &module.body(db).blocks(db)[0];
        let original_block_id = original_block.id(db);

        // Verify the mul operation uses block arg as first operand
        let ops = original_block.operations(db);
        assert_eq!(ops.len(), 2, "Should have const and mul operations");
        let mul_op = &ops[1];
        let mul_operands = mul_op.operands(db);
        assert_eq!(mul_operands.len(), 2);

        // First operand should be block arg
        let first_operand = mul_operands[0];
        match first_operand.def(db) {
            ValueDef::BlockArg(block_id) => {
                assert_eq!(block_id, original_block_id);
            }
            _ => panic!("First operand should be a block argument"),
        }

        // Apply the pattern (rewrites const(42) -> const(100))
        let (reached_fixpoint, result_module) = apply_to_block_arg_module(db, module);
        assert!(reached_fixpoint);

        // Verify the block ID is preserved
        let result_block = &result_module.body(db).blocks(db)[0];
        let result_block_id = result_block.id(db);
        assert_eq!(
            result_block_id, original_block_id,
            "Block ID should be preserved after rewrite"
        );

        // Verify the mul operation still references the correct block arg
        let result_ops = result_block.operations(db);
        assert_eq!(result_ops.len(), 2);
        let result_mul_op = &result_ops[1];
        let result_mul_operands = result_mul_op.operands(db);

        // First operand should still be block arg with same block ID
        let result_first_operand = result_mul_operands[0];
        match result_first_operand.def(db) {
            ValueDef::BlockArg(block_id) => {
                assert_eq!(
                    block_id, original_block_id,
                    "Block arg should still reference the original block"
                );
            }
            _ => panic!("First operand should still be a block argument"),
        }

        // Verify the const was actually rewritten
        let const_op = arith::Const::from_operation(db, result_ops[0]).unwrap();
        assert_eq!(
            const_op.value(db),
            Attribute::IntBits(100),
            "Const should have been rewritten"
        );
    }

    /// Test that block arguments in nested regions are handled correctly.
    #[salsa::tracked]
    fn make_module_with_nested_block_args(db: &dyn salsa::Database) -> Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test {
  ^bb0(%arg0: core.i32):
    %0 = arith.const {value = true} : core.i1
    %1 = scf.if %0 : core.i32 {
      %2 = arith.const {value = 42} : core.i32
      %3 = arith.add %arg0, %2 : core.i32
      scf.yield %3
    } {
      %4 = arith.const {value = 0} : core.i32
      scf.yield %4
    }
}"#,
        )
    }

    /// Apply patterns to module with nested regions.
    #[salsa::tracked]
    fn apply_to_nested_block_arg_module<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, Module<'db>) {
        use super::ConversionTarget;

        // Pattern that rewrites const(42) -> const(100)
        struct ConstRewritePattern;

        impl<'db> RewritePattern<'db> for ConstRewritePattern {
            fn match_and_rewrite(
                &self,
                db: &'db dyn salsa::Database,
                op: &Operation<'db>,
                _adaptor: &OpAdaptor<'db, '_>,
            ) -> RewriteResult<'db> {
                let Ok(const_op) = arith::Const::from_operation(db, *op) else {
                    return RewriteResult::Unchanged;
                };
                if const_op.value(db) != Attribute::IntBits(42) {
                    return RewriteResult::Unchanged;
                }
                let location = op.location(db);
                let result_ty = op.results(db)[0];
                let new_op = arith::r#const(db, location, result_ty, Attribute::IntBits(100));
                RewriteResult::Replace(new_op.as_operation())
            }
        }

        let target = ConversionTarget::new();
        let applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(ConstRewritePattern);
        let result = applicator.apply_partial(db, module, target);
        (result.reached_fixpoint, result.module)
    }

    #[salsa_test]
    fn test_outer_block_arg_referenced_in_nested_region(db: &salsa::DatabaseImpl) {
        use crate::ValueDef;

        let module = make_module_with_nested_block_args(db);

        // Get original outer block ID
        let original_outer_block = &module.body(db).blocks(db)[0];
        let original_outer_block_id = original_outer_block.id(db);

        // Verify the if operation's then region uses the outer block arg
        let if_op = &original_outer_block.operations(db)[1]; // const1, if_op
        let then_region = &if_op.regions(db)[0]; // then region
        let then_ops = then_region.blocks(db)[0].operations(db);
        // then_ops: const42, add, yield
        let add_op = &then_ops[1];
        let add_operands = add_op.operands(db);

        // First operand of add should be outer block arg
        let first_operand = add_operands[0];
        match first_operand.def(db) {
            ValueDef::BlockArg(block_id) => {
                assert_eq!(
                    block_id, original_outer_block_id,
                    "Add should reference outer block arg"
                );
            }
            _ => panic!("First operand should be outer block argument"),
        }

        // Apply pattern (rewrites const(42) -> const(100))
        let (reached_fixpoint, result_module) = apply_to_nested_block_arg_module(db, module);
        assert!(reached_fixpoint);

        // Verify outer block ID is preserved
        let result_outer_block = &result_module.body(db).blocks(db)[0];
        let result_outer_block_id = result_outer_block.id(db);
        assert_eq!(
            result_outer_block_id, original_outer_block_id,
            "Outer block ID should be preserved"
        );

        // Verify the add operation in nested region still references outer block arg
        let result_if_op = &result_outer_block.operations(db)[1];
        let result_then_region = &result_if_op.regions(db)[0];
        let result_then_ops = result_then_region.blocks(db)[0].operations(db);
        let result_add_op = &result_then_ops[1];
        let result_add_operands = result_add_op.operands(db);

        let result_first_operand = result_add_operands[0];
        match result_first_operand.def(db) {
            ValueDef::BlockArg(block_id) => {
                assert_eq!(
                    block_id, original_outer_block_id,
                    "Add in nested region should still reference outer block arg"
                );
            }
            _ => panic!("First operand should still be outer block argument"),
        }

        // Verify the const was actually rewritten
        let const_op = arith::Const::from_operation(db, result_then_ops[0]).unwrap();
        assert_eq!(
            const_op.value(db),
            Attribute::IntBits(100),
            "Const in then region should have been rewritten"
        );
    }

    // === Region Reconstruction Test ===
    // This tests the actual problem: when a pattern reconstructs a region with
    // fresh BlockIds, operations inside that region that reference the old
    // block arguments become stale.

    /// Pattern that reconstructs a region with fresh BlockIds.
    /// This simulates what can happen in lowering passes.
    /// Uses a marker attribute to avoid infinite loops.
    struct RegionReconstructPattern;

    impl<'db> RewritePattern<'db> for RegionReconstructPattern {
        fn match_and_rewrite(
            &self,
            db: &'db dyn salsa::Database,
            op: &Operation<'db>,
            _adaptor: &OpAdaptor<'db, '_>,
        ) -> RewriteResult<'db> {
            use crate::dialect::scf;

            // Match scf.if operations
            let Ok(if_op) = scf::If::from_operation(db, *op) else {
                return RewriteResult::Unchanged;
            };

            // Check if already transformed (marker attribute)
            let marker = Symbol::new("_reconstructed");
            if op.attributes(db).contains_key(&marker) {
                return RewriteResult::Unchanged;
            }

            // Get the then region
            let then_region = if_op.then(db);
            let then_blocks = then_region.blocks(db);
            if then_blocks.is_empty() {
                return RewriteResult::Unchanged;
            }

            let then_block = &then_blocks[0];

            // Reconstruct the then block with a FRESH BlockId
            // This is the problematic pattern!
            let fresh_block_id = BlockId::fresh();
            let new_then_block = Block::new(
                db,
                fresh_block_id, // ← Fresh ID instead of preserving original
                then_block.location(db),
                then_block.args(db).clone(),
                then_block.operations(db).clone(),
            );
            let new_then_region = Region::new(db, then_region.location(db), idvec![new_then_block]);

            // Keep else region unchanged
            let else_region = if_op.r#else(db);

            // Rebuild the if operation with a marker attribute
            let new_if = scf::r#if(
                db,
                op.location(db),
                if_op.cond(db),
                op.results(db)[0], // result type
                new_then_region,
                else_region,
            );

            // Add marker attribute to prevent infinite loop
            let new_if_op = new_if
                .as_operation()
                .modify(db)
                .attr(marker, Attribute::Bool(true))
                .build();

            RewriteResult::Replace(new_if_op)
        }
    }

    /// Apply the region reconstruction pattern.
    #[salsa::tracked]
    fn apply_region_reconstruct_pattern<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> (bool, Module<'db>) {
        use super::ConversionTarget;

        let target = ConversionTarget::new();
        let applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(RegionReconstructPattern);
        let result = applicator.apply_partial(db, module, target);
        (result.reached_fixpoint, result.module)
    }

    /// This test demonstrates the problem: when a region is reconstructed
    /// with a fresh BlockId, operations inside that reference the outer
    /// block argument become stale (pointing to an old BlockId).
    #[salsa_test]
    fn test_region_reconstruction_stale_block_arg_reference(db: &salsa::DatabaseImpl) {
        let module = make_module_with_nested_block_args(db);

        // Get original outer block ID
        let original_outer_block = &module.body(db).blocks(db)[0];
        let original_outer_block_id = original_outer_block.id(db);

        // Apply the pattern that reconstructs regions with fresh BlockIds
        let (reached_fixpoint, result_module) = apply_region_reconstruct_pattern(db, module);
        assert!(reached_fixpoint);

        // The outer block ID should still be preserved (we didn't touch it)
        let result_outer_block = &result_module.body(db).blocks(db)[0];
        let result_outer_block_id = result_outer_block.id(db);
        assert_eq!(
            result_outer_block_id, original_outer_block_id,
            "Outer block ID should be preserved"
        );

        // Now check the add operation in the then region
        let result_if_op = &result_outer_block.operations(db)[1];
        let result_then_region = &result_if_op.regions(db)[0];
        let result_then_ops = result_then_region.blocks(db)[0].operations(db);

        // The add operation (index 1) uses outer block arg as first operand
        let result_add_op = &result_then_ops[1];
        let result_add_operands = result_add_op.operands(db);

        // Check if the block arg reference is still valid
        let result_first_operand = result_add_operands[0];
        match result_first_operand.def(db) {
            ValueDef::BlockArg(block_id) => {
                // This is the key test: the add operation should still reference
                // the outer block's argument, which has the original block ID.
                // If the pattern incorrectly reconstructed the region, this might
                // be pointing to a stale BlockId.
                assert_eq!(
                    block_id, original_outer_block_id,
                    "Add operation should still reference the outer block arg. \
                     This test documents the issue: when a region is reconstructed \
                     with fresh BlockIds, inner operations may have stale references."
                );
            }
            _ => panic!("First operand should still be a block argument"),
        }
    }

    // === Successor Remap Tests ===

    /// Create a module with cf.br and cf.cond_br ops that have successor references.
    /// Tests that PatternApplicator correctly remaps successor block references
    /// when blocks are rewritten (getting new Block entities).
    #[salsa::tracked]
    fn make_module_with_successors(db: &dyn salsa::Database) -> Module<'_> {
        use crate::dialect::cf;

        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        // Block 2: target block with a block argument
        let block2_id = BlockId::fresh();
        let block2_arg = crate::BlockArg::of_type(db, i32_ty);
        let block2_arg_value = Value::new(db, ValueDef::BlockArg(block2_id), 0);
        // block2 uses its own arg
        let const99 = arith::r#const(db, location, i32_ty, Attribute::IntBits(99));
        let add_op = arith::add(db, location, block2_arg_value, const99.result(db), i32_ty);
        let block2 = Block::new(
            db,
            block2_id,
            location,
            idvec![block2_arg],
            idvec![const99.as_operation(), add_op.as_operation()],
        );

        // Block 3: another target block
        let block3_id = BlockId::fresh();
        let const0 = arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let block3 = Block::new(
            db,
            block3_id,
            location,
            idvec![],
            idvec![const0.as_operation()],
        );

        // Block 1 (entry): has cf.cond_br to block2 / block3
        let block1_id = BlockId::fresh();
        let cond = arith::r#const(db, location, i1_ty, Attribute::Bool(true));
        // Const for the arg passed to block2
        let const42 = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let cond_br = cf::cond_br(db, location, cond.result(db), block2, block3);

        let block1 = Block::new(
            db,
            block1_id,
            location,
            idvec![],
            idvec![
                cond.as_operation(),
                const42.as_operation(),
                cond_br.as_operation()
            ],
        );

        let region = Region::new(db, location, idvec![block1, block2, block3]);
        Module::create(db, location, Symbol::new("test_succ"), region)
    }

    /// Apply a no-op pattern (that still triggers block rewriting) to test successor remap.
    #[salsa::tracked]
    fn apply_to_successor_module<'db>(
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> Module<'db> {
        use super::ConversionTarget;

        // Use const(99) → const(100) pattern to force block rewriting
        struct Const99To100;
        impl<'db> RewritePattern<'db> for Const99To100 {
            fn match_and_rewrite(
                &self,
                db: &'db dyn salsa::Database,
                op: &Operation<'db>,
                _adaptor: &OpAdaptor<'db, '_>,
            ) -> RewriteResult<'db> {
                let Ok(const_op) = arith::Const::from_operation(db, *op) else {
                    return RewriteResult::Unchanged;
                };
                if const_op.value(db) != Attribute::IntBits(99) {
                    return RewriteResult::Unchanged;
                }
                let location = op.location(db);
                let result_ty = op.results(db)[0];
                let new_op = arith::r#const(db, location, result_ty, Attribute::IntBits(100));
                RewriteResult::Replace(new_op.as_operation())
            }
        }

        let target = ConversionTarget::new();
        let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(Const99To100);
        applicator.apply_partial(db, module, target).module
    }

    #[salsa_test]
    fn test_applicator_successor_remap(db: &salsa::DatabaseImpl) {
        use crate::dialect::cf;

        let module = make_module_with_successors(db);
        let result = apply_to_successor_module(db, module);

        let blocks = result.body(db).blocks(db);
        assert_eq!(blocks.len(), 3, "Should still have 3 blocks");

        // Get the new block2 and block3
        let new_block2 = blocks[1];
        let new_block3 = blocks[2];

        // Entry block should have cf.cond_br as last op
        let entry_ops = blocks[0].operations(db);
        let last_op = entry_ops.last().unwrap();
        let cond_br = cf::CondBr::from_operation(db, *last_op)
            .expect("Last op of entry should be cf.cond_br");

        // Successor references should point to the NEW blocks, not the old ones
        assert_eq!(
            cond_br.then_dest(db),
            new_block2,
            "cf.cond_br then_dest should be remapped to new block2"
        );
        assert_eq!(
            cond_br.else_dest(db),
            new_block3,
            "cf.cond_br else_dest should be remapped to new block3"
        );

        // Verify the pattern was applied (const 99 → 100)
        let block2_ops = new_block2.operations(db);
        let const_op = arith::Const::from_operation(db, block2_ops[0]).unwrap();
        assert_eq!(
            const_op.value(db),
            Attribute::IntBits(100),
            "const(99) should have been rewritten to const(100)"
        );
    }
}
