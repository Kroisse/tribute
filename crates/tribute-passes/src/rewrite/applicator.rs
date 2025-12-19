//! Pattern applicator for driving IR rewrites.
//!
//! The `PatternApplicator` manages a set of rewrite patterns and
//! applies them to a module until fixpoint is reached.

use tribute_trunk_ir::dialect::core::Module;
use tribute_trunk_ir::{Block, IdVec, Operation, Region};

use super::context::RewriteContext;
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

/// Applies a set of rewrite patterns to IR until fixpoint.
///
/// # Example
///
/// ```ignore
/// let applicator = PatternApplicator::new()
///     .add_pattern(ResolveSrcVar::new(env))
///     .add_pattern(ResolveSrcPath::new(env))
///     .with_max_iterations(50);
///
/// let result = applicator.apply(db, module);
/// assert!(result.reached_fixpoint);
/// ```
pub struct PatternApplicator {
    patterns: Vec<Box<dyn RewritePattern>>,
    max_iterations: usize,
}

impl PatternApplicator {
    /// Create a new empty pattern applicator.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            max_iterations: 100,
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
        P: RewritePattern + 'static,
    {
        self.patterns.push(Box::new(pattern));
        self
    }

    /// Apply all patterns to a module until fixpoint.
    pub fn apply<'db>(
        &self,
        db: &'db dyn salsa::Database,
        module: Module<'db>,
    ) -> ApplyResult<'db> {
        let mut current = module;
        let mut total_changes = 0;

        for iteration in 0..self.max_iterations {
            let mut ctx = RewriteContext::new();
            let new_module = self.rewrite_module(db, &current, &mut ctx);

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
    fn rewrite_module<'db>(
        &self,
        db: &'db dyn salsa::Database,
        module: &Module<'db>,
        ctx: &mut RewriteContext<'db>,
    ) -> Module<'db> {
        let body = module.body(db);
        let new_body = self.rewrite_region(db, &body, ctx);

        // Rebuild module with new body
        Module::create(db, module.location(db), module.name(db), new_body)
    }

    /// Rewrite a region.
    fn rewrite_region<'db>(
        &self,
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        ctx: &mut RewriteContext<'db>,
    ) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(db)
            .iter()
            .map(|block| self.rewrite_block(db, block, ctx))
            .collect();

        Region::new(db, region.location(db), new_blocks)
    }

    /// Rewrite a block.
    fn rewrite_block<'db>(
        &self,
        db: &'db dyn salsa::Database,
        block: &Block<'db>,
        ctx: &mut RewriteContext<'db>,
    ) -> Block<'db> {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(db)
            .iter()
            .flat_map(|op| self.rewrite_operation(db, op, ctx))
            .collect();

        Block::new(db, block.location(db), block.args(db).clone(), new_ops)
    }

    /// Rewrite a single operation.
    ///
    /// This is the core rewrite loop:
    /// 1. Remap operands using the current value map
    /// 2. Try each pattern in order
    /// 3. If a pattern matches, apply it and record mappings
    /// 4. Recursively rewrite any nested regions
    fn rewrite_operation<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        ctx: &mut RewriteContext<'db>,
    ) -> Vec<Operation<'db>> {
        // Step 1: Remap operands from previous transformations
        let remapped_op = ctx.remap_operands(db, op);

        // Step 2: Try each pattern
        for pattern in &self.patterns {
            match pattern.match_and_rewrite(db, &remapped_op, ctx) {
                RewriteResult::Unchanged => continue,

                RewriteResult::Replace(new_op) => {
                    ctx.record_change();
                    ctx.map_results(db, &remapped_op, &new_op);
                    // Recursively rewrite regions in the new operation
                    let final_op = self.rewrite_op_regions(db, &new_op, ctx);
                    return vec![final_op];
                }

                RewriteResult::Expand(ops) => {
                    ctx.record_change();
                    if let Some(first) = ops.first() {
                        ctx.map_results(db, &remapped_op, first);
                    }
                    // Recursively rewrite regions in all new operations
                    return ops
                        .into_iter()
                        .map(|op| self.rewrite_op_regions(db, &op, ctx))
                        .collect();
                }

                RewriteResult::Erase { replacement_values } => {
                    ctx.record_change();
                    // Map each result to its replacement value
                    for (i, val) in replacement_values.into_iter().enumerate() {
                        let old_val = remapped_op.result(db, i);
                        ctx.map_value(old_val, val);
                    }
                    return vec![];
                }
            }
        }

        // Step 3: No pattern matched - recursively process regions
        let final_op = self.rewrite_op_regions(db, &remapped_op, ctx);
        vec![final_op]
    }

    /// Rewrite nested regions within an operation.
    fn rewrite_op_regions<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        ctx: &mut RewriteContext<'db>,
    ) -> Operation<'db> {
        let regions = op.regions(db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.rewrite_region(db, region, ctx))
            .collect();

        op.modify(db).regions(new_regions).build()
    }
}

impl Default for PatternApplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::Database;
    use tribute_core::{Location, PathId, Span, TributeDatabaseImpl};
    use tribute_trunk_ir::{Attribute, idvec};

    /// A simple test pattern that rewrites `test.source` → `test.target`.
    struct TestRenamePattern;

    impl RewritePattern for TestRenamePattern {
        fn match_and_rewrite<'db>(
            &self,
            db: &'db dyn salsa::Database,
            op: &Operation<'db>,
            _ctx: &mut RewriteContext<'db>,
        ) -> RewriteResult<'db> {
            if op.dialect(db).text(db) != "test" || op.name(db).text(db) != "source" {
                return RewriteResult::Unchanged;
            }

            // Create replacement operation with same structure but different name
            let new_op = op
                .modify(db)
                .name_str("target")
                .build();

            RewriteResult::Replace(new_op)
        }
    }

    /// Create a test module with a test.source operation.
    #[salsa::tracked]
    fn make_source_module(db: &dyn salsa::Database) -> Module<'_> {
        let path = PathId::new(db, std::path::PathBuf::from("test.tr"));
        let location = Location::new(path, Span::new(0, 0));

        let op = Operation::of_name(db, location, "test.source")
            .attr("value", Attribute::IntBits(42))
            .build();
        let block = Block::new(db, location, idvec![], idvec![op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test", region)
    }

    /// Create a test module with an unrelated operation.
    #[salsa::tracked]
    fn make_other_module(db: &dyn salsa::Database) -> Module<'_> {
        let path = PathId::new(db, std::path::PathBuf::from("test.tr"));
        let location = Location::new(path, Span::new(0, 0));

        let op = Operation::of_name(db, location, "other.op").build();
        let block = Block::new(db, location, idvec![], idvec![op]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test", region)
    }

    /// Apply patterns and return results (tracked to enable IR creation during rewrite).
    #[salsa::tracked]
    fn apply_rename_pattern(
        db: &dyn salsa::Database,
        module: Module<'_>,
    ) -> (bool, usize, usize, String) {
        let applicator = PatternApplicator::new().add_pattern(TestRenamePattern);
        let result = applicator.apply(db, module);
        let body = result.module.body(db);
        let op_name = body.blocks(db)[0].operations(db)[0].full_name(db);
        (
            result.reached_fixpoint,
            result.total_changes,
            result.iterations,
            op_name,
        )
    }

    #[test]
    fn test_pattern_applicator_basic() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = make_source_module(db);
            let (reached_fixpoint, total_changes, _iterations, op_name) =
                apply_rename_pattern(db, module);

            // Should have made one change and reached fixpoint
            assert!(reached_fixpoint);
            assert_eq!(total_changes, 1);

            // The operation should now be test.target
            assert_eq!(op_name, "test.target");
        });
    }

    #[test]
    fn test_pattern_applicator_no_match() {
        TributeDatabaseImpl::default().attach(|db| {
            let module = make_other_module(db);
            let (reached_fixpoint, total_changes, iterations, _op_name) =
                apply_rename_pattern(db, module);

            // Should reach fixpoint immediately with no changes
            assert!(reached_fixpoint);
            assert_eq!(total_changes, 0);
            assert_eq!(iterations, 1);
        });
    }
}
