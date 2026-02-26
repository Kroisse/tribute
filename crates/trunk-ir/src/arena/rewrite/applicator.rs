//! PatternApplicator for arena IR.
//!
//! Visitor-based fixpoint iteration that applies rewrite patterns to all
//! operations in a module. Uses snapshots of block operations and checks
//! `parent_block` validity to skip deleted ops.

use super::ArenaModule;
use super::conversion_target::{ArenaConversionTarget, IllegalOp};
use super::pattern::ArenaRewritePattern;
use super::rewriter::{self, PatternRewriter};
use super::type_converter::ArenaTypeConverter;
use crate::arena::context::IrContext;
use crate::arena::refs::{BlockRef, OpRef, RegionRef};

/// Result of applying rewrite patterns.
pub struct ApplyResult {
    /// Number of fixpoint iterations performed.
    pub iterations: usize,
    /// Total number of pattern matches (mutations applied).
    pub total_changes: usize,
    /// Whether the fixpoint was reached (no changes in last iteration).
    pub reached_fixpoint: bool,
}

impl ApplyResult {
    /// Verify that no illegal operations remain.
    pub fn verify(
        &self,
        ctx: &IrContext,
        module: ArenaModule,
        target: &ArenaConversionTarget,
    ) -> Result<(), Vec<IllegalOp>> {
        let body = match module.body(ctx) {
            Some(r) => r,
            None => return Ok(()),
        };
        let illegal = target.verify(ctx, body);
        if illegal.is_empty() {
            Ok(())
        } else {
            Err(illegal)
        }
    }
}

/// Applies rewrite patterns to arena IR using visitor-based fixpoint iteration.
pub struct PatternApplicator {
    patterns: Vec<Box<dyn ArenaRewritePattern>>,
    max_iterations: usize,
    type_converter: ArenaTypeConverter,
}

impl PatternApplicator {
    /// Create a new applicator with the given type converter.
    pub fn new(type_converter: ArenaTypeConverter) -> Self {
        Self {
            patterns: Vec::new(),
            max_iterations: 10,
            type_converter,
        }
    }

    /// Add a rewrite pattern.
    pub fn add_pattern(mut self, pattern: impl ArenaRewritePattern + 'static) -> Self {
        self.patterns.push(Box::new(pattern));
        self
    }

    /// Set maximum fixpoint iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Get a reference to the type converter.
    pub fn type_converter(&self) -> &ArenaTypeConverter {
        &self.type_converter
    }

    /// Apply patterns and verify the result.
    pub fn apply(
        &self,
        ctx: &mut IrContext,
        module: ArenaModule,
        target: &ArenaConversionTarget,
    ) -> Result<ApplyResult, Vec<IllegalOp>> {
        let result = self.apply_partial(ctx, module);
        result.verify(ctx, module, target)?;
        Ok(result)
    }

    /// Apply patterns without verification.
    pub fn apply_partial(&self, ctx: &mut IrContext, module: ArenaModule) -> ApplyResult {
        let mut total_changes = 0;
        let mut iterations = 0;

        for _ in 0..self.max_iterations {
            iterations += 1;
            let changes = self.run_one_iteration(ctx, module);
            total_changes += changes;
            if changes == 0 {
                return ApplyResult {
                    iterations,
                    total_changes,
                    reached_fixpoint: true,
                };
            }
        }

        ApplyResult {
            iterations,
            total_changes,
            reached_fixpoint: false,
        }
    }

    /// Run a single iteration over all operations.
    fn run_one_iteration(&self, ctx: &mut IrContext, module: ArenaModule) -> usize {
        let mut changes = 0;
        let module_first_block = module.first_block(ctx);

        // Walk the module body region
        let body = match module.body(ctx) {
            Some(r) => r,
            None => return 0,
        };
        changes += self.visit_region(ctx, body, module_first_block);

        changes
    }

    fn visit_region(
        &self,
        ctx: &mut IrContext,
        region: RegionRef,
        module_first_block: Option<BlockRef>,
    ) -> usize {
        let mut changes = 0;
        let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
        for block in blocks {
            changes += self.visit_block(ctx, block, module_first_block);
        }
        changes
    }

    fn visit_block(
        &self,
        ctx: &mut IrContext,
        block: BlockRef,
        module_first_block: Option<BlockRef>,
    ) -> usize {
        let mut changes = 0;

        // Snapshot the ops in this block
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

        for op in ops {
            // Skip ops that have been removed from their block
            if ctx.op(op).parent_block != Some(block) {
                continue;
            }

            // First, recurse into nested regions
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for region in regions {
                changes += self.visit_region(ctx, region, module_first_block);
            }

            // Skip ops that were removed during nested processing
            if ctx.op(op).parent_block != Some(block) {
                continue;
            }

            // Try each pattern
            for pattern in &self.patterns {
                let mut rw = PatternRewriter::new();
                let matched = pattern.match_and_rewrite(ctx, op, &mut rw);
                if matched && rw.has_mutations() {
                    let mutations = rw.take_mutations();
                    rewriter::apply_mutations(ctx, op, mutations, module_first_block);
                    changes += 1;
                    break; // Only apply one pattern per op per iteration
                }
            }
        }

        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::rewrite::conversion_target::ArenaConversionTarget;
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i32"),
            params: smallvec![],
            attrs: BTreeMap::new(),
        })
    }

    fn make_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> ArenaModule {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        ArenaModule::new(ctx, module_op).expect("test module should be valid")
    }

    /// Pattern: rename test.source → test.target
    struct RenamePattern;

    impl ArenaRewritePattern for RenamePattern {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter,
        ) -> bool {
            let data = ctx.op(op);
            if data.dialect != Symbol::new("test") || data.name != Symbol::new("source") {
                return false;
            }

            let loc = data.location;
            let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

            let new_data =
                OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("target"))
                    .results(result_types)
                    .build(ctx);
            let new_op = ctx.create_op(new_data);
            rewriter.replace_op(new_op);
            true
        }
    }

    #[test]
    fn applicator_renames_op() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op_data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(op_data);
        let module = make_module(&mut ctx, loc, vec![op]);

        let applicator =
            PatternApplicator::new(ArenaTypeConverter::new()).add_pattern(RenamePattern);

        let mut target = ArenaConversionTarget::new();
        target.add_legal_dialect("test");
        target.add_illegal_op("test", "source");

        let result = applicator.apply(&mut ctx, module, &target).unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 1);

        // Verify the op was renamed
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ctx.op(ops[0]).name, Symbol::new("target"));
    }

    #[test]
    fn applicator_preserves_uses_via_rauw() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // op1: test.source -> %0
        let op1_data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(op1_data);
        let v1 = ctx.op_result(op1, 0);

        // op2: test.use(%0)
        let op2_data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("use"))
            .operand(v1)
            .build(&mut ctx);
        let op2 = ctx.create_op(op2_data);

        let module = make_module(&mut ctx, loc, vec![op1, op2]);

        let applicator =
            PatternApplicator::new(ArenaTypeConverter::new()).add_pattern(RenamePattern);

        let target = ArenaConversionTarget::new();
        applicator.apply(&mut ctx, module, &target).unwrap();

        // op2's operand should now point to the replacement op's result
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2);

        let new_result = ctx.op_result(ops[0], 0);
        let op2_operands = ctx.op_operands(ops[1]);
        assert_eq!(op2_operands[0], new_result);
    }
}
