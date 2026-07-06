//! PatternApplicator for arena IR.
//!
//! Visitor-based fixpoint iteration that applies rewrite patterns to all
//! operations in a module. Uses snapshots of block operations and checks
//! `parent_block` validity to skip deleted ops.

use super::Module;
use super::conversion_target::{
    ConversionError, ConversionMode, ConversionTarget, IllegalOp, LegalityCheck,
};
use super::pattern::RewritePattern;
use super::rewriter::{self, PatternRewriter};
use super::type_converter::TypeConverter;
use crate::context::IrContext;
use crate::dialect::{core, func};
use crate::ops::DialectOp;
use crate::refs::{BlockRef, OpRef, RegionRef};

/// Scope that bounds a pattern rewrite traversal.
///
/// A rewrite scope provides the regions to visit and, when the scope is a
/// module, the module block that may receive `PatternRewriter::add_module_op`
/// insertions. Function-scoped rewrites intentionally return no module block,
/// keeping anchored rewrites inside the current function body.
pub trait RewriteScope: Copy {
    type Regions: Iterator<Item = RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions;
    fn module_first_block(self, ctx: &IrContext) -> Option<BlockRef>;
}

impl RewriteScope for Module {
    type Regions = std::option::IntoIter<RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions {
        self.body(ctx).into_iter()
    }

    fn module_first_block(self, ctx: &IrContext) -> Option<BlockRef> {
        self.first_block(ctx)
    }
}

impl RewriteScope for func::Func {
    type Regions = std::iter::Once<RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions {
        std::iter::once(self.body(ctx))
    }

    fn module_first_block(self, _ctx: &IrContext) -> Option<BlockRef> {
        None
    }
}

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
        scope: impl RewriteScope,
        target: &ConversionTarget,
    ) -> Result<(), Vec<IllegalOp>> {
        self.verify_mode(ctx, scope, target, ConversionMode::Partial)
    }

    /// Verify that every operation is legal for the target.
    pub fn verify_full(
        &self,
        ctx: &IrContext,
        scope: impl RewriteScope,
        target: &ConversionTarget,
    ) -> Result<(), Vec<IllegalOp>> {
        self.verify_mode(ctx, scope, target, ConversionMode::Full)
    }

    /// Verify the result under a specific conversion mode.
    pub fn verify_mode(
        &self,
        ctx: &IrContext,
        scope: impl RewriteScope,
        target: &ConversionTarget,
        mode: ConversionMode,
    ) -> Result<(), Vec<IllegalOp>> {
        let illegal: Vec<IllegalOp> = scope
            .regions(ctx)
            .flat_map(|region| target.verify_mode(ctx, region, mode))
            .collect();
        if illegal.is_empty() {
            Ok(())
        } else {
            Err(illegal)
        }
    }
}

/// Applies rewrite patterns to arena IR using visitor-based fixpoint iteration.
pub struct PatternApplicator {
    patterns: Vec<Box<dyn RewritePattern>>,
    max_iterations: usize,
    type_converter: TypeConverter,
    /// Conversion target used for legality-aware rewriting and verification.
    target: ConversionTarget,
    /// Whether to automatically convert block argument types and insert
    /// `unrealized_conversion_cast` for operand type mismatches.
    auto_type_conversion: bool,
}

impl PatternApplicator {
    /// Create a new applicator with the given type converter.
    pub fn new(type_converter: TypeConverter) -> Self {
        Self {
            patterns: Vec::new(),
            max_iterations: 10,
            type_converter,
            target: ConversionTarget::default(),
            auto_type_conversion: false,
        }
    }

    /// Set the conversion target for legality-aware rewriting and verification.
    pub fn with_target(mut self, target: ConversionTarget) -> Self {
        self.target = target;
        self
    }

    /// Mutably access the configured conversion target.
    pub fn conversion_target_mut(&mut self) -> &mut ConversionTarget {
        &mut self.target
    }

    /// Add a rewrite pattern.
    pub fn add_pattern(mut self, pattern: impl RewritePattern + 'static) -> Self {
        self.patterns.push(Box::new(pattern));
        self
    }

    /// Add an already-boxed rewrite pattern.
    ///
    /// Useful when collecting heterogeneous patterns from multiple sources
    /// (e.g. per-dialect `canonicalization_patterns()` registries) into a
    /// single applicator without naming each concrete type at the call site.
    pub fn add_pattern_box(mut self, pattern: Box<dyn RewritePattern>) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Enable or disable automatic type conversion.
    ///
    /// When enabled, block argument types are converted and
    /// `unrealized_conversion_cast` ops are inserted for operand type
    /// mismatches before pattern matching. This is independent of conversion
    /// target legality and must be enabled explicitly by passes that need it.
    pub fn with_auto_type_conversion(mut self, enable: bool) -> Self {
        self.auto_type_conversion = enable;
        self
    }

    /// Set maximum fixpoint iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Get a reference to the type converter.
    pub fn type_converter(&self) -> &TypeConverter {
        &self.type_converter
    }

    /// Apply patterns within a rewrite scope without verification.
    ///
    /// The configured target still skips operations classified as legal.
    pub fn apply_partial<S: RewriteScope>(&self, ctx: &mut IrContext, scope: S) -> ApplyResult {
        let module_first_block = scope.module_first_block(ctx);
        let mut total_changes = 0;
        let mut iterations = 0;

        for _ in 0..self.max_iterations {
            iterations += 1;
            let changes = self.run_one_iteration(ctx, scope, module_first_block);
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

    /// Apply patterns using partial conversion semantics at a named boundary.
    ///
    /// Verification fails only for operations that are explicitly illegal in
    /// the target. Unknown operations may remain for later conversion passes.
    /// Failures are reported as [`ConversionError`] with `boundary` attached.
    pub fn apply_partial_conversion(
        &self,
        ctx: &mut IrContext,
        scope: impl RewriteScope,
        boundary: &'static str,
    ) -> Result<ApplyResult, ConversionError> {
        let result = self.apply_partial(ctx, scope);
        result
            .verify(ctx, scope, &self.target)
            .map_err(|operations| ConversionError::new(boundary, operations))?;
        Ok(result)
    }

    /// Apply patterns using full conversion semantics at a named boundary.
    ///
    /// Verification fails for both explicitly illegal operations and unknown
    /// operations. Use this at named pipeline boundaries. Failures are reported
    /// as [`ConversionError`] with `boundary` attached.
    pub fn apply_full_conversion(
        &self,
        ctx: &mut IrContext,
        scope: impl RewriteScope,
        boundary: &'static str,
    ) -> Result<ApplyResult, ConversionError> {
        let result = self.apply_partial(ctx, scope);
        result
            .verify_full(ctx, scope, &self.target)
            .map_err(|operations| ConversionError::new(boundary, operations))?;
        Ok(result)
    }

    /// Run a single iteration over all operations below the selected root.
    fn run_one_iteration(
        &self,
        ctx: &mut IrContext,
        scope: impl RewriteScope,
        module_first_block: Option<BlockRef>,
    ) -> usize {
        let mut changes = 0;

        for region in scope.regions(ctx) {
            changes += self.visit_region(ctx, region, module_first_block);
        }

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

        // Step 1: Convert block argument types before processing ops.
        // Only do this when auto_type_conversion is enabled (dialect conversion
        // passes), so that non-conversion passes don't inadvertently change types.
        if self.auto_type_conversion && !self.type_converter.is_empty() {
            let block_args = ctx.block_args(block).to_vec();
            for (i, arg_val) in block_args.iter().enumerate() {
                let raw_ty = ctx.value_ty(*arg_val);
                if let Some(new_ty) = self.type_converter.convert_type(ctx, raw_ty)
                    && new_ty != raw_ty
                {
                    ctx.set_block_arg_type(block, i as u32, new_ty);
                    changes += 1;
                }
            }
        }

        // Snapshot the ops in this block
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

        for op in ops {
            // Skip ops that have been removed from their block
            if ctx.op(op).parent_block != Some(block) {
                continue;
            }

            if self.target.has_constraints() && self.target.is_recursively_legal(ctx, op) {
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

            // Skip cast insertion and pattern matching for legal operations
            let is_legal = self.target.has_constraints()
                && self.target.is_legal(ctx, op) == LegalityCheck::Legal;

            if !is_legal {
                // Step 2: Insert conversion casts for operand type mismatches.
                // Only insert casts when auto_type_conversion is enabled, so that
                // non-conversion passes don't accidentally box/cast operands of
                // unrelated ops.
                if self.auto_type_conversion && !self.type_converter.is_empty() {
                    changes += self.insert_conversion_casts(ctx, block, op);
                }

                // Try each pattern
                for pattern in &self.patterns {
                    let mut rw = PatternRewriter::new(&self.type_converter);
                    let matched = pattern.match_and_rewrite(ctx, op, &mut rw);
                    if matched && rw.has_mutations() {
                        let mutations = rw.take_mutations();
                        rewriter::apply_mutations(ctx, op, mutations, module_first_block);
                        changes += 1;
                        break; // Only apply one pattern per op per iteration
                    }
                }
            }
        }

        changes
    }

    /// Insert `core.unrealized_conversion_cast` for operands whose types
    /// need conversion. Skips `unrealized_conversion_cast` itself to avoid
    /// infinite loops.
    fn insert_conversion_casts(&self, ctx: &mut IrContext, block: BlockRef, op: OpRef) -> usize {
        // Skip unrealized_conversion_cast itself to avoid infinite loops
        if core::UnrealizedConversionCast::from_op(ctx, op).is_ok() {
            return 0;
        }

        let mut cast_count = 0;
        let operands = ctx.op_operands(op).to_vec();
        for (i, &operand) in operands.iter().enumerate() {
            let raw_ty = ctx.value_ty(operand);
            if let Some(target_ty) = self.type_converter.convert_type(ctx, raw_ty)
                && target_ty != raw_ty
            {
                let loc = ctx.op(op).location;
                let cast = core::unrealized_conversion_cast(ctx, loc, operand, target_ty);
                let cast_ref = cast.op_ref();
                let cast_result = cast.result(ctx);
                ctx.insert_op_before(block, op, cast_ref);
                ctx.set_op_operand(op, i as u32, cast_result);
                cast_count += 1;
            }
        }
        cast_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::func;
    use crate::location::Span;
    use crate::ops::DialectOp;
    use crate::rewrite::conversion_target::ConversionTarget;
    use crate::symbol::Symbol;
    use crate::*;
    use smallvec::smallvec;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> Module {
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
        Module::new(ctx, module_op).expect("test module should be valid")
    }

    fn make_container(ctx: &mut IrContext, loc: Location, nested_ops: Vec<OpRef>) -> OpRef {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in nested_ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let container_data =
            OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("container"))
                .region(region)
                .build(ctx);
        ctx.create_op(container_data)
    }

    fn make_func_container(
        ctx: &mut IrContext,
        loc: Location,
        name: &'static str,
        nested_ops: Vec<OpRef>,
    ) -> func::Func {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in nested_ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let nil_ty = crate::dialect::core::nil(ctx).as_type_ref();
        let func_ty = crate::dialect::core::func(ctx, nil_ty, []).as_type_ref();
        let func_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new(name)))
            .attr("type", Attribute::Type(func_ty))
            .region(region)
            .build(ctx);
        let func_op = ctx.create_op(func_data);
        func::Func::from_op(ctx, func_op).expect("test func should be valid")
    }

    fn first_nested_op(ctx: &IrContext, op: OpRef) -> OpRef {
        let region = ctx.op(op).regions[0];
        let block = ctx.region(region).blocks[0];
        ctx.block(block).ops[0]
    }

    /// Pattern: rename test.source → test.target
    struct RenamePattern;

    impl RewritePattern for RenamePattern {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
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

        let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(RenamePattern);

        let mut target = ConversionTarget::new();
        target.add_legal_dialect("test");
        target.add_illegal_op("test", "source");

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
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

        let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(RenamePattern);

        let target = ConversionTarget::new();
        applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        // op2's operand should now point to the replacement op's result
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2);

        let new_result = ctx.op_result(ops[0], 0);
        let op2_operands = ctx.op_operands(ops[1]);
        assert_eq!(op2_operands[0], new_result);
    }

    #[test]
    fn full_conversion_rejects_unknown_ops() {
        let (mut ctx, loc) = test_ctx();

        let op_data = OperationDataBuilder::new(loc, Symbol::new("unknown"), Symbol::new("op"))
            .build(&mut ctx);
        let op = ctx.create_op(op_data);
        let module = make_module(&mut ctx, loc, vec![op]);

        let applicator = PatternApplicator::new(TypeConverter::new());
        let target = ConversionTarget::new();

        let error = match applicator.with_target(target).apply_full_conversion(
            &mut ctx,
            module,
            "test-boundary",
        ) {
            Ok(_) => panic!("full conversion should reject unknown ops"),
            Err(error) => error,
        };
        assert_eq!(error.boundary(), "test-boundary");
        assert_eq!(error.operations().len(), 1);
        assert_eq!(error.operations()[0].legality, LegalityCheck::Unknown);
    }

    #[test]
    fn mutable_conversion_target_skips_legal_ops() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op_data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(op_data);
        let module = make_module(&mut ctx, loc, vec![op]);

        let mut applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(RenamePattern);
        applicator
            .conversion_target_mut()
            .add_legal_op("test", "source");

        let result = applicator
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        assert_eq!(result.total_changes, 0);
        assert_eq!(ctx.op(module.ops(&ctx)[0]).name, Symbol::new("source"));
    }

    #[test]
    fn legal_container_without_recursive_marker_rewrites_descendants() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let nested_data =
            OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
                .result(i32_ty)
                .build(&mut ctx);
        let nested = ctx.create_op(nested_data);
        let container = make_container(&mut ctx, loc, vec![nested]);
        let module = make_module(&mut ctx, loc, vec![container]);

        let target = ConversionTarget::new().legal_op("test", "container");
        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        assert_eq!(result.total_changes, 1);
        let rewritten = first_nested_op(&ctx, container);
        assert_eq!(ctx.op(rewritten).name, Symbol::new("target"));
    }

    #[test]
    fn recursive_legal_container_skips_descendant_rewrites() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let nested_data =
            OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
                .result(i32_ty)
                .build(&mut ctx);
        let nested = ctx.create_op(nested_data);
        let container = make_container(&mut ctx, loc, vec![nested]);
        let module = make_module(&mut ctx, loc, vec![container]);

        let target = ConversionTarget::new()
            .legal_op("test", "container")
            .recursive_legal_op("test", "container");
        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        assert_eq!(result.total_changes, 0);
        let skipped = first_nested_op(&ctx, container);
        assert_eq!(ctx.op(skipped).name, Symbol::new("source"));
    }

    #[test]
    fn apply_partial_rewrites_only_selected_typed_scope_regions() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let selected_nested =
            OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
                .result(i32_ty)
                .build(&mut ctx);
        let selected_nested = ctx.create_op(selected_nested);
        let selected_func = make_func_container(&mut ctx, loc, "selected", vec![selected_nested]);

        let sibling_nested =
            OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("source"))
                .result(i32_ty)
                .build(&mut ctx);
        let sibling_nested = ctx.create_op(sibling_nested);
        let sibling_func = make_func_container(&mut ctx, loc, "sibling", vec![sibling_nested]);
        let _module = make_module(
            &mut ctx,
            loc,
            vec![selected_func.op_ref(), sibling_func.op_ref()],
        );

        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .apply_partial(&mut ctx, selected_func);

        assert_eq!(result.total_changes, 1);
        assert_eq!(
            ctx.op(first_nested_op(&ctx, selected_func.op_ref())).name,
            Symbol::new("target")
        );
        assert_eq!(
            ctx.op(first_nested_op(&ctx, sibling_func.op_ref())).name,
            Symbol::new("source")
        );
    }

    #[test]
    fn conversion_target_does_not_enable_auto_type_conversion() {
        let applicator = PatternApplicator::new(TypeConverter::new());
        let applicator = applicator.with_target(ConversionTarget::new());

        assert!(!applicator.auto_type_conversion);
    }
}
