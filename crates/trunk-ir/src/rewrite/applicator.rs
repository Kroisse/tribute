//! PatternApplicator for arena IR.
//!
//! Applies rewrite patterns to a fixpoint: one sweep over every operation in
//! the scope, then rounds over only the operations earlier rewrites queued.

use super::Module;
use super::conversion_target::{
    ConversionError, ConversionMode, ConversionTarget, IllegalOp, LegalityCheck,
};
use super::pattern::RewritePattern;
use super::rewriter::{self, PatternRewriter};
use super::type_converter::TypeConverter;
use crate::OpName;
use crate::context::{IrContext, Use};
use crate::dialect::{clif, core, func, wasm};
use crate::ops::DialectOp;
use crate::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

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

impl RewriteScope for core::Module {
    type Regions = std::option::IntoIter<RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions {
        Module::from(self).regions(ctx)
    }

    fn module_first_block(self, ctx: &IrContext) -> Option<BlockRef> {
        Module::from(self).module_first_block(ctx)
    }
}

impl RewriteScope for func::Func {
    type Regions = std::option::IntoIter<RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions {
        ctx.op_region(self.op_ref(), 0).into_iter()
    }

    fn module_first_block(self, _ctx: &IrContext) -> Option<BlockRef> {
        None
    }
}

impl RewriteScope for clif::Func {
    type Regions = std::option::IntoIter<RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions {
        ctx.op_region(self.op_ref(), 0).into_iter()
    }

    fn module_first_block(self, _ctx: &IrContext) -> Option<BlockRef> {
        None
    }
}

impl RewriteScope for wasm::Func {
    type Regions = std::option::IntoIter<RegionRef>;

    fn regions(self, ctx: &IrContext) -> Self::Regions {
        ctx.op_region(self.op_ref(), 0).into_iter()
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

/// Applies rewrite patterns to arena IR until no pattern matches.
///
/// The first round sweeps every operation in the scope, each after the
/// operations nested in it. Each later round offers only the operations the
/// previous round queued, in the order it queued them, and the applicator
/// stops at a round that changes nothing. A rewrite of an operation queues
/// the operations it inserted (after those nested in them), the users of the
/// values that replace its results, the definers of its operands, and its
/// ancestors; a cast or type conversion the applicator makes queues the
/// operations it changed. An operation gets at most one rewrite per round,
/// from the first pattern in insertion order that matches it.
///
/// A pattern therefore changes attached IR only through its
/// [`PatternRewriter`]; it may shape operations it built itself freely before
/// recording them. A change made directly on attached IR is not queued, and
/// a pattern it enables may never run. Debug builds check this: at the
/// fixpoint they offer every operation again on a copy of the context and
/// panic if a pattern still matches or a type would still convert. A
/// converted block argument queues its users but not the branches into its
/// block, so a pattern that matches on a successor's argument types relies
/// on that check.
///
/// The applicator holds its patterns for the lifetime `'p`, so patterns
/// borrow data the driver computed, such as a symbol table or a lowering
/// plan, instead of sharing it through `Rc` or `Arc`.
///
/// A pattern decides only from the IR and that read-only data. A pattern
/// that reports to the driver, such as a count or a collected list, writes
/// through a `Cell` or `RefCell` the driver owns and reads back after
/// applying the patterns; what it writes never decides a later rewrite, and
/// it reports only the rewrites it records, since the debug check offers it
/// operations again. A
/// pattern must not cache facts about the IR, such as uses, types, or the
/// operations a region holds, between calls: later rewrites and the
/// applicator's own mutations make them stale.
pub struct PatternApplicator<'p> {
    patterns: Vec<Box<dyn RewritePattern + 'p>>,
    /// Indices of the patterns offered each operation name a pattern
    /// declares as a root, in insertion order.
    by_root: HashMap<OpName, Vec<usize>>,
    /// Indices of the patterns that declare no roots, offered every other
    /// operation.
    any_root: Vec<usize>,
    max_iterations: usize,
    type_converter: TypeConverter,
    /// Conversion target used for legality-aware rewriting and verification.
    target: ConversionTarget,
    /// Whether to automatically convert block argument types and insert
    /// `unrealized_conversion_cast` for operand type mismatches.
    auto_type_conversion: bool,
}

impl<'p> PatternApplicator<'p> {
    /// Create a new applicator with the given type converter.
    pub fn new(type_converter: TypeConverter) -> Self {
        Self {
            patterns: Vec::new(),
            by_root: HashMap::default(),
            any_root: Vec::new(),
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
    pub fn add_pattern(self, pattern: impl RewritePattern + 'p) -> Self {
        self.add_pattern_box(Box::new(pattern))
    }

    /// Add an already-boxed rewrite pattern.
    ///
    /// Useful when collecting heterogeneous patterns from multiple sources
    /// (e.g. per-dialect `canonicalization_patterns()` registries) into a
    /// single applicator without naming each concrete type at the call site.
    pub fn add_pattern_box(mut self, pattern: Box<dyn RewritePattern + 'p>) -> Self {
        let index = self.patterns.len();
        match pattern.roots() {
            Some(roots) => {
                for &root in roots {
                    let indices = self
                        .by_root
                        .entry(root)
                        .or_insert_with(|| self.any_root.clone());
                    if indices.last() != Some(&index) {
                        indices.push(index);
                    }
                }
            }
            None => {
                self.any_root.push(index);
                for indices in self.by_root.values_mut() {
                    indices.push(index);
                }
            }
        }
        self.patterns.push(pattern);
        self
    }

    /// The indices of the patterns offered `op`, in insertion order.
    fn patterns_for(&self, ctx: &IrContext, op: OpRef) -> &[usize] {
        self.by_root.get(&ctx.op(op).name).unwrap_or(&self.any_root)
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
        let regions: crate::RegionList = scope.regions(ctx).collect();
        let mut worklist = Worklist::default();
        let mut total_changes = 0;
        let mut iterations = 0;

        while iterations < self.max_iterations {
            iterations += 1;
            let changes = if iterations == 1 {
                regions
                    .iter()
                    .map(|&region| {
                        self.visit_region(ctx, region, module_first_block, &mut worklist)
                    })
                    .sum()
            } else {
                self.revisit(ctx, &regions, module_first_block, &mut worklist)
            };
            total_changes += changes;
            if changes == 0 {
                // A first round that changes nothing was a full sweep.
                if cfg!(debug_assertions) && total_changes > 0 {
                    self.check_fixpoint(ctx, &regions);
                }
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

    /// Offer the operations queued by the previous round, skipping those a
    /// later rewrite removed or that left the scope.
    fn revisit(
        &self,
        ctx: &mut IrContext,
        regions: &[RegionRef],
        module_first_block: Option<BlockRef>,
        worklist: &mut Worklist,
    ) -> usize {
        let mut changes = 0;
        for op in worklist.take() {
            if !self.is_offered(ctx, op, regions) {
                continue;
            }
            let block = ctx.op(op).parent_block.expect("offered op is attached");
            changes += self.rewrite_op(ctx, block, op, module_first_block, worklist);
        }
        changes
    }

    /// Whether `op` is attached below one of `regions` and no operation on
    /// the way, including `op`, is recursively legal.
    fn is_offered(&self, ctx: &IrContext, op: OpRef, regions: &[RegionRef]) -> bool {
        let mut current = op;
        loop {
            if self.target.has_constraints() && self.target.is_recursively_legal(ctx, current) {
                return false;
            }
            let Some(block) = ctx.op(current).parent_block else {
                return false;
            };
            let Some(region) = ctx.block(block).parent_region else {
                return false;
            };
            if regions.contains(&region) {
                return true;
            }
            let Some(parent) = ctx.region(region).parent_op else {
                return false;
            };
            current = parent;
        }
    }

    fn visit_region(
        &self,
        ctx: &mut IrContext,
        region: RegionRef,
        module_first_block: Option<BlockRef>,
        worklist: &mut Worklist,
    ) -> usize {
        let mut changes = 0;
        let blocks = ctx.region(region).blocks.clone();
        for block in blocks {
            changes += self.visit_block(ctx, block, module_first_block, worklist);
        }
        changes
    }

    fn visit_block(
        &self,
        ctx: &mut IrContext,
        block: BlockRef,
        module_first_block: Option<BlockRef>,
        worklist: &mut Worklist,
    ) -> usize {
        let mut changes = self.convert_block_args(ctx, block, worklist);

        // Snapshot the ops in this block
        let ops = ctx.block(block).ops.clone();

        for op in ops {
            // Skip ops that have been removed from their block
            if ctx.op(op).parent_block != Some(block) {
                continue;
            }

            if self.target.has_constraints() && self.target.is_recursively_legal(ctx, op) {
                continue;
            }

            // First, recurse into nested regions
            let regions = ctx.op_regions(op).collect::<crate::RegionList>();
            for region in regions {
                changes += self.visit_region(ctx, region, module_first_block, worklist);
            }

            // Skip ops that were removed during nested processing
            if ctx.op(op).parent_block != Some(block) {
                continue;
            }

            changes += self.rewrite_op(ctx, block, op, module_first_block, worklist);
        }

        changes
    }

    /// Convert the argument types of `block`, queueing the users of each
    /// converted argument.
    ///
    /// Only an applicator with automatic type conversion converts them, so
    /// that other passes do not change types inadvertently.
    fn convert_block_args(
        &self,
        ctx: &mut IrContext,
        block: BlockRef,
        worklist: &mut Worklist,
    ) -> usize {
        if !self.auto_type_conversion || self.type_converter.is_empty() {
            return 0;
        }
        let mut changes = 0;
        let block_args = ctx.block_args(block).to_vec();
        for (i, arg_val) in block_args.iter().enumerate() {
            let raw_ty = ctx.value_ty(*arg_val);
            if let Some(new_ty) = self.type_converter.convert_type(ctx, raw_ty)
                && new_ty != raw_ty
            {
                ctx.set_block_arg_type(block, i as u32, new_ty);
                let uses = ctx.uses(*arg_val).to_vec();
                worklist.extend(uses.iter().map(|use_| use_.user));
                self.restore_unconverted_uses(ctx, *arg_val, raw_ty, &uses, worklist);
                changes += 1;
            }
        }
        changes
    }

    /// Cast the operands of `op` and offer it to its patterns, applying the
    /// first that matches and queueing every operation the rewrite may have
    /// enabled another pattern on.
    fn rewrite_op(
        &self,
        ctx: &mut IrContext,
        block: BlockRef,
        op: OpRef,
        module_first_block: Option<BlockRef>,
        worklist: &mut Worklist,
    ) -> usize {
        // Skip cast insertion and pattern matching for legal operations
        if self.target.has_constraints() && self.target.is_legal(ctx, op) == LegalityCheck::Legal {
            return 0;
        }
        let mut changes = 0;
        // Only an applicator with automatic type conversion casts operands, so
        // that other passes do not box or cast operands of unrelated ops.
        if self.converts_operands_of(ctx, op) {
            let casts = self.insert_conversion_casts(ctx, block, op, worklist);
            if casts > 0 {
                worklist.push(op);
            }
            changes += casts;
        }

        for &index in self.patterns_for(ctx, op) {
            let mut rw = PatternRewriter::new(&self.type_converter);
            let matched = self.patterns[index]
                .match_and_rewrite(ctx, op, &mut rw)
                .is_some();
            if !matched && let Some(reason) = rw.take_no_match() {
                tracing::debug!(
                    pattern = self.patterns[index].name(),
                    op = %ctx.op(op).name.display(ctx),
                    "no match: {reason}",
                );
            }
            if matched && rw.has_mutations() {
                let mutations = rw.take_mutations();
                self.queue_neighbors(ctx, op, worklist);
                let replaced = self.replaced_results(ctx, op, &mutations);
                let mut new_ops = mutations.prefix_ops.clone();
                new_ops.extend(mutations.replacement);
                if module_first_block.is_some() {
                    new_ops.extend(mutations.module_ops.iter().copied());
                }
                let new_values = match (&mutations.replacement, &mutations.erase_values) {
                    (Some(replacement), _) => ctx.op_results(*replacement).to_vec(),
                    (None, Some(values)) => values.clone(),
                    (None, None) => Vec::new(),
                };
                rewriter::apply_mutations(ctx, op, mutations, module_first_block);
                for new_op in new_ops {
                    changes += self.queue_new_op(ctx, new_op, worklist);
                }
                for value in new_values {
                    worklist.extend(ctx.uses(value).iter().map(|use_| use_.user));
                }
                for (value, old_ty, uses) in replaced {
                    self.restore_unconverted_uses(ctx, value, old_ty, &uses, worklist);
                }
                changes += 1;
                break; // Only apply one pattern per op per iteration
            }
        }
        changes
    }

    /// Queue the operations whose match may depend on `op`, which is about
    /// to be rewritten: the definers of its operands and its ancestors.
    fn queue_neighbors(&self, ctx: &IrContext, op: OpRef, worklist: &mut Worklist) {
        for &operand in ctx.op_operands(op) {
            if let ValueDef::OpResult(definer, _) = ctx.value_def(operand) {
                worklist.push(definer);
            }
        }
        let mut current = op;
        while let Some(parent) = ctx
            .op(current)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region)
            .and_then(|region| ctx.region(region).parent_op)
        {
            worklist.push(parent);
            current = parent;
        }
    }

    /// Queue an operation a rewrite inserted, after the operations nested
    /// in it, converting the argument types of its blocks first.
    fn queue_new_op(&self, ctx: &mut IrContext, op: OpRef, worklist: &mut Worklist) -> usize {
        // A sweep skips a recursively legal operation and everything in it.
        if self.target.has_constraints() && self.target.is_recursively_legal(ctx, op) {
            return 0;
        }
        let mut changes = 0;
        for region in ctx.op_regions(op).collect::<crate::RegionList>() {
            for block in ctx.region(region).blocks.clone() {
                changes += self.convert_block_args(ctx, block, worklist);
                for nested in ctx.block(block).ops.clone() {
                    changes += self.queue_new_op(ctx, nested, worklist);
                }
            }
        }
        worklist.push(op);
        changes
    }

    /// Panic if another sweep over `regions` would still change the IR.
    ///
    /// The applicator revisits only the operations a rewrite queued, so a
    /// pattern that changes attached IR other than through its rewriter
    /// can leave a match behind. Debug builds probe a copy of the context
    /// so that the check cannot change the result.
    fn check_fixpoint(&self, ctx: &IrContext, regions: &[RegionRef]) {
        let mut probe = ctx.clone();
        let mut ops = Vec::new();
        for &region in regions {
            if self.auto_type_conversion && !self.type_converter.is_empty() {
                for &block in &probe.region(region).blocks {
                    assert!(
                        !self.converts_any(&probe, probe.block_args(block).iter().copied()),
                        "a block argument in the scope still converts after the applicator \
                         reached a fixpoint",
                    );
                }
            }
            collect_ops_post_order(&probe, region, &mut ops);
        }
        for op in ops {
            if !self.is_offered(&probe, op, regions) {
                continue;
            }
            if self.auto_type_conversion && !self.type_converter.is_empty() {
                for region in probe.op_regions(op) {
                    for &block in &probe.region(region).blocks {
                        assert!(
                            !self.converts_any(&probe, probe.block_args(block).iter().copied()),
                            "a block argument in `{}` still converts after the applicator \
                             reached a fixpoint",
                            probe.op(op).name.display(&probe),
                        );
                    }
                }
            }
            if self.target.has_constraints()
                && self.target.is_legal(&probe, op) == LegalityCheck::Legal
            {
                continue;
            }
            assert!(
                !(self.converts_operands_of(&probe, op)
                    && core::UnrealizedConversionCast::from_op(&probe, op).is_err()
                    && self.converts_any(&probe, probe.op_operands(op).iter().copied())),
                "an operand of `{}` still converts after the applicator reached a fixpoint",
                probe.op(op).name.display(&probe),
            );
            for &index in self.patterns_for(&probe, op) {
                let pattern = &self.patterns[index];
                let mut rw = PatternRewriter::new(&self.type_converter);
                if pattern.match_and_rewrite(&mut probe, op, &mut rw).is_some()
                    && rw.has_mutations()
                {
                    panic!(
                        "pattern `{}` still matches `{}` after the applicator reached a \
                         fixpoint; a rewrite changed it without queueing it",
                        pattern.name(),
                        probe.op(op).name.display(&probe),
                    );
                }
            }
        }
    }

    /// Whether the type converter changes the type of any of `values`.
    fn converts_any(&self, ctx: &IrContext, values: impl IntoIterator<Item = ValueRef>) -> bool {
        values.into_iter().any(|value| {
            let ty = ctx.value_ty(value);
            self.type_converter
                .convert_type(ctx, ty)
                .is_some_and(|converted| converted != ty)
        })
    }

    /// Whether this conversion casts the operands of `op` to their converted
    /// types. With a conversion target, only an operation the target declares
    /// illegal is converted here; one it has no rule for is left to a later
    /// conversion and keeps the operand types its results agree with.
    fn converts_operands_of(&self, ctx: &IrContext, op: OpRef) -> bool {
        self.auto_type_conversion
            && !self.type_converter.is_empty()
            && (!self.target.has_constraints()
                || self.target.is_legal(ctx, op) == LegalityCheck::Illegal)
    }

    /// The values that will replace the results of `op` with a different
    /// type, each with the result's type and the uses it has before the
    /// replacement.
    fn replaced_results(
        &self,
        ctx: &IrContext,
        op: OpRef,
        mutations: &rewriter::Mutations,
    ) -> Vec<(ValueRef, TypeRef, Vec<Use>)> {
        if !self.auto_type_conversion || !self.target.has_constraints() {
            return Vec::new();
        }
        let new_values = match (&mutations.replacement, &mutations.erase_values) {
            (Some(replacement), _) => ctx.op_results(*replacement).to_vec(),
            (None, Some(values)) => values.clone(),
            (None, None) => return Vec::new(),
        };
        ctx.op_results(op)
            .iter()
            .zip(new_values)
            .filter(|&(&old, new)| ctx.value_ty(old) != ctx.value_ty(new))
            .map(|(&old, new)| (new, ctx.value_ty(old), ctx.uses(old).to_vec()))
            .collect()
    }

    /// Cast `value`, whose type this conversion changed from `old_ty`, back to
    /// `old_ty` at each of `uses` by an operation that the target has no rule
    /// for and whose schema the new type violates.
    ///
    /// Such an operation is converted by a later pass and still relates the
    /// old type to its other operands and results, so the IR stays
    /// schema-valid until then. An operation that accepts the new type keeps
    /// using the value directly.
    fn restore_unconverted_uses(
        &self,
        ctx: &mut IrContext,
        value: ValueRef,
        old_ty: TypeRef,
        uses: &[Use],
        worklist: &mut Worklist,
    ) {
        if !self.target.has_constraints() {
            return;
        }
        for use_ in uses {
            let user = use_.user;
            if self.target.is_legal(ctx, user) != LegalityCheck::Unknown
                || !violates_schema(ctx, user)
            {
                continue;
            }
            let Some(block) = ctx.op(user).parent_block else {
                continue;
            };
            let loc = ctx.op(user).location;
            let cast = core::UnrealizedConversionCast::operands(value)
                .results(old_ty)
                .build(ctx, loc);
            ctx.insert_op_before(block, user, cast.op_ref());
            ctx.set_op_operand(user, use_.operand_index, cast.result(ctx));
            worklist.push(cast.op_ref());
        }
    }

    /// Insert `core.unrealized_conversion_cast` for operands whose types
    /// need conversion. Skips `unrealized_conversion_cast` itself to avoid
    /// infinite loops.
    fn insert_conversion_casts(
        &self,
        ctx: &mut IrContext,
        block: BlockRef,
        op: OpRef,
        worklist: &mut Worklist,
    ) -> usize {
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
                let cast = core::UnrealizedConversionCast::operands(operand)
                    .results(target_ty)
                    .build(ctx, loc);
                let cast_ref = cast.op_ref();
                let cast_result = cast.result(ctx);
                ctx.insert_op_before(block, op, cast_ref);
                ctx.set_op_operand(op, i as u32, cast_result);
                worklist.push(cast_ref);
                cast_count += 1;
            }
        }
        cast_count
    }
}

/// The operations a round changed or that may match after its changes,
/// offered again in the next round in the order they were queued.
#[derive(Default)]
struct Worklist {
    ops: Vec<OpRef>,
    queued: HashSet<OpRef>,
}

impl Worklist {
    fn push(&mut self, op: OpRef) {
        if self.queued.insert(op) {
            self.ops.push(op);
        }
    }

    fn extend(&mut self, ops: impl IntoIterator<Item = OpRef>) {
        for op in ops {
            self.push(op);
        }
    }

    fn take(&mut self) -> Vec<OpRef> {
        self.queued.clear();
        std::mem::take(&mut self.ops)
    }
}

/// Collect the operations below `region`, each after those nested in it.
fn collect_ops_post_order(ctx: &IrContext, region: RegionRef, ops: &mut Vec<OpRef>) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            for nested in ctx.op_regions(op) {
                collect_ops_post_order(ctx, nested, ops);
            }
            ops.push(op);
        }
    }
}

/// Whether `op` violates the declarative schema of its definition.
fn violates_schema(ctx: &IrContext, op: OpRef) -> bool {
    crate::op_def::OpDef::of(ctx, op).is_some_and(|def| !def.schema.verify(ctx, op).is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{func, wasm};
    use crate::location::Span;
    use crate::ops::DialectOp;
    use crate::printer::print_module;
    use crate::rewrite::conversion_target::ConversionTarget;

    use crate::*;
    use smallvec::smallvec;
    use std::cell::RefCell;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.trb");
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
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
        let module_data = OperationDataBuilder::new(loc, "core.module")
            .attr("sym_name", Attribute::String(ctx.intern_str("test")))
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
        let container_data = OperationDataBuilder::new(loc, "test.container")
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
        let nil_ty = crate::dialect::core::Nil::params()
            .intern(ctx)
            .as_type_ref();
        let func_ty = crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref();
        let func_data = OperationDataBuilder::new(loc, "func.func")
            .attr("sym_name", Attribute::String(ctx.intern_str(name)))
            .attr("type", Attribute::Type(func_ty))
            .region(region)
            .build(ctx);
        let func_op = ctx.create_op(func_data);
        func::Func::from_op(ctx, func_op).expect("test func should be valid")
    }

    fn make_wasm_func_container(
        ctx: &mut IrContext,
        loc: Location,
        name: &'static str,
        nested_ops: Vec<OpRef>,
    ) -> wasm::Func {
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
        let nil_ty = crate::dialect::core::Nil::params()
            .intern(ctx)
            .as_type_ref();
        let func_ty = crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref();
        wasm::Func::operands()
            .sym_name(name)
            .r#type(func_ty)
            .regions(region)
            .build(ctx, loc)
    }

    fn make_bodyless_func(
        ctx: &mut IrContext,
        loc: Location,
        func_op: &'static crate::OpDef,
        name: &'static str,
    ) -> OpRef {
        let nil_ty = crate::dialect::core::Nil::params()
            .intern(ctx)
            .as_type_ref();
        let func_ty = crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref();
        let op_data = OperationDataBuilder::new(loc, func_op)
            .attr("sym_name", Attribute::String(ctx.intern_str(name)))
            .attr("type", Attribute::Type(func_ty))
            .build(ctx);
        ctx.create_op(op_data)
    }

    fn first_nested_op(ctx: &IrContext, op: OpRef) -> OpRef {
        let region = ctx.op_region(op, 0).unwrap();
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
        ) -> Option<()> {
            let data = ctx.op(op);
            if !data.name.is(ctx, "test.source") {
                return None;
            }

            let loc = data.location;
            let result_types: Vec<TypeRef> = ctx.op_result_types(op).to_vec();

            let new_data = OperationDataBuilder::new(loc, "test.target")
                .results(result_types)
                .build(ctx);
            let new_op = ctx.create_op(new_data);
            rewriter.replace_op(new_op);
            Some(())
        }
    }

    #[test]
    fn applicator_renames_op() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(op_data);
        let module = make_module(&mut ctx, loc, vec![op]);

        let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(RenamePattern);

        let mut target = ConversionTarget::new();
        target.add_legal_dialect("test");
        target.add_illegal_op("test.source");

        let result = applicator
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();
        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 1);

        // Verify the op was renamed
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        assert!(ctx.op(ops[0]).name.is(&ctx, "test.target"));
    }

    #[test]
    fn applicator_preserves_uses_via_rauw() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // op1: test.source -> %0
        let op1_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(op1_data);
        let v1 = ctx.op_result(op1, 0);

        // op2: test.use(%0)
        let op2_data = OperationDataBuilder::new(loc, "test.use")
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

        let op_data = OperationDataBuilder::new(loc, "unknown.op").build(&mut ctx);
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

        let op_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let op = ctx.create_op(op_data);
        let module = make_module(&mut ctx, loc, vec![op]);

        let mut applicator =
            PatternApplicator::new(TypeConverter::new()).add_pattern(RenamePattern);
        applicator
            .conversion_target_mut()
            .add_legal_op("test.source");

        let result = applicator
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        assert_eq!(result.total_changes, 0);
        assert!(ctx.op(module.ops(&ctx)[0]).name.is(&ctx, "test.source"));
    }

    #[test]
    fn legal_container_without_recursive_marker_rewrites_descendants() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let nested_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let nested = ctx.create_op(nested_data);
        let container = make_container(&mut ctx, loc, vec![nested]);
        let module = make_module(&mut ctx, loc, vec![container]);

        let target = ConversionTarget::new().legal_op("test.container");
        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        assert_eq!(result.total_changes, 1);
        let rewritten = first_nested_op(&ctx, container);
        assert!(ctx.op(rewritten).name.is(&ctx, "test.target"));
    }

    #[test]
    fn recursive_legal_container_skips_descendant_rewrites() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let nested_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let nested = ctx.create_op(nested_data);
        let container = make_container(&mut ctx, loc, vec![nested]);
        let module = make_module(&mut ctx, loc, vec![container]);

        let target = ConversionTarget::new()
            .legal_op("test.container")
            .recursive_legal_op("test.container");
        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .with_target(target)
            .apply_partial_conversion(&mut ctx, module, "test-boundary")
            .unwrap();

        assert_eq!(result.total_changes, 0);
        let skipped = first_nested_op(&ctx, container);
        assert!(ctx.op(skipped).name.is(&ctx, "test.source"));
    }

    #[test]
    fn apply_partial_rewrites_only_selected_typed_scope_regions() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let selected_nested = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let selected_nested = ctx.create_op(selected_nested);
        let selected_func = make_func_container(&mut ctx, loc, "selected", vec![selected_nested]);

        let sibling_nested = OperationDataBuilder::new(loc, "test.source")
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
        assert!(
            ctx.op(first_nested_op(&ctx, selected_func.op_ref()))
                .name
                .is(&ctx, "test.target")
        );
        assert!(
            ctx.op(first_nested_op(&ctx, sibling_func.op_ref()))
                .name
                .is(&ctx, "test.source")
        );
    }

    #[test]
    fn func_scope_skips_bodyless_and_visits_body() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let bodyless = make_bodyless_func(&mut ctx, loc, crate::dialect::func::Func::DEF, "extern");
        let nested_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let nested = ctx.create_op(nested_data);
        let bodyful = make_func_container(&mut ctx, loc, "defined", vec![nested]);
        let module = make_module(&mut ctx, loc, vec![bodyless, bodyful.op_ref()]);

        let before = print_module(&ctx, module.op());
        let bodyless = func::Func::from_op(&ctx, bodyless).unwrap();
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .apply_partial(&mut ctx, bodyless);
        assert_eq!(print_module(&ctx, module.op()), before);

        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .apply_partial(&mut ctx, bodyful);
        assert_eq!(result.total_changes, 1);
        assert!(
            ctx.op(first_nested_op(&ctx, bodyful.op_ref()))
                .name
                .is(&ctx, "test.target")
        );
    }

    #[test]
    fn wasm_scope_skips_bodyless_and_visits_body() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let bodyless = make_bodyless_func(&mut ctx, loc, crate::dialect::wasm::Func::DEF, "extern");
        let nested_data = OperationDataBuilder::new(loc, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let nested = ctx.create_op(nested_data);
        let bodyful = make_wasm_func_container(&mut ctx, loc, "defined", vec![nested]);
        let module = make_module(&mut ctx, loc, vec![bodyless, bodyful.op_ref()]);

        let before = print_module(&ctx, module.op());
        let bodyless = wasm::Func::from_op(&ctx, bodyless).unwrap();
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .apply_partial(&mut ctx, bodyless);
        assert_eq!(print_module(&ctx, module.op()), before);

        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenamePattern)
            .apply_partial(&mut ctx, bodyful);
        assert_eq!(result.total_changes, 1);
        assert!(
            ctx.op(first_nested_op(&ctx, bodyful.op_ref()))
                .name
                .is(&ctx, "test.target")
        );
    }

    /// Convert `core.i1` to `core.i32` in `input`, with `test.*` as the only
    /// dialect the conversion declares illegal, and print the result.
    fn convert_booleans(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let i32_ty = i32_type(&mut ctx);
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |ctx, ty| {
            crate::dialect::core::I1::matches(ctx, ty).then_some(i32_ty)
        });
        let mut target = ConversionTarget::new();
        target.add_illegal_dialect("test");
        PatternApplicator::new(converter)
            .with_auto_type_conversion(true)
            .with_target(target)
            .apply_partial(&mut ctx, module);
        let schemas = crate::validation::validate_op_schemas(&ctx, module.op());
        assert!(schemas.is_ok(), "{schemas}");
        print_module(&ctx, module.op())
    }

    #[test]
    fn retyped_value_is_cast_back_only_where_a_later_conversion_still_needs_it() {
        let printed = convert_booleans(
            r#"core.module @test {
  func.func @main(%flag: core.i1, %other: core.i1) {
    ^entry(%flag: core.i1, %other: core.i1):
      %both = arith.and %flag, %other : core.i1
      cf.br %flag [^exit]
    ^exit(%forwarded: core.i1):
      func.return
  }
}"#,
        );
        assert_eq!(
            printed.matches("core.unrealized_conversion_cast").count(),
            2,
            "only the operands of arith.and return to core.i1: {printed}"
        );
        assert!(
            printed.contains(": core.i1\n      cf.br %0 [^bb1]"),
            "{printed}"
        );
    }

    #[test]
    fn operands_of_an_operation_left_for_a_later_conversion_are_not_cast() {
        let printed = convert_booleans(
            r#"core.module @test {
  func.func @main(%left: core.i32, %right: core.i32) {
    %less = arith.cmpi %left, %right {predicate = "slt"} : core.i1
    %both = arith.and %less, %less : core.i1
    func.return
  }
}"#,
        );
        assert!(
            !printed.contains("core.unrealized_conversion_cast"),
            "{printed}"
        );
    }

    /// Renames each `test.source` whose `tag` the driver selects to
    /// `test.target`, and records the tag in a log the driver owns.
    struct RenameSelected<'a> {
        selected: &'a [&'a str],
        renamed: &'a RefCell<Vec<String>>,
    }

    impl RewritePattern for RenameSelected<'_> {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            if !ctx.op(op).name.is(ctx, "test.source") {
                return None;
            }
            let tag = ctx.op_attrs(op).get_str("tag").unwrap().to_owned();
            if !self.selected.contains(&tag.as_str()) {
                return None;
            }
            self.renamed.borrow_mut().push(tag);
            let loc = ctx.op(op).location;
            let result_types = ctx.op_result_types(op).to_vec();
            let new_data = OperationDataBuilder::new(loc, "test.target")
                .results(result_types)
                .build(ctx);
            let new_op = ctx.create_op(new_data);
            rewriter.replace_op(new_op);
            Some(())
        }
    }

    #[test]
    fn pattern_borrows_data_from_the_driver() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  %0 = test.source {tag = "first"} : core.i32
  %1 = test.source {tag = "second"} : core.i32
  %2 = test.source {tag = "third"} : core.i32
}"#,
        );

        let selected = ["first", "third"];
        let renamed = RefCell::new(Vec::new());
        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(RenameSelected {
                selected: &selected,
                renamed: &renamed,
            })
            .apply_partial(&mut ctx, module);

        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 2);
        assert_eq!(renamed.into_inner(), ["first", "third"]);
        let printed = print_module(&ctx, module.op());
        assert_eq!(printed.matches("test.target").count(), 2, "{printed}");
        assert!(
            printed.contains(r#"test.source {tag = "second"}"#),
            "{printed}"
        );
    }

    /// Records each operation offered to it and never matches.
    struct RecordOffers<'a> {
        roots: Option<&'a [OpName]>,
        offered: &'a RefCell<Vec<String>>,
    }

    impl RewritePattern for RecordOffers<'_> {
        fn roots(&self) -> Option<&[OpName]> {
            self.roots
        }

        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            _rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            let name = ctx.op(op).name.display(ctx).to_string();
            self.offered.borrow_mut().push(name);
            None
        }
    }

    #[test]
    fn patterns_are_offered_only_their_roots_in_insertion_order() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = arith.addi %x, %x : core.i32
    func.return %r
  }
}"#,
        );

        let offered = RefCell::new(Vec::new());
        let roots = [crate::dialect::arith::Addi::NAME];
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(RecordOffers {
                roots: Some(&roots),
                offered: &offered,
            })
            .add_pattern(RecordOffers {
                roots: None,
                offered: &offered,
            })
            .apply_partial(&mut ctx, module);

        assert_eq!(
            offered.into_inner(),
            ["arith.addi", "arith.addi", "func.return", "func.func"]
        );
    }

    /// Rewrites `from` to `to`; with `only_dead`, only once its results have
    /// no uses.
    struct Rename {
        from: &'static str,
        to: &'static str,
        only_dead: bool,
    }

    impl RewritePattern for Rename {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            if !ctx.op(op).name.is(ctx, self.from)
                || (self.only_dead && ctx.op_results(op).iter().any(|&v| ctx.has_uses(v)))
            {
                return None;
            }
            let loc = ctx.op(op).location;
            let result_types = ctx.op_result_types(op).to_vec();
            let new_data = OperationDataBuilder::new(loc, self.to)
                .results(result_types)
                .build(ctx);
            let new_op = ctx.create_op(new_data);
            rewriter.replace_op(new_op);
            Some(())
        }
    }

    /// Erases `test.sink`.
    struct EraseSink;

    impl RewritePattern for EraseSink {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            if !ctx.op(op).name.is(ctx, "test.sink") {
                return None;
            }
            rewriter.erase_op(vec![]);
            Some(())
        }
    }

    #[test]
    fn rewrite_revisits_the_definers_of_its_operands() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  %0 = test.source : core.i32
  test.sink %0
}"#,
        );

        let result = PatternApplicator::new(TypeConverter::new())
            .add_pattern(Rename {
                from: "test.source",
                to: "test.dead",
                only_dead: true,
            })
            .add_pattern(EraseSink)
            .apply_partial(&mut ctx, module);

        assert!(result.reached_fixpoint);
        assert_eq!(result.total_changes, 2);
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("test.dead"), "{printed}");
        assert!(!printed.contains("test.sink"), "{printed}");
    }

    /// Rewrites `test.source` and marks every `test.waiting` in its block
    /// ready directly, without queueing them.
    struct MarkSiblingsReady;

    impl RewritePattern for MarkSiblingsReady {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            if !ctx.op(op).name.is(ctx, "test.source") {
                return None;
            }
            let block = ctx.op(op).parent_block.unwrap();
            for sibling in ctx.block(block).ops.clone() {
                if ctx.op(sibling).name.is(ctx, "test.waiting") {
                    ctx.op_attrs_mut(sibling)
                        .insert("ready", Attribute::Bool(true));
                }
            }
            let loc = ctx.op(op).location;
            let new_data = OperationDataBuilder::new(loc, "test.target").build(ctx);
            let new_op = ctx.create_op(new_data);
            rewriter.replace_op(new_op);
            Some(())
        }
    }

    /// Rewrites a `test.waiting` marked ready.
    struct FinishReady;

    impl RewritePattern for FinishReady {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            if !ctx.op(op).name.is(ctx, "test.waiting") || !ctx.op_attrs(op).contains_key("ready") {
                return None;
            }
            let loc = ctx.op(op).location;
            let new_data = OperationDataBuilder::new(loc, "test.done").build(ctx);
            let new_op = ctx.create_op(new_data);
            rewriter.replace_op(new_op);
            Some(())
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "still matches `test.waiting`")]
    fn debug_builds_reject_a_change_made_outside_the_rewriter() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  test.waiting
  test.source
}"#,
        );

        PatternApplicator::new(TypeConverter::new())
            .add_pattern(MarkSiblingsReady)
            .add_pattern(FinishReady)
            .apply_partial(&mut ctx, module);
    }

    /// Replaces `test.source` with a `test.container` whose block takes a
    /// `core.i32` argument.
    struct InsertContainer;

    impl RewritePattern for InsertContainer {
        fn match_and_rewrite(
            &self,
            ctx: &mut IrContext,
            op: OpRef,
            rewriter: &mut PatternRewriter<'_>,
        ) -> Option<()> {
            if !ctx.op(op).name.is(ctx, "test.source") {
                return None;
            }
            let loc = ctx.op(op).location;
            let i32_ty = i32_type(ctx);
            let block = ctx.create_block(BlockData {
                location: loc,
                args: vec![BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                }],
                ops: smallvec![],
                parent_region: None,
            });
            let region = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![block],
                parent_op: None,
            });
            let data = OperationDataBuilder::new(loc, "test.container")
                .region(region)
                .build(ctx);
            let container = ctx.create_op(data);
            rewriter.replace_op(container);
            Some(())
        }
    }

    #[test]
    fn inserted_recursively_legal_op_keeps_its_block_argument_types() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  test.source
}"#,
        );
        let i32_ty = i32_type(&mut ctx);
        let i64_ty = TypeDataBuilder::new("core.i64").intern(&mut ctx);
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| (ty == i32_ty).then_some(i64_ty));
        let mut target = ConversionTarget::new();
        target.add_illegal_op("test.source");
        target.add_legal_op("test.container");
        target.add_recursive_legal_op("test.container");

        PatternApplicator::new(converter)
            .with_target(target)
            .with_auto_type_conversion(true)
            .add_pattern(InsertContainer)
            .apply_partial(&mut ctx, module);

        let container = ctx.block(module.first_block(&ctx).unwrap()).ops[0];
        let region = ctx.op_region(container, 0).unwrap();
        let block = ctx.region(region).blocks[0];
        assert_eq!(ctx.value_ty(ctx.block_arg(block, 0)), i32_ty);
    }

    #[test]
    fn conversion_target_does_not_enable_auto_type_conversion() {
        let applicator = PatternApplicator::new(TypeConverter::new());
        let applicator = applicator.with_target(ConversionTarget::new());

        assert!(!applicator.auto_type_conversion);
    }
}
