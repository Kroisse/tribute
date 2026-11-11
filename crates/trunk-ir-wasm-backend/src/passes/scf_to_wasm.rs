//! Lower scf dialect operations to wasm dialect (arena IR).
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.index_switch` -> one `wasm.br_table` in nested `wasm.block`s when
//!   dense, otherwise nested `wasm.if` comparisons
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br` to the enclosing `wasm.loop`
//! - `scf.break` -> `wasm.br` to the `wasm.block` around that loop
//!
//! A loop lowers its exits after the structures around them, since a Wasm
//! branch names its target by the number of labels in between.

use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockData, IrContext, OpList, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::dialect::scf;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::op_interface::{DynRegionBranch, DynRegionBranchTerminator};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, IllegalOp, LegalityCheck, Module, OpRewritePattern,
    PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::transforms::scf_control_flow::StructuredControlAnalysis;
use trunk_ir::walk::{WalkAction, walk_op};

const SCF_TO_WASM_BOUNDARY: &str = "scf-to-wasm";

/// Require successful conversion to remove every SCF operation.
pub fn scf_to_wasm_target() -> ConversionTarget {
    ConversionTarget::new().illegal_dialect(scf::DIALECT_NAME)
}

/// Lower scf dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
    analyses: &mut AnalysisCache,
) -> Result<(), ConversionError> {
    let analysis = analyses.require::<StructuredControlAnalysis>(ctx, module.op());
    let mut plan = ScfLoweringPlan::default();
    validate_structured_control(ctx, module, &analysis, |op, decision| match decision {
        ControlLowering::DropNeverResult => {
            plan.drop_never_results.insert(op);
            plan.terminal_controls.insert(op);
        }
        ControlLowering::ResultlessSwitch => {
            plan.resultless_switches.insert(op);
            plan.terminal_controls.insert(op);
        }
        ControlLowering::TerminalResultless => {
            plan.terminal_controls.insert(op);
        }
    })?;
    // Patterns consume decisions about source operations. They must not
    // query cached source facts after nested operations have been rewritten.
    analyses.invalidate::<StructuredControlAnalysis>(module.op());
    PatternApplicator::new(type_converter)
        .add_pattern(ScfIfPattern(&plan))
        .add_pattern(ScfSwitchPattern(&plan))
        .add_pattern(ScfLoopPattern(&plan))
        .add_pattern(ScfYieldPattern)
        .with_target(scf_to_wasm_target())
        .apply_partial_conversion(ctx, module, SCF_TO_WASM_BOUNDARY)?;
    Ok(())
}

/// Validate structured control before any target pipeline mutation.
pub fn validate_lowerable_structured_control(
    ctx: &IrContext,
    module: Module,
    analyses: &mut AnalysisCache,
) -> Result<(), ConversionError> {
    let analysis = analyses.require::<StructuredControlAnalysis>(ctx, module.op());
    validate_structured_control(ctx, module, &analysis, |_, _| {})
}

/// Wasm-specific decisions for the original operations, not cached IR facts.
#[derive(Default)]
struct ScfLoweringPlan {
    drop_never_results: HashSet<OpRef>,
    resultless_switches: HashSet<OpRef>,
    terminal_controls: HashSet<OpRef>,
}

enum ControlLowering {
    DropNeverResult,
    ResultlessSwitch,
    TerminalResultless,
}

/// Apply target legality to common facts. Validation-only callers need not
/// allocate a rewrite plan; lowering records accepted decisions before mutation.
fn validate_structured_control(
    ctx: &IrContext,
    module: Module,
    analysis: &StructuredControlAnalysis,
    mut accept: impl FnMut(OpRef, ControlLowering),
) -> Result<(), ConversionError> {
    validate_lowerable_switches(ctx, module)?;
    let result = walk_op(ctx, module.op(), &mut |op| {
        let has_never = DynRegionBranch::is_implemented(ctx, op)
            && ctx.op_results(op).iter().any(|&value| {
                let ty = ctx.get_type(ctx.value_ty(value));
                ty.name.is(ctx, "core.never")
            });
        if has_never {
            if analysis.has_terminal_unused_never_result(op) {
                accept(op, ControlLowering::DropNeverResult);
            } else {
                return ControlFlow::Break(ConversionError::new(SCF_TO_WASM_BOUNDARY, vec![IllegalOp {
                    op,
                    name: ctx.op(op).name.to_text(ctx),
                    legality: LegalityCheck::Illegal,
                    reason: Some("Never control requires one unused result, final block position, and terminal region successors".into()),
                }]));
            }
        } else if analysis.is_terminal_resultless_switch(op) {
            accept(op, ControlLowering::ResultlessSwitch);
        } else if (scf::If::matches(ctx, op) || scf::Loop::matches(ctx, op))
            && ctx.op_results(op).is_empty()
            && analysis.has_only_terminal_region_successors(op)
        {
            accept(op, ControlLowering::TerminalResultless);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    match result {
        ControlFlow::Break(error) => Err(error),
        ControlFlow::Continue(()) => Ok(()),
    }
}

/// Reject switches this target cannot lower without mutating the module.
pub fn validate_lowerable_switches(ctx: &IrContext, module: Module) -> Result<(), ConversionError> {
    if let Some((op, reason)) = find_nonlowerable_switch(ctx, module.op()) {
        let data = ctx.op(op);
        return Err(ConversionError::new(
            SCF_TO_WASM_BOUNDARY,
            vec![
                IllegalOp {
                    op,
                    name: data.name.to_text(ctx),
                    legality: LegalityCheck::Illegal,
                    reason: None,
                }
                .with_reason(reason.to_string()),
            ],
        ));
    }
    Ok(())
}

/// Validated data for an `scf.index_switch` that this target can lower.
///
/// Wasm compares integers natively only as `i32` or `i64`, so the discriminant
/// must be `core.i32` or `core.i64` and every case value must be representable
/// in the discriminant's type. Other source switch shapes remain for an
/// earlier or richer lowering rather than being partially rewritten here.
struct ScfSwitchArms {
    discriminant: ValueRef,
    width: SwitchWidth,
    cases: Vec<(i64, RegionRef)>,
    default: RegionRef,
}

/// The Wasm integer type a switch compares its discriminant in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwitchWidth {
    I32,
    I64,
}

struct ScfSwitchShape {
    discriminant: ValueRef,
    cases: Vec<(i64, RegionRef)>,
    default: RegionRef,
}

#[derive(Debug)]
enum SwitchLoweringReason {
    MalformedShape,
    UnsupportedDiscriminantType(String),
    CaseValueOutsideRange(SwitchWidth),
    TooManyResults(usize),
    ResultDefaultWithoutYield,
}

impl std::fmt::Display for SwitchLoweringReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedShape => write!(f, "malformed switch shape"),
            Self::UnsupportedDiscriminantType(ty) => {
                write!(
                    f,
                    "unsupported discriminant type `{ty}`; expected `core.i32`, `core.i64`, or `core.index`"
                )
            }
            Self::CaseValueOutsideRange(SwitchWidth::I32) => {
                write!(f, "case integer value is outside the i32 range")
            }
            Self::CaseValueOutsideRange(SwitchWidth::I64) => {
                write!(f, "case integer value is outside the i64 range")
            }
            Self::TooManyResults(count) => {
                write!(f, "switch has {count} results; at most one is supported")
            }
            Self::ResultDefaultWithoutYield => {
                write!(
                    f,
                    "a switch with a result and no cases needs a default ending in scf.yield"
                )
            }
        }
    }
}

fn switch_width(ctx: &IrContext, value: ValueRef) -> Option<SwitchWidth> {
    let ty = ctx.value_ty(value);
    // A Wasm `core.index` is a `core.i32`.
    if core::I32::matches(ctx, ty) || core::Index::matches(ctx, ty) {
        Some(SwitchWidth::I32)
    } else if core::I64::matches(ctx, ty) {
        Some(SwitchWidth::I64)
    } else {
        None
    }
}

/// Validate every region of the switch before rewriting any of them. This
/// keeps a malformed region from being partly lowered by the recursive
/// pattern walk.
fn switch_shape(ctx: &IrContext, op: OpRef) -> Option<ScfSwitchShape> {
    let switch = scf::IndexSwitch::from_op(ctx, op).ok()?;
    let cases: Vec<(i64, RegionRef)> = switch.cases(ctx).zip(switch.case_regions(ctx)).collect();
    let default = switch.default_region(ctx);
    for region in cases.iter().map(|(_, region)| *region).chain([default]) {
        let [entry] = ctx.region(region).blocks.as_array()?;
        if !ctx.block_args(*entry).is_empty() {
            return None;
        }
    }
    Some(ScfSwitchShape {
        discriminant: switch.discriminant(ctx),
        cases,
        default,
    })
}

/// The yield that ends a region, before or after `ScfYieldPattern` rewrote
/// it: the applicator lowers a region's terminator before its parent.
fn default_yield(ctx: &IrContext, region: RegionRef) -> Option<OpRef> {
    let entry = *ctx.region(region).blocks.first()?;
    let last = *ctx.block(entry).ops.last()?;
    (scf::Yield::matches(ctx, last) || wasm_dialect::Yield::matches(ctx, last)).then_some(last)
}

fn switch_arms(ctx: &IrContext, op: OpRef) -> Result<ScfSwitchArms, SwitchLoweringReason> {
    let shape = switch_shape(ctx, op).ok_or(SwitchLoweringReason::MalformedShape)?;
    let result_count = ctx.op_results(op).len();
    if result_count > 1 {
        return Err(SwitchLoweringReason::TooManyResults(result_count));
    }
    // Without cases the default's ops replace the switch, so its yield must
    // provide the result. A Never result is instead dropped, and the Never
    // control check requires the default to be terminal.
    let yields_result = ctx
        .op_results(op)
        .first()
        .is_some_and(|&result| !core::Never::matches(ctx, ctx.value_ty(result)));
    if yields_result && shape.cases.is_empty() && default_yield(ctx, shape.default).is_none() {
        return Err(SwitchLoweringReason::ResultDefaultWithoutYield);
    }
    let Some(width) = switch_width(ctx, shape.discriminant) else {
        let ty = ctx.get_type(ctx.value_ty(shape.discriminant));
        return Err(SwitchLoweringReason::UnsupportedDiscriminantType(format!(
            "{}",
            ty.name.display(ctx)
        )));
    };
    let mut cases = Vec::with_capacity(shape.cases.len());
    for (value, body) in shape.cases {
        let value = match width {
            SwitchWidth::I32 => i32::try_from(value).map(i64::from).ok(),
            SwitchWidth::I64 => Some(value),
        }
        .ok_or(SwitchLoweringReason::CaseValueOutsideRange(width))?;
        cases.push((value, body));
    }
    Ok(ScfSwitchArms {
        discriminant: shape.discriminant,
        width,
        cases,
        default: shape.default,
    })
}

/// Find the first switch rejected by the same acceptance contract as lowering.
fn find_nonlowerable_switch(ctx: &IrContext, op: OpRef) -> Option<(OpRef, SwitchLoweringReason)> {
    if scf::IndexSwitch::matches(ctx, op)
        && let Err(reason) = switch_arms(ctx, op)
    {
        return Some((op, reason));
    }
    for region in ctx.op_regions(op) {
        for block in ctx.region(region).blocks.iter().copied() {
            for nested in ctx.block(block).ops.iter().copied() {
                if let Some(rejected) = find_nonlowerable_switch(ctx, nested) {
                    return Some(rejected);
                }
            }
        }
    }
    None
}

fn region_with_ops(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    ops: Vec<OpRef>,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    for op in ops {
        ctx.push_op(block, op);
    }
    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn take_region_ops(ctx: &mut IrContext, region: RegionRef) -> OpList {
    let [block] = ctx.region(region).blocks.as_slice() else {
        unreachable!("switch regions are preflighted as single-block");
    };
    let ops = ctx.block(*block).ops.clone();
    for &op in &ops {
        ctx.detach_op(op);
    }
    ops
}

/// Pattern for an `scf.index_switch` -> a `wasm.br_table` dispatch when the
/// switch is dense, otherwise nested `wasm.if` comparisons.
struct ScfSwitchPattern<'a>(&'a ScfLoweringPlan);

impl OpRewritePattern for ScfSwitchPattern<'_> {
    type Op = scf::IndexSwitch;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        switch: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = switch.op_ref();
        let arms = switch_arms(ctx, op).ok()?;
        let loc = ctx.op(op).location;
        // A switch with a result yields it from each nested `wasm.if`, so
        // the created ifs declare its target type.
        let drop_result = self.0.drop_never_results.contains(&op);
        let switch_results = rewriter.result_types(ctx, op);
        let has_result = !switch_results.is_empty() && !drop_result;
        let result_types = if has_result {
            switch_results
        } else if drop_result || self.0.resultless_switches.contains(&op) {
            vec![]
        } else {
            vec![core::Nil::params().intern(ctx).as_type_ref()]
        };

        for &(_, body) in &arms.cases {
            ctx.detach_region(body);
        }
        ctx.detach_region(arms.default);

        if arms.cases.is_empty() {
            let terminator = default_yield(ctx, arms.default).filter(|_| has_result);
            let values = terminator.map_or_else(Vec::new, |last| ctx.op_operands(last).to_vec());
            let mut children = take_region_ops(ctx, arms.default);
            if let Some(terminator) = terminator {
                children.pop();
                ctx.remove_op(terminator);
            }
            // A dropped Never result leaves the default's terminal op in the
            // switch's place, so it stays last in the block.
            let last = if drop_result { children.pop() } else { None };
            for child in children {
                rewriter.insert_op(child);
            }
            if let Some(last) = last {
                assert!(rewriter.replace_op_dropping_unused_results(ctx, op, last));
            } else {
                rewriter.erase_op(values);
            }
            return Some(());
        }

        let mut discriminant = arms.discriminant;
        if core::Index::matches(ctx, ctx.value_ty(discriminant)) {
            let i32_ty = core::I32::type_ref(ctx);
            let cast = core::UnrealizedConversionCast::operands(discriminant)
                .results(i32_ty)
                .build(ctx, loc);
            rewriter.insert_op(cast.op_ref());
            discriminant = cast.result(ctx);
        }
        let (outer_ops, outer) = if let Some(table) = JumpTable::of(arms.width, &arms.cases) {
            let exit = lower_jump_table(ctx, loc, discriminant, table, &arms, &result_types);
            (vec![exit], exit)
        } else {
            lower_comparison_chain(ctx, loc, discriminant, arms, &result_types, has_result)
        };
        for inserted in outer_ops {
            rewriter.insert_op(inserted);
        }
        if drop_result {
            let unreachable = wasm_dialect::Unreachable::operands().build(ctx, loc);
            assert!(rewriter.replace_op_dropping_unused_results(ctx, op, unreachable.op_ref()));
            return Some(());
        }
        if self.0.terminal_controls.contains(&op) {
            let unreachable = wasm_dialect::Unreachable::operands().build(ctx, loc);
            rewriter.insert_op(unreachable.op_ref());
        }
        let values = if has_result {
            vec![ctx.op_result(outer, 0)]
        } else {
            vec![]
        };
        rewriter.erase_op(values);
        Some(())
    }
}

/// The `br_table` dispatch of a dense switch: its smallest case value, and
/// for each value from there the position of the first case with that value,
/// or `default`, the number of cases, for a value no case has.
struct JumpTable {
    min: i32,
    targets: Vec<u32>,
    default: u32,
}

impl JumpTable {
    /// Binaryen's `RemoveUnusedBrs` thresholds for turning a comparison
    /// chain into a `br_table`.
    const MIN_CASES: usize = 3;
    const MAX_RANGE: i64 = 1024;
    const MAX_RANGE_PER_CASE: i64 = 3;

    /// Choose a table for an `i32` switch with enough cases that cover a
    /// small, dense range. Any other switch keeps the comparison chain.
    fn of(width: SwitchWidth, cases: &[(i64, RegionRef)]) -> Option<Self> {
        if width != SwitchWidth::I32 || cases.len() < Self::MIN_CASES {
            return None;
        }
        let min = cases.iter().map(|&(value, _)| value).min()?;
        let max = cases.iter().map(|&(value, _)| value).max()?;
        let range = max - min + 1;
        if range > Self::MAX_RANGE || range > Self::MAX_RANGE_PER_CASE * cases.len() as i64 {
            return None;
        }
        let default = u32::try_from(cases.len()).ok()?;
        let mut targets = vec![default; usize::try_from(range).ok()?];
        // An earlier case wins a repeated value, as in the comparison chain.
        for (position, &(value, _)) in cases.iter().enumerate().rev() {
            targets[usize::try_from(value - min).ok()?] = u32::try_from(position).ok()?;
        }
        Some(Self {
            min: i32::try_from(min).ok()?,
            targets,
            default,
        })
    }
}

/// Lower a dense switch to one `br_table` inside a `wasm.block` per case:
///
/// ```text
/// %r = wasm.block -> (T) {          // exit
///   wasm.block {                     // default label
///     wasm.block {                   // label of the last case
///       ...
///         wasm.block {               // label of the first case
///           %i = wasm.i32_sub %d, %min
///           wasm.br_table %i, default n [..]
///         }
///         <first case>
///         wasm.yield %v
///         wasm.br {target = n}       // to the exit
///       ...
///     }
///     <last case>
///     wasm.yield %v
///     wasm.br {target = 1}
///   }
///   <default>
///   wasm.yield %v
/// }
/// ```
///
/// The index wraps below the smallest case value, so `br_table` sends every
/// value outside the table to the default label. Each case follows the end
/// of its own label and branches to the exit with its result; the default
/// falls through to the end of the exit, which yields its result.
fn lower_jump_table(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    discriminant: ValueRef,
    table: JumpTable,
    arms: &ScfSwitchArms,
    result_types: &[TypeRef],
) -> OpRef {
    let i32_ty = core::I32::type_ref(ctx);
    let mut ops = Vec::new();
    let mut index = discriminant;
    if table.min != 0 {
        let min = wasm_dialect::I32Const::operands()
            .value(table.min)
            .results(i32_ty)
            .build(ctx, loc);
        let sub = wasm_dialect::I32Sub::operands(discriminant, min.result(ctx))
            .results(i32_ty)
            .build(ctx, loc);
        index = sub.result(ctx);
        ops.extend([min.op_ref(), sub.op_ref()]);
    }
    let case_count = table.default;
    let dispatch = wasm_dialect::BrTable::operands(index)
        .targets(table.targets)
        .default(case_count)
        .build(ctx, loc);
    ops.push(dispatch.op_ref());

    // A case's result reaches the exit as the branch's operand, which a
    // result without a Wasm value does not have.
    let carries_value = result_types
        .first()
        .is_some_and(|&ty| !core::Nil::matches(ctx, ty));
    for (position, &(_, body)) in arms.cases.iter().enumerate() {
        let label = wasm_dialect::Block::operands()
            .results(Vec::<TypeRef>::new())
            .regions(region_with_ops(ctx, loc, ops))
            .build(ctx, loc);
        ops = vec![label.op_ref()];
        ops.extend(take_region_ops(ctx, body));
        if !carries_value
            && let Some(&last) = ops.last()
            && wasm_dialect::Yield::matches(ctx, last)
        {
            ops.pop();
            ctx.remove_op(last);
        }
        // A case that leaves through its loop's `scf.continue` or
        // `scf.break` never reaches the exit.
        if ops
            .last()
            .is_some_and(|&last| DynRegionBranchTerminator::is_implemented(ctx, last))
        {
            continue;
        }
        let to_exit = wasm_dialect::Br::operands()
            .target(case_count - position as u32)
            .build(ctx, loc);
        ops.push(to_exit.op_ref());
    }
    let default_label = wasm_dialect::Block::operands()
        .results(Vec::<TypeRef>::new())
        .regions(region_with_ops(ctx, loc, ops))
        .build(ctx, loc);
    let mut exit_ops = vec![default_label.op_ref()];
    exit_ops.extend(take_region_ops(ctx, arms.default));
    wasm_dialect::Block::operands()
        .results(result_types.to_vec())
        .regions(region_with_ops(ctx, loc, exit_ops))
        .build(ctx, loc)
        .op_ref()
}

/// Lower a switch to a `wasm.if` per case, each testing one case value and
/// holding the rest of the chain in its else region. Returns the operations
/// that replace the switch and the outermost `wasm.if`.
fn lower_comparison_chain(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    discriminant: ValueRef,
    arms: ScfSwitchArms,
    result_types: &[TypeRef],
    has_result: bool,
) -> (Vec<OpRef>, OpRef) {
    let discriminant_ty = ctx.value_ty(discriminant);
    let condition_ty = core::I32::type_ref(ctx);
    let case_count = arms.cases.len();
    let mut next = arms.default;
    let mut outer_ops = None;
    for (index, (value, body)) in arms.cases.into_iter().rev().enumerate() {
        let (case, matches) = match arms.width {
            SwitchWidth::I32 => {
                let value = i32::try_from(value)
                    .expect("i32 switch case values are validated before lowering");
                let case = wasm_dialect::I32Const::operands()
                    .value(value)
                    .results(discriminant_ty)
                    .build(ctx, loc);
                let case_value = case.result(ctx);
                let matches = wasm_dialect::I32Eq::operands(discriminant, case_value)
                    .results(condition_ty)
                    .build(ctx, loc);
                (case.op_ref(), matches.op_ref())
            }
            SwitchWidth::I64 => {
                let case = wasm_dialect::I64Const::operands()
                    .value(value)
                    .results(discriminant_ty)
                    .build(ctx, loc);
                let case_value = case.result(ctx);
                let matches = wasm_dialect::I64Eq::operands(discriminant, case_value)
                    .results(condition_ty)
                    .build(ctx, loc);
                (case.op_ref(), matches.op_ref())
            }
        };
        let condition = ctx.op_result(matches, 0);
        let branch = wasm_dialect::If::operands(condition)
            .results(result_types.to_vec())
            .regions(body, next)
            .build(ctx, loc);
        let mut ops = vec![case, matches, branch.op_ref()];
        if index + 1 == case_count {
            outer_ops = Some((ops, branch.op_ref()));
        } else {
            if has_result {
                let value = ctx.op_result(branch.op_ref(), 0);
                ops.push(
                    wasm_dialect::Yield::operands(value)
                        .build(ctx, loc)
                        .op_ref(),
                );
            }
            next = region_with_ops(ctx, loc, ops);
        }
    }

    outer_ops.expect("nonempty switch cases build an outer dispatch")
}

/// Pattern for `scf.if` -> `wasm.if`
struct ScfIfPattern<'a>(&'a ScfLoweringPlan);

impl OpRewritePattern for ScfIfPattern<'_> {
    type Op = scf::If;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        scf_if_op: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = scf_if_op.op_ref();
        let loc = ctx.op(op).location;

        // The created `wasm.if` must declare target types, so read them through
        // the converter instead of copying the shared IR spelling.
        let mut result_types = rewriter.result_types(ctx, op);
        let drop_result = self.0.drop_never_results.contains(&op);
        if drop_result {
            result_types.clear();
        }

        // Get the condition operand
        let cond = scf_if_op.cond(ctx);

        // Get then/else regions and detach them from the original op
        let then_region = scf_if_op.then_region(ctx);
        let else_region = scf_if_op.else_region(ctx);
        ctx.detach_region(then_region);
        ctx.detach_region(else_region);

        let new_op = wasm_dialect::If::operands(cond)
            .results(result_types)
            .regions(then_region, else_region)
            .build(ctx, loc);
        replace_control(
            ctx,
            op,
            new_op.op_ref(),
            self.0.terminal_controls.contains(&op),
            drop_result,
            rewriter,
        );
        Some(())
    }
}

/// Pattern for `scf.loop` -> `wasm.block(wasm.loop(...))`
///
/// The loop is wrapped in a block to provide a break target. The loop
/// lowers its own `scf.continue` and `scf.break` to `wasm.br`, as
/// structured-to-CFG lowering does: the applicator lowers nested controls
/// first, so the labels between each exit and the loop are known here.
struct ScfLoopPattern<'a>(&'a ScfLoweringPlan);

impl OpRewritePattern for ScfLoopPattern<'_> {
    type Op = scf::Loop;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        loop_op: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = loop_op.op_ref();
        let loc = ctx.op(op).location;

        // The created `wasm.block`/`wasm.loop` must declare target types.
        let mut result_types = rewriter.result_types(ctx, op);
        let drop_result = self.0.drop_never_results.contains(&op);
        if drop_result {
            result_types.clear();
        }

        // Get init operands
        let init: Vec<_> = loop_op.init(ctx).to_vec();

        let body = loop_op.body(ctx);
        let exits = rewriter.require(
            collect_loop_exits(ctx, body),
            "a loop exit is nested in an operation that is not a Wasm label",
        )?;
        rewriter.ensure(
            exits.iter().all(|exit| ctx.op_operands(exit.op).len() <= 1),
            "multiple loop-carried values are not supported",
        )?;

        // Detach the body region from the original loop op
        ctx.detach_region(body);
        lower_loop_exits(ctx, &exits);

        // The created `wasm.loop` owns the detached body, so its block
        // arguments become Wasm-level parameters. Declare them with target
        // types instead of leaving the SCF spelling on the boundary.
        let body_blocks = ctx.region(body).blocks.clone();
        for block in body_blocks {
            let block_args = ctx.block_args(block).to_vec();
            for (index, arg) in block_args.into_iter().enumerate() {
                let raw_ty = ctx.value_ty(arg);
                let converted = rewriter.get_value_type(ctx, arg);
                if converted != raw_ty {
                    ctx.set_block_arg_type(block, index as u32, converted);
                }
            }
        }

        // Create wasm.loop with init operands and the body region
        let wasm_loop = wasm_dialect::Loop::operands(init)
            .results(result_types.clone())
            .regions(body)
            .build(ctx, loc);

        // Create a block containing just the wasm.loop, to serve as the break target
        let block_body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block_body_block, wasm_loop.op_ref());

        let block_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_body_block],
            parent_op: None,
        });

        let wasm_block = wasm_dialect::Block::operands()
            .results(result_types)
            .regions(block_body)
            .build(ctx, loc);
        replace_control(
            ctx,
            op,
            wasm_block.op_ref(),
            self.0.terminal_controls.contains(&op),
            drop_result,
            rewriter,
        );
        Some(())
    }
}

/// Pattern for `scf.yield` -> `wasm.yield`
///
/// In wasm, block results are implicit - the last value on the stack is the result.
/// We convert scf.yield to wasm.yield to track which value should be the region's
/// result. This is especially important for handler dispatch where the result value
/// may be defined outside the region (e.g., the scrutinee in `{ result } -> result`).
///
/// At emit time, wasm.yield is handled specially: its operand is emitted as a
/// local.get, and the wasm.yield itself produces no Wasm instruction.
struct ScfYieldPattern;

impl OpRewritePattern for ScfYieldPattern {
    type Op = scf::Yield;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        yield_op: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = yield_op.op_ref();
        // Get yield values (variadic operands)
        let operands = ctx.op_operands(op).to_vec();

        if operands.is_empty() {
            // No value to yield - just erase
            rewriter.erase_op(vec![]);
            return Some(());
        }

        if operands.len() > 1 {
            // Multi-value yields are not yet supported; leave unlowered.
            return None;
        }

        let value = operands[0];
        let loc = ctx.op(op).location;
        let new_op = wasm_dialect::Yield::operands(value).build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// A loop's own `scf.continue` or `scf.break`, with the number of Wasm
/// labels between it and the loop body.
struct LoopExit {
    op: OpRef,
    labels: u32,
}

/// Collect the exits of the loop whose body is `body`, skipping the bodies
/// of nested loops, which own their exits. The applicator lowers nested
/// controls first, so every label between an exit and the body is a Wasm
/// `block`, `loop`, or `if`; `None` reports any other enclosing operation.
fn collect_loop_exits(ctx: &IrContext, body: RegionRef) -> Option<Vec<LoopExit>> {
    /// `labels` is `None` below an operation that is not a Wasm label.
    fn walk(
        ctx: &IrContext,
        region: RegionRef,
        labels: Option<u32>,
        exits: &mut Vec<LoopExit>,
    ) -> Option<()> {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                if scf::Continue::matches(ctx, op) || scf::Break::matches(ctx, op) {
                    exits.push(LoopExit {
                        op,
                        labels: labels?,
                    });
                } else if !(wasm_dialect::Loop::matches(ctx, op) || scf::Loop::matches(ctx, op)) {
                    let nested_labels = if wasm_dialect::Block::matches(ctx, op)
                        || wasm_dialect::If::matches(ctx, op)
                    {
                        labels.map(|labels| labels + 1)
                    } else {
                        None
                    };
                    for nested in ctx.op_regions(op) {
                        walk(ctx, nested, nested_labels, exits)?;
                    }
                }
            }
        }
        Some(())
    }
    let mut exits = Vec::new();
    walk(ctx, body, Some(0), &mut exits)?;
    Some(exits)
}

/// Replace each exit with `wasm.br`. Inside the body, `wasm.loop` is one
/// label past the exit's enclosing labels and the `wasm.block` around it,
/// the break target, is one more. A carried value is marked by `wasm.yield`
/// before the branch: the emitter stores a continue's value into the loop
/// argument and leaves a break's value on the stack for the block.
fn lower_loop_exits(ctx: &mut IrContext, exits: &[LoopExit]) {
    for exit in exits {
        let loc = ctx.op(exit.op).location;
        let block = ctx
            .op(exit.op)
            .parent_block
            .expect("collected loop exits are in a block");
        let target = if scf::Break::matches(ctx, exit.op) {
            exit.labels + 1
        } else {
            exit.labels
        };
        if let Some(&value) = ctx.op_operands(exit.op).first() {
            let yield_op = wasm_dialect::Yield::operands(value).build(ctx, loc);
            ctx.insert_op_before(block, exit.op, yield_op.op_ref());
        }
        let br = wasm_dialect::Br::operands().target(target).build(ctx, loc);
        ctx.insert_op_before(block, exit.op, br.op_ref());
        trunk_ir::rewrite::erase_op(ctx, exit.op);
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Make proven non-returning structured control visible after Wasm's `end`.
fn replace_control(
    ctx: &mut IrContext,
    op: OpRef,
    lowered: OpRef,
    terminal: bool,
    drop_result: bool,
    rewriter: &mut PatternRewriter<'_>,
) {
    let replacement = if terminal {
        rewriter.insert_op(lowered);
        wasm_dialect::Unreachable::operands()
            .build(ctx, ctx.op(op).location)
            .op_ref()
    } else {
        lowered
    };
    if drop_result {
        assert!(rewriter.replace_op_dropping_unused_results(ctx, op, replacement));
    } else {
        rewriter.replace_op(replacement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    fn lower_text(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .expect("test module should lower to wasm");
        let use_chains = trunk_ir::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let verifiers = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
        assert!(verifiers.is_ok(), "{verifiers}");
        let output = module.op().display(&ctx).to_string();
        assert!(
            !output.contains("scf."),
            "residual scf operation:\n{output}"
        );
        output
    }

    fn assert_switch_rejected_unchanged(case: &str, input: &str, reason: &str) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = module.op().display(&ctx).to_string();

        let error = lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .expect_err("nonlowerable switch should reject the entire conversion");

        assert_eq!(error.boundary(), SCF_TO_WASM_BOUNDARY, "{case}");
        assert_eq!(error.operations().len(), 1, "{case}: {error}");
        assert_eq!(
            error.operations()[0].legality,
            LegalityCheck::Illegal,
            "{case}"
        );
        assert_eq!(
            error.operations()[0].reason.as_deref(),
            Some(reason),
            "{case}"
        );
        assert!(
            error.to_string().contains("scf.index_switch"),
            "{case}: {error}"
        );
        assert!(error.to_string().contains(reason), "{case}: {error}");
        assert_eq!(module.op().display(&ctx).to_string(), before, "{case}");
    }

    fn assert_no_scf_switch(output: &str) {
        for name in ["scf.index_switch", "scf.yield"] {
            assert!(!output.contains(name), "residual {name}:\n{output}");
        }
    }

    #[test]
    fn lowers_resultless_switch_with_explicit_default() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {test.marker = 2} {
    scf.index_switch %choice : (core.i32) -> ()
    case 0 {
      func.tail_call_indirect %callee(%unit) attributes {test.marker = 2} : (core.nil) -> core.never
    }
    default {
      func.unreachable
    }
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(output.contains("wasm.i32_eq"), "{output}");
        assert!(output.contains("wasm.if"), "{output}");
        assert!(output.contains("func.tail_call_indirect"), "{output}");
        assert!(output.contains("func.unreachable"), "{output}");
    }

    #[test]
    fn lowers_resultless_switch_without_default_to_fallthrough() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.index_switch %choice : (core.i32) -> ()
    case 0 {
      scf.yield
    }
    default {
      scf.yield
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(output.contains("wasm.i32_eq"), "{output}");
        assert!(output.contains("wasm.if"), "{output}");
        assert!(output.contains("func.return"), "{output}");
    }

    #[test]
    fn lowers_multiple_i32_switch_cases_in_source_order() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.index_switch %choice : (core.i32) -> ()
    case 1 {
      scf.yield
    }
    case 2 {
      scf.yield
    }
    default {
      scf.yield
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert_eq!(output.matches("wasm.i32_eq").count(), 2, "{output}");
        assert!(
            output.find("wasm.i32_const 1") < output.find("wasm.i32_const 2"),
            "case dispatch order changed: {output}"
        );
    }

    #[test]
    fn index_switch_compares_its_discriminant_as_i32() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.index) -> core.nil {
    scf.index_switch %choice : (core.index) -> ()
    case 0 {
      scf.yield
    }
    case 2 {
      scf.yield
    }
    default {
      scf.yield
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(
            output.contains("core.unrealized_conversion_cast %0 : core.index to core.i32"),
            "{output}"
        );
        assert_eq!(output.matches("wasm.i32_eq").count(), 2, "{output}");
    }

    #[test]
    fn lowers_i64_switch_cases_with_i64_comparisons() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i64) -> core.nil {
    scf.index_switch %choice : (core.i64) -> ()
    case -1 {
      scf.yield
    }
    case 4294967296 {
      scf.yield
    }
    default {
      scf.yield
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(!output.contains("wasm.i32_eq"), "{output}");
        assert_eq!(output.matches("wasm.i64_eq").count(), 2, "{output}");
        assert!(
            output.find("wasm.i64_const -1") < output.find("wasm.i64_const 4294967296"),
            "case dispatch order changed: {output}"
        );
        for line in output.lines().filter(|line| line.contains("wasm.i64_eq")) {
            assert!(
                line.ends_with(": core.i32"),
                "i64 comparison yields i32: {line}"
            );
        }
    }

    #[test]
    fn lowered_i64_switch_emits_a_valid_module() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @choose(%choice: core.i64) -> (core.i32) {
      scf.index_switch %choice : (core.i64) -> ()
      case 4294967296 {
        %one = wasm.i32_const 1 : core.i32
        wasm.return %one
      }
      default {
        scf.yield
      }
      %zero = wasm.i32_const 0 : core.i32
      wasm.return %zero
  }
}"#,
        );
        lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .expect("i64 switch should lower to wasm");
        let bytes = crate::emit_module_to_wasm(&mut ctx, module)
            .expect("lowered i64 switch must emit")
            .bytes;
        wasmparser::Validator::new()
            .validate_all(&bytes)
            .expect("lowered i64 switch must validate");
    }

    #[test]
    fn lowers_nested_proper_tail_switch_arms() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32, %cond: core.i1, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {test.marker = 2} {
    scf.index_switch %choice : (core.i32) -> ()
    case 0 {
      %never = scf.if %cond -> (core.never) {
        func.unreachable
      } else {
        func.tail_call_indirect %callee(%unit) attributes {test.marker = 2} : (core.nil) -> core.never
      }
    }
    default {
      func.tail_call_indirect %callee(%unit) attributes {test.marker = 2} : (core.nil) -> core.never
    }
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(output.contains("wasm.i32_eq"), "{output}");
        assert_eq!(output.matches("wasm.if").count(), 2, "{output}");
        for line in output.lines().filter(|line| line.contains("wasm.if")) {
            assert!(
                !line.contains(" = ") && !line.contains(" -> "),
                "both the switch dispatch and nested tail-transfer if must be resultless: {line}"
            );
        }
        assert_eq!(
            output.matches("func.tail_call_indirect").count(),
            2,
            "{output}"
        );
    }

    #[test]
    fn nonlowerable_switches_are_rejected_unchanged() {
        // (case, input, rejection reason)
        for (case, input, reason) in [
            (
                "leaves malformed switch arm entry args unchanged",
                r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.index_switch %choice : (core.i32) -> ()
    case 0 {
      ^case_entry(%unexpected: core.i32):
        scf.yield
    }
    default { scf.yield }
    func.return
  }
}"#,
                "malformed switch shape",
            ),
            (
                "rejects shape valid narrow integer switch without mutating",
                r#"core.module @test {
  func.func @main(%cond: core.i1, %choice: core.i16) -> core.nil {
    scf.index_switch %choice : (core.i16) -> ()
    case 0 {
      scf.if %cond -> (core.nil) {
        scf.yield
      } else {
        scf.yield
      }
      scf.yield
    }
    default { scf.yield }
    func.return
  }
}"#,
                "unsupported discriminant type `core.i16`; expected `core.i32`, `core.i64`, or `core.index`",
            ),
            (
                "rejects shape valid out of range case without mutating",
                r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.index_switch %choice : (core.i32) -> ()
    case 2147483648 { scf.yield }
    default { scf.yield }
    func.return
  }
}"#,
                "case integer value is outside the i32 range",
            ),
            (
                "rejects a switch with two results without mutating",
                r#"core.module @test {
  func.func @main(%choice: core.i32, %a: core.i32) -> core.nil {
    %r, %s = scf.index_switch %choice : (core.i32) -> (core.i32, core.i32)
    case 0 { scf.yield %a, %a }
    default { scf.yield %a, %a }
    func.return
  }
}"#,
                "switch has 2 results; at most one is supported",
            ),
            (
                "rejects a caseless result switch whose default does not yield",
                r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.i32 {
    %r = scf.index_switch %choice : (core.i32) -> core.i32
    default { func.unreachable }
  }
}"#,
                "a switch with a result and no cases needs a default ending in scf.yield",
            ),
        ] {
            assert_switch_rejected_unchanged(case, input, reason);
        }
    }

    #[test]
    fn switch_results_flow_out_of_the_nested_ifs() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32, %a: core.i64, %b: core.i64, %c: core.i64) -> core.i64 {
    %r = scf.index_switch %choice : (core.i32) -> core.i64
    case 0 {
      scf.yield %a
    }
    case 1 {
      scf.yield %b
    }
    default {
      scf.yield %c
    }
    func.return %r
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert_eq!(output.matches("wasm.if").count(), 2, "{output}");
        assert_eq!(output.matches("wasm.yield").count(), 4, "{output}");
    }

    #[test]
    fn default_only_switch_yields_its_result() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32, %a: core.i64) -> core.i64 {
    %r = scf.index_switch %choice : (core.i32) -> core.i64
    default {
      %b = wasm.i64_add %a, %a : core.i64
      scf.yield %b
    }
    func.return %r
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(!output.contains("wasm.yield"), "{output}");
        assert!(output.contains("func.return %2"), "{output}");
    }

    #[test]
    fn lowers_default_only_switch() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.index_switch %choice : (core.i32) -> ()
    default { scf.yield }
    func.return
  }
}"#,
        );

        assert_no_scf_switch(&output);
        assert!(!output.contains("wasm.i32_eq"), "{output}");
        assert!(output.contains("func.return"), "{output}");
    }

    #[test]
    fn converts_if_result_through_the_type_converter() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @select(%cond: core.i1, %value: wasm_gc.array<core.i32>) -> wasm_gc.array<core.i32> {
    %result = scf.if %cond -> (wasm_gc.array<core.i32>) {
      scf.yield %value
    } else {
      scf.yield %value
    }
    func.return %result
  }
}"#,
        );

        let type_converter = array_to_arrayref_converter(&mut ctx);
        lower(&mut ctx, module, type_converter, &mut Default::default())
            .expect("test module should lower to wasm");

        let use_chains = trunk_ir::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("scf."), "{output}");
        assert!(
            output.contains("wasm.if") && output.contains("-> (wasm.arrayref)"),
            "the created wasm.if must declare the converted result: {output}"
        );
    }

    #[test]
    fn converts_loop_body_block_argument_through_the_type_converter() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @carry(%init: wasm_gc.array<core.i32>) -> core.nil {
    scf.loop(%init) -> (core.nil) {
      ^header(%iter: wasm_gc.array<core.i32>):
        scf.continue %iter
    }
    func.return
  }
}"#,
        );

        let type_converter = array_to_arrayref_converter(&mut ctx);
        lower(&mut ctx, module, type_converter, &mut Default::default())
            .expect("test module should lower to wasm");

        let use_chains = trunk_ir::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("scf."), "{output}");
        assert!(
            output.contains("wasm.loop") && output.contains(": wasm.arrayref)"),
            "the created wasm.loop body argument must declare the converted type: {output}"
        );
        assert!(
            output.contains("wasm_gc.array<core.i32>"),
            "operand types must keep their producer spelling: {output}"
        );
    }

    /// The `wasm.br` targets in printed order.
    fn br_targets(output: &str) -> Vec<&str> {
        output
            .lines()
            .filter_map(|line| line.trim().strip_prefix("wasm.br {target = "))
            .map(|rest| rest.trim_end_matches('}'))
            .collect()
    }

    #[test]
    fn loop_exits_branch_past_every_enclosing_label() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%init: core.i32, %a: core.i1, %b: core.i1) -> core.i32 {
    %r = scf.loop(%init) -> (core.i32) {
      ^body(%acc: core.i32):
        scf.if %a {
          scf.if %b {
            scf.break %acc
          } else {
            scf.continue %acc
          }
          scf.yield
        } else {
          scf.yield
        }
        scf.continue %acc
    }
    func.return %r
  }
}"#,
        );

        // Inner if, outer if, loop, then the block around the loop.
        assert_eq!(br_targets(&output), ["3", "2", "0"], "{output}");
    }

    #[test]
    fn loop_body_continue_targets_the_loop() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%init: core.i32) -> core.nil {
    scf.loop(%init) {
      ^body(%acc: core.i32):
        scf.continue %acc
    }
    func.return
  }
}"#,
        );

        assert_eq!(br_targets(&output), ["0"], "{output}");
    }

    #[test]
    fn loop_exits_in_switch_cases_count_the_case_dispatch() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%init: core.i32, %choice: core.i32) -> core.i32 {
    %r = scf.loop(%init) -> (core.i32) {
      ^body(%acc: core.i32):
        scf.index_switch %choice : (core.i32) -> ()
        case 0 {
          scf.continue %acc
        }
        case 1 {
          scf.break %acc
        }
        default {
          scf.continue %acc
        }
        scf.continue %acc
    }
    func.return %r
  }
}"#,
        );

        // Case 0 is in the outer comparison; case 1 and the default are in
        // the nested one.
        assert_eq!(br_targets(&output), ["1", "3", "2", "0"], "{output}");
    }

    #[test]
    fn lowered_loop_exits_in_nested_controls_emit_a_valid_module() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @count(%init: core.i32, %choice: core.i32) -> (core.i32) {
      %r = scf.loop(%init) -> (core.i32) {
        ^body(%acc: core.i32):
          %zero = wasm.i32_const 0 : core.i32
          %done = "wasm.i32_eq"(%acc, %zero) : core.i32
          scf.if %done {
            scf.break %acc
          } else {
            scf.yield
          }
          scf.index_switch %choice : (core.i32) -> ()
          case 0 {
            %one = wasm.i32_const 1 : core.i32
            %next = "wasm.i32_sub"(%acc, %one) : core.i32
            scf.continue %next
          }
          case 1 {
            scf.break %acc
          }
          default {
            scf.yield
          }
          scf.continue %acc
      }
      wasm.return %r
  }
}"#,
        );
        lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .expect("loop exits should lower to wasm");
        let bytes = crate::emit_module_to_wasm(&mut ctx, module)
            .expect("lowered loop exits must emit")
            .bytes;
        wasmparser::Validator::new()
            .validate_all(&bytes)
            .expect("lowered loop exits must validate");
    }

    /// Lower `ir`, then emit it and validate the module with `wasmparser`.
    fn lower_and_validate(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .expect("test module should lower to wasm");
        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("scf."), "residual scf:\n{output}");
        let bytes = crate::emit_module_to_wasm(&mut ctx, module)
            .expect("lowered module must emit")
            .bytes;
        wasmparser::Validator::new()
            .validate_all(&bytes)
            .unwrap_or_else(|error| panic!("{error}:\n{output}"));
        output
    }

    #[test]
    fn dense_switch_dispatches_through_one_br_table() {
        let output = lower_and_validate(
            r#"core.module @test {
  wasm.func @pick(%choice: core.i32) -> (core.i32) {
      %r = scf.index_switch %choice : (core.i32) -> (core.i32)
      case 1 {
        %a = wasm.i32_const 10 : core.i32
        scf.yield %a
      }
      case 2 {
        %b = wasm.i32_const 20 : core.i32
        scf.yield %b
      }
      case 4 {
        %c = wasm.i32_const 40 : core.i32
        scf.yield %c
      }
      default {
        %d = wasm.i32_const 0 : core.i32
        scf.yield %d
      }
      wasm.return %r
  }
}"#,
        );

        assert!(!output.contains("wasm.if"), "{output}");
        assert!(output.contains("wasm.i32_sub"), "{output}");
        // Values 1, 2, 3, 4 from the smallest case; 3 has no case.
        assert!(output.contains(", default 3 [0, 1, 3, 2]"), "{output}");
        // Each case branches past the later case labels and the default
        // label to the exit; the default falls through to it.
        assert_eq!(br_targets(&output), ["3", "2", "1"], "{output}");
    }

    #[test]
    fn dense_switch_from_zero_indexes_without_subtracting() {
        let output = lower_and_validate(
            r#"core.module @test {
  wasm.func @run(%choice: core.i32) {
      scf.index_switch %choice : (core.i32) -> ()
      case 0 {
        scf.yield
      }
      case 1 {
        scf.yield
      }
      case 2 {
        scf.yield
      }
      default {
        scf.yield
      }
      wasm.return
  }
}"#,
        );

        assert!(!output.contains("wasm.i32_sub"), "{output}");
        assert!(output.contains(", default 3 [0, 1, 2]"), "{output}");
    }

    #[test]
    fn sparse_or_small_switches_keep_the_comparison_chain() {
        for cases in [&[0, 10, 20][..], &[0, 1][..]] {
            let arms = cases
                .iter()
                .map(|value| format!("case {value} {{\n scf.yield\n }}\n"))
                .collect::<String>();
            let output = lower_text(&format!(
                r#"core.module @test {{
  func.func @main(%choice: core.i32) -> core.nil {{
    scf.index_switch %choice : (core.i32) -> ()
    {arms}default {{
      scf.yield
    }}
    func.return
  }}
}}"#
            ));

            assert!(!output.contains("wasm.br_table"), "{cases:?}: {output}");
            assert!(output.contains("wasm.if"), "{cases:?}: {output}");
        }
    }

    #[test]
    fn loop_exits_in_jump_table_cases_count_the_case_labels() {
        let output = lower_and_validate(
            r#"core.module @test {
  wasm.func @count(%init: core.i32, %choice: core.i32) -> (core.i32) {
      %r = scf.loop(%init) -> (core.i32) {
        ^body(%acc: core.i32):
          scf.index_switch %choice : (core.i32) -> ()
          case 0 {
            %one = wasm.i32_const 1 : core.i32
            %next = "wasm.i32_sub"(%acc, %one) : core.i32
            scf.continue %next
          }
          case 1 {
            scf.break %acc
          }
          case 2 {
            scf.yield
          }
          default {
            scf.break %acc
          }
          scf.continue %acc
      }
      wasm.return %r
  }
}"#,
        );

        // Case 0 continues past the later case labels, the default label,
        // and the exit; case 1 breaks one label further; case 2 leaves the
        // switch; the default breaks from the exit.
        assert_eq!(br_targets(&output), ["4", "4", "1", "2", "0"], "{output}");
    }

    fn control_fixture(body: &str) -> String {
        format!(
            "core.module @test {{ func.func @main(%cond: core.i1, %choice: core.i32, %unit: core.nil) -> core.nil {{ {body} }} }}"
        )
    }

    #[test]
    fn an_if_whose_arm_traps_keeps_the_result_of_its_other_arm() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %choice: core.i32) -> core.i32 {
    %outer = scf.if %cond -> (core.i32) {
      func.trap "out_of_range"
    } else {
      %inner = scf.if %cond -> (core.i32) {
        func.trap "out_of_range"
      } else {
        scf.yield %choice
      }
      scf.yield %inner
    }
    func.return %outer
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .unwrap();
        let printed = module.op().display(&ctx).to_string();
        let mut results = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            if wasm_dialect::If::matches(&ctx, op) {
                results.push(ctx.op_results(op).len());
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        assert_eq!(results, [1, 1], "{printed}");

        // The emitted `if`s keep their result, so the value of the arm that
        // does not trap reaches the caller.
        crate::passes::func_to_wasm::lower(&mut ctx, module, TypeConverter::new());
        let bytes = crate::emit_module_to_wasm(&mut ctx, module)
            .expect("module emits")
            .bytes;
        let file = tempfile::NamedTempFile::new().expect("temporary Wasm file");
        std::fs::write(file.path(), bytes).expect("write Wasm module");
        let run = |cond: &str| {
            std::process::Command::new("wasmtime")
                .args(["-W", "gc=y", "--invoke", "main"])
                .arg(file.path())
                .args([cond, "42"])
                .output()
                .expect("Wasmtime CLI is required")
        };
        let kept = run("0");
        assert!(
            kept.status.success(),
            "{}",
            String::from_utf8_lossy(&kept.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&kept.stdout), "42\n");
        assert!(!run("1").status.success());
    }

    #[test]
    fn terminal_never_controls_have_no_target_results() {
        use trunk_ir::dialect::wasm::{Block, If, Loop};
        let controls_named = [If::NAME, Block::NAME, Loop::NAME];
        let cases = [
            (
                "scf.if %cond { func.return } else { func.unreachable }",
                vec![If::NAME],
            ),
            ("scf.loop() { func.return }", vec![Block::NAME, Loop::NAME]),
            (
                "%n = scf.if %cond -> (core.never) { func.tail_call @main(%cond, %choice, %unit) : (core.i1, core.i32, core.nil) -> () } else { func.unreachable }",
                vec![If::NAME],
            ),
            (
                "%n = scf.if %cond -> (core.never) { func.return } else { func.unreachable }",
                vec![If::NAME],
            ),
            (
                "%n = scf.loop() -> (core.never) { func.return }",
                vec![Block::NAME, Loop::NAME],
            ),
            (
                "%n = scf.loop(%choice) -> (core.never) { ^body(%carried: core.i32): func.return }",
                vec![Block::NAME, Loop::NAME],
            ),
            (
                "scf.index_switch %choice : (core.i32) -> () case 0 { func.return } default { func.unreachable }",
                vec![If::NAME],
            ),
            (
                "%n = scf.index_switch %choice : (core.i32) -> core.never case 0 { func.return } case 1 { func.unreachable } default { func.return }",
                vec![If::NAME, If::NAME],
            ),
            (
                "%n = scf.index_switch %choice : (core.i32) -> core.never default { func.return }",
                vec![],
            ),
            (
                "%outer = scf.if %cond -> (core.never) { %inner = scf.if %cond -> (core.never) { func.return } else { func.unreachable } } else { func.return }",
                vec![If::NAME, If::NAME],
            ),
            (
                "%outer = scf.if %cond -> (core.never) { scf.if %cond { func.return } else { func.unreachable } } else { func.return }",
                vec![If::NAME, If::NAME],
            ),
            (
                "%outer = scf.if %cond -> (core.never) { scf.loop() { func.return } } else { func.return }",
                vec![If::NAME, Block::NAME, Loop::NAME],
            ),
            (
                "%outer = scf.if %cond -> (core.never) { %inner = scf.loop() -> (core.never) { func.return } } else { func.return }",
                vec![If::NAME, Block::NAME, Loop::NAME],
            ),
            (
                "scf.index_switch %choice : (core.i32) -> () case 0 { %n = scf.if %cond -> (core.never) { func.return } else { func.unreachable } } default { func.return }",
                vec![If::NAME, If::NAME],
            ),
            (
                "%n = scf.loop() -> (core.never) { %inner = scf.if %cond -> (core.never) { func.return } else { func.unreachable } }",
                vec![Block::NAME, Loop::NAME, If::NAME],
            ),
            (
                "%n = scf.if %cond -> (core.never) { scf.index_switch %choice : (core.i32) -> () case 0 { func.return } default { func.unreachable } } else { func.return }",
                vec![If::NAME, If::NAME],
            ),
        ];
        for (body, expected) in cases {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &control_fixture(body));
            lower(
                &mut ctx,
                module,
                TypeConverter::new(),
                &mut Default::default(),
            )
            .unwrap();
            let mut controls = Vec::new();
            let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
                let data = ctx.op(op);
                assert_ne!(data.name.dialect(&ctx), "scf");
                if controls_named.contains(&data.name) {
                    assert!(ctx.op_results(op).is_empty(), "{body}");
                    controls.push(data.name);
                }
                for &ty in ctx.op_result_types(op) {
                    assert!(!(ctx.get_type(ty).name.is(&ctx, "core.never")));
                }
                for region in ctx.op_regions(op) {
                    for &block in &ctx.region(region).blocks {
                        for &arg in ctx.block_args(block) {
                            let ty = ctx.get_type(ctx.value_ty(arg));
                            assert!(!(ty.name.is(&ctx, "core.never")));
                        }
                    }
                }
                std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
            });
            assert_eq!(controls, expected, "{body}");
            assert!(trunk_ir::validation::validate_use_chains(&ctx, module).is_ok());
            assert!(trunk_ir::validation::validate_operation_verifiers(&ctx, module).is_ok());
            crate::passes::func_to_wasm::lower(&mut ctx, module, TypeConverter::new());
            let bytes = crate::emit_module_to_wasm(&mut ctx, module)
                .expect(body)
                .bytes;
            wasmparser::Validator::new()
                .validate_all(&bytes)
                .expect(body);
        }
    }

    #[test]
    fn terminal_controls_preserve_unreachability_in_value_returning_functions() {
        for body in [
            "scf.if %cond { func.return %choice } else { func.unreachable }",
            "%n = scf.if %cond -> (core.never) { func.return %choice } else { func.unreachable }",
            "scf.loop() { func.return %choice }",
            "%n = scf.loop() -> (core.never) { func.return %choice }",
            "scf.index_switch %choice : (core.i32) -> () case 0 { func.return %choice } default { func.unreachable }",
            "scf.if %cond { scf.if %cond { func.return %choice } else { func.unreachable } } else { func.return %choice }",
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{ func.func @main(%cond: core.i1, %choice: core.i32) -> core.i32 {{ {body} }} }}"
                ),
            );
            lower(
                &mut ctx,
                module,
                TypeConverter::new(),
                &mut Default::default(),
            )
            .unwrap();
            crate::passes::func_to_wasm::lower(&mut ctx, module, TypeConverter::new());
            let bytes = crate::emit_module_to_wasm(&mut ctx, module).unwrap().bytes;
            wasmparser::Validator::new()
                .validate_all(&bytes)
                .expect(body);
        }

        // A reachable fallthrough must not become an implicit trap that hides
        // a missing return in a value-returning function.
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
            func.func @main(%cond: core.i1, %unit: core.nil) -> core.i32 {
                scf.if %cond { scf.yield %unit } else { scf.yield %unit }
            }
        }"#,
        );
        lower(
            &mut ctx,
            module,
            TypeConverter::new(),
            &mut Default::default(),
        )
        .unwrap();
        crate::passes::func_to_wasm::lower(&mut ctx, module, TypeConverter::new());
        let bytes = crate::emit_module_to_wasm(&mut ctx, module).unwrap().bytes;
        assert!(wasmparser::Validator::new().validate_all(&bytes).is_err());
    }

    #[test]
    fn invalid_never_controls_reject_before_any_mutation() {
        for body in [
            "%n = scf.if %cond -> (core.never) { func.return } else { func.return } func.return %n",
            "%n = scf.if %cond -> (core.never) { func.return } else { func.return } func.unreachable",
            "%n = scf.if %cond -> (core.never) { scf.yield } else { func.return }",
            "%n, %v = scf.if %cond -> (core.never, core.i32) { func.return } else { func.return }",
            "%n = scf.if %cond -> (core.never) { } else { func.return }",
            "%n = scf.if %cond -> (core.never) { ^a: func.return ^b: func.return } else { func.return }",
            "%n = scf.if %cond -> (core.never) { scf.index_switch %choice : (core.i32) -> () case 0 { func.return } default { scf.yield } } else { func.return }",
            "%n = scf.loop() -> (core.never) { func.return } func.return %n",
            "%n = scf.loop() -> (core.never) { func.return } func.unreachable",
            "%n = scf.loop() -> (core.never) { scf.continue }",
            "%n = scf.loop() -> (core.never) { scf.break %unit }",
            "%n = scf.loop() -> (core.never) { }",
            "%n = scf.if %cond -> (core.never) { scf.if %cond { func.return } else { scf.yield } } else { func.return }",
            "%n = scf.if %cond -> (core.never) { scf.loop() { \"scf.break\"() } } else { func.return }",
            "%n = scf.if %cond -> (core.never) { scf.loop() { scf.continue } } else { func.return }",
            "%n = scf.loop() -> (core.never) { ^a: func.return ^b: func.return }",
        ] {
            let input = control_fixture(&format!(
                "scf.if %cond -> (core.nil) {{ scf.yield %unit }} else {{ scf.yield %unit }} {body}"
            ));
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &input);
            let before = module.op().display(&ctx).to_string();
            let error = lower(
                &mut ctx,
                module,
                TypeConverter::new(),
                &mut Default::default(),
            )
            .expect_err(body);
            assert_eq!(error.boundary(), SCF_TO_WASM_BOUNDARY);
            assert!(
                error.to_string().contains("Never control requires"),
                "{error}"
            );
            assert_eq!(module.op().display(&ctx).to_string(), before, "{body}");
        }
    }

    #[test]
    fn preserves_used_nil_and_numeric_control_results() {
        for (ty, value) in [("core.nil", "%unit"), ("core.i32", "%choice")] {
            for body in [
                format!(
                    "%v = scf.if %cond -> ({ty}) {{ scf.yield {value} }} else {{ scf.yield {value} }} func.return %v"
                ),
                format!("%v = scf.loop() -> ({ty}) {{ scf.break {value} }} func.return %v"),
            ] {
                let mut ctx = IrContext::new();
                let input = control_fixture(&body).replace("-> core.nil", &format!("-> {ty}"));
                let module = parse_test_module(&mut ctx, &input);
                lower(
                    &mut ctx,
                    module,
                    TypeConverter::new(),
                    &mut Default::default(),
                )
                .unwrap();
                let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
                    if wasm_dialect::If::matches(&ctx, op)
                        || wasm_dialect::Block::matches(&ctx, op)
                        || wasm_dialect::Loop::matches(&ctx, op)
                    {
                        assert_eq!(ctx.op_result_types(op).len(), 1);
                        let result = ctx.get_type(ctx.op_result_types(op)[0]);
                        assert_eq!(format!("{}", result.name.display(&ctx)), ty);
                    }
                    std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
                });
                assert!(trunk_ir::validation::validate_use_chains(&ctx, module).is_ok());
            }
        }
    }

    /// Convert `wasm_gc.array` to the abstract `wasm.arrayref` type.
    fn array_to_arrayref_converter(ctx: &mut IrContext) -> TypeConverter {
        let arrayref_ty = wasm_dialect::Arrayref::params().intern(ctx).as_type_ref();
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            ctx.get_type(ty)
                .name
                .is(ctx, "wasm_gc.array")
                .then_some(arrayref_ty)
        });
        type_converter
    }
}
