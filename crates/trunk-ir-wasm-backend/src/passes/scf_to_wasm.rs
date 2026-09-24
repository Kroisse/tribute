//! Lower scf dialect operations to wasm dialect (arena IR).
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=2)` (branch to outer block, past if and loop)

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::sync::Arc;

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::dialect::scf;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, IllegalOp, LegalityCheck, Module, PatternApplicator,
    PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::transforms::scf_control_flow::StructuredControlAnalysis;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_op};

const SCF_TO_WASM_BOUNDARY: &str = "scf-to-wasm";

/// Require successful conversion to remove every SCF operation.
pub fn scf_to_wasm_target() -> ConversionTarget {
    ConversionTarget::new().illegal_dialect("scf")
}

/// Lower scf dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    AnalysisCache::scope(ctx, |ctx, cache| {
        let analysis = cache
            .get::<StructuredControlAnalysis>(ctx, module.op())
            .expect("structured control analysis is infallible");
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
        cache.invalidate::<StructuredControlAnalysis>(module.op());
        let plan = Arc::new(plan);
        PatternApplicator::new(type_converter)
            .add_pattern(ScfIfPattern(plan.clone()))
            .add_pattern(ScfSwitchPattern(plan.clone()))
            .add_pattern(ScfLoopPattern(plan))
            .add_pattern(ScfYieldPattern)
            .add_pattern(ScfContinuePattern)
            .add_pattern(ScfBreakPattern)
            .with_target(scf_to_wasm_target())
            .apply_partial_conversion(ctx, module, SCF_TO_WASM_BOUNDARY)?;
        Ok(())
    })
}

/// Validate structured control before any target pipeline mutation.
pub fn validate_lowerable_structured_control(
    ctx: &IrContext,
    module: Module,
) -> Result<(), ConversionError> {
    let analysis = AnalysisCache::new()
        .get::<StructuredControlAnalysis>(ctx, module.op())
        .expect("structured control analysis is infallible");
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
        let has_never = (scf::If::matches(ctx, op) || scf::Loop::matches(ctx, op))
            && ctx.op_results(op).iter().any(|&value| {
                let ty = ctx.get_type(ctx.value_ty(value));
                ty.dialect == "core" && ty.name == "never"
            });
        if has_never {
            if analysis.has_terminal_unused_never_result(op) {
                accept(op, ControlLowering::DropNeverResult);
            } else {
                let data = ctx.op(op);
                return ControlFlow::Break(ConversionError::new(SCF_TO_WASM_BOUNDARY, vec![IllegalOp {
                    op,
                    dialect: data.dialect,
                    name: data.name,
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
                    dialect: data.dialect,
                    name: data.name,
                    legality: LegalityCheck::Illegal,
                    reason: None,
                }
                .with_reason(reason.to_string()),
            ],
        ));
    }
    Ok(())
}

/// Validated data for a resultless `scf.switch` that this target can lower.
///
/// Wasm branch conditions and the available concrete comparison operations are
/// `i32`, so other source switch shapes remain for an earlier or richer
/// lowering rather than being partially rewritten here.
struct ScfSwitchArms {
    discriminant: ValueRef,
    cases: Vec<(i32, RegionRef)>,
    default: Option<RegionRef>,
}

struct ScfSwitchShape {
    discriminant: ValueRef,
    cases: Vec<(Attribute, RegionRef)>,
    default: Option<RegionRef>,
}

#[derive(Debug)]
enum SwitchLoweringReason {
    MalformedShape,
    UnsupportedDiscriminantType(String),
    NonIntegerCaseAttribute,
    CaseValueOutsideI32Range,
}

impl std::fmt::Display for SwitchLoweringReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedShape => write!(f, "malformed resultless switch shape"),
            Self::UnsupportedDiscriminantType(ty) => {
                write!(
                    f,
                    "unsupported discriminant type `{ty}`; expected `core.i32`"
                )
            }
            Self::NonIntegerCaseAttribute => {
                write!(f, "case attribute `value` must be an integer")
            }
            Self::CaseValueOutsideI32Range => {
                write!(f, "case integer value is outside the i32 range")
            }
        }
    }
}

fn is_i32(ctx: &IrContext, value: ValueRef) -> bool {
    let ty = ctx.get_type(ctx.value_ty(value));
    ty.dialect == core::DIALECT_NAME() && ty.name == "i32"
}

/// Validate the complete declarative switch container before rewriting any
/// nested region. This keeps malformed arms from being partly lowered by the
/// recursive pattern walk.
fn switch_shape(ctx: &IrContext, op: OpRef) -> Option<ScfSwitchShape> {
    if !ctx.op_results(op).is_empty() {
        return None;
    }
    let [discriminant] = ctx.op_operands(op) else {
        return None;
    };
    let [switch_body] = ctx.op(op).regions.as_slice() else {
        return None;
    };
    let [body_block] = ctx.region(*switch_body).blocks.as_slice() else {
        return None;
    };
    if !ctx.block_args(*body_block).is_empty() {
        return None;
    }

    let mut cases = Vec::new();
    let mut default = None;
    for &arm in &ctx.block(*body_block).ops {
        if !ctx.op_results(arm).is_empty() || !ctx.op_operands(arm).is_empty() {
            return None;
        }
        let [body] = ctx.op(arm).regions.as_slice() else {
            return None;
        };
        let [entry] = ctx.region(*body).blocks.as_slice() else {
            return None;
        };
        if !ctx.block_args(*entry).is_empty() {
            return None;
        }
        if scf::Case::matches(ctx, arm) {
            cases.push((ctx.op(arm).attributes.get("value")?.clone(), *body));
        } else if scf::Default::matches(ctx, arm) {
            if default.replace(*body).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    if cases.is_empty() && default.is_none() {
        return None;
    }

    Some(ScfSwitchShape {
        discriminant: *discriminant,
        cases,
        default,
    })
}

fn switch_arms(ctx: &IrContext, op: OpRef) -> Result<ScfSwitchArms, SwitchLoweringReason> {
    let shape = switch_shape(ctx, op).ok_or(SwitchLoweringReason::MalformedShape)?;
    if !is_i32(ctx, shape.discriminant) {
        let ty = ctx.get_type(ctx.value_ty(shape.discriminant));
        return Err(SwitchLoweringReason::UnsupportedDiscriminantType(format!(
            "{}.{}",
            ty.dialect, ty.name
        )));
    }
    let mut cases = Vec::with_capacity(shape.cases.len());
    for (value, body) in shape.cases {
        let Attribute::Int(value) = value else {
            return Err(SwitchLoweringReason::NonIntegerCaseAttribute);
        };
        let value =
            i32::try_from(value).map_err(|_| SwitchLoweringReason::CaseValueOutsideI32Range)?;
        cases.push((value, body));
    }
    Ok(ScfSwitchArms {
        discriminant: shape.discriminant,
        cases,
        default: shape.default,
    })
}

/// Find the first switch rejected by the same acceptance contract as lowering.
fn find_nonlowerable_switch(ctx: &IrContext, op: OpRef) -> Option<(OpRef, SwitchLoweringReason)> {
    if scf::Switch::matches(ctx, op)
        && let Err(reason) = switch_arms(ctx, op)
    {
        return Some((op, reason));
    }
    for region in ctx.op(op).regions.iter().copied() {
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

fn take_region_ops(ctx: &mut IrContext, region: RegionRef) -> Vec<OpRef> {
    let [block] = ctx.region(region).blocks.as_slice() else {
        unreachable!("switch regions are preflighted as single-block");
    };
    let ops = ctx.block(*block).ops.to_vec();
    for &op in &ops {
        ctx.detach_op(op);
    }
    ops
}

/// Pattern for a resultless `scf.switch` -> nested `wasm.if` comparisons.
struct ScfSwitchPattern(Arc<ScfLoweringPlan>);

impl RewritePattern for ScfSwitchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_switch) = scf::Switch::from_op(ctx, op) else {
            return false;
        };
        let Ok(arms) = switch_arms(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let nil_ty = core::nil(ctx).as_type_ref();
        let result_types = if self.0.resultless_switches.contains(&op) {
            vec![]
        } else {
            vec![nil_ty]
        };

        for &(_, body) in &arms.cases {
            ctx.detach_region(body);
        }
        if let Some(body) = arms.default {
            ctx.detach_region(body);
        }

        if arms.cases.is_empty() {
            if let Some(default) = arms.default {
                for child in take_region_ops(ctx, default) {
                    rewriter.insert_op(child);
                }
            }
            rewriter.erase_op(vec![]);
            return true;
        }

        let discriminant = arms.discriminant;
        let case_count = arms.cases.len();
        let mut next = arms
            .default
            .unwrap_or_else(|| region_with_ops(ctx, loc, vec![]));
        let mut outer_ops = None;
        for (index, (value, body)) in arms.cases.into_iter().rev().enumerate() {
            let case = wasm_dialect::I32Const::operands()
                .value(value)
                .results(ctx.value_ty(discriminant))
                .build(ctx, loc);
            let matches = wasm_dialect::I32Eq::operands(discriminant, case.result(ctx))
                .results(ctx.value_ty(discriminant))
                .build(ctx, loc);
            let branch = wasm_dialect::If::operands(matches.result(ctx))
                .results(result_types.clone())
                .regions(body, next)
                .build(ctx, loc);
            let ops = vec![case.op_ref(), matches.op_ref(), branch.op_ref()];
            if index + 1 == case_count {
                outer_ops = Some(ops);
            } else {
                next = region_with_ops(ctx, loc, ops);
            }
        }

        for inserted in outer_ops.expect("nonempty switch cases build an outer dispatch") {
            rewriter.insert_op(inserted);
        }
        if self.0.terminal_controls.contains(&op) {
            let unreachable = wasm_dialect::Unreachable::operands().build(ctx, loc);
            rewriter.insert_op(unreachable.op_ref());
        }
        rewriter.erase_op(vec![]);
        true
    }
}

/// Pattern for `scf.if` -> `wasm.if`
struct ScfIfPattern(Arc<ScfLoweringPlan>);

impl RewritePattern for ScfIfPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(scf_if_op) = scf::If::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;

        // Preserve the result list; reject multi-result. The
        // created `wasm.if` must declare target types, so read them through the
        // converter instead of copying the shared IR spelling.
        let mut result_types = rewriter.result_types(ctx, op);
        if result_types.len() > 1 {
            return false;
        }
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
        true
    }
}

/// Pattern for `scf.loop` -> `wasm.block(wasm.loop(...))`
///
/// The loop is wrapped in a block to provide a break target.
/// From inside a `wasm.if` within the loop body:
/// - `wasm.br(target=1)` branches to the loop (continue)
/// - `wasm.br(target=2)` branches to the block (break)
struct ScfLoopPattern(Arc<ScfLoweringPlan>);

impl RewritePattern for ScfLoopPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(loop_op) = scf::Loop::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;

        // Get result type; reject multi-result loops. The created
        // `wasm.block`/`wasm.loop` must declare target types.
        let mut result_types = rewriter.result_types(ctx, op);
        if result_types.len() > 1 {
            return false;
        }
        let drop_result = self.0.drop_never_results.contains(&op);
        if drop_result {
            result_types.clear();
        }

        // Get init operands
        let init: Vec<_> = loop_op.init(ctx).to_vec();

        // Detach the body region from the original loop op
        let body = loop_op.body(ctx);
        ctx.detach_region(body);

        // The created `wasm.loop` owns the detached body, so its block
        // arguments become Wasm-level parameters. Declare them with target
        // types instead of leaving the SCF spelling on the boundary.
        let body_blocks: Vec<_> = ctx.region(body).blocks.to_vec();
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
        true
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

impl RewritePattern for ScfYieldPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !scf::Yield::matches(ctx, op) {
            return false;
        }

        // Get yield values (variadic operands)
        let operands = ctx.op_operands(op).to_vec();

        if operands.is_empty() {
            // No value to yield - just erase
            rewriter.erase_op(vec![]);
            return true;
        }

        if operands.len() > 1 {
            // Multi-value yields are not yet supported; leave unlowered.
            return false;
        }

        let value = operands[0];
        let loc = ctx.op(op).location;
        let new_op = wasm_dialect::Yield::operands(value).build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `scf.continue` -> `wasm.br(target=1)`
///
/// Branches to the enclosing wasm.loop. Depth 1 is correct because
/// `scf.continue` is always inside a `scf.if` (depth 0 = wasm.if,
/// depth 1 = wasm.loop) within a `scf.loop`.
struct ScfContinuePattern;

impl RewritePattern for ScfContinuePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !scf::Continue::matches(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;

        // Get loop-carried values (variadic operands)
        let values = ctx.op_operands(op).to_vec();
        if values.len() > 1 {
            // Multiple loop-carried values not yet supported; leave unlowered.
            return false;
        }

        if values.is_empty() {
            // No loop-carried values -- simple branch
            let br_op = wasm_dialect::Br::operands().target(1).build(ctx, loc);
            rewriter.replace_op(br_op.op_ref());
            return true;
        }

        // Emit wasm.yield(value) + wasm.br(1) for each loop-carried value.
        // The emit layer will translate yield+br targeting a loop into
        // local.set for the loop arg followed by br.
        let value = values[0];
        let yield_op = wasm_dialect::Yield::operands(value).build(ctx, loc);
        let br_op = wasm_dialect::Br::operands().target(1).build(ctx, loc);

        rewriter.insert_op(yield_op.op_ref());
        rewriter.replace_op(br_op.op_ref());
        true
    }
}

/// Pattern for `scf.break` -> `wasm.yield(value) + wasm.br(target=2)`
///
/// Branches to the enclosing wasm.block with a result value.
/// `scf.break` is always inside a `scf.if` within a `scf.loop`, so after
/// lowering the nesting is: wasm.block > wasm.loop > wasm.if. From inside
/// the wasm.if, depth 2 targets the outer wasm.block (break out of loop).
///
/// According to WASM spec, `br` instruction takes no operands - values are
/// passed via the stack. We use `wasm.yield` to mark the break value as the
/// region's result, then branch without operands.
struct ScfBreakPattern;

impl RewritePattern for ScfBreakPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(break_op) = scf::Break::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let value = break_op.value(ctx);

        // Emit the break value via wasm.yield (marks it as region result)
        let yield_op = wasm_dialect::Yield::operands(value).build(ctx, loc);

        // Branch to outer block (depth 2: if=0, loop=1, block=2)
        let br_op = wasm_dialect::Br::operands().target(2).build(ctx, loc);

        rewriter.insert_op(yield_op.op_ref());
        rewriter.replace_op(br_op.op_ref());
        true
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
    use trunk_ir::printer::print_module;

    fn lower_text(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(&mut ctx, module, TypeConverter::new()).expect("test module should lower to wasm");
        let use_chains = trunk_ir::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let verifiers = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
        assert!(verifiers.is_ok(), "{verifiers}");
        let output = print_module(&ctx, module.op());
        assert!(
            !output.contains("scf."),
            "residual scf operation:\n{output}"
        );
        output
    }

    fn assert_switch_rejected_unchanged(input: &str, reason: &str) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = lower(&mut ctx, module, TypeConverter::new())
            .expect_err("nonlowerable switch should reject the entire conversion");

        assert_eq!(error.boundary(), SCF_TO_WASM_BOUNDARY);
        assert_eq!(error.operations().len(), 1);
        assert_eq!(error.operations()[0].legality, LegalityCheck::Illegal);
        assert_eq!(error.operations()[0].reason.as_deref(), Some(reason));
        assert!(error.to_string().contains("scf.switch"), "{error}");
        assert!(error.to_string().contains(reason), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    fn assert_no_scf_switch_wrappers(output: &str) {
        for name in ["scf.switch", "scf.case", "scf.default", "scf.yield"] {
            assert!(!output.contains(name), "residual {name}:\n{output}");
        }
    }

    #[test]
    fn lowers_resultless_switch_with_explicit_default() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
      }
      scf.default {
        func.unreachable
      }
    }
  }
}"#,
        );

        assert_no_scf_switch_wrappers(&output);
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
    scf.switch %choice {
      scf.case {value = 0} {
        scf.yield
      }
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch_wrappers(&output);
        assert!(output.contains("wasm.i32_eq"), "{output}");
        assert!(output.contains("wasm.if"), "{output}");
        assert!(output.contains("func.return"), "{output}");
    }

    #[test]
    fn lowers_multiple_i32_switch_cases_in_source_order() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.case {value = 1} {
        scf.yield
      }
      scf.case {value = 2} {
        scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch_wrappers(&output);
        assert_eq!(output.matches("wasm.i32_eq").count(), 2, "{output}");
        assert!(
            output.find("wasm.i32_const {value = 1}") < output.find("wasm.i32_const {value = 2}"),
            "case dispatch order changed: {output}"
        );
    }

    #[test]
    fn lowers_nested_proper_tail_switch_arms() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32, %cond: core.i1, %callee: func.func_sig<(core.nil) -> core.never>, %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        %never = scf.if %cond : core.never {
          func.unreachable
        } {
          func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
        }
      }
      scf.default {
        func.tail_call_indirect %callee, %unit {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = 2}
      }
    }
  }
}"#,
        );

        assert_no_scf_switch_wrappers(&output);
        assert!(output.contains("wasm.i32_eq"), "{output}");
        assert_eq!(output.matches("wasm.if").count(), 2, "{output}");
        for line in output.lines().filter(|line| line.contains("wasm.if")) {
            assert!(
                !line.contains(" = ") && !line.contains(" : "),
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
    fn leaves_malformed_switch_unchanged() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.default { scf.yield }
      scf.default { scf.yield }
    }
    func.return
  }
}"#;
        assert_switch_rejected_unchanged(input, "malformed resultless switch shape");
    }

    #[test]
    fn leaves_malformed_switch_arm_operands_and_entry_args_unchanged() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.case %choice {value = 0} {
        ^case_entry(%unexpected: core.i32):
          scf.yield
      }
      scf.default %choice {
        scf.yield
      }
    }
    func.return
  }
}"#;
        assert_switch_rejected_unchanged(input, "malformed resultless switch shape");
    }

    #[test]
    fn rejects_shape_valid_non_i32_switch_without_mutating() {
        let input = r#"core.module @test {
  func.func @main(%cond: core.i1, %choice: core.i64) -> core.nil {
    scf.switch %choice {
      scf.case {value = 0} {
        scf.if %cond : core.nil {
          scf.yield
        } {
          scf.yield
        }
        scf.yield
      }
      scf.default { scf.yield }
    }
    func.return
  }
}"#;
        assert_switch_rejected_unchanged(
            input,
            "unsupported discriminant type `core.i64`; expected `core.i32`",
        );
    }

    #[test]
    fn rejects_shape_valid_out_of_range_case_without_mutating() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.case {value = 2147483648} { scf.yield }
      scf.default { scf.yield }
    }
    func.return
  }
}"#;
        assert_switch_rejected_unchanged(input, "case integer value is outside the i32 range");
    }

    #[test]
    fn rejects_non_integer_case_attribute_without_mutating() {
        let input = r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.case {value = @not_an_integer} { scf.yield }
      scf.default { scf.yield }
    }
    func.return
  }
}"#;
        assert_switch_rejected_unchanged(input, "case attribute `value` must be an integer");
    }

    #[test]
    fn lowers_default_only_switch() {
        let output = lower_text(
            r#"core.module @test {
  func.func @main(%choice: core.i32) -> core.nil {
    scf.switch %choice {
      scf.default { scf.yield }
    }
    func.return
  }
}"#,
        );

        assert_no_scf_switch_wrappers(&output);
        assert!(!output.contains("wasm.i32_eq"), "{output}");
        assert!(output.contains("func.return"), "{output}");
    }

    #[test]
    fn converts_if_result_through_the_type_converter() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @select(%cond: core.i1, %value: core.array(core.i32)) -> core.array(core.i32) {
    %result = scf.if %cond : core.array(core.i32) {
      scf.yield %value
    } {
      scf.yield %value
    }
    func.return %result
  }
}"#,
        );

        let type_converter = array_to_arrayref_converter(&mut ctx);
        lower(&mut ctx, module, type_converter).expect("test module should lower to wasm");

        let use_chains = trunk_ir::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let output = print_module(&ctx, module.op());
        assert!(!output.contains("scf."), "{output}");
        assert!(
            output.contains("wasm.if") && output.contains(": wasm.arrayref"),
            "the created wasm.if must declare the converted result: {output}"
        );
    }

    #[test]
    fn converts_loop_body_block_argument_through_the_type_converter() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @carry(%init: core.array(core.i32)) -> core.nil {
    scf.loop %init : core.nil {
      ^header(%iter: core.array(core.i32)):
        scf.continue %iter
    }
    func.return
  }
}"#,
        );

        let type_converter = array_to_arrayref_converter(&mut ctx);
        lower(&mut ctx, module, type_converter).expect("test module should lower to wasm");

        let use_chains = trunk_ir::validation::validate_use_chains(&ctx, module);
        assert!(use_chains.is_ok(), "{use_chains}");
        let output = print_module(&ctx, module.op());
        assert!(!output.contains("scf."), "{output}");
        assert!(
            output.contains("wasm.loop") && output.contains(": wasm.arrayref)"),
            "the created wasm.loop body argument must declare the converted type: {output}"
        );
        assert!(
            output.contains("core.array(core.i32)"),
            "operand types must keep their producer spelling: {output}"
        );
    }

    fn control_fixture(body: &str) -> String {
        format!(
            "core.module @test {{ func.func @main(%cond: core.i1, %choice: core.i32, %unit: core.nil) -> core.nil {{ {body} }} }}"
        )
    }

    #[test]
    fn terminal_never_controls_have_no_target_results() {
        let cases = [
            (
                "scf.if %cond { func.return } { func.unreachable }",
                vec!["if"],
            ),
            ("scf.loop { func.return }", vec!["block", "loop"]),
            (
                "%n = scf.if %cond : core.never { func.tail_call %cond, %choice, %unit {callee = @main} } { func.unreachable }",
                vec!["if"],
            ),
            (
                "%n = scf.if %cond : core.never { func.return } { func.unreachable }",
                vec!["if"],
            ),
            (
                "%n = scf.loop : core.never { func.return }",
                vec!["block", "loop"],
            ),
            (
                "%n = scf.loop %choice : core.never { ^body(%carried: core.i32): func.return }",
                vec!["block", "loop"],
            ),
            (
                "scf.switch %choice { scf.case {value = 0} { func.return } scf.default { func.unreachable } }",
                vec!["if"],
            ),
            (
                "%outer = scf.if %cond : core.never { %inner = scf.if %cond : core.never { func.return } { func.unreachable } } { func.return }",
                vec!["if", "if"],
            ),
            (
                "%outer = scf.if %cond : core.never { scf.if %cond { func.return } { func.unreachable } } { func.return }",
                vec!["if", "if"],
            ),
            (
                "%outer = scf.if %cond : core.never { scf.loop { func.return } } { func.return }",
                vec!["if", "block", "loop"],
            ),
            (
                "%outer = scf.if %cond : core.never { %inner = scf.loop : core.never { func.return } } { func.return }",
                vec!["if", "block", "loop"],
            ),
            (
                "scf.switch %choice { scf.case {value = 0} { %n = scf.if %cond : core.never { func.return } { func.unreachable } } scf.default { func.return } }",
                vec!["if", "if"],
            ),
            (
                "%n = scf.loop : core.never { %inner = scf.if %cond : core.never { func.return } { func.unreachable } }",
                vec!["block", "loop", "if"],
            ),
            (
                "%n = scf.if %cond : core.never { scf.switch %choice { scf.case {value = 0} { func.return } scf.default { func.unreachable } } } { func.return }",
                vec!["if", "if"],
            ),
        ];
        for (body, expected) in cases {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &control_fixture(body));
            lower(&mut ctx, module, TypeConverter::new()).unwrap();
            let mut controls = Vec::new();
            let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
                let data = ctx.op(op);
                assert_ne!(data.dialect, "scf");
                if data.dialect == "wasm"
                    && (data.name == "if" || data.name == "block" || data.name == "loop")
                {
                    assert!(ctx.op_results(op).is_empty(), "{body}");
                    controls.push(data.name.to_string());
                }
                for &ty in ctx.op_result_types(op) {
                    assert!(
                        !(ctx.get_type(ty).dialect == "core" && ctx.get_type(ty).name == "never")
                    );
                }
                for &region in &data.regions {
                    for &block in &ctx.region(region).blocks {
                        for &arg in ctx.block_args(block) {
                            let ty = ctx.get_type(ctx.value_ty(arg));
                            assert!(!(ty.dialect == "core" && ty.name == "never"));
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
            "scf.if %cond { func.return %choice } { func.unreachable }",
            "%n = scf.if %cond : core.never { func.return %choice } { func.unreachable }",
            "scf.loop { func.return %choice }",
            "%n = scf.loop : core.never { func.return %choice }",
            "scf.switch %choice { scf.case {value = 0} { func.return %choice } scf.default { func.unreachable } }",
            "scf.if %cond { scf.if %cond { func.return %choice } { func.unreachable } } { func.return %choice }",
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{ func.func @main(%cond: core.i1, %choice: core.i32) -> core.i32 {{ {body} }} }}"
                ),
            );
            lower(&mut ctx, module, TypeConverter::new()).unwrap();
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
                scf.if %cond { scf.yield %unit } { scf.yield %unit }
            }
        }"#,
        );
        lower(&mut ctx, module, TypeConverter::new()).unwrap();
        crate::passes::func_to_wasm::lower(&mut ctx, module, TypeConverter::new());
        let bytes = crate::emit_module_to_wasm(&mut ctx, module).unwrap().bytes;
        assert!(wasmparser::Validator::new().validate_all(&bytes).is_err());
    }

    #[test]
    fn invalid_never_controls_reject_before_any_mutation() {
        for body in [
            "%n = scf.if %cond : core.never { func.return } { func.return } func.return %n",
            "%n = scf.if %cond : core.never { func.return } { func.return } func.unreachable",
            "%n = scf.if %cond : core.never { scf.yield } { func.return }",
            "%n, %v = scf.if %cond : core.never, core.i32 { func.return } { func.return }",
            "%n, %v = scf.loop : core.never, core.i32 { func.return }",
            "%n = scf.if %cond : core.never { } { func.return }",
            "%n = scf.if %cond : core.never { ^a: func.return ^b: func.return } { func.return }",
            "%n = scf.if %cond : core.never { func.unreachable %unit } { func.return }",
            "%n = scf.if %cond : core.never { scf.switch %choice { scf.case {value = 0} { func.return } } } { func.return }",
            "%n = scf.loop : core.never { func.return } func.return %n",
            "%n = scf.loop : core.never { func.return } func.unreachable",
            "%n = scf.loop : core.never { scf.continue }",
            "%n = scf.loop : core.never { scf.break %unit }",
            "%n = scf.loop : core.never { }",
            "%n = scf.if %cond : core.never { scf.if %cond { func.return } { scf.yield } } { func.return }",
            "%n = scf.if %cond : core.never { scf.loop { scf.break } } { func.return }",
            "%n = scf.if %cond : core.never { scf.loop { scf.continue } } { func.return }",
            "%n = scf.loop : core.never { ^a: func.return ^b: func.return }",
        ] {
            let input = control_fixture(&format!(
                "scf.if %cond : core.nil {{ scf.yield %unit }} {{ scf.yield %unit }} {body}"
            ));
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &input);
            let before = print_module(&ctx, module.op());
            let error = lower(&mut ctx, module, TypeConverter::new()).expect_err(body);
            assert_eq!(error.boundary(), SCF_TO_WASM_BOUNDARY);
            assert!(
                error.to_string().contains("Never control requires"),
                "{error}"
            );
            assert_eq!(print_module(&ctx, module.op()), before, "{body}");
        }
    }

    #[test]
    fn multiple_value_results_are_rejected_without_truncation() {
        for control in ["scf.if %cond", "scf.loop"] {
            let body = if control.starts_with("scf.if") {
                "{ scf.yield %choice, %unit } { scf.yield %choice, %unit }"
            } else {
                "{ scf.continue }"
            };
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &control_fixture(&format!(
                    "%a, %b = {control} : core.i32, core.nil {body} func.return"
                )),
            );
            let function = module.ops(&ctx)[0];
            let block = ctx.region(ctx.op(function).regions[0]).blocks[0];
            let original = ctx.block(block).ops[0];
            let result_types = ctx.op_result_types(original).to_vec();
            assert!(lower(&mut ctx, module, TypeConverter::new()).is_err());
            assert_eq!(ctx.op_result_types(original), result_types);
        }
    }

    #[test]
    fn preserves_used_nil_and_numeric_control_results() {
        for (ty, value) in [("core.nil", "%unit"), ("core.i32", "%choice")] {
            for body in [
                format!(
                    "%v = scf.if %cond : {ty} {{ scf.yield {value} }} {{ scf.yield {value} }} func.return %v"
                ),
                format!("%v = scf.loop : {ty} {{ scf.break {value} }} func.return %v"),
            ] {
                let mut ctx = IrContext::new();
                let input = control_fixture(&body).replace("-> core.nil", &format!("-> {ty}"));
                let module = parse_test_module(&mut ctx, &input);
                lower(&mut ctx, module, TypeConverter::new()).unwrap();
                let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
                    let data = ctx.op(op);
                    if data.dialect == "wasm"
                        && (data.name == "if" || data.name == "block" || data.name == "loop")
                    {
                        assert_eq!(ctx.op_result_types(op).len(), 1);
                        let result = ctx.get_type(ctx.op_result_types(op)[0]);
                        assert_eq!(format!("{}.{}", result.dialect, result.name), ty);
                    }
                    std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
                });
                assert!(trunk_ir::validation::validate_use_chains(&ctx, module).is_ok());
            }
        }
    }

    /// Convert `core.array` to the abstract `wasm.arrayref` type.
    fn array_to_arrayref_converter(ctx: &mut IrContext) -> TypeConverter {
        let arrayref_ty = ctx.intern_type(
            trunk_ir::types::TypeDataBuilder::new(
                trunk_ir::Symbol::new("wasm"),
                trunk_ir::Symbol::new("arrayref"),
            )
            .build(),
        );
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            ctx.types()
                .is_dialect(
                    ty,
                    trunk_ir::Symbol::new("core"),
                    trunk_ir::Symbol::new("array"),
                )
                .then_some(arrayref_ty)
        });
        type_converter
    }
}
