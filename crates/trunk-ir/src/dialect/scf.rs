//! Arena-based scf dialect.

#[trunk_ir::dialect]
mod scf {
    fn r#if(cond: ()) -> result {
        #[region(then_region)]
        {}
        #[region(else_region)]
        {}
    }

    fn switch(discriminant: ()) {
        #[region(body)]
        {}
    }

    #[attr(value: any)]
    fn case() {
        #[region(body)]
        {}
    }

    fn default() {
        #[region(body)]
        {}
    }

    fn r#yield(#[rest] values: ()) {}

    fn r#loop(#[rest] init: ()) -> result {
        #[region(body)]
        {}
    }

    fn r#continue(#[rest] values: ()) {}

    fn r#break(value: ()) {}
}

// =========================================================================
// Control-flow interfaces
// =========================================================================

use crate::op_interface::{
    ControlFlowInterfaceError, ForwardedValues, RegionBranchOps, RegionBranchPoint,
    RegionBranchTerminatorOps, RegionSuccessor, RegionSuccessors,
};

fn interface_error(detail: impl Into<String>) -> ControlFlowInterfaceError {
    ControlFlowInterfaceError::new(detail)
}

fn not_applicable(detail: impl Into<String>) -> ControlFlowInterfaceError {
    ControlFlowInterfaceError::not_applicable(detail)
}

fn op_parent_region(ctx: &IrContext, op: OpRef) -> Option<crate::refs::RegionRef> {
    ctx.op(op)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region)
}

fn nearest_enclosing_loop(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let mut region = op_parent_region(ctx, op);
    while let Some(current) = region {
        let owner = ctx.region(current).parent_op?;
        if Loop::matches(ctx, owner) {
            return Some(owner);
        }
        region = op_parent_region(ctx, owner);
    }
    None
}

fn immediate_region_owner(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    op_parent_region(ctx, op).and_then(|region| ctx.region(region).parent_op)
}

fn if_successors(
    ctx: &IrContext,
    op: OpRef,
    point: RegionBranchPoint,
) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
    if ctx.op_operands(op).len() != 1 || ctx.op(op).regions.len() != 2 {
        return Err(interface_error(
            "scf.if requires one condition and exactly two regions",
        ));
    }
    let if_op = If::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.if: {error:?}")))?;
    match point {
        RegionBranchPoint::Parent => Ok(RegionSuccessors::new([
            RegionSuccessor::Region(if_op.then_region(ctx)),
            RegionSuccessor::Region(if_op.else_region(ctx)),
        ])),
        RegionBranchPoint::Terminator(terminator)
            if Yield::matches(ctx, terminator)
                && immediate_region_owner(ctx, terminator) == Some(op) =>
        {
            Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
        }
        RegionBranchPoint::Terminator(terminator) => Err(not_applicable(format!(
            "{terminator} is not an scf.if region yield"
        ))),
    }
}

fn if_entry_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    if ctx.op_operands(op).len() != 1 || ctx.op(op).regions.len() != 2 {
        return Err(interface_error(
            "scf.if requires one condition and exactly two regions",
        ));
    }
    let if_op = If::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.if: {error:?}")))?;
    match successor {
        RegionSuccessor::Region(region)
            if region == if_op.then_region(ctx) || region == if_op.else_region(ctx) =>
        {
            Ok(ForwardedValues::default())
        }
        _ => Err(interface_error("successor is not an scf.if entry region")),
    }
}

fn loop_successors(
    ctx: &IrContext,
    op: OpRef,
    point: RegionBranchPoint,
) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
    if ctx.op(op).regions.len() != 1 {
        return Err(interface_error("scf.loop requires exactly one body region"));
    }
    let loop_op = Loop::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.loop: {error:?}")))?;
    match point {
        RegionBranchPoint::Parent => Ok(RegionSuccessors::new([RegionSuccessor::Region(
            loop_op.body(ctx),
        )])),
        RegionBranchPoint::Terminator(terminator)
            if nearest_enclosing_loop(ctx, terminator) == Some(op)
                && Continue::matches(ctx, terminator) =>
        {
            Ok(RegionSuccessors::new([RegionSuccessor::Region(
                loop_op.body(ctx),
            )]))
        }
        RegionBranchPoint::Terminator(terminator)
            if nearest_enclosing_loop(ctx, terminator) == Some(op)
                && Break::matches(ctx, terminator) =>
        {
            Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
        }
        RegionBranchPoint::Terminator(terminator) => Err(not_applicable(format!(
            "{terminator} is not a continue or break for this scf.loop"
        ))),
    }
}

fn loop_entry_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    if ctx.op(op).regions.len() != 1 {
        return Err(interface_error("scf.loop requires exactly one body region"));
    }
    let loop_op = Loop::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.loop: {error:?}")))?;
    match successor {
        RegionSuccessor::Region(region) if region == loop_op.body(ctx) => {
            Ok(ForwardedValues::new(loop_op.init(ctx).iter().copied()))
        }
        _ => Err(interface_error("successor is not the scf.loop body")),
    }
}

fn case_regions(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Vec<crate::refs::RegionRef>, ControlFlowInterfaceError> {
    if ctx.op_operands(op).len() != 1 || ctx.op(op).regions.len() != 1 {
        return Err(interface_error(
            "scf.switch requires one discriminant and exactly one body region",
        ));
    }
    let switch = Switch::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.switch: {error:?}")))?;
    let body = ctx.region(switch.body(ctx));
    let [block] = body.blocks.as_slice() else {
        return Err(interface_error(format!(
            "scf.switch body must contain one block, found {}",
            body.blocks.len()
        )));
    };
    let mut regions = Vec::new();
    let mut default_count = 0usize;
    for &child in &ctx.block(*block).ops {
        if let Ok(case) = Case::from_op(ctx, child) {
            regions.push(case.body(ctx));
        } else if let Ok(default) = Default::from_op(ctx, child) {
            default_count += 1;
            regions.push(default.body(ctx));
        } else {
            return Err(interface_error(format!(
                "scf.switch body contains unsupported operation {child}"
            )));
        }
    }
    if default_count > 1 {
        return Err(interface_error(
            "scf.switch contains multiple default regions",
        ));
    }
    Ok(regions)
}

fn switch_successors(
    ctx: &IrContext,
    op: OpRef,
    point: RegionBranchPoint,
) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
    let regions = case_regions(ctx, op)?;
    match point {
        RegionBranchPoint::Parent => {
            let mut successors: Vec<_> = regions
                .iter()
                .copied()
                .map(RegionSuccessor::Region)
                .collect();
            let has_default = regions.iter().any(|region| {
                ctx.region(*region)
                    .parent_op
                    .is_some_and(|parent| Default::matches(ctx, parent))
            });
            if !has_default {
                successors.push(RegionSuccessor::Parent);
            }
            Ok(RegionSuccessors::new(successors))
        }
        RegionBranchPoint::Terminator(terminator)
            if Yield::matches(ctx, terminator)
                && op_parent_region(ctx, terminator)
                    .is_some_and(|region| regions.contains(&region)) =>
        {
            Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
        }
        RegionBranchPoint::Terminator(terminator) => Err(not_applicable(format!(
            "{terminator} is not a yield from this scf.switch"
        ))),
    }
}

fn switch_entry_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    let regions = case_regions(ctx, op)?;
    match successor {
        RegionSuccessor::Region(region) if regions.contains(&region) => {
            Ok(ForwardedValues::default())
        }
        RegionSuccessor::Parent
            if !regions.iter().any(|region| {
                ctx.region(*region)
                    .parent_op
                    .is_some_and(|parent| Default::matches(ctx, parent))
            }) =>
        {
            Ok(ForwardedValues::default())
        }
        _ => Err(interface_error(
            "successor is not an scf.switch entry target",
        )),
    }
}

fn wrapper_successors(
    ctx: &IrContext,
    op: OpRef,
    point: RegionBranchPoint,
) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
    if ctx.op(op).regions.len() != 1 {
        return Err(interface_error(
            "scf.case/default requires exactly one body region",
        ));
    }
    let body = if let Ok(case) = Case::from_op(ctx, op) {
        case.body(ctx)
    } else {
        Default::from_op(ctx, op)
            .map_err(|error| interface_error(format!("malformed switch arm: {error:?}")))?
            .body(ctx)
    };
    match point {
        RegionBranchPoint::Parent => Ok(RegionSuccessors::new([RegionSuccessor::Region(body)])),
        RegionBranchPoint::Terminator(terminator)
            if Yield::matches(ctx, terminator)
                && immediate_region_owner(ctx, terminator) == Some(op) =>
        {
            Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
        }
        RegionBranchPoint::Terminator(terminator) => Err(not_applicable(format!(
            "{terminator} is not a yield from this switch arm"
        ))),
    }
}

fn wrapper_entry_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    if ctx.op(op).regions.len() != 1 {
        return Err(interface_error(
            "scf.case/default requires exactly one body region",
        ));
    }
    let body = if let Ok(case) = Case::from_op(ctx, op) {
        case.body(ctx)
    } else {
        Default::from_op(ctx, op)
            .map_err(|error| interface_error(format!("malformed switch arm: {error:?}")))?
            .body(ctx)
    };
    match successor {
        RegionSuccessor::Region(region) if region == body => Ok(ForwardedValues::default()),
        _ => Err(interface_error("successor is not this switch arm's body")),
    }
}

fn yield_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    let yield_op = Yield::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.yield: {error:?}")))?;
    match successor {
        RegionSuccessor::Parent => Ok(ForwardedValues::new(yield_op.values(ctx).iter().copied())),
        RegionSuccessor::Region(_) => Err(interface_error("scf.yield can only return to a parent")),
    }
}

fn continue_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    let continue_op = Continue::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.continue: {error:?}")))?;
    match successor {
        RegionSuccessor::Region(_) => Ok(ForwardedValues::new(
            continue_op.values(ctx).iter().copied(),
        )),
        RegionSuccessor::Parent => Err(interface_error("scf.continue cannot exit its loop")),
    }
}

fn break_operands(
    ctx: &IrContext,
    op: OpRef,
    successor: RegionSuccessor,
) -> Result<ForwardedValues, ControlFlowInterfaceError> {
    if ctx.op_operands(op).len() != 1 {
        return Err(interface_error("scf.break requires exactly one value"));
    }
    let break_op = Break::from_op(ctx, op)
        .map_err(|error| interface_error(format!("malformed scf.break: {error:?}")))?;
    match successor {
        RegionSuccessor::Parent => {
            let owner = nearest_enclosing_loop(ctx, op)
                .ok_or_else(|| interface_error("scf.break has no enclosing scf.loop"))?;
            if ctx.op_results(owner).is_empty() {
                Ok(ForwardedValues::default())
            } else {
                Ok(ForwardedValues::new([break_op.value(ctx)]))
            }
        }
        RegionSuccessor::Region(_) => Err(interface_error("scf.break cannot enter a region")),
    }
}

inventory::submit! {
    RegionBranchOps::register("scf", "if", if_successors, if_entry_operands)
}
inventory::submit! {
    RegionBranchOps::register("scf", "switch", switch_successors, switch_entry_operands)
}
inventory::submit! {
    RegionBranchOps::register("scf", "case", wrapper_successors, wrapper_entry_operands)
}
inventory::submit! {
    RegionBranchOps::register("scf", "default", wrapper_successors, wrapper_entry_operands)
}
inventory::submit! {
    RegionBranchOps::register("scf", "loop", loop_successors, loop_entry_operands)
}
inventory::submit! {
    RegionBranchTerminatorOps::register("scf", "yield", yield_operands)
}
inventory::submit! {
    RegionBranchTerminatorOps::register("scf", "continue", continue_operands)
}
inventory::submit! {
    RegionBranchTerminatorOps::register("scf", "break", break_operands)
}

// =========================================================================
// Canonicalization folds
// =========================================================================

use crate::context::IrContext;
use crate::dialect::arith::{const_int_value, core_int_width};
use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueRef};
use crate::transforms::canonicalize::FoldResult;

/// `scf.if(arith.const Int(c) : core.i1)` → splice the chosen region's
/// body into the parent block.
///
/// When the condition is a compile-time constant, exactly one branch
/// runs; the other is dead. The chosen region's `scf.yield <values>`
/// supplies the if op's results, so the fold names the body ops to
/// keep and the values that should take over the if op's result slots
/// — the canonicalize dispatcher does the splice + cleanup of the
/// dead branch and the chosen region's yield via [`FoldResult::Splice`].
///
/// Multi-block regions are left alone — they only appear
/// post-`scf_to_cf`, where this pass doesn't run anyway. Bails out on
/// any structural mismatch (yield arity, non-`i1` const, malformed
/// regions).
#[trunk_ir::canonicalize_fold(scf.r#if)]
pub(crate) fn fold_if(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let if_op = If::from_op(ctx, op).ok()?;
    let cond = if_op.cond(ctx);
    // Guard against malformed IR: only fold when the cond's type is
    // `core.i1`. Without this, an `arith.const value=2 : core.i32`
    // wired into the cond slot would be treated as truthy here,
    // burning the dead branch before the validator gets to reject it.
    if core_int_width(ctx, ctx.value_ty(cond)) != Some(1) {
        return None;
    }
    let cond_value = const_int_value(ctx, cond)?;
    // We've gated on cond's type being `core.i1`; the only valid
    // payloads are 0 and 1. Reject anything else (e.g.
    // `arith.const value=2 : core.i1`) so the validator surfaces the
    // malformed const instead of the fold silently picking a branch.
    let active_region = match cond_value {
        0 => if_op.else_region(ctx),
        1 => if_op.then_region(ctx),
        _ => return None,
    };

    // Active region must be a single block whose terminator is `scf.yield`.
    let blocks = ctx.region(active_region).blocks.to_vec();
    let [active_block] = blocks.as_slice() else {
        return None;
    };
    let region_ops: Vec<OpRef> = ctx.block(*active_block).ops.to_vec();
    let (yield_op, body_ops) = region_ops.split_last()?;
    if !Yield::matches(ctx, *yield_op) {
        return None;
    }

    // Yield arity must match the if op's result count.
    let yield_operands: Vec<ValueRef> = ctx.op_operands(*yield_op).to_vec();
    if yield_operands.len() != ctx.op_results(op).len() {
        return None;
    }

    Some(FoldResult::Splice {
        body: body_ops.to_vec(),
        results: yield_operands,
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::symbol::Symbol;
    use crate::transforms::canonicalize::{FoldDispatchPattern, folds_for_dialect};
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only this dialect's folds on `module` via a single
    /// [`FoldDispatchPattern`] — mirrors `arith.rs::run_arith_patterns`
    /// to keep per-dialect tests isolated from other dialects' folds.
    fn run_scf_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_folds(folds_for_dialect("scf"));
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
            .apply_partial(ctx, module)
    }

    fn count_ops(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> usize {
        let dialect_sym = Symbol::from_dynamic(dialect);
        let name_sym = Symbol::from_dynamic(name);
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect_sym && data.name == name_sym {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn if_const_true_splices_then_region() {
        // `scf.if(const true) { %a = arith.addi %x, %x; yield %a } { yield %x }`
        // → splice the addi into the parent block, replace the if's
        // result with %a (the yield's operand).
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.const {value = 1} : core.i1
    %r = scf.if %t : core.i32 {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        // The then-region's `arith.addi` is now at the parent block.
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn if_const_false_splices_else_region() {
        // Mirror of the previous test but with const false: the else
        // region is the active one.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %f = arith.const {value = 0} : core.i1
    %r = scf.if %f : core.i32 {
      scf.yield %x
    } {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn if_const_does_not_match_non_const_cond() {
        // The condition is a block argument — the pattern can't decide
        // which branch runs at compile time, so the if must stay.
        let input = r#"core.module @test {
  func.func @f(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      scf.yield %x
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 1);
    }

    #[test]
    fn if_non_i1_const_cond_is_not_folded() {
        // Malformed IR: the cond slot is wired to a `core.i32` constant
        // instead of `core.i1`. The fold must bail and leave the op for
        // validation rather than picking a branch by truthiness.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %bad = arith.const {value = 2} : core.i32
    %r = scf.if %bad : core.i32 {
      scf.yield %x
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 1);
    }

    #[test]
    fn if_malformed_i1_payload_is_not_folded() {
        // Cond is i1-typed but carries a payload outside {0, 1}.
        // The fold must bail so validation flags the bad const rather
        // than the rewrite picking the then-branch by truthiness.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %bad = arith.const {value = 2} : core.i1
    %r = scf.if %bad : core.i32 {
      scf.yield %x
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 1);
    }

    #[test]
    fn if_const_with_empty_active_region_just_forwards_yield() {
        // No body ops in the active region — the rewrite still fires:
        // the if is erased and its result becomes the yield's operand.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.const {value = 1} : core.i1
    %r = scf.if %t : core.i32 {
      scf.yield %x
    } {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        // The else region's addi was dropped along with the if op (it
        // was unreachable). Only the parent's `func.return` remains for
        // `arith` ops, plus the const cond.
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
    }
}
