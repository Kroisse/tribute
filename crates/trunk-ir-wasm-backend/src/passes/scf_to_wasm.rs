//! Lower scf dialect operations to wasm dialect (arena IR).
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=2)` (branch to outer block, past if and loop)

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
use trunk_ir::types::Attribute;

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
    validate_lowerable_switches(ctx, module)?;
    PatternApplicator::new(type_converter)
        .add_pattern(ScfIfPattern)
        .add_pattern(ScfSwitchPattern)
        .add_pattern(ScfLoopPattern)
        .add_pattern(ScfYieldPattern)
        .add_pattern(ScfContinuePattern)
        .add_pattern(ScfBreakPattern)
        .with_target(scf_to_wasm_target())
        .apply_partial_conversion(ctx, module, SCF_TO_WASM_BOUNDARY)?;
    Ok(())
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
    let ty = ctx.types.get(ctx.value_ty(value));
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
        let ty = ctx.types.get(ctx.value_ty(shape.discriminant));
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
struct ScfSwitchPattern;

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
            let case = wasm_dialect::i32_const(ctx, loc, ctx.value_ty(discriminant), value);
            let matches = wasm_dialect::i32_eq(
                ctx,
                loc,
                discriminant,
                case.result(ctx),
                ctx.value_ty(discriminant),
            );
            let branch = wasm_dialect::r#if(ctx, loc, matches.result(ctx), nil_ty, body, next);
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
        rewriter.erase_op(vec![]);
        true
    }
}

/// Pattern for `scf.if` -> `wasm.if`
struct ScfIfPattern;

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

        // Get result type (default to nil if none); reject multi-result
        let result_types = ctx.op_result_types(op);
        if result_types.len() > 1 {
            return false;
        }
        let result_ty = result_types
            .first()
            .copied()
            .unwrap_or_else(|| core::nil(ctx).as_type_ref());

        // Get the condition operand
        let cond = scf_if_op.cond(ctx);

        // Get then/else regions and detach them from the original op
        let then_region = scf_if_op.then_region(ctx);
        let else_region = scf_if_op.else_region(ctx);
        ctx.detach_region(then_region);
        ctx.detach_region(else_region);

        let new_op = wasm_dialect::r#if(ctx, loc, cond, result_ty, then_region, else_region);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `scf.loop` -> `wasm.block(wasm.loop(...))`
///
/// The loop is wrapped in a block to provide a break target.
/// From inside a `wasm.if` within the loop body:
/// - `wasm.br(target=1)` branches to the loop (continue)
/// - `wasm.br(target=2)` branches to the block (break)
struct ScfLoopPattern;

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

        // Get result type; reject multi-result loops
        let result_types = ctx.op_result_types(op);
        if result_types.len() > 1 {
            return false;
        }
        let result_ty = result_types
            .first()
            .copied()
            .unwrap_or_else(|| core::nil(ctx).as_type_ref());

        // Get init operands
        let init: Vec<_> = loop_op.init(ctx).to_vec();

        // Detach the body region from the original loop op
        let body = loop_op.body(ctx);
        ctx.detach_region(body);

        // Create wasm.loop with init operands and the body region
        let wasm_loop = wasm_dialect::r#loop(ctx, loc, init, result_ty, body);

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

        let wasm_block = wasm_dialect::block(ctx, loc, result_ty, block_body);
        rewriter.replace_op(wasm_block.op_ref());
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
        let new_op = wasm_dialect::r#yield(ctx, loc, value);
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
            let br_op = wasm_dialect::br(ctx, loc, 1);
            rewriter.replace_op(br_op.op_ref());
            return true;
        }

        // Emit wasm.yield(value) + wasm.br(1) for each loop-carried value.
        // The emit layer will translate yield+br targeting a loop into
        // local.set for the loop arg followed by br.
        let value = values[0];
        let yield_op = wasm_dialect::r#yield(ctx, loc, value);
        let br_op = wasm_dialect::br(ctx, loc, 1);

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
        let yield_op = wasm_dialect::r#yield(ctx, loc, value);

        // Branch to outer block (depth 2: if=0, loop=1, block=2)
        let br_op = wasm_dialect::br(ctx, loc, 2);

        rewriter.insert_op(yield_op.op_ref());
        rewriter.replace_op(br_op.op_ref());
        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

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
  func.func @main(%choice: core.i32, %callee: core.func(core.never, core.nil), %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        func.tail_call_indirect %callee, %unit {signature = core.func(core.never, core.nil), tribute.calling_convention = 2}
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
  func.func @main(%choice: core.i32, %cond: core.i1, %callee: core.func(core.never, core.nil), %unit: core.nil) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %choice {
      scf.case {value = 0} {
        %never = scf.if %cond : core.never {
          func.unreachable
        } {
          func.tail_call_indirect %callee, %unit {signature = core.func(core.never, core.nil), tribute.calling_convention = 2}
        }
      }
      scf.default {
        func.tail_call_indirect %callee, %unit {signature = core.func(core.never, core.nil), tribute.calling_convention = 2}
      }
    }
  }
}"#,
        );

        assert_no_scf_switch_wrappers(&output);
        assert!(output.contains("wasm.i32_eq"), "{output}");
        assert_eq!(output.matches("wasm.if").count(), 2, "{output}");
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
}
