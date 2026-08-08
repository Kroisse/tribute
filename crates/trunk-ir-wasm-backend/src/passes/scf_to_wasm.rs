//! Lower scf dialect operations to wasm dialect (arena IR).
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - resultless `scf.switch` -> nested `wasm.if` comparisons
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=2)` (branch to outer block, past if and loop)

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::dialect::scf;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Attribute;

/// Lower scf dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(ScfIfPattern)
        .add_pattern(ScfSwitchPattern)
        .add_pattern(ScfLoopPattern)
        .add_pattern(ScfYieldPattern)
        .add_pattern(ScfContinuePattern)
        .add_pattern(ScfBreakPattern);
    applicator.apply_partial(ctx, module);
}

/// A fully validated `scf.switch` shape, collected before any region is
/// detached.  Post-CPS switches are resultless and their arms end in proper
/// tail transfers, which maps directly to nested resultless `wasm.if`s.
struct SwitchShape {
    discriminant: ValueRef,
    discriminant_ty: trunk_ir::TypeRef,
    cases: Vec<(i32, RegionRef)>,
    default: RegionRef,
}

/// Pattern for resultless `scf.switch` -> source-order nested `wasm.if`.
struct ScfSwitchPattern;

impl RewritePattern for ScfSwitchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !scf::Switch::matches(ctx, op) {
            return false;
        }
        let Some(shape) = validate_switch(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        for (_, region) in &shape.cases {
            ctx.detach_region(*region);
        }
        ctx.detach_region(shape.default);

        if let Some(((value, then_region), remaining_cases)) = shape.cases.split_first() {
            let else_region = build_switch_chain(
                ctx,
                loc,
                shape.discriminant,
                shape.discriminant_ty,
                remaining_cases,
                shape.default,
            );
            let (constant, condition) =
                switch_condition(ctx, loc, shape.discriminant, shape.discriminant_ty, *value);
            rewriter.insert_op(constant);
            rewriter.insert_op(condition);
            let nil = core::nil(ctx).as_type_ref();
            let wasm_if = wasm_dialect::r#if(
                ctx,
                loc,
                ctx.op_results(condition)[0],
                nil,
                *then_region,
                else_region,
            );
            // `wasm.if` carries a logical nil result even when its physical
            // block type is empty, whereas resultless `scf.switch` has no IR
            // result slot.  Insert then erase rather than asking the generic
            // replacement API to RAUW incompatible result vectors.
            rewriter.insert_op(wasm_if.op_ref());
            rewriter.erase_op(vec![]);
        } else {
            // A default-only switch is the degenerate structured chain.  Its
            // sole arm has already been validated as a zero-argument block,
            // so splice its operations in place without introducing a CFG.
            let default_block = ctx.region(shape.default).blocks[0];
            let default_ops = ctx.block(default_block).ops.to_vec();
            for default_op in default_ops {
                ctx.detach_op(default_op);
                rewriter.insert_op(default_op);
            }
            rewriter.erase_op(vec![]);
        }
        true
    }
}

fn validate_switch(ctx: &IrContext, op: OpRef) -> Option<SwitchShape> {
    let data = ctx.op(op);
    if ctx.op_operands(op).len() != 1 || !ctx.op_result_types(op).is_empty() {
        return None;
    }
    let discriminant = ctx.op_operands(op)[0];
    let discriminant_ty = ctx.value_ty(discriminant);
    let discriminant_data = ctx.types.get(discriminant_ty);
    if discriminant_data.dialect != Symbol::new("core")
        || discriminant_data.name != Symbol::new("i32")
        || data.regions.len() != 1
    {
        return None;
    }
    let body_blocks = &ctx.region(data.regions[0]).blocks;
    let [body] = body_blocks.as_slice() else {
        return None;
    };
    if !ctx.block_args(*body).is_empty() {
        return None;
    }

    let mut cases = Vec::new();
    let mut values = HashSet::new();
    let mut default = None;
    for &wrapper in &ctx.block(*body).ops {
        let wrapper_data = ctx.op(wrapper);
        let region = match wrapper_data.regions.as_slice() {
            [region] => *region,
            _ => return None,
        };
        let arm_blocks = &ctx.region(region).blocks;
        let [arm] = arm_blocks.as_slice() else {
            return None;
        };
        if !ctx.block_args(*arm).is_empty()
            || !ctx.op_operands(wrapper).is_empty()
            || !ctx.op_result_types(wrapper).is_empty()
        {
            return None;
        }
        if scf::Case::matches(ctx, wrapper) {
            let Attribute::Int(value) = wrapper_data.attributes.get(Symbol::new("value"))? else {
                return None;
            };
            let value = i32::try_from(*value).ok()?;
            if !values.insert(value) {
                return None;
            }
            cases.push((value, region));
        } else if scf::Default::matches(ctx, wrapper) {
            if default.replace(region).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(SwitchShape {
        discriminant,
        discriminant_ty,
        cases,
        default: default?,
    })
}

fn switch_condition(
    ctx: &mut IrContext,
    loc: trunk_ir::Location,
    discriminant: ValueRef,
    discriminant_ty: trunk_ir::TypeRef,
    value: i32,
) -> (OpRef, OpRef) {
    let constant = wasm_dialect::i32_const(ctx, loc, discriminant_ty, value);
    let comparison = wasm_dialect::i32_eq(
        ctx,
        loc,
        discriminant,
        constant.result(ctx),
        discriminant_ty,
    );
    (constant.op_ref(), comparison.op_ref())
}

fn build_switch_chain(
    ctx: &mut IrContext,
    loc: trunk_ir::Location,
    discriminant: ValueRef,
    discriminant_ty: trunk_ir::TypeRef,
    cases: &[(i32, RegionRef)],
    default: RegionRef,
) -> RegionRef {
    let mut fallback = default;
    for (value, then_region) in cases.iter().rev() {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let (constant, comparison) =
            switch_condition(ctx, loc, discriminant, discriminant_ty, *value);
        ctx.push_op(block, constant);
        ctx.push_op(block, comparison);
        let condition = ctx.op_results(comparison)[0];
        let nil = core::nil(ctx).as_type_ref();
        let nested = wasm_dialect::r#if(ctx, loc, condition, nil, *then_region, fallback);
        ctx.push_op(block, nested.op_ref());
        fallback = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
    }
    fallback
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
    use std::ops::ControlFlow;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::walk::{WalkAction, walk_op};

    fn lower_text(input: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        lower(&mut ctx, module, TypeConverter::new());
        let printed = print_module(&ctx, module.op());
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &printed);
        printed
    }

    fn count_ops(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> usize {
        let mut count = 0;
        let dialect = Symbol::from_dynamic(dialect);
        let name = Symbol::from_dynamic(name);
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect && data.name == name {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn lowers_ordered_resultless_switch_to_nested_wasm_ifs() {
        let input = r#"core.module @test {
  func.func @one() -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @two() -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @dispatch(%tag: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %tag {
      scf.case {value = 7} {
        func.tail_call {callee = @one, tribute.calling_convention = 2}
      }
      scf.case {value = 9} {
        func.tail_call {callee = @two, tribute.calling_convention = 2}
      }
      scf.default {
        func.unreachable
      }
    }
  }
}"#;

        let printed = lower_text(input);
        assert!(!printed.contains("scf.switch"), "{printed}");
        assert!(!printed.contains("scf.case"), "{printed}");
        assert!(!printed.contains("scf.default"), "{printed}");
        assert_eq!(printed.matches("wasm.if").count(), 2, "{printed}");
        assert_eq!(printed.matches("wasm.i32_eq").count(), 2, "{printed}");
        assert!(
            printed.find("wasm.i32_const {value = 7}").unwrap()
                < printed.find("wasm.i32_const {value = 9}").unwrap(),
            "case comparisons must retain source order:\n{printed}"
        );
        assert_eq!(printed.matches("func.tail_call").count(), 2, "{printed}");
        assert_eq!(printed.matches("func.unreachable").count(), 3, "{printed}");
    }

    #[test]
    fn lowers_default_only_switch_by_splicing_its_tail_arm() {
        let input = r#"core.module @test {
  func.func @dispatch(%tag: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %tag {
      scf.default {
        func.unreachable
      }
    }
  }
}"#;

        let printed = lower_text(input);
        assert!(!printed.contains("scf.switch"), "{printed}");
        assert_eq!(printed.matches("wasm.if").count(), 0, "{printed}");
        assert_eq!(printed.matches("func.unreachable").count(), 1, "{printed}");
    }

    #[test]
    fn malformed_switches_remain_byte_identical() {
        let malformed = [
            (
                r#"core.module @test {
  func.func @dispatch(%tag: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %tag {
      scf.case {value = 1} { func.unreachable }
    }
  }
}"#,
                "missing default",
            ),
            (
                r#"core.module @test {
  func.func @dispatch(%tag: core.i64) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %tag {
      scf.default { func.unreachable }
    }
  }
}"#,
                "non-i32 discriminant",
            ),
            (
                r#"core.module @test {
  func.func @dispatch(%tag: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    scf.switch %tag {
      scf.case {value = 1} { func.unreachable }
      scf.case {value = 1} { func.unreachable }
      scf.default { func.unreachable }
    }
  }
}"#,
                "duplicate case value",
            ),
        ];

        for (input, description) in malformed {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            let before = print_module(&ctx, module.op());
            lower(&mut ctx, module, TypeConverter::new());
            assert_eq!(
                print_module(&ctx, module.op()),
                before,
                "{description} must fail before mutation"
            );
            assert_eq!(count_ops(&ctx, module, "scf", "switch"), 1);
        }
    }
}
