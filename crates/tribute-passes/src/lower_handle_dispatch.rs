//! Lower `ability.handle_dispatch` to inline done-handler application.
//!
//! In the tail-call CPS design, effect operations are handled via tail calls
//! to handler_dispatch closures (see `lower_ability_perform`). By the time
//! `ability.handle_dispatch` is reached, the body result is already the final
//! value. This pass simply applies the done handler to the body result.
//!
//! Uses `PatternApplicator` for declarative op-level rewriting.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{cont, core, scf};
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Location;

use tribute_ir::dialect::ability;

/// Lower all `ability.handle_dispatch` ops in the module.
pub fn lower_handle_dispatch(ctx: &mut IrContext, module: Module) {
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(LowerHandleDispatchPattern);
    applicator.apply_partial(ctx, module);
}

/// Pattern: `ability.handle_dispatch` → inline done handler body.
struct LowerHandleDispatchPattern;

impl RewritePattern for LowerHandleDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = ability::HandleDispatch::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        // operand[0] = body result (anyref), operand[1] = handler_fn (unused here)
        let body_result = ctx.op_operands(op)[0];
        let user_result_ty = dispatch_op.result_type(ctx);
        let handler_body = dispatch_op.body(ctx);

        // In the tail-call CPS design, the body result is the final value
        // (effects are handled via tail calls, not YieldResult dispatch).
        // Just apply the done handler to the body result.
        let done_region = get_done_region(ctx, handler_body);

        let block = ctx.op(op).parent_block.unwrap();
        let final_result = if let Some(done_body) = done_region {
            inline_done_body(ctx, block, location, done_body, body_result, op)
        } else {
            body_result
        };

        // Cast to user result type if needed
        let result_val = if ctx.value_ty(final_result) != user_result_ty {
            let cast =
                core::unrealized_conversion_cast(ctx, location, final_result, user_result_ty);
            rewriter.insert_op(cast.op_ref());
            cast.result(ctx)
        } else {
            final_result
        };

        rewriter.erase_op(vec![result_val]);
        true
    }
}

/// Get the done region from handler_dispatch's body.
///
/// Finds the first `cont.done` child op and returns its body region.
fn get_done_region(ctx: &IrContext, body: RegionRef) -> Option<RegionRef> {
    let blocks = &ctx.region(body).blocks;
    let &first_block = blocks.first()?;

    for &op in &ctx.block(first_block).ops {
        if let Ok(done_op) = cont::Done::from_op(ctx, op) {
            return Some(done_op.body(ctx));
        }
    }

    None
}

/// Inline the `cont.done` region's body before `insert_before`.
///
/// The done region has a single block argument (the body result value).
/// We map that argument to `done_value` and clone the ops into `dest_block`.
/// `scf.yield` terminators are skipped — their operand becomes the result.
fn inline_done_body(
    ctx: &mut IrContext,
    dest_block: BlockRef,
    _location: Location,
    done_body: trunk_ir::refs::RegionRef,
    done_value: ValueRef,
    insert_before: OpRef,
) -> ValueRef {
    let done_blocks = &ctx.region(done_body).blocks;
    let Some(&done_block) = done_blocks.first() else {
        return done_value;
    };

    let mut mapping = IrMapping::new();
    let done_block_args = ctx.block_args(done_block).to_vec();
    if !done_block_args.is_empty() {
        mapping.map_value(done_block_args[0], done_value);
    }

    let mut final_result = done_value;
    let done_ops: Vec<OpRef> = ctx.block(done_block).ops.clone().to_vec();
    for &done_op in &done_ops {
        if scf::Yield::matches(ctx, done_op) {
            let yielded = ctx.op_operands(done_op).to_vec();
            if let Some(&result) = yielded.first() {
                final_result = mapping.lookup_value_or_default(result);
            }
            continue;
        }
        let cloned = ctx.clone_op(done_op, &mut mapping);
        ctx.insert_op_before(dest_block, insert_before, cloned);
        let cloned_results = ctx.op_results(cloned);
        if !cloned_results.is_empty() {
            final_result = cloned_results[0];
        }
    }

    final_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    /// Basic handle_dispatch with a done handler that passes through the result.
    #[test]
    fn test_lower_handle_dispatch_identity_done() {
        let mut ctx = IrContext::new();

        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run() -> tribute_rt.anyref {
    %body = arith.const {value = 42} : tribute_rt.anyref
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %handler_fn {tag = 1, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      cont.done {
        ^bb0(%v: tribute_rt.anyref):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: tribute_rt.anyref, %sv: tribute_rt.anyref):
          scf.yield %k
      }
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }

    /// Handle_dispatch with a done handler that transforms the result.
    #[test]
    fn test_lower_handle_dispatch_transforming_done() {
        let mut ctx = IrContext::new();

        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run() -> core.i32 {
    %body = arith.const {value = 10} : tribute_rt.anyref
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %handler_fn {tag = 1, result_type = core.i32} : core.i32 {
      cont.done {
        ^bb0(%v: tribute_rt.anyref):
          %one = arith.const {value = 1} : core.i32
          %cast = core.unrealized_conversion_cast %v : core.i32
          %sum = arith.add %cast, %one : core.i32
          scf.yield %sum
      }
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }

    /// Handle_dispatch without a done handler — body result passes through.
    #[test]
    fn test_lower_handle_dispatch_no_done() {
        let mut ctx = IrContext::new();

        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run() -> tribute_rt.anyref {
    %body = arith.const {value = 42} : tribute_rt.anyref
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %handler_fn {tag = 1, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: tribute_rt.anyref, %sv: tribute_rt.anyref):
          scf.yield %k
      }
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }
}
