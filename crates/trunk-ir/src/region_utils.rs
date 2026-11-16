//! Queries over the values regions use, after MLIR's `RegionUtils`.

use std::ops::ControlFlow;

use rustc_hash::FxHashSet as HashSet;

use crate::context::IrContext;
use crate::refs::{RegionRef, ValueDef, ValueRef};
use crate::walk::{WalkAction, walk_region};

/// The values that operations in `regions`, nested ones included, use and
/// that are defined outside every region of `regions`, as MLIR's
/// `getUsedValuesDefinedAbove`.
///
/// Each value appears once, in order of first use: `regions` in order, each
/// walked in pre-order. A value defined in one region of `regions` and used
/// in another is not defined above them.
pub fn values_defined_above(ctx: &IrContext, regions: &[RegionRef]) -> Vec<ValueRef> {
    let mut seen = HashSet::default();
    let mut above = Vec::new();
    for &region in regions {
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            for &operand in ctx.op_operands(op) {
                if !is_defined_in(ctx, operand, regions) && seen.insert(operand) {
                    above.push(operand);
                }
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    above
}

/// Whether `value` is defined in one of `regions` or in a region nested in
/// one of them.
fn is_defined_in(ctx: &IrContext, value: ValueRef, regions: &[RegionRef]) -> bool {
    let block = match ctx.value_def(value) {
        ValueDef::OpResult(op, _) => ctx.op(op).parent_block,
        ValueDef::BlockArg(block, _) => Some(block),
    };
    let mut region = block.and_then(|block| ctx.block(block).parent_region);
    while let Some(current) = region {
        if regions.contains(&current) {
            return true;
        }
        region = ctx
            .region(current)
            .parent_op
            .and_then(|op| ctx.op(op).parent_block)
            .and_then(|block| ctx.block(block).parent_region);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::values_defined_above;
    use crate::IrContext;
    use crate::parser::parse_test_module;
    use crate::refs::{RegionRef, ValueRef};

    const INPUT: &str = r#"core.module @test {
  %a = "test.make"() : core.i32
  %b = "test.make"() : core.i32
  "test.lambda"() ({
  ^bb0(%x: core.i32):
    %c = "test.use"(%b, %x) : core.i32
    "test.nested"() ({
    ^bb0(%y: core.i32):
      "test.use"(%c, %a, %y, %b)
    })
  }, {
    "test.use"(%a)
    "test.use"(%b)
  })
}"#;

    /// The values `%a`, `%b`, and the two regions of `test.lambda`.
    fn parse(ctx: &mut IrContext) -> ([ValueRef; 2], [RegionRef; 2]) {
        let module = parse_test_module(ctx, INPUT);
        let ops = module.ops(ctx);
        let values = [ctx.op_result(ops[0], 0), ctx.op_result(ops[1], 0)];
        let regions = [ctx.op_region(ops[2], 0), ctx.op_region(ops[2], 1)]
            .map(|region| region.expect("test.lambda has two regions"));
        (values, regions)
    }

    #[test]
    fn values_used_in_a_region_and_its_nested_ones_in_first_use_order() {
        let mut ctx = IrContext::new();
        let ([a, b], [first, _]) = parse(&mut ctx);
        assert_eq!(values_defined_above(&ctx, &[first]), [b, a]);
    }

    #[test]
    fn values_used_in_several_regions_appear_once_in_region_order() {
        let mut ctx = IrContext::new();
        let ([a, b], [first, second]) = parse(&mut ctx);
        assert_eq!(values_defined_above(&ctx, &[second]), [a, b]);
        assert_eq!(values_defined_above(&ctx, &[first, second]), [b, a]);
        assert_eq!(values_defined_above(&ctx, &[second, first]), [a, b]);
    }

    #[test]
    fn a_value_of_an_enclosing_region_is_defined_above_a_nested_one() {
        let mut ctx = IrContext::new();
        let ([a, b], [first, _]) = parse(&mut ctx);
        let ops = ctx.block(ctx.region(first).blocks[0]).ops.clone();
        let c = ctx.op_result(ops[0], 0);
        let nested = ctx.op_region(ops[1], 0).expect("test.nested has a region");
        assert_eq!(values_defined_above(&ctx, &[nested]), [c, a, b]);
    }

    #[test]
    fn no_regions_use_no_values() {
        let ctx = IrContext::new();
        assert_eq!(values_defined_above(&ctx, &[]), []);
    }
}
