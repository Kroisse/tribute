//! Lower cf dialect operations to clif dialect.
//!
//! This pass converts CFG-based control flow operations to Cranelift equivalents:
//! - `cf.br` -> `clif.jump`
//! - `cf.cond_br` -> `clif.brif`
//! - `cf.switch` -> `clif.switch`

use trunk_ir::OperationDataBuilder;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{cf, clif, core};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};

/// Lower cf dialect to clif dialect.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(CfBrPattern)
        .add_pattern(CfCondBrPattern)
        .add_pattern(CfSwitchPattern)
        .with_target(cf_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "cf-to-clif")?;
    Ok(())
}

fn cf_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("cf")
}

/// Pattern: `cf.br` -> `clif.jump`
struct CfBrPattern;

impl RewritePattern for CfBrPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if cf::Br::from_op(ctx, op).is_err() {
            return false;
        }

        let new_op = rebuild_op_as(ctx, op, Symbol::new("clif"), Symbol::new("jump"));
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `cf.cond_br` -> `clif.brif`
struct CfCondBrPattern;

impl RewritePattern for CfCondBrPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if cf::CondBr::from_op(ctx, op).is_err() {
            return false;
        }

        let new_op = rebuild_op_as(ctx, op, Symbol::new("clif"), Symbol::new("brif"));
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `cf.switch` -> `clif.switch`
///
/// A `cf.switch` case is a value of the discriminant's type. `clif.switch`
/// compares unsigned, so each case becomes its bit pattern at the
/// discriminant's width.
struct CfSwitchPattern;

impl RewritePattern for CfSwitchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(switch) = cf::Switch::from_op(ctx, op) else {
            return false;
        };
        let discriminant = switch.discriminant(ctx);
        let Some(width) = core::IntegerLike::width(ctx, ctx.value_ty(discriminant)) else {
            return false;
        };
        let Some(cases) = switch
            .cases(ctx)
            .map(|case| unsigned_case(width, case))
            .collect::<Option<Vec<u64>>>()
        else {
            return false;
        };
        let default = switch.default(ctx);
        let targets: trunk_ir::BlockList = switch.targets(ctx).collect();
        let location = ctx.op(op).location;
        let new_op = clif::Switch::operands(discriminant)
            .cases(cases)
            .successors(default, targets)
            .build(ctx, location);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// The unsigned value `clif.switch` compares for a `cf.switch` case on a
/// discriminant `width` bits wide: its bit pattern at that width.
///
/// A `clif.switch` case is a `u64`, so a negative case of a wider
/// discriminant, whose pattern has bits above the 64th, has none.
fn unsigned_case(width: u32, case: i64) -> Option<u64> {
    match width {
        0..64 => Some(case as u64 & ((1u64 << width) - 1)),
        64 => Some(case as u64),
        _ => u64::try_from(case).ok(),
    }
}

/// Rebuild an operation with a new dialect/name, transferring all operands,
/// results, attributes, regions, and successors from the original.
///
/// Regions are detached from the original operation and re-attached to the new one.
pub fn rebuild_op_as(ctx: &mut IrContext, op: OpRef, dialect: Symbol, name: Symbol) -> OpRef {
    let data = ctx.op(op);
    let loc = data.location;
    let attrs = data.attributes.clone();
    let regions: trunk_ir::RegionList = ctx.op_regions(op).collect();
    let successors: trunk_ir::BlockList = ctx.op_successors(op).collect();
    let operands: Vec<_> = ctx.op_operands(op).to_vec();
    let result_types: Vec<_> = ctx.op_result_types(op).to_vec();

    // Detach regions from the old operation so they can be owned by the new one
    for &r in &regions {
        ctx.detach_region(r);
    }

    let mut builder = OperationDataBuilder::new(loc, dialect, name)
        .operands(operands)
        .results(result_types);
    for (k, v) in attrs {
        builder = builder.attr(k, v);
    }
    for r in regions {
        builder = builder.region(r);
    }
    for s in successors {
        builder = builder.successor(s);
    }
    let data = builder.build(ctx);
    ctx.create_op(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn unsigned_cases_are_the_bit_patterns_clif_switch_can_hold() {
        assert_eq!(unsigned_case(8, -1), Some(255));
        assert_eq!(unsigned_case(64, -1), Some(u64::MAX));
        assert_eq!(unsigned_case(128, 7), Some(7));
        assert_eq!(unsigned_case(128, -1), None);
    }

    #[test]
    fn switch_cases_become_bit_patterns_of_the_discriminant_width() {
        for (discriminant_ty, expected) in [
            ("core.i8", "[7, 255]"),
            ("core.i32", "[7, 4294967295]"),
            ("core.i64", "[7, 18446744073709551615]"),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  clif.func {{sym_name = "select", type = clif.func_sig<({discriminant_ty}) -> ()>}} {{
    ^entry(%choice: {discriminant_ty}):
      cf.switch %choice [^fallback, ^first, ^second] {{cases = [7, -1]}}
    ^fallback:
      clif.return
    ^first:
      clif.return
    ^second:
      clif.return
  }}
}}"#
                ),
            );

            lower(&mut ctx, module, TypeConverter::new()).expect("cf.switch lowers");

            let printed = print_module(&ctx, module.op());
            assert!(
                printed.contains(&format!(
                    "clif.switch %0 [^bb1, ^bb2, ^bb3] {{cases = {expected}}}"
                )),
                "{printed}"
            );
        }
    }
}
