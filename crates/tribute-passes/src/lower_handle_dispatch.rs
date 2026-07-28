//! Lower `ability.handle_dispatch` to inline done-handler application.
//!
//! In the tail-call CPS design, effect operations are handled via tail calls
//! to handler_dispatch closures (see `lower_ability_perform`). By the time
//! `ability.handle_dispatch` is reached, the body result is already the final
//! value. This pass applies the done handler to the body result and, as the
//! final shared ability conversion, establishes the `ability-lowered` boundary.
//!
//! Uses `PatternApplicator` for declarative op-level rewriting.

use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, core, func, scf};
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern,
    RewriteScope, TypeConverter,
};
use trunk_ir::types::{Location, TypeDataBuilder};

use tribute_ir::dialect::ability;

const ABILITY_LOWERED_BOUNDARY: &str = "ability-lowered";

fn anyref_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("anyref")).build())
}

fn i1_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build())
}

fn i32_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Conversion target for IR after shared ability lowering.
pub fn ability_lowered_target() -> ConversionTarget {
    ConversionTarget::new().illegal_dialect("ability")
}

/// Lower all `ability.handle_dispatch` ops and establish the ability boundary.
///
/// The final partial conversion rejects every residual `ability.*` operation
/// while allowing unknown operations owned by later lowering stages.
pub(crate) fn lower_handle_dispatch(
    ctx: &mut IrContext,
    scope: impl RewriteScope,
) -> Result<(), ConversionError> {
    let applicator = PatternApplicator::new(TypeConverter::new())
        .with_target(ability_lowered_target())
        .add_pattern(LowerHandleDispatchPattern);
    applicator.apply_partial_conversion(ctx, scope, ABILITY_LOWERED_BOUNDARY)?;
    Ok(())
}

/// PassManager-friendly wrapper for [`lower_handle_dispatch`].
pub struct LowerHandleDispatch;

impl Pass for LowerHandleDispatch {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-handle-dispatch"
    }

    fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
        lower_handle_dispatch(ctx, target).map_err(Into::into)
    }
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
        // operand[0] = body result and is proven by frontend construction to
        // be the private #815 carrier. owner_tag is a dynamic i32 token, never
        // a source value or a syntactic prompt tag.
        let body_result = ctx.op_operands(op)[0];
        let owner_tag = dispatch_op.owner_tag(ctx);
        let user_result_ty = dispatch_op.result_type(ctx);
        let handler_body = dispatch_op.body(ctx);
        let escape_body = dispatch_op.escape(ctx);

        // The frontend proves that this operand is the private #815 carrier:
        // the body normal continuation constructs `Normal`, and general op
        // arms construct `Escape`. Never apply these probes to a source SSA
        // value or a public ADT.
        let done_region = get_done_region(ctx, handler_body);
        let control_ty = ability::cps_control_type_ref(ctx);
        let i1_ty = i1_type(ctx);
        let normal = adt::variant_is(
            ctx,
            location,
            body_result,
            i1_ty,
            control_ty,
            Symbol::new(ability::CPS_CONTROL_NORMAL_VARIANT),
        );
        rewriter.insert_op(normal.op_ref());

        let normal_region = completion_region(
            ctx,
            location,
            body_result,
            control_ty,
            Symbol::new(ability::CPS_CONTROL_NORMAL_VARIANT),
            user_result_ty,
            done_region,
        );
        let escape_region = escape_region(
            ctx,
            location,
            body_result,
            control_ty,
            owner_tag,
            user_result_ty,
            escape_body,
        );
        let dispatch = scf::r#if(
            ctx,
            location,
            normal.result(ctx),
            user_result_ty,
            normal_region,
            escape_region,
        );
        rewriter.insert_op(dispatch.op_ref());
        rewriter.erase_op(vec![dispatch.result(ctx)]);
        true
    }
}

/// Build the Normal branch. Its carrier proof comes from the matching
/// `ability.handle_dispatch` operand and the preceding private variant test.
fn completion_region(
    ctx: &mut IrContext,
    location: Location,
    carrier: ValueRef,
    control_ty: trunk_ir::refs::TypeRef,
    tag: Symbol,
    user_result_ty: trunk_ir::refs::TypeRef,
    done_body: Option<RegionRef>,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let anyref_ty = anyref_type(ctx);
    let cast = adt::variant_cast(ctx, location, carrier, anyref_ty, control_ty, tag);
    ctx.push_op(block, cast.op_ref());
    let payload = adt::variant_get(
        ctx,
        location,
        cast.result(ctx),
        anyref_ty,
        control_ty,
        tag,
        0,
    );
    ctx.push_op(block, payload.op_ref());
    let payload = payload.result(ctx);
    let result = if let Some(done_body) = done_body {
        inline_done_body(ctx, block, done_body, payload)
    } else {
        payload
    };
    let result = cast_result_if_needed(ctx, block, location, result, user_result_ty);
    let yield_op = scf::r#yield(ctx, location, [result]);
    ctx.push_op(block, yield_op.op_ref());
    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build the Escape branch. A foreign owner returns the exact carrier without
/// entering a source continuation or `do` region; the matching owner inlines
/// the frontend-built completion region.
fn escape_region(
    ctx: &mut IrContext,
    location: Location,
    carrier: ValueRef,
    control_ty: trunk_ir::refs::TypeRef,
    owner_tag: ValueRef,
    user_result_ty: trunk_ir::refs::TypeRef,
    escape_body: RegionRef,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let anyref_ty = anyref_type(ctx);
    let i32_ty = i32_type(ctx);
    let i1_ty = i1_type(ctx);
    let cast = adt::variant_cast(
        ctx,
        location,
        carrier,
        anyref_ty,
        control_ty,
        Symbol::new(ability::CPS_CONTROL_ESCAPE_VARIANT),
    );
    ctx.push_op(block, cast.op_ref());
    let owner = adt::variant_get(
        ctx,
        location,
        cast.result(ctx),
        i32_ty,
        control_ty,
        Symbol::new(ability::CPS_CONTROL_ESCAPE_VARIANT),
        0,
    );
    ctx.push_op(block, owner.op_ref());
    let payload = adt::variant_get(
        ctx,
        location,
        cast.result(ctx),
        anyref_ty,
        control_ty,
        Symbol::new(ability::CPS_CONTROL_ESCAPE_VARIANT),
        1,
    );
    ctx.push_op(block, payload.op_ref());
    let same_owner = trunk_ir::dialect::arith::cmpi(
        ctx,
        location,
        owner.result(ctx),
        owner_tag,
        i1_ty,
        Symbol::new("eq"),
    );
    ctx.push_op(block, same_owner.op_ref());

    let own_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let own_result = inline_done_body(ctx, own_block, escape_body, payload.result(ctx));
    let own_result = cast_result_if_needed(ctx, own_block, location, own_result, user_result_ty);
    let own_yield = scf::r#yield(ctx, location, [own_result]);
    ctx.push_op(own_block, own_yield.op_ref());
    let own_region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![own_block],
        parent_op: None,
    });

    let foreign_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    let foreign = cast_result_if_needed(ctx, foreign_block, location, carrier, user_result_ty);
    let foreign_yield = scf::r#yield(ctx, location, [foreign]);
    ctx.push_op(foreign_block, foreign_yield.op_ref());
    let foreign_region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![foreign_block],
        parent_op: None,
    });

    let dispatch = scf::r#if(
        ctx,
        location,
        same_owner.result(ctx),
        user_result_ty,
        own_region,
        foreign_region,
    );
    ctx.push_op(block, dispatch.op_ref());
    let yield_op = scf::r#yield(ctx, location, [dispatch.result(ctx)]);
    ctx.push_op(block, yield_op.op_ref());
    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

fn cast_result_if_needed(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    result: ValueRef,
    user_result_ty: trunk_ir::refs::TypeRef,
) -> ValueRef {
    if ctx.value_ty(result) == user_result_ty {
        result
    } else {
        let cast = core::unrealized_conversion_cast(ctx, location, result, user_result_ty);
        ctx.push_op(block, cast.op_ref());
        cast.result(ctx)
    }
}

/// Get the done region from handler_dispatch's body.
///
/// Finds the first `ability.done` child op and returns its body region.
fn get_done_region(ctx: &IrContext, body: RegionRef) -> Option<RegionRef> {
    let blocks = &ctx.region(body).blocks;
    let &first_block = blocks.first()?;

    for &op in &ctx.block(first_block).ops {
        if let Ok(done_op) = ability::Done::from_op(ctx, op) {
            return Some(done_op.body(ctx));
        }
    }

    None
}

/// Inline the `ability.done` region's body before `insert_before`.
///
/// The done region has a single block argument (the body result value).
/// We map that argument to `done_value` and clone the ops into `dest_block`.
/// `scf.yield` terminators are skipped — their operand becomes the result.
fn inline_done_body(
    ctx: &mut IrContext,
    dest_block: BlockRef,
    done_body: trunk_ir::refs::RegionRef,
    done_value: ValueRef,
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
        ctx.push_op(dest_block, cloned);
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
    use trunk_ir::Symbol;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::LegalityCheck;

    /// Basic handle_dispatch with a done handler that passes through the result.
    #[test]
    fn test_lower_handle_dispatch_identity_done() {
        let mut ctx = IrContext::new();

        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !__tribute_cps_control = adt.enum() {name = @__tribute_cps_control, variants = [[@Normal, [tribute_rt.anyref]], [@Escape, [core.i32, tribute_rt.anyref]]]}
  func.func @run() -> tribute_rt.anyref {
    %value = arith.const {value = 42} : tribute_rt.anyref
    %body = adt.variant_new %value {tag = @Normal, type = !__tribute_cps_control} : tribute_rt.anyref
    %owner = arith.const {value = 1} : core.i32
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %tr_dispatch_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %owner, %handler_fn, %tr_dispatch_fn {tag = 1, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      ability.done {
        ^bb0(%v: tribute_rt.anyref):
          scf.yield %v
      }
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: tribute_rt.anyref, %sv: tribute_rt.anyref):
          scf.yield %k
      }
    } {
      ^bb0(%v: tribute_rt.anyref):
        scf.yield %v
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module).unwrap();

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
  !__tribute_cps_control = adt.enum() {name = @__tribute_cps_control, variants = [[@Normal, [tribute_rt.anyref]], [@Escape, [core.i32, tribute_rt.anyref]]]}
  func.func @run() -> core.i32 {
    %value = arith.const {value = 10} : tribute_rt.anyref
    %body = adt.variant_new %value {tag = @Normal, type = !__tribute_cps_control} : tribute_rt.anyref
    %owner = arith.const {value = 1} : core.i32
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %tr_dispatch_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %owner, %handler_fn, %tr_dispatch_fn {tag = 1, result_type = core.i32} : core.i32 {
      ability.done {
        ^bb0(%v: tribute_rt.anyref):
          %one = arith.const {value = 1} : core.i32
          %cast = core.unrealized_conversion_cast %v : core.i32
          %sum = arith.addi %cast, %one : core.i32
          scf.yield %sum
      }
    } {
      ^bb0(%v: tribute_rt.anyref):
        %cast = core.unrealized_conversion_cast %v : core.i32
        scf.yield %cast
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module).unwrap();

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
  !__tribute_cps_control = adt.enum() {name = @__tribute_cps_control, variants = [[@Normal, [tribute_rt.anyref]], [@Escape, [core.i32, tribute_rt.anyref]]]}
  func.func @run() -> tribute_rt.anyref {
    %value = arith.const {value = 42} : tribute_rt.anyref
    %body = adt.variant_new %value {tag = @Normal, type = !__tribute_cps_control} : tribute_rt.anyref
    %owner = arith.const {value = 1} : core.i32
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %tr_dispatch_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %owner, %handler_fn, %tr_dispatch_fn {tag = 1, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      ability.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: tribute_rt.anyref, %sv: tribute_rt.anyref):
          scf.yield %k
      }
    } {
      ^bb0(%v: tribute_rt.anyref):
        scf.yield %v
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module).unwrap();

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }

    /// A different dynamic owner is not a completed source result. The final
    /// handle pass must forward that exact private carrier without entering
    /// either completion region.
    #[test]
    fn test_lower_handle_dispatch_forwards_foreign_escape() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !__tribute_cps_control = adt.enum() {name = @__tribute_cps_control, variants = [[@Normal, [tribute_rt.anyref]], [@Escape, [core.i32, tribute_rt.anyref]]]}
  func.func @run() -> tribute_rt.anyref {
    %payload = arith.const {value = 42} : tribute_rt.anyref
    %foreign_owner = arith.const {value = 2} : core.i32
    %body = adt.variant_new %foreign_owner, %payload {tag = @Escape, type = !__tribute_cps_control} : tribute_rt.anyref
    %owner = arith.const {value = 1} : core.i32
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %tr_dispatch_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %owner, %handler_fn, %tr_dispatch_fn {tag = 1, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      ability.done {
        ^bb0(%v: tribute_rt.anyref):
          scf.yield %v
      }
    } {
      ^bb0(%v: tribute_rt.anyref):
        %marker = arith.const {value = 99} : tribute_rt.anyref
        scf.yield %marker
    }
    func.return %result
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module).unwrap();

        let ir_text = print_module(&ctx, module.op());
        // The reviewed snapshot verifies that the foreign branch forwards the
        // original Escape carrier rather than its payload or a done-arm result.
        assert_snapshot!(ir_text);
    }

    #[test]
    fn final_conversion_allows_unknown_non_ability_ops() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run() -> core.i32 {
    %value = arith.const {value = 42} : core.i32
    func.return %value
  }
}"#,
        );

        lower_handle_dispatch(&mut ctx, module).unwrap();
    }

    #[test]
    fn function_scope_converts_only_selected_function() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !__tribute_cps_control = adt.enum() {name = @__tribute_cps_control, variants = [[@Normal, [tribute_rt.anyref]], [@Escape, [core.i32, tribute_rt.anyref]]]}
  func.func @selected() -> tribute_rt.anyref {
    %value = arith.const {value = 42} : tribute_rt.anyref
    %body = adt.variant_new %value {tag = @Normal, type = !__tribute_cps_control} : tribute_rt.anyref
    %owner = arith.const {value = 1} : core.i32
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %tr_dispatch_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %owner, %handler_fn, %tr_dispatch_fn {tag = 1, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      ability.done {
        ^bb0(%v: tribute_rt.anyref):
          scf.yield %v
      }
    } {
      ^bb0(%v: tribute_rt.anyref):
        scf.yield %v
    }
    func.return %result
  }
  func.func @untouched() -> tribute_rt.anyref {
    %value = arith.const {value = 7} : tribute_rt.anyref
    %body = adt.variant_new %value {tag = @Normal, type = !__tribute_cps_control} : tribute_rt.anyref
    %owner = arith.const {value = 2} : core.i32
    %handler_fn = arith.const {value = 0} : tribute_rt.anyref
    %tr_dispatch_fn = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.handle_dispatch %body, %owner, %handler_fn, %tr_dispatch_fn {tag = 2, result_type = tribute_rt.anyref} : tribute_rt.anyref {
      ability.done {
        ^bb0(%v: tribute_rt.anyref):
          scf.yield %v
      }
    } {
      ^bb0(%v: tribute_rt.anyref):
        scf.yield %v
    }
    func.return %result
  }
}"#,
        );
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected function");

        lower_handle_dispatch(&mut ctx, selected).unwrap();

        let ir_text = print_module(&ctx, module.op());
        assert_eq!(ir_text.matches("ability.handle_dispatch").count(), 1);
        assert!(ir_text.contains("func.func @untouched"));
        assert!(ir_text.contains("tag = 2"));
    }

    #[test]
    fn final_conversion_reports_residual_ability_op() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%k: tribute_rt.anyref) -> tribute_rt.anyref {
    %result = ability.perform %k {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : tribute_rt.anyref
    func.return %result
  }
}"#,
        );

        let error = lower_handle_dispatch(&mut ctx, module)
            .expect_err("residual ability operation should fail final conversion");

        assert_eq!(error.boundary(), "ability-lowered");
        assert_eq!(error.operations().len(), 1);
        assert_eq!(error.operations()[0].dialect, Symbol::new("ability"));
        assert_eq!(error.operations()[0].name, Symbol::new("perform"));
        assert_eq!(error.operations()[0].legality, LegalityCheck::Illegal);
    }
}
