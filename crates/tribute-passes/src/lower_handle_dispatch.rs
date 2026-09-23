//! Lower resultless `ability.handle_dispatch` delimiters.
//!
//! The final resultless form is structurally spliced after evidence resolution.
//! As the final shared ability conversion, this pass establishes the `ability-lowered` boundary.
//!
//! Uses `PatternApplicator` for declarative op-level rewriting.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern,
    RewriteScope, TypeConverter,
};

use tribute_ir::dialect::ability;

const ABILITY_LOWERED_BOUNDARY: &str = "ability-lowered";

/// Conversion target for IR after shared ability lowering.
pub fn ability_lowered_target() -> ConversionTarget {
    ConversionTarget::new().illegal_dialect("ability")
}

/// Lower resultless handle delimiters and establish the ability boundary.
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

/// Splice a resultless delimiter after its evidence argument has been resolved.
struct LowerHandleDispatchPattern;

impl RewritePattern for LowerHandleDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if let Ok(dispatch_op) = ability::HandleDispatch::from_op(ctx, op) {
            let body = dispatch_op.body(ctx);
            let blocks = ctx.region(body).blocks.to_vec();
            let [body_block] = blocks.as_slice() else {
                return false;
            };

            // `resolve_evidence` has already replaced the body's evidence
            // argument. The final delimiter is resultless, so exhausting it is
            // a structural splice only: no carrier inspection and no returned
            // control value are involved.
            if ctx
                .block_args(*body_block)
                .iter()
                .any(|arg| !ctx.uses(*arg).is_empty())
            {
                return false;
            }
            let body_ops = ctx.block(*body_block).ops.to_vec();
            for body_op in body_ops {
                ctx.detach_op(body_op);
                rewriter.insert_op(body_op);
            }
            rewriter.erase_op(vec![]);
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Symbol;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::LegalityCheck;

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
  func.func @selected() -> core.never {
    %ev = arith.const {value = 0} : core.ptr
    %prompt = arith.const {value = 1} : core.i32
    ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      func.tail_call {callee = @finish}
    }
  }
  func.func @untouched() -> core.never {
    %ev = arith.const {value = 0} : core.ptr
    %prompt = arith.const {value = 2} : core.i32
    ability.handle_dispatch %ev, %prompt {ability_refs = []} {
      func.tail_call {callee = @finish}
    }
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
        assert!(trunk_ir::printer::print_op(&ctx, selected.op_ref()).contains("func.tail_call"));
        assert!(
            !trunk_ir::printer::print_op(&ctx, selected.op_ref())
                .contains("ability.handle_dispatch")
        );
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

    #[test]
    fn malformed_final_delimiters_are_not_structurally_spliced() {
        let malformed = [
            r#"ability.handle_dispatch %ev {ability_refs = []} {
      ^first(%inner: !evidence):
        func.unreachable
      ^second:
        func.unreachable
    }"#,
            r#"ability.handle_dispatch %ev {ability_refs = []} {
      ^body(%inner: !evidence):
        test.consume %inner
        func.unreachable
    }"#,
        ];
        for operation in malformed {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !marker = adt.struct() {{fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker}}
  !evidence = core.array(!marker)
  func.func @run(%ev: !evidence) -> core.never {{
    {operation}
  }}
}}"#
                ),
            );
            let before = print_module(&ctx, module.op());
            let error = lower_handle_dispatch(&mut ctx, module).unwrap_err();
            assert_eq!(error.boundary(), "ability-lowered");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }
}
