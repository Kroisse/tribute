//! Lower the public prelude List construction intrinsic to shared `list.*` IR.

use tribute_ir::dialect::list;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

const PREPEND_INTRINSIC: &str = "List::__tribute_list_prepend_intrinsic";

fn is_prepend_intrinsic(name: Symbol) -> bool {
    name.with_str(|name| {
        name == PREPEND_INTRINSIC
            || name
                .strip_prefix(PREPEND_INTRINSIC)
                .is_some_and(|suffix| suffix.starts_with('$'))
    })
}

pub struct LowerListIntrinsics;

impl Pass for LowerListIntrinsics {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-list-intrinsics"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        PatternApplicator::new(TypeConverter::new())
            .add_pattern(PrependCallPattern)
            .add_pattern(PrependDeclarationPattern)
            .apply_partial(ctx, Module::from(target));
        Ok(())
    }
}

struct PrependCallPattern;

impl RewritePattern for PrependCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = func::Call::from_op(ctx, op) else {
            return false;
        };
        if !is_prepend_intrinsic(call.callee(ctx)) {
            return false;
        }
        let ([element, tail], [result_ty]) = (ctx.op_operands(op), ctx.op_result_types(op)) else {
            return false;
        };
        let (element, tail, result_ty) = (*element, *tail, *result_ty);
        let element_ty = ctx.value_ty(element);
        let prepend = list::prepend(
            ctx,
            ctx.op(op).location,
            element,
            tail,
            result_ty,
            element_ty,
        );
        rewriter.replace_op(prepend.op_ref());
        true
    }
}

struct PrependDeclarationPattern;

impl RewritePattern for PrependDeclarationPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            return false;
        };
        if !is_prepend_intrinsic(function.sym_name(ctx))
            || ctx.op(op).attributes.get_str("abi") != Some("intrinsic")
        {
            return false;
        }
        rewriter.erase_op(vec![]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn lowers_specialized_prepend_and_removes_declaration() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"List::__tribute_list_prepend_intrinsic"(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @caller(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"List::__tribute_list_prepend_intrinsic$String"} : tribute_rt.anyref
                    func.return %2
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerListIntrinsics.run(&mut ctx, core).unwrap();

        let output = print_module(&ctx, module.op());
        assert!(output.contains("list.prepend"), "{output}");
        assert!(
            !output.contains("List::__tribute_list_prepend_intrinsic"),
            "{output}"
        );
    }

    #[test]
    fn leaves_same_spelled_public_source_function_untouched() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"List::prepend"(%0: tribute_rt.int, %1: tribute_rt.int) -> tribute_rt.int {
                ^bb0:
                    func.return %0
                }
                func.func @caller(%0: tribute_rt.int, %1: tribute_rt.int) -> tribute_rt.int {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"List::prepend"} : tribute_rt.int
                    func.return %2
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerListIntrinsics.run(&mut ctx, core).unwrap();

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("list.prepend"), "{output}");
        assert!(output.contains("List::prepend"), "{output}");
    }
}
