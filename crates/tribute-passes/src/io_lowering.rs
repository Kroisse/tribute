//! Lower embedded standard-library I/O intrinsics to the shared I/O dialect.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

use tribute_ir::dialect::tribute_io;

const WRITE_INTRINSIC: &str = "std::io::__tribute_io_write";
const READ_LINE_INTRINSIC: &str = "std::io::__tribute_io_read_line";

pub struct LowerIoIntrinsics;

impl Pass for LowerIoIntrinsics {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-io-intrinsics"
    }

    fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
        let applicator = PatternApplicator::new(TypeConverter::new())
            .add_pattern(IoCallPattern)
            .add_pattern(IoDeclarationPattern);
        applicator.apply_partial(ctx, Module::from(target));
        Ok(())
    }
}

struct IoCallPattern;

impl RewritePattern for IoCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = func::Call::from_op(ctx, op) else {
            return false;
        };
        let callee = call.callee(ctx);
        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op).first().copied();
        let operands = ctx.op_operands(op).to_vec();

        let replacement = if callee == WRITE_INTRINSIC {
            let (Some(result_ty), [bytes, newline]) = (result_ty, operands.as_slice()) else {
                return false;
            };
            tribute_io::write(ctx, loc, *bytes, *newline, result_ty).op_ref()
        } else if callee == READ_LINE_INTRINSIC {
            let (Some(result_ty), []) = (result_ty, operands.as_slice()) else {
                return false;
            };
            tribute_io::read_line(ctx, loc, result_ty).op_ref()
        } else {
            return false;
        };

        rewriter.replace_op(replacement);
        true
    }
}

struct IoDeclarationPattern;

impl RewritePattern for IoDeclarationPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            return false;
        };
        let name = function.sym_name(ctx);
        if name != WRITE_INTRINSIC && name != READ_LINE_INTRINSIC {
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
    use trunk_ir::pass::Pass;
    use trunk_ir::printer::print_module;

    #[test]
    fn lowers_calls_and_removes_private_declarations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::io::__tribute_io_write"(%0: core.ptr, %1: core.i1) -> core.ptr
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @"std::io::__tribute_io_read_line"() -> core.ptr
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @caller(%0: core.ptr, %1: core.i1) -> core.ptr {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::io::__tribute_io_write"} : core.ptr
                    %3 = func.call {callee = @"std::io::__tribute_io_read_line"} : core.ptr
                    func.return %3
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerIoIntrinsics.run(&mut ctx, core).unwrap();

        let output = print_module(&ctx, module.op());
        assert!(output.contains("tribute_io.write"), "{output}");
        assert!(output.contains("tribute_io.read_line"), "{output}");
        assert!(!output.contains("__tribute_io_"), "{output}");
    }
}
