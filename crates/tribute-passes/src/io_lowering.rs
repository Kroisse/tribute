//! Lower embedded standard-library I/O intrinsics to the shared I/O dialect.

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, func};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
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

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let applicator = PatternApplicator::new(TypeConverter::new())
            .add_pattern(IoCallPattern)
            .add_pattern(IoDeclarationPattern);
        applicator.apply_partial(ctx, Module::from(target));
        Ok(())
    }
}

struct IoCallPattern;

impl OpRewritePattern for IoCallPattern {
    type Op = func::Call;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        call: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = call.op_ref();
        let callee = call.callee(ctx);
        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op).first().copied();

        let replacement = if callee == [WRITE_INTRINSIC] {
            let &[data, start, len, newline] = ctx.op_operands(op) else {
                return None;
            };
            // `write` takes its range as `core.index` and has no result: the
            // stub's `Nil` result is a nil value (`design/io.md`
            // § "Runtime Boundary").
            let index_ty = core::Index::type_ref(ctx);
            let [start, len] = [start, len].map(|position| {
                let cast = arith::IndexCastui::operands(position)
                    .results(index_ty)
                    .build(ctx, loc);
                rewriter.insert_op(cast.op_ref());
                cast.result(ctx)
            });
            let write = tribute_io::Write::operands(data, start, len, newline).build(ctx, loc);
            rewriter.insert_op(write.op_ref());
            let results = match result_ty {
                Some(_) => {
                    let nil = core::NilValue::operands().build(ctx, loc);
                    rewriter.insert_op(nil.op_ref());
                    vec![nil.result(ctx)]
                }
                None => vec![],
            };
            rewriter.erase_op(results);
            return Some(());
        } else if callee == [READ_LINE_INTRINSIC] {
            let (Some(result_ty), []) = (result_ty, ctx.op_operands(op)) else {
                return None;
            };
            tribute_io::ReadLine::operands()
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else {
            return None;
        };

        rewriter.replace_op(replacement);
        Some(())
    }
}

struct IoDeclarationPattern;

impl OpRewritePattern for IoDeclarationPattern {
    type Op = func::Func;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        function: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let name = function.sym_name(ctx);
        if name != WRITE_INTRINSIC && name != READ_LINE_INTRINSIC {
            return None;
        }

        rewriter.erase_op(vec![]);
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::pass::Pass;

    #[test]
    fn lowers_calls_and_removes_private_declarations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::io::__tribute_io_write"(%0: adt.array<core.i8>, %s: core.i32, %n: core.i32, %1: core.i1) -> core.nil
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @"std::io::__tribute_io_read_line"() -> core.ptr
                    attributes {abi = "intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @write(%0: adt.array<core.i8>, %s: core.i32, %n: core.i32, %1: core.i1) -> core.nil {
                ^bb0:
                    %2 = func.call @"std::io::__tribute_io_write"(%0, %s, %n, %1) : (adt.array<core.i8>, core.i32, core.i32, core.i1) -> core.nil
                    func.return %2
                }
                func.func @read() -> core.ptr {
                ^bb0:
                    %3 = func.call @"std::io::__tribute_io_read_line"() : () -> core.ptr
                    func.return %3
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerIoIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();

        let output = module.op().display(&ctx).to_string();
        let result = trunk_ir::validation::validate_op_schemas(&ctx, module.op());
        assert!(result.is_ok(), "{result}\n{output}");
        // The range widens to `core.index`, and a nil value stands for the
        // stub's result.
        assert_eq!(output.matches("arith.index_castui").count(), 2, "{output}");
        assert!(output.contains("tribute_io.write"), "{output}");
        assert!(output.contains("core.nil_value"), "{output}");
        assert!(output.contains("tribute_io.read_line"), "{output}");
        assert!(!output.contains("__tribute_io_"), "{output}");
    }

    #[test]
    fn leaves_malformed_and_unrelated_calls_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @other(%0: core.ptr) -> core.ptr {
                ^bb0:
                    func.return %0
                }
                func.func @caller(%0: core.ptr) -> core.ptr {
                ^bb0:
                    %1 = func.call @"std::io::__tribute_io_write"(%0) : (core.ptr) -> core.ptr
                    %2 = func.call @"std::io::__tribute_io_read_line"(%0) : (core.ptr) -> core.ptr
                    %3 = func.call @other(%0) : (core.ptr) -> core.ptr
                    func.return %3
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerIoIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();

        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("tribute_io."), "{output}");
        assert!(output.contains("std::io::__tribute_io_write"), "{output}");
        assert!(
            output.contains("std::io::__tribute_io_read_line"),
            "{output}"
        );
        assert!(output.contains("func.func @other"), "{output}");
    }
}
