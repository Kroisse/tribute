//! Lower the public prelude List construction intrinsic to shared `list.*` IR.

use rustc_hash::FxHashSet as HashSet;

use tribute_ir::dialect::list;
use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::{Symbol, SymbolPath};

const PREPEND_INTRINSIC: &str = "std::collections::List::__tribute_list_prepend_intrinsic";

/// Whether `name` is the prepend intrinsic or one of its specializations,
/// which append a `$` suffix.
fn is_prepend_intrinsic(name: &SymbolPath) -> bool {
    name.as_simple()
        .and_then(|name| name.as_str().strip_prefix(PREPEND_INTRINSIC))
        .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with('$'))
}

fn intrinsic_declaration(name: &SymbolPath) -> Option<SymbolPath> {
    is_prepend_intrinsic(name).then(|| SymbolPath::new([PREPEND_INTRINSIC]))
}

#[derive(Default)]
struct IntrinsicDeclarations {
    all: HashSet<SymbolPath>,
    eligible: HashSet<SymbolPath>,
}

pub struct LowerListIntrinsics;

impl Pass for LowerListIntrinsics {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-list-intrinsics"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let module = Module::from(target);
        let mut intrinsic_declarations = IntrinsicDeclarations::default();
        for &op in module.ops(ctx) {
            let Ok(function) = func::Func::from_op(ctx, op) else {
                continue;
            };
            let name = Symbol::new(function.sym_name(ctx));
            intrinsic_declarations
                .all
                .insert(SymbolPath::new([name.clone()]));
            if is_prepend_intrinsic(&SymbolPath::new([name.clone()]))
                && ctx.op_attrs(op).get_str(COMPILER_INTRINSIC_ATTR) == Some(PREPEND_INTRINSIC)
                && {
                    func::FuncSig::from_type_ref(ctx, function.r#type(ctx)).is_some_and(
                        |signature| {
                            let [_, _] = signature.inputs(ctx) else {
                                return false;
                            };
                            signature.single_result(ctx).is_some()
                        },
                    )
                        // Pre-CPS validation authenticates the exact semantic
                        // identity and complete concrete signature. Keep that
                        // signature intact for a specialized generic intrinsic;
                        // this target-neutral lowering only needs its callable
                        // arity and must not recreate trust from the symbol.
                        && ctx.op_attrs(op).get_str("abi") == Some("intrinsic")
                }
            {
                intrinsic_declarations
                    .eligible
                    .insert(SymbolPath::new([name]));
            }
        }

        PatternApplicator::new(TypeConverter::new())
            .add_pattern(PrependCallPattern {
                intrinsic_declarations: &intrinsic_declarations,
            })
            .add_pattern(PrependDeclarationPattern {
                intrinsic_declarations: &intrinsic_declarations,
            })
            .apply_partial(ctx, module);
        Ok(())
    }
}

struct PrependCallPattern<'a> {
    intrinsic_declarations: &'a IntrinsicDeclarations,
}

impl OpRewritePattern for PrependCallPattern<'_> {
    type Op = func::Call;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        call: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = call.op_ref();
        let callee = call.callee(ctx);
        let base_declaration = intrinsic_declaration(callee)?;
        // The frontend keeps generic extern declarations unmangled while
        // specializing their calls. An exact declaration still takes priority
        // so an ordinary same-spelled function cannot inherit the base ABI.
        let eligible = if self.intrinsic_declarations.all.contains(callee) {
            self.intrinsic_declarations.eligible.contains(callee)
        } else {
            self.intrinsic_declarations
                .eligible
                .contains(&base_declaration)
        };
        if !eligible {
            return None;
        }
        let ([element, tail], [_]) = (
            ctx.op_operands(op).as_array()?,
            ctx.op_result_types(op).as_array()?,
        );
        let (element, tail) = (*element, *tail);
        let element_ty = ctx.value_ty(element);
        let prepend = list::Prepend::operands(element, tail)
            .element_type(element_ty)
            .build(ctx, ctx.op(op).location);
        rewriter.replace_op(prepend.op_ref());
        Some(())
    }
}

struct PrependDeclarationPattern<'a> {
    intrinsic_declarations: &'a IntrinsicDeclarations,
}

impl OpRewritePattern for PrependDeclarationPattern<'_> {
    type Op = func::Func;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        function: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        if !self
            .intrinsic_declarations
            .eligible
            .contains(&SymbolPath::new([
                Symbol::new(function.sym_name(ctx)).clone()
            ]))
        {
            return None;
        }
        rewriter.erase_op(vec![]);
        Some(())
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
                func.func @"std::collections::List::__tribute_list_prepend_intrinsic"(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::collections::List::__tribute_list_prepend_intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @caller(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::collections::List::__tribute_list_prepend_intrinsic$String"} : tribute_rt.anyref
                    func.return %2
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerListIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();

        let output = print_module(&ctx, module.op());
        assert!(output.contains("list.prepend"), "{output}");
        assert!(
            !output.contains("std::collections::List::__tribute_list_prepend_intrinsic"),
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

        LowerListIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("list.prepend"), "{output}");
        assert!(output.contains("List::prepend"), "{output}");
    }

    #[test]
    fn leaves_same_spelled_hidden_source_function_untouched() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::collections::List::__tribute_list_prepend_intrinsic"(%0: tribute_rt.int, %1: tribute_rt.int) -> tribute_rt.int
                    attributes {tribute.compiler_intrinsic = "std::collections::List::__tribute_list_prepend_intrinsic"} {
                ^bb0:
                    func.return %0
                }
                func.func @caller(%0: tribute_rt.int, %1: tribute_rt.int) -> tribute_rt.int {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::collections::List::__tribute_list_prepend_intrinsic"} : tribute_rt.int
                    func.return %2
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerListIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("list.prepend"), "{output}");
        assert!(
            output.contains("std::collections::List::__tribute_list_prepend_intrinsic"),
            "{output}"
        );
    }

    #[test]
    fn exact_specialized_declaration_takes_priority_over_intrinsic_base() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
            core.module @test {
                func.func @"std::collections::List::__tribute_list_prepend_intrinsic"(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref
                    attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::collections::List::__tribute_list_prepend_intrinsic"} {
                ^bb0:
                    func.unreachable
                }
                func.func @"std::collections::List::__tribute_list_prepend_intrinsic$String"(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref {
                ^bb0:
                    func.return %0
                }
                func.func @caller(%0: tribute_rt.anyref, %1: tribute_rt.anyref) -> tribute_rt.anyref {
                ^bb0:
                    %2 = func.call %0, %1 {callee = @"std::collections::List::__tribute_list_prepend_intrinsic$String"} : tribute_rt.anyref
                    func.return %2
                }
            }
            "#,
        );
        let core = core::Module::from_op(&ctx, module.op()).expect("core.module");

        LowerListIntrinsics
            .run(&mut ctx, core, &mut Default::default())
            .unwrap();

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("list.prepend"), "{output}");
        assert!(output.contains("func.call"), "{output}");
        assert!(
            output.contains(
                r#"func.func @"std::collections::List::__tribute_list_prepend_intrinsic$String""#
            ),
            "{output}"
        );
        assert!(
            !output.contains(
                r#"func.func @"std::collections::List::__tribute_list_prepend_intrinsic"("#
            ),
            "{output}"
        );
    }
}
