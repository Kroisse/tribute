//! Target evidence lowering (arena-based).
//!
//! Evidence lowering runs inside the representation/ABI boundary on every
//! target. It lowers `effect.extend`, the evidence selection operations,
//! `effect.dispatch_tail`, and `effect.dispatch_cps` to calls of the evidence
//! runtime written in Tribute source, `std::runtime::evidence`
//! (`new-plans/cps-effects.md`), which it finds by canonical path. The runtime
//! functions are compiled like any other source function; marker
//! construction and field access stay inside them. The lowering is the same
//! on every target.

use std::ops::ControlFlow;
use std::rc::Rc;

use tribute_ir::dialect::ability;
use tribute_ir::dialect::adt;
use tribute_ir::dialect::effect;
use tribute_ir::runtime_items::{RuntimeFunction, RuntimeFunctionError, evidence};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::walk::{WalkAction, walk_op};

use crate::effect_dispatch;

/// The evidence runtime functions the module's `effect.*` operations call.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvidenceRuntime {
    lookup: Option<RuntimeFunction>,
    lookup_tr: Option<RuntimeFunction>,
    extend: Option<RuntimeFunction>,
    mask: Option<RuntimeFunction>,
    dup: Option<RuntimeFunction>,
    outer: Option<RuntimeFunction>,
    tail: Option<RuntimeFunction>,
    with_tail: Option<RuntimeFunction>,
    push: Option<RuntimeFunction>,
}

/// Find the evidence runtime functions that the module's `effect.*`
/// operations will call.
///
/// Run once on the module before [`LowerEvidence`]. A function no operation
/// needs is not looked up, so a missing one fails only when used.
pub(crate) fn prepare_evidence_runtime(
    ctx: &IrContext,
    module: Module,
) -> Result<EvidenceRuntime, RuntimeFunctionError> {
    let needs = evidence_runtime_requirements(ctx, module);
    let find = |needed: bool, path| {
        needed
            .then(|| RuntimeFunction::lookup(ctx, module, path))
            .transpose()
    };
    Ok(EvidenceRuntime {
        lookup: find(needs.lookup, evidence::LOOKUP)?,
        lookup_tr: find(needs.lookup_tr, evidence::LOOKUP_TR)?,
        extend: find(needs.extend, evidence::EXTEND)?,
        mask: find(needs.mask, evidence::MASK)?,
        dup: find(needs.dup, evidence::DUP)?,
        outer: find(needs.outer, evidence::OUTER)?,
        tail: find(needs.tail, evidence::TAIL)?,
        with_tail: find(needs.with_tail, evidence::WITH_TAIL)?,
        push: find(needs.push, evidence::PUSH)?,
    })
}

/// Lower evidence operations in one function body with the functions of
/// `runtime`, from [`prepare_evidence_runtime`] on the containing module.
pub(crate) fn lower_evidence_func(
    ctx: &mut IrContext,
    func_op: func::Func,
    runtime: &EvidenceRuntime,
) -> Result<(), ConversionError> {
    let runtime = Rc::new(*runtime);
    PatternApplicator::new(TypeConverter::new())
        .with_target(effect_abi_target())
        .add_pattern(LowerEffectExtend {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectStackOp {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectTailOp {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectDispatchTail {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectDispatchCps { runtime })
        .apply_partial_conversion(ctx, func_op, "evidence-effect-abi")?;
    Ok(())
}

/// Lower evidence operations in every function of the module with the
/// evidence runtime functions the module defines, then remove the ability
/// declarations, which nothing names afterwards.
pub struct LowerEvidence;

impl Pass for LowerEvidence {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-evidence"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let module = Module::from(target);
        let runtime = prepare_evidence_runtime(ctx, module)?;
        let mut functions = Vec::new();
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(function) = func::Func::from_op(ctx, op) {
                functions.push(function);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        for function in functions {
            lower_evidence_func(ctx, function, &runtime)?;
        }
        ability::remove_declarations(ctx, module);
        Ok(())
    }
}

fn effect_abi_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_op("func.func")
        .recursive_legal_op("func.func")
        .illegal_op("effect.extend")
        .illegal_op("effect.mask")
        .illegal_op("effect.dup")
        .illegal_op("effect.outer")
        .illegal_op("effect.tail")
        .illegal_op("effect.with_tail")
        .illegal_op("effect.push")
        .illegal_op("effect.dispatch_tail")
        .illegal_op("effect.dispatch_cps")
}

/// The runtime functions the module's `effect.*` operations call.
#[derive(Default)]
struct RuntimeRequirements {
    lookup: bool,
    lookup_tr: bool,
    extend: bool,
    mask: bool,
    dup: bool,
    outer: bool,
    tail: bool,
    with_tail: bool,
    push: bool,
}

fn evidence_runtime_requirements(ctx: &IrContext, module: Module) -> RuntimeRequirements {
    fn visit(ctx: &IrContext, region: RegionRef, needs: &mut RuntimeRequirements) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                needs.lookup |= effect::DispatchCps::matches(ctx, op);
                needs.lookup_tr |= effect::DispatchTail::matches(ctx, op);
                needs.extend |= effect::Extend::matches(ctx, op);
                needs.mask |= effect::Mask::matches(ctx, op);
                needs.dup |= effect::Dup::matches(ctx, op);
                needs.outer |= effect::Outer::matches(ctx, op);
                needs.tail |= effect::Tail::matches(ctx, op);
                needs.with_tail |= effect::WithTail::matches(ctx, op);
                needs.push |= effect::Push::matches(ctx, op);
                for nested in ctx.op_regions(op) {
                    visit(ctx, nested, needs);
                }
            }
        }
    }

    let mut needs = RuntimeRequirements::default();
    if let Some(body) = module.body(ctx) {
        visit(ctx, body, &mut needs);
    }
    needs
}

/// Call `function`, which [`prepare_evidence_runtime`] found because an
/// operation of this kind needs it, and replace `op` with the call's result.
fn replace_with_call(
    ctx: &mut IrContext,
    op: OpRef,
    function: Option<RuntimeFunction>,
    args: Vec<ValueRef>,
    rewriter: &mut PatternRewriter<'_>,
) {
    let function = function.expect("the evidence runtime was prepared for this operation");
    let location = ctx.op(op).location;
    let call = function.call(ctx, location, args);
    rewriter.insert_op(call.op_ref());
    rewriter.erase_op(vec![call.result(ctx)]);
}

/// `tr_dispatch_fn` as the `Some` of the runtime's `Option(Dispatch)`
/// parameter `option_ty`.
fn some_dispatch(
    ctx: &mut IrContext,
    location: Location,
    tr_dispatch_fn: ValueRef,
    option_ty: TypeRef,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let dispatch = if ctx.value_ty(tr_dispatch_fn) == closure_ty {
        tr_dispatch_fn
    } else {
        let cast = core::UnrealizedConversionCast::operands(tr_dispatch_fn)
            .results(closure_ty)
            .build(ctx, location);
        rewriter.insert_op(cast.op_ref());
        cast.result(ctx)
    };
    // The parameter is a reference to the published `Option` instance.
    let option_name = ctx
        .type_attrs(option_ty)
        .get_str("name")
        .expect("the Option(Dispatch) parameter is a nominal reference")
        .to_owned();
    let option_layout = ctx
        .type_alias_by_text(&option_name)
        .expect("the Option(Dispatch) layout is published");
    let some = adt::VariantNew::operands([dispatch])
        .r#type(option_layout)
        .tag("Some")
        .results(option_ty)
        .build(ctx, location);
    rewriter.insert_op(some.op_ref());
    some.result(ctx)
}

/// `effect.extend` → `std::runtime::evidence::extend`.
struct LowerEffectExtend {
    runtime: Rc<EvidenceRuntime>,
}

impl RewritePattern for LowerEffectExtend {
    fn match_and_rewrite(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(extend_op) = effect::Extend::from_op(ctx, op) else {
            return false;
        };
        let function = self
            .runtime
            .extend
            .expect("the evidence runtime was prepared for effect.extend");
        let loc = ctx.op(op).location;
        let ability_id = {
            let ability = extend_op.ability(ctx).clone();
            effect_dispatch::insert_ability_id(ctx, loc, &ability, rewriter)
        };
        let option_ty = function.signature(ctx).inputs(ctx)[3];
        let tr_dispatch =
            some_dispatch(ctx, loc, extend_op.tr_dispatch_fn(ctx), option_ty, rewriter);
        let args = vec![
            extend_op.evidence(ctx),
            ability_id,
            extend_op.prompt_tag(ctx),
            tr_dispatch,
            extend_op.outer(ctx),
        ];
        replace_with_call(ctx, op, Some(function), args, rewriter);
        true
    }
}

/// `effect.mask` / `effect.dup` / `effect.outer` → the runtime function of
/// the same name.
struct LowerEffectStackOp {
    runtime: Rc<EvidenceRuntime>,
}

impl RewritePattern for LowerEffectStackOp {
    fn match_and_rewrite(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let runtime = &self.runtime;
        let (function, ability, evidence) = if let Ok(mask) = effect::Mask::from_op(ctx, op) {
            (runtime.mask, mask.ability(ctx).clone(), mask.evidence(ctx))
        } else if let Ok(dup) = effect::Dup::from_op(ctx, op) {
            (runtime.dup, dup.ability(ctx).clone(), dup.evidence(ctx))
        } else if let Ok(outer) = effect::Outer::from_op(ctx, op) {
            (
                runtime.outer,
                outer.ability(ctx).clone(),
                outer.evidence(ctx),
            )
        } else {
            return false;
        };
        let loc = ctx.op(op).location;
        let ability_id = effect_dispatch::insert_ability_id(ctx, loc, &ability, rewriter);
        replace_with_call(ctx, op, function, vec![evidence, ability_id], rewriter);
        true
    }
}

/// `effect.tail` / `effect.with_tail` / `effect.push` → the runtime
/// function of the same name.
struct LowerEffectTailOp {
    runtime: Rc<EvidenceRuntime>,
}

impl RewritePattern for LowerEffectTailOp {
    fn match_and_rewrite(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let runtime = &self.runtime;
        let loc = ctx.op(op).location;
        let i32_ty = effect_dispatch::i32_type(ctx);
        let mut slot = |ctx: &mut IrContext, index: u32| {
            let slot = trunk_ir::dialect::arith::Const::operands()
                .value(Attribute::Int(i128::from(ability::tail_slot_id(index))))
                .results(i32_ty)
                .build(ctx, loc);
            rewriter.insert_op(slot.op_ref());
            slot.result(ctx)
        };
        let (function, args) = if let Ok(tail) = effect::Tail::from_op(ctx, op) {
            let slot = slot(ctx, tail.index(ctx));
            (runtime.tail, vec![tail.evidence(ctx), slot])
        } else if let Ok(with_tail) = effect::WithTail::from_op(ctx, op) {
            let slot = slot(ctx, with_tail.index(ctx));
            (
                runtime.with_tail,
                vec![with_tail.evidence(ctx), slot, with_tail.tail(ctx)],
            )
        } else if let Ok(push) = effect::Push::from_op(ctx, op) {
            let ability_id = {
                let ability = push.ability(ctx).clone();
                effect_dispatch::insert_ability_id(ctx, loc, &ability, rewriter)
            };
            (
                runtime.push,
                vec![push.evidence(ctx), push.source(ctx), ability_id],
            )
        } else {
            return false;
        };
        replace_with_call(ctx, op, function, args, rewriter);
        true
    }
}

/// `effect.dispatch_tail` → `std::runtime::evidence::lookup_tr` + indirect
/// call.
struct LowerEffectDispatchTail {
    runtime: Rc<EvidenceRuntime>,
}

impl RewritePattern for LowerEffectDispatchTail {
    fn match_and_rewrite(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchTail::from_op(ctx, op) else {
            return false;
        };
        if !effect_dispatch::is_valid_tail_dispatch(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;
        let ability_id = {
            let ability = dispatch_op.ability(ctx).clone();
            effect_dispatch::insert_ability_id(ctx, loc, &ability, rewriter)
        };
        let lookup_tr = self
            .runtime
            .lookup_tr
            .expect("the evidence runtime was prepared for effect.dispatch_tail");
        let dispatch_closure = lookup_tr.call(ctx, loc, [dispatch_op.evidence(ctx), ability_id]);
        rewriter.insert_op(dispatch_closure.op_ref());
        effect_dispatch::lower_tail_dispatch(ctx, op, dispatch_closure.result(ctx), rewriter);
        true
    }
}

/// `effect.dispatch_cps` → `std::runtime::evidence::lookup` + proper-tail
/// transfer.
struct LowerEffectDispatchCps {
    runtime: Rc<EvidenceRuntime>,
}

impl RewritePattern for LowerEffectDispatchCps {
    fn match_and_rewrite(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchCps::from_op(ctx, op) else {
            return false;
        };
        if !effect_dispatch::is_valid_cps_dispatch(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;
        let ability_id = {
            let ability = dispatch_op.ability(ctx).clone();
            effect_dispatch::insert_ability_id(ctx, loc, &ability, rewriter)
        };
        let lookup = self
            .runtime
            .lookup
            .expect("the evidence runtime was prepared for effect.dispatch_cps");
        let prompt = lookup.call(ctx, loc, [dispatch_op.evidence(ctx), ability_id]);
        rewriter.insert_op(prompt.op_ref());
        effect_dispatch::lower_cps_dispatch(ctx, op, ability_id, prompt.result(ctx), rewriter);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::SymbolPath;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    /// The evidence type and stand-ins for the runtime functions.
    const RUNTIME: &str = r#"  !Evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  !Closure = adt.struct<_closure(entries: tribute_rt.convention_table_ref), {layout = "closure"}>
  !"std::Option$std::runtime::Dispatch" = adt.enum<"std::Option$std::runtime::Dispatch" { None(), Some(!Closure) }>
  !OptionRef = adt.typeref<{name = "std::Option$std::runtime::Dispatch"}>
  func.func @"std::runtime::evidence::lookup"(%ev: !Evidence, %id: core.i32) -> core.i32 {
    func.return %id
  }
  func.func @"std::runtime::evidence::lookup_tr"(%ev: !Evidence, %id: core.i32) -> !Closure {
    %f = adt.ref_null {type = !Closure} : !Closure
    func.return %f
  }
  func.func @"std::runtime::evidence::extend"(%ev: !Evidence, %id: core.i32, %prompt: core.i32, %tr: !OptionRef, %outer: !Evidence) -> !Evidence {
    func.return %ev
  }
  func.func @"std::runtime::evidence::mask"(%ev: !Evidence, %id: core.i32) -> !Evidence {
    func.return %ev
  }
  func.func @"std::runtime::evidence::dup"(%ev: !Evidence, %id: core.i32) -> !Evidence {
    func.return %ev
  }"#;

    fn module_text(body: &str) -> String {
        format!("core.module @test {{\n{RUNTIME}\n{body}\n}}")
    }

    fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
        let runtime = prepare_evidence_runtime(ctx, module).expect("evidence runtime");
        for op in module.ops_snapshot(ctx) {
            if let Ok(function) = func::Func::from_op(ctx, op)
                && !function.sym_name(ctx).starts_with("std::runtime::")
            {
                lower_evidence_func(ctx, function, &runtime)?;
            }
        }
        Ok(())
    }

    fn lower_text(body: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(body));
        lower(&mut ctx, module).expect("evidence lowering");
        print_module(&ctx, module.op())
    }

    fn calls(ctx: &IrContext, module: Module) -> Vec<(SymbolPath, usize)> {
        let mut calls = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(call) = func::Call::from_op(ctx, op) {
                calls.push((call.callee(ctx).clone(), ctx.op_operands(op).len()));
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        calls
    }

    fn assert_shared_dialect_only(printed: &str) {
        assert!(!printed.contains("effect."), "{printed}");
        assert!(!printed.contains("tribute.calling_convention"), "{printed}");
    }

    const TAIL: &str = r#"  func.func @tail(%ev: !Evidence, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability = @Console, op_name = "read"} : tribute_rt.anyref
    func.return %result
  }"#;

    const CPS: &str = r#"  func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
    effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability = @State, op_name = "get", answer_type = core.i32}
  }"#;

    const EXTEND: &str = r#"  func.func @install(%ev: !Evidence, %prompt: core.i32, %tr: !Closure) -> !Evidence {
    %extended = effect.extend %ev, %prompt, %tr, %ev {ability = @State} : !Evidence
    func.return %extended
  }"#;

    const SELECT: &str = r#"  func.func @select(%ev: !Evidence) -> !Evidence {
    %masked = effect.mask %ev {ability = @State} : !Evidence
    %dup = effect.dup %masked {ability = @State} : !Evidence
    func.return %dup
  }"#;

    fn runtime_callee(name: &str) -> SymbolPath {
        SymbolPath::new([format!("std::runtime::evidence::{name}").as_str()])
    }

    #[test]
    fn mask_and_dup_call_the_runtime_with_the_ability_id() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(SELECT));
        lower(&mut ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());

        assert_shared_dialect_only(&printed);
        assert_eq!(
            calls(&ctx, module),
            [(runtime_callee("mask"), 2), (runtime_callee("dup"), 2)],
            "{printed}"
        );
    }

    #[test]
    fn tail_dispatch_calls_the_selected_closure_indirectly() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(TAIL));
        lower(&mut ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());

        assert_shared_dialect_only(&printed);
        assert_eq!(
            calls(&ctx, module),
            [(runtime_callee("lookup_tr"), 2)],
            "{printed}"
        );
        assert!(printed.contains("func.call_indirect"), "{printed}");
    }

    #[test]
    fn cps_dispatch_transfers_through_a_tail_signature() {
        let printed = lower_text(CPS);

        assert_shared_dialect_only(&printed);
        assert!(
            printed.contains(r#"callee = @"std::runtime::evidence::lookup"}"#),
            "{printed}"
        );
        assert!(printed.contains("func.tail_call_indirect"), "{printed}");
        assert!(printed.contains("call_conv = \"tail\""), "{printed}");
    }

    #[test]
    fn extend_passes_the_dispatch_closure_as_some() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(EXTEND));
        lower(&mut ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());

        assert_shared_dialect_only(&printed);
        assert_eq!(
            calls(&ctx, module),
            [(runtime_callee("extend"), 5)],
            "{printed}"
        );
        assert!(
            printed.contains(r#"adt.variant_new %2 {tag = "Some""#),
            "{printed}"
        );
    }

    #[test]
    fn a_missing_runtime_function_fails_only_when_needed() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &module_text(
                r#"  func.func @tail_slot(%ev: !Evidence) -> !Evidence {
    %tail = effect.tail %ev {index = 0} : !Evidence
    func.return %tail
  }"#,
            ),
        );
        let error = prepare_evidence_runtime(&ctx, module).expect_err("tail is missing");
        assert_eq!(error.path, "std::runtime::evidence::tail");

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(SELECT));
        assert!(prepare_evidence_runtime(&ctx, module).is_ok());
    }

    #[test]
    fn malformed_cps_dispatch_is_rejected_without_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &module_text(
                r#"  func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
    %bad = effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability = @State, op_name = "get", answer_type = core.i32} : core.i32
  }"#,
            ),
        );
        let runtime = prepare_evidence_runtime(&ctx, module).unwrap();
        let before = print_module(&ctx, module.op());
        let function = module
            .ops(&ctx)
            .iter()
            .copied()
            .find_map(|op| {
                func::Func::from_op(&ctx, op)
                    .ok()
                    .filter(|f| f.sym_name(&ctx) == "cps")
            })
            .unwrap();

        let error = lower_evidence_func(&mut ctx, function, &runtime)
            .expect_err("a result-bearing final dispatch is malformed");

        assert!(error.to_string().contains("effect.dispatch_cps"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
