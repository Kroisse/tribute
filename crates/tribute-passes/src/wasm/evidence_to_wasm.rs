//! Wasm evidence lowering (arena-based).
//!
//! Evidence lowering runs inside the representation/ABI boundary. It lowers
//! `effect.extend`, the evidence selection operations, `effect.dispatch_tail`,
//! and `effect.dispatch_cps` to calls of the evidence runtime written in
//! Tribute source, `std::runtime::evidence` (`new-plans/cps-effects.md`),
//! which it finds by canonical path. The runtime functions are compiled like
//! any other source function; marker construction and field access stay
//! inside them.
//!
//! After the boundary exit, [`bind_wasm_prompt_tag_runtime`] gives the
//! prompt tag allocator `__tribute_next_tag` its Wasm implementation.

use std::rc::Rc;

use tribute_ir::dialect::ability;
use tribute_ir::dialect::adt;
use tribute_ir::dialect::effect;
use tribute_ir::runtime_items::{RuntimeFunction, RuntimeFunctionError, evidence};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func, wasm as wasm_dialect};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location};

use crate::effect_dispatch;

/// C link name of the helper that allocates a fresh prompt tag.
pub const NEXT_TAG: &str = "__tribute_next_tag";

// =============================================================================
// Boundary: effect lowering
// =============================================================================

/// The evidence runtime functions the module's `effect.*` operations call.
#[derive(Clone, Copy, Debug, Default)]
pub struct WasmEvidenceRuntime {
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
/// Run once on the module before [`LowerEvidenceToWasm`]. A function no
/// operation needs is not looked up, so a missing one fails only when used.
pub fn prepare_wasm_evidence_runtime(
    ctx: &IrContext,
    module: Module,
) -> Result<WasmEvidenceRuntime, RuntimeFunctionError> {
    let needs = evidence_runtime_requirements(ctx, module);
    let find = |needed: bool, path| {
        needed
            .then(|| RuntimeFunction::lookup(ctx, module, path))
            .transpose()
    };
    Ok(WasmEvidenceRuntime {
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
/// `runtime`, from [`prepare_wasm_evidence_runtime`] on the containing module.
pub fn lower_evidence_to_wasm_func(
    ctx: &mut IrContext,
    func_op: func::Func,
    runtime: &WasmEvidenceRuntime,
) -> Result<(), ConversionError> {
    let runtime = Rc::new(*runtime);
    PatternApplicator::new(TypeConverter::new())
        .with_target(wasm_effect_abi_target())
        .add_pattern(LowerEffectExtendToWasm {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectStackOpToWasm {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectTailOpToWasm {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectDispatchTailToWasm {
            runtime: Rc::clone(&runtime),
        })
        .add_pattern(LowerEffectDispatchCpsToWasm { runtime })
        .apply_partial_conversion(ctx, func_op, "wasm-evidence-effect-abi")?;
    Ok(())
}

/// PassManager-friendly Wasm evidence lowering pass.
///
/// This pass is function-scoped; it takes the runtime functions
/// [`prepare_wasm_evidence_runtime`] found on the module.
pub struct LowerEvidenceToWasm {
    runtime: WasmEvidenceRuntime,
}

impl LowerEvidenceToWasm {
    pub fn new(runtime: WasmEvidenceRuntime) -> Self {
        Self { runtime }
    }
}

impl Pass for LowerEvidenceToWasm {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-evidence-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: func::Func,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_evidence_to_wasm_func(ctx, target, &self.runtime)?;
        Ok(())
    }
}

fn wasm_effect_abi_target() -> ConversionTarget {
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

/// Call `function`, which [`prepare_wasm_evidence_runtime`] found because an
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
struct LowerEffectExtendToWasm {
    runtime: Rc<WasmEvidenceRuntime>,
}

impl RewritePattern for LowerEffectExtendToWasm {
    fn match_and_rewrite(
        &self,
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
struct LowerEffectStackOpToWasm {
    runtime: Rc<WasmEvidenceRuntime>,
}

impl RewritePattern for LowerEffectStackOpToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let (function, ability, evidence) = if let Ok(mask) = effect::Mask::from_op(ctx, op) {
            (
                self.runtime.mask,
                mask.ability(ctx).clone(),
                mask.evidence(ctx),
            )
        } else if let Ok(dup) = effect::Dup::from_op(ctx, op) {
            (
                self.runtime.dup,
                dup.ability(ctx).clone(),
                dup.evidence(ctx),
            )
        } else if let Ok(outer) = effect::Outer::from_op(ctx, op) {
            (
                self.runtime.outer,
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
struct LowerEffectTailOpToWasm {
    runtime: Rc<WasmEvidenceRuntime>,
}

impl RewritePattern for LowerEffectTailOpToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
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
            (self.runtime.tail, vec![tail.evidence(ctx), slot])
        } else if let Ok(with_tail) = effect::WithTail::from_op(ctx, op) {
            let slot = slot(ctx, with_tail.index(ctx));
            (
                self.runtime.with_tail,
                vec![with_tail.evidence(ctx), slot, with_tail.tail(ctx)],
            )
        } else if let Ok(push) = effect::Push::from_op(ctx, op) {
            let ability_id = {
                let ability = push.ability(ctx).clone();
                effect_dispatch::insert_ability_id(ctx, loc, &ability, rewriter)
            };
            (
                self.runtime.push,
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
struct LowerEffectDispatchTailToWasm {
    runtime: Rc<WasmEvidenceRuntime>,
}

impl RewritePattern for LowerEffectDispatchTailToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchTail::from_op(ctx, op) else {
            return false;
        };
        let converter = super::type_converter::wasm_type_converter(ctx);
        if !effect_dispatch::is_valid_tail_dispatch(ctx, op, &converter) {
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
struct LowerEffectDispatchCpsToWasm {
    runtime: Rc<WasmEvidenceRuntime>,
}

impl RewritePattern for LowerEffectDispatchCpsToWasm {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchCps::from_op(ctx, op) else {
            return false;
        };
        let converter = super::type_converter::wasm_type_converter(ctx);
        if !effect_dispatch::is_valid_cps_dispatch(ctx, op, &converter) {
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

// =============================================================================
// After the exit: prompt tag allocator binding
// =============================================================================

/// Replace a bodyless `__tribute_next_tag` declaration with its Wasm
/// implementation: a mutable `i32` counter global that the function returns
/// and increments.
///
/// Runs after Wasm dialect lowering has converted the declaration's
/// signature; the implementation adopts it.
pub fn bind_wasm_prompt_tag_runtime(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    for op in module.ops_snapshot(ctx) {
        let is_function = wasm_dialect::Func::matches(ctx, op) || func::Func::matches(ctx, op);
        if !is_function || ctx.op_has_regions(op) {
            continue;
        }
        let attrs = ctx.op_attrs(op);
        if attrs.get_str("sym_name") != Some(NEXT_TAG) {
            continue;
        }
        let name = attrs
            .get_string_ref("sym_name")
            .expect("a declaration has a name");
        let signature = attrs
            .get_type("type")
            .expect("a function declaration has a signature");
        let location = ctx.op(op).location;
        let global = add_tag_counter(ctx, module, location);
        let implementation = next_tag_function(ctx, location, name, signature, global);
        ctx.insert_op_before(block, op, implementation);
        ctx.remove_op_from_block(block, op);
        ctx.remove_op(op);
    }
}

/// Append a mutable `i32` tag counter global, returning its index.
fn add_tag_counter(ctx: &mut IrContext, module: Module, location: Location) -> u32 {
    let index = module
        .ops(ctx)
        .iter()
        .filter(|&&op| wasm_dialect::Global::matches(ctx, op))
        .count() as u32;
    let global = wasm_dialect::Global::operands()
        .valtype("i32")
        .mutable(true)
        .init(Attribute::Int(0))
        .build(ctx, location);
    let block = module
        .first_block(ctx)
        .expect("a module with declarations has a body block");
    ctx.push_op(block, global.op_ref());
    index
}

/// The `wasm.func` that returns the counter `global` and increments it, with
/// the declaration's (converted) signature.
fn next_tag_function(
    ctx: &mut IrContext,
    location: Location,
    name: trunk_ir::StringRef,
    signature: TypeRef,
    global: u32,
) -> OpRef {
    let (inputs, results) = if let Some(sig) = wasm_dialect::FuncSig::from_type_ref(ctx, signature)
    {
        (sig.inputs(ctx).to_vec(), sig.results(ctx).to_vec())
    } else {
        let sig = func::FuncSig::from_type_ref(ctx, signature)
            .expect("runtime helper declaration must have a function signature");
        (sig.inputs(ctx).to_vec(), sig.results(ctx).to_vec())
    };
    let result_ty = *results
        .first()
        .expect("runtime helper declaration must have one result");
    let block = ctx.create_block(BlockData {
        location,
        args: inputs
            .iter()
            .map(|&ty| BlockArgData {
                ty,
                attrs: Default::default(),
            })
            .collect(),
        ops: smallvec![],
        parent_region: None,
    });
    let current = wasm_dialect::GlobalGet::operands()
        .index(global)
        .results(result_ty)
        .build(ctx, location);
    ctx.push_op(block, current.op_ref());
    let one = wasm_dialect::I32Const::operands()
        .value(1)
        .results(result_ty)
        .build(ctx, location);
    ctx.push_op(block, one.op_ref());
    let next =
        wasm_dialect::I32Add::operands(current.result(ctx), one.result(ctx)).build(ctx, location);
    ctx.push_op(block, next.op_ref());
    let set = wasm_dialect::GlobalSet::operands(next.result(ctx))
        .index(global)
        .build(ctx, location);
    ctx.push_op(block, set.op_ref());
    let ret = wasm_dialect::Return::operands(vec![current.result(ctx)]).build(ctx, location);
    ctx.push_op(block, ret.op_ref());

    let body = ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    });
    let func_ty = wasm_dialect::func_sig(ctx, inputs, [result_ty]).as_type_ref();
    wasm_dialect::Func::operands()
        .sym_name(name)
        .r#type(func_ty)
        .regions(body)
        .build(ctx, location)
        .op_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::SymbolPath;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    /// The evidence type and stand-ins for the runtime functions.
    const RUNTIME: &str = r#"  !Evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  !Closure = adt.struct<_closure(code: core.i32), {layout = "closure"}>
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
        let runtime = prepare_wasm_evidence_runtime(ctx, module).expect("evidence runtime");
        for op in module.ops_snapshot(ctx) {
            if let Ok(function) = func::Func::from_op(ctx, op)
                && !function.sym_name(ctx).starts_with("std::runtime::")
            {
                lower_evidence_to_wasm_func(ctx, function, &runtime)?;
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
        assert!(!printed.contains("wasm."), "{printed}");
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
        let error = prepare_wasm_evidence_runtime(&ctx, module).expect_err("tail is missing");
        assert_eq!(error.path, "std::runtime::evidence::tail");

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &module_text(SELECT));
        assert!(prepare_wasm_evidence_runtime(&ctx, module).is_ok());
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
        let runtime = prepare_wasm_evidence_runtime(&ctx, module).unwrap();
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

        let error = lower_evidence_to_wasm_func(&mut ctx, function, &runtime)
            .expect_err("a result-bearing final dispatch is malformed");

        assert!(error.to_string().contains("effect.dispatch_cps"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn next_tag_counts_in_a_global_appended_after_existing_ones() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.global {valtype = "i32", mutable = false, init = 7}
  func.func @__tribute_next_tag() -> core.i32 attributes {abi = "C"}
}"#,
        );

        bind_wasm_prompt_tag_runtime(&mut ctx, module);

        assert_eq!(
            print_module(&ctx, module.op()),
            r#"core.module @test {
  wasm.global {init = 7, mutable = false, valtype = "i32"}
  wasm.func {sym_name = "__tribute_next_tag", type = wasm.func_sig<() -> core.i32>} {
      %0 = wasm.global_get {index = 1} : core.i32
      %1 = wasm.i32_const {value = 1} : core.i32
      %2 = wasm.i32_add %0, %1 : core.i32
      wasm.global_set %2 {index = 1}
      wasm.return %0
  }
  wasm.global {init = 0, mutable = true, valtype = "i32"}
}
"#
        );
    }

    #[test]
    fn binding_leaves_modules_without_a_tag_declaration_unchanged() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main() -> core.nil {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        bind_wasm_prompt_tag_runtime(&mut ctx, module);

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
