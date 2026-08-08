//! Lower `ability.perform` and `ability.call` operations to the effect ABI.
//!
//! In CPS-based effect handling, `ability.perform` carries an explicit
//! continuation closure. This pass converts it to:
//!
//! ```text
//! // Input:
//! %yr = ability.perform %continuation, [%args...]
//!   { ability_ref: @State, op_name: @get }
//!
//! // Output:
//! %payload = pack %args into the canonical operation product
//! %cont = cast %continuation to anyref
//! effect.dispatch_cps %evidence, %cont, %payload
//!   { ability_ref: @State, op_name: @get }
//! ```
//!
//! The explicitly named legacy operations retain the old null-or-single-value
//! carrier payload ABI until the frontend/pipeline migration is complete. Uses
//! `PatternApplicator` for declarative op-level rewriting. The function pass
//! validates every final `ability.perform` candidate before mutation; the
//! final `ability-lowered` dialect boundary is established by
//! `LowerHandleDispatch` after evidence resolution.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult, VerifyError};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    PatternApplicator, PatternRewriter, RewritePattern, RewriteScope, TypeConverter,
};
use trunk_ir::walk::{WalkAction, walk_op};

use tribute_core::{
    CLOSURE_CALLABLE_TYPE_ATTR, cps_closure_function_type, get_closure_callable_type,
    has_canonical_cps_parent_layout,
};
use tribute_ir::dialect::ability;
use tribute_ir::dialect::effect;
use tribute_ir::dialect::tribute_rt;

/// Cached common type references used by the perform lowering pattern.
#[derive(Clone, Copy)]
struct CommonTypes {
    anyref: TypeRef,
}

impl CommonTypes {
    fn new(ctx: &mut IrContext) -> Self {
        Self {
            anyref: tribute_rt::anyref(ctx).as_type_ref(),
        }
    }
}

/// Lower all currently legalizable `ability.perform` and `ability.call` ops.
///
/// Residual ability operations are allowed here and rejected at the final
/// `ability-lowered` boundary.
pub(crate) fn lower_ability_perform<S: RewriteScope>(ctx: &mut IrContext, scope: S) {
    let types = CommonTypes::new(ctx);
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(LowerPerformPattern { types })
        .add_pattern(LowerCallPattern { types });
    applicator.apply_partial(ctx, scope);
}

/// PassManager-friendly wrapper for [`lower_ability_perform`].
pub struct LowerAbilityPerform;

impl Pass for LowerAbilityPerform {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-ability-perform"
    }

    fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
        if ctx.op(target.op_ref()).regions.is_empty() {
            return Ok(());
        }
        prevalidate_final_performs(ctx, target)?;
        lower_ability_perform(ctx, target);
        let mut residual = None;
        let _ = walk_op::<()>(ctx, target.op_ref(), &mut |op| {
            if ability::Perform::from_op(ctx, op).is_ok() {
                let reason = explain_invalid_final_perform(ctx, op)
                    .unwrap_or("lowering pattern rejected a prevalidated final perform");
                residual = Some(format!("residual ability.perform: {reason}"));
                return std::ops::ControlFlow::Break(());
            }
            std::ops::ControlFlow::Continue(WalkAction::Advance)
        });
        if let Some(message) = residual {
            return Err(Box::new(VerifyError { message }));
        }
        consume_closure_callable_provenance(ctx, target);
        Ok(())
    }
}

fn prevalidate_final_performs(ctx: &IrContext, target: func::Func) -> PassRunResult {
    let mut invalid = None;
    let _ = walk_op::<()>(ctx, target.op_ref(), &mut |op| {
        if let Some(callable) = get_closure_callable_type(ctx, op)
            && !crate::closure_lower::is_lowered_closure_pack(ctx, op, callable)
        {
            invalid = Some(
                "invalid temporary closure callable provenance on a non-canonical closure pack"
                    .to_owned(),
            );
            return std::ops::ControlFlow::Break(());
        }
        if ability::Perform::from_op(ctx, op).is_ok()
            && let Some(reason) = explain_invalid_final_perform(ctx, op)
        {
            invalid = Some(format!("invalid final ability.perform: {reason}"));
            return std::ops::ControlFlow::Break(());
        }
        std::ops::ControlFlow::Continue(WalkAction::Advance)
    });
    match invalid {
        Some(message) => Err(Box::new(VerifyError { message })),
        None => Ok(()),
    }
}

/// Remove the proof-only closure type after every final perform in this
/// function has lowered successfully. The marker must not reach backend-ready
/// IR, where it would retain an abstract closure type in an attribute.
fn consume_closure_callable_provenance(ctx: &mut IrContext, target: func::Func) {
    let mut packs = Vec::new();
    let _ = walk_op::<()>(ctx, target.op_ref(), &mut |op| {
        if get_closure_callable_type(ctx, op).is_some() {
            packs.push(op);
        }
        std::ops::ControlFlow::Continue(WalkAction::Advance)
    });
    for pack in packs {
        ctx.op_mut(pack)
            .attributes
            .remove(Symbol::new(CLOSURE_CALLABLE_TYPE_ATTR));
    }
}

#[derive(Clone)]
struct FinalPerformShape {
    evidence: ValueRef,
    dispatch: ValueRef,
    resume: ValueRef,
    values: Vec<ValueRef>,
}

/// Validate the complete final `ability.perform` ABI before either the pass
/// wrapper or the local rewrite pattern mutates IR. The exact callback types
/// are proof-bearing control values; raw `anyref` never substitutes for them.
fn validate_final_perform_shape(
    ctx: &IrContext,
    op: OpRef,
) -> Result<FinalPerformShape, &'static str> {
    let operands = ctx.op_operands(op);
    if operands.len() < 3 {
        return Err("missing explicit dynamic evidence, dispatch, or resume operand");
    }
    let evidence = operands[0];
    if !ability::is_evidence_type_ref(ctx, ctx.value_ty(evidence)) {
        return Err("dynamic evidence operand has the wrong exact type");
    }
    let dispatch = operands[1];
    let Some(dispatch_closure) = proven_callback_closure_type(ctx, dispatch) else {
        return Err("dispatch operand is not a convention-proven CPS closure");
    };
    let Some(dispatch_function) = cps_closure_function_type(ctx, dispatch_closure) else {
        return Err("dispatch operand is not a convention-proven CPS closure");
    };
    let dispatch_params = &ctx.types.get(dispatch_function).params;
    if dispatch_params.len() != 7
        || !is_type(ctx, dispatch_params[0], "core", "never")
        || dispatch_params[1] != ctx.value_ty(evidence)
        || !is_type(ctx, dispatch_params[3], "core", "i32")
        || !is_type(ctx, dispatch_params[4], "core", "i32")
        || !is_type(ctx, dispatch_params[5], "core", "i32")
        || !is_type(ctx, dispatch_params[6], "tribute_rt", "anyref")
    {
        return Err("dispatch operand does not have the exact Dispatch<R> ABI");
    }
    let expected_resume = dispatch_params[2];
    let resume = operands[2];
    if proven_callback_closure_type(ctx, resume) != Some(expected_resume) {
        return Err("resume operand does not match Dispatch<R>'s exact Resume<R> type");
    }
    let Some(resume_function) = cps_closure_function_type(ctx, expected_resume) else {
        return Err("Dispatch<R> does not name a convention-proven Resume<R> closure");
    };
    let resume_params = &ctx.types.get(resume_function).params;
    if resume_params.len() != 4
        || !is_type(ctx, resume_params[0], "core", "never")
        || resume_params[1] != ctx.value_ty(evidence)
        || !has_canonical_cps_parent_layout(ctx, resume_params[2], dispatch_closure)
        || !is_type(ctx, resume_params[3], "tribute_rt", "anyref")
    {
        return Err("Dispatch<R> names a malformed Resume<R> callback");
    }
    let [result] = ctx.op_results(op) else {
        return Err("expected exactly one core.never marker result");
    };
    let result_type = ctx.types.get(ctx.value_ty(*result));
    if result_type.dialect != Symbol::new("core") || result_type.name != Symbol::new("never") {
        return Err("marker result is not core.never");
    }
    if ctx.has_uses(*result) {
        return Err("core.never marker result is used");
    }
    let Some(block) = ctx.op(op).parent_block else {
        return Err("operation has no parent block");
    };
    if ctx.block(block).ops.last().copied() != Some(op) {
        return Err("operation is not in proper-tail position");
    }
    Ok(FinalPerformShape {
        evidence,
        dispatch,
        resume,
        values: operands[3..].to_vec(),
    })
}

fn proven_callback_closure_type(ctx: &IrContext, value: ValueRef) -> Option<TypeRef> {
    let value_type = ctx.value_ty(value);
    if cps_closure_function_type(ctx, value_type).is_some() {
        return Some(value_type);
    }
    let trunk_ir::ValueDef::OpResult(def, _) = ctx.value_def(value) else {
        return None;
    };
    let closure_type = get_closure_callable_type(ctx, def)?;
    crate::closure_lower::is_lowered_closure_pack(ctx, def, closure_type)
        .then_some(closure_type)
        .filter(|closure| cps_closure_function_type(ctx, *closure).is_some())
}

fn is_type(ctx: &IrContext, ty: TypeRef, dialect: &'static str, name: &'static str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

fn explain_invalid_final_perform(ctx: &IrContext, op: OpRef) -> Option<&'static str> {
    validate_final_perform_shape(ctx, op).err()
}

/// Pattern: `ability.perform` → resultless `effect.dispatch_cps`.
struct LowerPerformPattern {
    types: CommonTypes,
}

impl RewritePattern for LowerPerformPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if ability::Perform::from_op(ctx, op).is_err() {
            // The explicitly named carrier form remains on the legacy path
            // until its dedicated cleanup; the final effect ABI has no
            // result-producing overload.
            return false;
        }

        let Ok(shape) = validate_final_perform_shape(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let ability_ref_type = ctx.op(op).attributes.get_type("ability_ref").unwrap();
        let op_name_sym = ctx.op(op).attributes.get_symbol("op_name").unwrap();

        let t = &self.types;

        // === 2. Build the canonical payload product. ===
        let shift_value_val = pack_payload(
            ctx,
            rewriter,
            location,
            ability_ref_type,
            op_name_sym,
            &shape.values,
            t.anyref,
        );

        let dispatch = effect::dispatch_cps(
            ctx,
            location,
            shape.evidence,
            shape.dispatch,
            shape.resume,
            shift_value_val,
            ability_ref_type,
            op_name_sym,
        );
        rewriter.insert_op(dispatch.op_ref());
        rewriter.erase_op_with_unused_results();
        true
    }
}

/// Pattern: `ability.call` → `effect.dispatch_tail` (no CPS).
struct LowerCallPattern {
    types: CommonTypes,
}

impl RewritePattern for LowerCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let legacy = if ability::Call::from_op(ctx, op).is_ok() {
            false
        } else if ability::LegacyCall::from_op(ctx, op).is_ok() {
            true
        } else {
            return false;
        };

        let location = ctx.op(op).location;
        let ability_ref_type = ctx.op(op).attributes.get_type("ability_ref").unwrap();
        let op_name_sym = ctx.op(op).attributes.get_symbol("op_name").unwrap();
        let source_result_ty = ctx.op_result_types(op)[0];

        // Operands: [...values]
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let value_operands = &operands[..];

        let t = &self.types;

        // Find evidence parameter from enclosing func's entry block.
        let evidence_val = find_evidence_from_op(ctx, op);

        // === 1. Find evidence ===
        let Some(evidence_val) = evidence_val else {
            // Missing evidence means the frontend detected unhandled effects
            // and emitted a diagnostic. Skip this op gracefully.
            return false;
        };

        // === 2. Build the canonical payload product. ===
        let shift_value_val = if legacy {
            pack_legacy_payload(ctx, rewriter, location, value_operands, t.anyref)
        } else {
            pack_payload(
                ctx,
                rewriter,
                location,
                ability_ref_type,
                op_name_sym,
                value_operands,
                t.anyref,
            )
        };

        // === 3. Dispatch through target-independent effect ABI ===
        let dispatch_op = effect::dispatch_tail(
            ctx,
            location,
            evidence_val,
            shift_value_val,
            t.anyref,
            ability_ref_type,
            op_name_sym,
        );
        rewriter.insert_op(dispatch_op.op_ref());

        // === 4. Restore the source result after the erased effect ABI. ===
        // `effect.dispatch_tail` always returns the runtime-erased `anyref`
        // produced by the handler dispatcher. The source `ability.call`
        // result remains its exact declared type, so preserve the conversion
        // boundary explicitly for target materialization.
        let replacement = if source_result_ty == t.anyref {
            dispatch_op.result(ctx)
        } else {
            let cast = core::unrealized_conversion_cast(
                ctx,
                location,
                dispatch_op.result(ctx),
                source_result_ty,
            );
            let result = cast.result(ctx);
            rewriter.insert_op(cast.op_ref());
            result
        };

        rewriter.erase_op(vec![replacement]);

        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn pack_payload(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    location: trunk_ir::types::Location,
    ability_ref: TypeRef,
    op_name: Symbol,
    values: &[ValueRef],
    anyref: TypeRef,
) -> ValueRef {
    let payload_type = ability::operation_payload_type_ref(
        ctx,
        ability_ref,
        op_name,
        values
            .iter()
            .map(|value| ctx.value_ty(*value))
            .collect::<Vec<_>>(),
    );
    let payload = adt::struct_new(
        ctx,
        location,
        values.iter().copied(),
        payload_type,
        payload_type,
    );
    rewriter.insert_op(payload.op_ref());
    let erased = core::unrealized_conversion_cast(ctx, location, payload.result(ctx), anyref);
    rewriter.insert_op(erased.op_ref());
    erased.result(ctx)
}

fn pack_legacy_payload(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    location: trunk_ir::types::Location,
    values: &[ValueRef],
    anyref: TypeRef,
) -> ValueRef {
    assert!(
        values.len() <= 1,
        "legacy ability payloads must already be tuple-packed"
    );
    if let Some(&value) = values.first() {
        let erased = core::unrealized_conversion_cast(ctx, location, value, anyref);
        rewriter.insert_op(erased.op_ref());
        erased.result(ctx)
    } else {
        let null = adt::ref_null(ctx, location, anyref, anyref);
        rewriter.insert_op(null.op_ref());
        null.result(ctx)
    }
}

/// Find the evidence parameter by walking up from the op to its enclosing func.
fn find_evidence_from_op(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    let mut current_op = op;
    loop {
        let block = ctx.op(current_op).parent_block?;
        let region = ctx.block(block).parent_region?;
        let parent = ctx.region(region).parent_op?;
        if let Ok(handle) = ability::HandleDispatch::from_op(ctx, parent) {
            // A final delimiter owns an exact dynamically extended evidence
            // value in its body entry block. A perform lexically inside that
            // body must look up its prompt through that value, rather than
            // through the enclosing function's outer evidence parameter.
            // Handler arms are lowered into separate closures, so they do not
            // satisfy this exact body-region provenance check.
            if region == handle.body(ctx) {
                let [evidence] = ctx.block_args(block) else {
                    return None;
                };
                return ability::is_evidence_type_ref(ctx, ctx.value_ty(*evidence))
                    .then_some(*evidence);
            }
        }
        if func::Func::matches(ctx, parent) {
            // Found the enclosing func — check entry block args.
            let func_body = func::Func::from_op(ctx, parent).ok()?.body(ctx);
            let entry = ctx.region(func_body).blocks[0];
            if let Some(evidence) = ctx
                .block_args(entry)
                .iter()
                .find(|&&arg| ability::is_evidence_type_ref(ctx, ctx.value_ty(arg)))
                .copied()
            {
                return Some(evidence);
            }
            return captured_evidence_from_closure_environment(ctx, entry);
        }
        current_op = parent;
    }
}

/// Recover evidence only from the canonical leading closure-environment
/// extraction sequence produced by `lower_closure_lambda`.
///
/// The environment is the exact entry argument named `__env`; its first
/// operation is an `adt.ref_cast`, followed immediately by the capture
/// `adt.struct_get`s. Restricting lookup to that prefix proves provenance and
/// dominance and rejects unrelated evidence-producing operations later in the
/// entry block.
fn captured_evidence_from_closure_environment(
    ctx: &IrContext,
    entry: trunk_ir::refs::BlockRef,
) -> Option<ValueRef> {
    let env_indices = ctx
        .block(entry)
        .args
        .iter()
        .enumerate()
        .filter_map(|(index, arg)| {
            (arg.attrs.get_symbol("bind_name") == Some(Symbol::new("__env"))).then_some(index)
        })
        .collect::<Vec<_>>();
    let [env_index] = env_indices.as_slice() else {
        return None;
    };
    let env_arg = ctx.block_arg(entry, *env_index as u32);

    let mut ops = ctx.block(entry).ops.iter().copied();
    let cast = adt::RefCast::from_op(ctx, ops.next()?).ok()?;
    if cast.r#ref(ctx) != env_arg {
        return None;
    }
    let cast_result = cast.result(ctx);

    let mut candidates = Vec::new();
    for op in ops {
        let Ok(get) = adt::StructGet::from_op(ctx, op) else {
            break;
        };
        if get.r#ref(ctx) != cast_result {
            break;
        }
        let result = get.result(ctx);
        if ability::is_evidence_type_ref(ctx, ctx.value_ty(result)) {
            candidates.push(result);
        }
    }
    match candidates.as_slice() {
        [evidence] => Some(*evidence),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    /// Initialize common types used by the pass.
    fn init_common_types(ctx: &mut IrContext) {
        let _ = CommonTypes::new(ctx);
    }

    /// Build the canonical evidence type string for use in test IR.
    fn evidence_type_str() -> &'static str {
        "core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})"
    }

    fn control_type_decls(ev_ty: &str) -> String {
        format!(
            "  !parent = adt.typeref() {{name = @Parent, tribute.cps_parent_result = core.i32}}\n  \
             !done = closure.closure(core.func(core.never, core.i32)) {{tribute.calling_convention = 2}}\n  \
             !resume = closure.closure(core.func(core.never, {ev_ty}, !parent, tribute_rt.anyref)) \
             {{tribute.calling_convention = 2}}\n  \
             !dispatch = closure.closure(core.func(core.never, {ev_ty}, !resume, core.i32, core.i32, core.i32, tribute_rt.anyref)) \
             {{tribute.calling_convention = 2}}\n  \
             !parent_layout = adt.struct() {{fields = [[@done, !done], [@dispatch, !dispatch]], name = @Parent, tribute.cps_parent_result = core.i32}}\n"
        )
    }

    #[test]
    fn test_lower_perform_basic() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
{control}  func.func @test_fn(%ev: {ev_ty}, %dispatch: !dispatch, %resume: !resume) -> core.never {{
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert!(!ir_text.contains("ability.perform"), "{ir_text}");
        assert!(ir_text.contains("effect.dispatch_cps"), "{ir_text}");
        assert!(!ir_text.contains("func.return"), "{ir_text}");
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir_text);
    }

    #[test]
    fn test_lower_perform_with_args() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
{control}  func.func @test_fn(%ev: {ev_ty}, %dispatch: !dispatch, %resume: !resume) -> core.never {{
    %val = arith.const {{value = 42}} : core.i32
    %never = ability.perform %ev, %dispatch, %resume, %val {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @set}} : core.never
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert!(!ir_text.contains("ability.perform"), "{ir_text}");
        assert!(ir_text.contains("effect.dispatch_cps"), "{ir_text}");
        assert!(ir_text.contains("@arg0"), "{ir_text}");
    }

    #[test]
    fn perform_in_final_handle_body_uses_dynamic_body_evidence() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @run(%outer: {ev_ty}) -> core.never {{
    %prompt = arith.const {{value = 7}} : core.i32
    ability.handle_dispatch %outer, %prompt {{ability_refs = []}} {{
      ^body(%dynamic: {ev_ty}):
        %dispatch = arith.const {{value = 0}} : tribute_rt.anyref
        %resume = arith.const {{value = 0}} : tribute_rt.anyref
        %never = ability.perform %dynamic, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
    }}
  }}
}}"#
            ),
        );
        let run = module
            .ops(&ctx)
            .iter()
            .find_map(|op| func::Func::from_op(&ctx, *op).ok())
            .expect("run function");
        let entry = ctx.region(run.body(&ctx)).blocks[0];
        let handle = ctx.block(entry).ops[1];
        let handle = ability::HandleDispatch::from_op(&ctx, handle).expect("delimiter");
        let body = ctx.region(handle.body(&ctx)).blocks[0];
        let dynamic = ctx.block_arg(body, 0);
        let perform = ctx.block(body).ops[2];

        assert_eq!(find_evidence_from_op(&ctx, perform), Some(dynamic));
    }

    #[test]
    fn test_lower_call_to_effect_dispatch_tail() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref {{
    %msg = arith.const {{value = 1}} : tribute_rt.anyref
    %result = ability.call %msg {{ability_ref = core.ability_ref() {{name = @Console}}, op_name = @print}} : tribute_rt.anyref
    func.return %result
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }

    #[test]
    fn tail_dispatch_restores_the_exact_source_result_after_erasure() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @call_nat(%ev: {ev_ty}) -> core.i32 {{
    %result = ability.call {{ability_ref = core.ability_ref() {{name = @Ask}}, op_name = @ask}} : core.i32
    func.return %result
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(!ir.contains("ability.call"), "{ir}");
        assert!(ir.contains("effect.dispatch_tail"), "{ir}");
        assert!(
            ir.contains("core.unrealized_conversion_cast") && ir.contains(": core.i32"),
            "the erased handler result must be restored to the source Nat type:\n{ir}"
        );
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir);
    }

    #[test]
    fn legacy_operations_keep_the_carrier_payload_abi() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @legacy_perform(%ev: {ev_ty}) -> tribute_rt.anyref {{
    %k = arith.const {{value = 0}} : tribute_rt.anyref
    %result = ability.legacy_perform %k {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : tribute_rt.anyref
    func.return %result
  }}
  func.func @legacy_call(%ev: {ev_ty}, %value: core.i32) -> tribute_rt.anyref {{
    %result = ability.legacy_call %value {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @set}} : tribute_rt.anyref
    func.return %result
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(ir.contains("ability.legacy_perform"));
        assert!(!ir.contains("ability.legacy_call"));
        assert!(!ir.contains("__tribute_ability_payload_"));
        assert_eq!(ir.matches("effect.dispatch_").count(), 1);
        assert_eq!(ir.matches("effect.dispatch_tail").count(), 1);
        assert!(ir.contains("core.unrealized_conversion_cast %"));
        assert!(!ir.contains("effect.legacy"));
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir);
    }

    #[test]
    fn explicit_dynamic_evidence_is_used_without_environment_provenance_scan() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  !env = adt.struct() {{fields = [[@value, core.i32]], name = @env}}
{control}  func.func @run(%__env: tribute_rt.anyref, %dispatch: !dispatch, %resume: !resume) -> core.never {{
    %env_cast = adt.ref_cast %__env {{type = !env}} : !env
    %captured_value = adt.struct_get %env_cast {{field = 0, type = !env}} : core.i32
    %unrelated = adt.ref_null {{type = {ev_ty}}} : {ev_ty}
    %never = ability.perform %unrelated, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
            ),
        );
        let run = module
            .ops(&ctx)
            .iter()
            .find_map(|op| func::Func::from_op(&ctx, *op).ok())
            .expect("run function");
        let entry = ctx.region(run.body(&ctx)).blocks[0];
        let env_arg = ctx.block_arg(entry, 0);
        assert_eq!(
            ctx.block(entry).args[0].attrs.get_symbol("bind_name"),
            Some(Symbol::new("__env"))
        );
        let entry_ops = ctx.block(entry).ops.clone();
        let env_cast = adt::RefCast::from_op(&ctx, entry_ops[0]).expect("leading environment cast");
        assert_eq!(env_cast.r#ref(&ctx), env_arg);
        let captured_value =
            adt::StructGet::from_op(&ctx, entry_ops[1]).expect("leading environment field");
        assert_eq!(captured_value.r#ref(&ctx), env_cast.result(&ctx));
        assert!(!ability::is_evidence_type_ref(
            &ctx,
            ctx.value_ty(captured_value.result(&ctx))
        ));
        let unrelated =
            adt::RefNull::from_op(&ctx, entry_ops[2]).expect("later unrelated evidence producer");
        assert!(ability::is_evidence_type_ref(
            &ctx,
            ctx.value_ty(unrelated.result(&ctx))
        ));
        lower_ability_perform(&mut ctx, module);

        let after = print_module(&ctx, module.op());
        assert!(!after.contains("ability.perform"), "{after}");
        assert!(after.contains("effect.dispatch_cps"), "{after}");
    }

    #[test]
    fn prevalidation_preserves_mixed_valid_and_invalid_function() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
{control}  func.func @run(%ev: {ev_ty}, %dispatch: !dispatch, %resume: !resume, %condition: core.i1) -> core.never {{
    %never = scf.if %condition : core.never {{
      %valid = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
    }} {{
      %invalid = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
      func.unreachable
    }}
  }}
}}"#
            ),
        );
        let run = module
            .ops(&ctx)
            .iter()
            .find_map(|op| func::Func::from_op(&ctx, *op).ok())
            .expect("run function");
        let before = print_module(&ctx, module.op());

        let error = LowerAbilityPerform.run(&mut ctx, run).unwrap_err();

        assert!(error.to_string().contains("not in proper-tail position"));
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn final_perform_prevalidation_rejects_unproven_control_values_without_mutation() {
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);
        let cases = [
            (
                format!(
                    r#"core.module @test {{
{control}  func.func @run(%wrong: core.i32, %dispatch: !dispatch, %resume: !resume) -> core.never {{
    %never = ability.perform %wrong, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
                ),
                "dynamic evidence operand has the wrong exact type",
            ),
            (
                format!(
                    r#"core.module @test {{
{control}  func.func @run(%ev: {ev_ty}, %dispatch: tribute_rt.anyref, %resume: !resume) -> core.never {{
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
                ),
                "dispatch operand is not a convention-proven CPS closure",
            ),
            (
                format!(
                    r#"core.module @test {{
{control}  func.func @run(%ev: {ev_ty}, %dispatch: !dispatch, %resume: tribute_rt.anyref) -> core.never {{
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
                ),
                "resume operand does not match Dispatch<R>'s exact Resume<R> type",
            ),
            (
                format!(
                    r#"core.module @test {{
{control}  !other_parent = adt.typeref() {{name = @Other, tribute.cps_parent_result = core.i64}}
  !other_resume = closure.closure(core.func(core.never, {ev_ty}, !other_parent, tribute_rt.anyref)) {{tribute.calling_convention = 2}}
  func.func @run(%ev: {ev_ty}, %dispatch: !dispatch, %resume: !other_resume) -> core.never {{
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
                ),
                "resume operand does not match Dispatch<R>'s exact Resume<R> type",
            ),
            (
                format!(
                    r#"core.module @test {{
{control}  !missing_layout_parent = adt.typeref() {{name = @MissingLayout, tribute.cps_parent_result = core.i32}}
  !missing_layout_resume = closure.closure(core.func(core.never, {ev_ty}, !missing_layout_parent, tribute_rt.anyref)) {{tribute.calling_convention = 2}}
  !missing_layout_dispatch = closure.closure(core.func(core.never, {ev_ty}, !missing_layout_resume, core.i32, core.i32, core.i32, tribute_rt.anyref)) {{tribute.calling_convention = 2}}
  func.func @run(%ev: {ev_ty}, %dispatch: !missing_layout_dispatch, %resume: !missing_layout_resume) -> core.never {{
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
                ),
                "Dispatch<R> names a malformed Resume<R> callback",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = IrContext::new();
            init_common_types(&mut ctx);
            let module = parse_test_module(&mut ctx, &input);
            let run = module
                .ops(&ctx)
                .iter()
                .find_map(|op| func::Func::from_op(&ctx, *op).ok())
                .expect("run function");
            let before = print_module(&ctx, module.op());
            let error = LowerAbilityPerform.run(&mut ctx, run).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn final_perform_consumes_canonical_closure_pack_provenance() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
{control}  !closure = adt.struct(core.i32, tribute_rt.anyref) {{name = @_closure}}
  func.func @resume_impl(%ev: {ev_ty}, %parent: !parent, %input: tribute_rt.anyref) -> core.never {{
    func.unreachable
  }}
  func.func @run(%ev: {ev_ty}, %dispatch: !dispatch, %env: tribute_rt.anyref) -> core.never {{
    %resume_fn = func.constant @resume_impl : core.func(core.never, {ev_ty}, !parent, tribute_rt.anyref)
    %resume = adt.struct_new %resume_fn, %env {{type = !closure, tribute.closure_callable_type = !resume}} : !closure
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
            ),
        );
        let run = module
            .ops(&ctx)
            .iter()
            .filter_map(|op| func::Func::from_op(&ctx, *op).ok())
            .find(|func_op| func_op.sym_name(&ctx) == Symbol::new("run"))
            .expect("run function");

        LowerAbilityPerform.run(&mut ctx, run).unwrap();

        let ir = print_module(&ctx, module.op());
        assert!(!ir.contains("ability.perform"), "{ir}");
        assert!(!ir.contains(CLOSURE_CALLABLE_TYPE_ATTR), "{ir}");
    }

    #[test]
    fn final_perform_rejects_spoofed_closure_pack_provenance_without_mutation() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let control = control_type_decls(ev_ty);
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
{control}  !closure = adt.struct(core.i32, tribute_rt.anyref) {{name = @_closure}}
  func.func @run(%ev: {ev_ty}, %dispatch: !dispatch, %env: tribute_rt.anyref) -> core.never {{
    %not_a_function = arith.const {{value = 0}} : core.i32
    %resume = adt.struct_new %not_a_function, %env {{type = !closure, tribute.closure_callable_type = !resume}} : !closure
    %never = ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : core.never
  }}
}}"#
            ),
        );
        let run = module
            .ops(&ctx)
            .iter()
            .find_map(|op| func::Func::from_op(&ctx, *op).ok())
            .expect("run function");
        let before = print_module(&ctx, module.op());

        let error = LowerAbilityPerform.run(&mut ctx, run).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid temporary closure callable provenance"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn test_lower_perform_no_evidence_skips_gracefully() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);

        // Function without evidence parameter — should skip without panicking.
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.never {
    %dispatch = arith.const {value = 0} : tribute_rt.anyref
    %resume = arith.const {value = 0} : tribute_rt.anyref
    %never = ability.perform %dispatch, %resume {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : core.never
  }
}"#,
        );

        // Should not panic; the perform op is left unchanged.
        lower_ability_perform(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(
            ir.contains("ability.perform"),
            "perform op should remain unchanged when evidence is missing, got:\n{ir}"
        );
    }
}
