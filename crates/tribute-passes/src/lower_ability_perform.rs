//! Lower `ability.perform` and `ability.call` operations to the effect ABI.
//!
//! In CPS-based effect handling, `ability.perform` carries an explicit
//! continuation closure. This pass converts it to:
//!
//! ```text
//! // Input:
//! ability.perform %evidence, %dispatch, %resume, [%args...]
//!   { ability_ref: @State, op_name: @get }
//!
//! // Output:
//! %payload = pack %args into the canonical operation product
//! effect.dispatch_cps %evidence, %dispatch, %resume, %payload
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
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    PatternApplicator, PatternRewriter, RewritePattern, RewriteScope, TypeConverter,
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
        lower_ability_perform(ctx, target);
        Ok(())
    }
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

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 3 || !ctx.op_result_types(op).is_empty() {
            return false;
        }

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
            &operands[3..],
            t.anyref,
        );

        let dispatch = effect::dispatch_cps(
            ctx,
            location,
            operands[0],
            operands[1],
            operands[2],
            shift_value_val,
            ability_ref_type,
            op_name_sym,
        );
        rewriter.insert_op(dispatch.op_ref());
        rewriter.erase_op(vec![]);
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
    ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}}
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
    ability.perform %ev, %dispatch, %resume, %val {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @set}}
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
        ability.perform %dynamic, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}}
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
    fn test_lower_call_no_evidence_skips_gracefully() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);

        // Function without evidence parameter — should skip without panicking.
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> core.never {
    %value = arith.const {value = 0} : tribute_rt.anyref
    %result = ability.call %value {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : tribute_rt.anyref
    func.unreachable
  }
}"#,
        );

        // Should not panic; the perform op is left unchanged.
        lower_ability_perform(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(
            ir.contains("ability.call"),
            "call op should remain unchanged when evidence is missing, got:\n{ir}"
        );
    }
}
