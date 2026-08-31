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
//! `PatternApplicator` for declarative op-level rewriting. This is an
//! intermediate best-effort pass: the final `ability-lowered` boundary is
//! established by `LowerHandleDispatch` after evidence resolution.

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
        .add_pattern(LowerLegacyPerformPattern { types })
        .add_pattern(LowerLegacyCallPattern { types })
        .add_pattern(LowerCallPattern { types });
    applicator.apply_partial(ctx, scope);
}

/// Pattern: explicit `ability.legacy_perform` → result-producing legacy ABI.
struct LowerLegacyPerformPattern {
    types: CommonTypes,
}

impl RewritePattern for LowerLegacyPerformPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if ability::LegacyPerform::from_op(ctx, op).is_err() {
            return false;
        }
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let Some((&continuation, values)) = operands.split_first() else {
            return false;
        };
        if values.len() > 1 {
            return false;
        }
        let Some(evidence) = find_evidence_from_op(ctx, op) else {
            return false;
        };
        let result_types = ctx.op_result_types(op).to_vec();
        let [result_ty] = result_types.as_slice() else {
            return false;
        };
        let location = ctx.op(op).location;
        let ability_ref = ctx.op(op).attributes.get_type("ability_ref").unwrap();
        let op_name = ctx.op(op).attributes.get_symbol("op_name").unwrap();
        let payload = pack_legacy_payload(ctx, rewriter, location, values, self.types.anyref);
        let continuation =
            core::unrealized_conversion_cast(ctx, location, continuation, self.types.anyref);
        let continuation_value = continuation.result(ctx);
        rewriter.insert_op(continuation.op_ref());
        let dispatch = effect::legacy_dispatch_cps(
            ctx,
            location,
            evidence,
            continuation_value,
            payload,
            *result_ty,
            ability_ref,
            op_name,
        );
        let result = dispatch.result(ctx);
        rewriter.insert_op(dispatch.op_ref());
        rewriter.erase_op(vec![result]);
        true
    }
}

/// PassManager-friendly wrapper for [`lower_ability_perform`].
pub struct LowerAbilityPerform;

impl Pass for LowerAbilityPerform {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-ability-perform"
    }

    fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
        lower_ability_perform(ctx, target);
        Ok(())
    }
}

/// Pattern: final `ability.perform` → resultless `effect.dispatch_cps`.
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
            // The explicit legacy carrier route is deliberately not coerced
            // into the final ABI before #825/#826 own its migration.
            return false;
        }

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        if operands.len() < 3 || !ctx.op_result_types(op).is_empty() {
            return false;
        }

        let location = ctx.op(op).location;
        let ability_ref_type = ctx.op(op).attributes.get_type("ability_ref").unwrap();
        let op_name_sym = ctx.op(op).attributes.get_symbol("op_name").unwrap();

        // Operands: [evidence, exact dispatch, exact resume, ...values]
        let evidence_val = operands[0];
        let dispatch_val = operands[1];
        let resume_val = operands[2];
        let value_operands = &operands[3..];

        let t = &self.types;

        let shift_value_val = pack_payload(
            ctx,
            rewriter,
            location,
            ability_ref_type,
            op_name_sym,
            value_operands,
            t.anyref,
        );

        // === 3. Dispatch through the target-independent effect ABI ===
        // Keep both control closures typed. Only the payload crosses the
        // target-independent dynamic-storage boundary.
        let dispatch_op = effect::dispatch_cps(
            ctx,
            location,
            evidence_val,
            dispatch_val,
            resume_val,
            shift_value_val,
            ability_ref_type,
            op_name_sym,
        );
        rewriter.insert_op(dispatch_op.op_ref());
        rewriter.erase_op(vec![]);
        true
    }
}

/// Pattern: `ability.legacy_call` → the pre-CPS carrier dispatch ABI.
///
/// Preserve its historical direct result mapping so the compatibility bridge
/// remains byte-for-byte transparent to later legacy consumers.
struct LowerLegacyCallPattern {
    types: CommonTypes,
}

impl RewritePattern for LowerLegacyCallPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if ability::LegacyCall::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let ability_ref_type = ctx.op(op).attributes.get_type("ability_ref").unwrap();
        let op_name_sym = ctx.op(op).attributes.get_symbol("op_name").unwrap();
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let Some(evidence_val) = find_evidence_from_op(ctx, op) else {
            return false;
        };
        let payload = pack_legacy_payload(ctx, rewriter, location, &operands, self.types.anyref);
        let dispatch = effect::dispatch_tail(
            ctx,
            location,
            evidence_val,
            payload,
            self.types.anyref,
            ability_ref_type,
            op_name_sym,
        );
        let result = dispatch.result(ctx);
        rewriter.insert_op(dispatch.op_ref());
        rewriter.erase_op(vec![result]);
        true
    }
}

/// Pattern: final `ability.call` → `effect.dispatch_tail`.
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
        if ability::Call::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let ability_ref_type = ctx.op(op).attributes.get_type("ability_ref").unwrap();
        let op_name_sym = ctx.op(op).attributes.get_symbol("op_name").unwrap();
        let result_types = ctx.op_result_types(op).to_vec();
        let [result_type] = result_types.as_slice() else {
            return false;
        };

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
        let shift_value_val = pack_payload(
            ctx,
            rewriter,
            location,
            ability_ref_type,
            op_name_sym,
            value_operands,
            t.anyref,
        );

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

        // The target-independent dispatch ABI erases the operation result.
        // Restore its exact source type before replacing the typed call result;
        // later CPS continuations still consume that logical value directly.
        let typed_result =
            core::unrealized_conversion_cast(ctx, location, dispatch_op.result(ctx), *result_type);
        rewriter.insert_op(typed_result.op_ref());

        // === 4. Erase ability.call, mapping its result to the typed dispatch result ===
        rewriter.erase_op(vec![typed_result.result(ctx)]);

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
        values.iter().map(|_| anyref),
    );
    let dynamic_values = values
        .iter()
        .map(|&value| {
            let cast = core::unrealized_conversion_cast(ctx, location, value, anyref);
            let result = cast.result(ctx);
            rewriter.insert_op(cast.op_ref());
            result
        })
        .collect::<Vec<_>>();
    let payload = adt::struct_new(ctx, location, dynamic_values, payload_type, payload_type);
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
        if func::Func::matches(ctx, parent) {
            // Found the enclosing func — check entry block args.
            let func_body = func::Func::from_op(ctx, parent).ok()?.body(ctx);
            let entry = ctx.region(func_body).blocks[0];
            return ctx
                .block_args(entry)
                .iter()
                .find(|&&arg| ability::is_evidence_type_ref(ctx, ctx.value_ty(arg)))
                .copied();
        }
        current_op = parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_lower_perform_basic() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> core.never {{
    %dispatch = arith.const {{value = 0}} : tribute_rt.anyref
    %resume = arith.const {{value = 1}} : tribute_rt.anyref
    ability.perform %ev, %dispatch, %resume {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}}
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert!(!ir_text.contains("ability.perform"), "{ir_text}");
        assert!(ir_text.contains("effect.dispatch_cps"), "{ir_text}");
        assert!(!ir_text.contains(" -> tribute_rt.anyref"), "{ir_text}");
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir_text);
    }

    #[test]
    fn test_lower_perform_with_args() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> core.never {{
    %val = arith.const {{value = 42}} : core.i32
    %dispatch = arith.const {{value = 0}} : tribute_rt.anyref
    %resume = arith.const {{value = 1}} : tribute_rt.anyref
    ability.perform %ev, %dispatch, %resume, %val {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @set}}
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert!(!ir_text.contains("ability.perform"), "{ir_text}");
        assert!(ir_text.contains("effect.dispatch_cps"), "{ir_text}");
        assert!(ir_text.contains("adt.struct_new"), "{ir_text}");
        assert!(
            ir_text.contains("fields = [[@arg0, tribute_rt.anyref]]"),
            "payload fields must use the canonical dynamic storage contract:\n{ir_text}"
        );
        assert!(
            ir_text.matches("core.unrealized_conversion_cast").count() >= 2,
            "the argument and packed product must both retain explicit dynamic views:\n{ir_text}"
        );
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir_text);
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
        assert!(!ir_text.contains("ability.call"), "{ir_text}");
        assert!(ir_text.contains("effect.dispatch_tail"), "{ir_text}");
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir_text);
    }

    #[test]
    fn lower_call_restores_the_exact_typed_result() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> core.i32 {{
    %result = ability.call {{ability_ref = core.ability_ref() {{name = @Counter}}, op_name = @next}} : core.i32
    func.return %result
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(!ir.contains("ability.call"), "{ir}");
        assert!(ir.contains("effect.dispatch_tail"), "{ir}");
        assert!(ir.contains("core.i32"), "{ir}");
        assert!(ir.contains("core.unrealized_conversion_cast"), "{ir}");
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir);
    }

    #[test]
    fn legacy_perform_uses_only_the_explicit_legacy_dispatch_abi() {
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
        assert!(!ir.contains("ability.legacy_perform"));
        assert!(!ir.contains("ability.legacy_call"));
        assert!(ir.contains("effect.legacy_dispatch_cps"));
        assert!(ir.contains("effect.dispatch_tail"));
        assert!(!ir.contains("effect.dispatch_cps"));
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir);
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
    %k = arith.const {value = 0} : tribute_rt.anyref
    ability.perform %k {ability_ref = core.ability_ref() {name = @State}, op_name = @get}
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

    #[test]
    fn result_bearing_final_perform_is_unchanged() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();
        let source = format!(
            r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref {{
    %k = arith.const {{value = 0}} : tribute_rt.anyref
    %result = ability.perform %ev, %k {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : tribute_rt.anyref
    func.return %result
  }}
}}"#
        );
        let module = parse_test_module(&mut ctx, &source);
        lower_ability_perform(&mut ctx, module);
        let output = print_module(&ctx, module.op());
        assert!(output.contains("ability.perform"), "{output}");
        assert!(!output.contains("effect.dispatch_cps"), "{output}");
    }
}
