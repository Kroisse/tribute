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
//! Uses `PatternApplicator` for declarative op-level rewriting. This is an
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

use tribute_core::calling_convention::CLOSURE_ENVIRONMENT_INDEX_ATTR;
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

        let Ok(answer_type) =
            crate::target_abi::dispatch_answer_type(ctx, evidence_val, dispatch_val, resume_val)
        else {
            return false;
        };
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
        let dispatch_op =
            effect::DispatchCps::operands(evidence_val, dispatch_val, resume_val, shift_value_val)
                .ability_ref(ability_ref_type)
                .op_name(op_name_sym)
                .answer_type(answer_type)
                .build(ctx, location);
        rewriter.insert_op(dispatch_op.op_ref());
        rewriter.erase_op(vec![]);
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

        // Consume the enclosing callable's exact hidden-parameter contract.
        let Some(evidence_val) = enclosing_callable_evidence(ctx, op) else {
            // Leave malformed input for the final ability boundary to reject.
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
        let dispatch_op = effect::DispatchTail::operands(evidence_val, shift_value_val)
            .ability_ref(ability_ref_type)
            .op_name(op_name_sym)
            .results(t.anyref)
            .build(ctx, location);
        rewriter.insert_op(dispatch_op.op_ref());

        // The target-independent dispatch ABI erases the operation result.
        // Restore its exact source type before replacing the typed call result;
        // later CPS continuations still consume that logical value directly.
        let typed_result = core::UnrealizedConversionCast::operands(dispatch_op.result(ctx))
            .results(*result_type)
            .build(ctx, location);
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
            let cast = core::UnrealizedConversionCast::operands(value)
                .results(anyref)
                .build(ctx, location);
            let result = cast.result(ctx);
            rewriter.insert_op(cast.op_ref());
            result
        })
        .collect::<Vec<_>>();
    let payload = adt::StructNew::operands(dynamic_values)
        .r#type(payload_type)
        .results(payload_type)
        .build(ctx, location);
    rewriter.insert_op(payload.op_ref());
    let erased = core::UnrealizedConversionCast::operands(payload.result(ctx))
        .results(anyref)
        .build(ctx, location);
    rewriter.insert_op(erased.op_ref());
    erased.result(ctx)
}

/// Read the canonical evidence slot of the nearest callable with a declared ABI.
///
/// A lifted closure whose type records environment index 0 stores that
/// environment ahead of the hidden evidence parameter, so evidence then
/// occupies the following slot.
fn enclosing_callable_evidence(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    let mut current = op;
    loop {
        let block = ctx.op(current).parent_block?;
        let region = ctx.block(block).parent_region?;
        let parent = ctx.region(region).parent_op?;
        if func::Func::matches(ctx, parent) {
            if !tribute_core::get_calling_convention(ctx, parent)?.needs_evidence() {
                return None;
            }
            let environment_index = ctx
                .op(parent)
                .attributes
                .get_u32(CLOSURE_ENVIRONMENT_INDEX_ATTR)
                .ok()
                .flatten();
            let evidence_index = usize::from(environment_index == Some(0));
            let body = *ctx.op(parent).regions.first()?;
            let entry = *ctx.region(body).blocks.first()?;
            let &evidence = ctx.block_args(entry).get(evidence_index)?;
            return ability::is_evidence_type_ref(ctx, ctx.value_ty(evidence)).then_some(evidence);
        }
        current = parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;
    use trunk_ir::ops::DialectType;
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

    fn attach_exact_perform_types(ctx: &mut IrContext, module: trunk_ir::rewrite::Module) {
        use tribute_core::calling_convention::*;
        let answer = ctx.intern_type(
            trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build(),
        );
        let evidence = ability::evidence_adt_type_ref(ctx);
        let anyref = tribute_rt::anyref(ctx).as_type_ref();
        let frame_name = Symbol::new("test_frame");
        let frame = cps_continuation_frame_ref_type(ctx, frame_name, answer);
        let done = cps_done_type(ctx, answer);
        let dispatch = cps_dispatch_type(ctx, evidence, frame, anyref, answer);
        let resume =
            func::FuncSig::from_type_ref(ctx, cps_closure_function_type(ctx, dispatch).unwrap())
                .unwrap()
                .inputs(ctx)[1];
        let layout = cps_continuation_frame_layout_type(ctx, frame_name, answer, done, dispatch);
        ctx.register_type_alias(frame_name, layout);
        let mut performs = Vec::new();
        let _ = trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
            if ability::Perform::matches(ctx, op) {
                performs.push(op);
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        for op in performs {
            for (value, ty) in [
                (ctx.op_operands(op)[1], dispatch),
                (ctx.op_operands(op)[2], resume),
            ] {
                let trunk_ir::refs::ValueDef::OpResult(producer, index) = ctx.value_def(value)
                else {
                    panic!("fixture constant")
                };
                ctx.set_op_result_type(producer, index, ty);
            }
        }
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

        attach_exact_perform_types(&mut ctx, module);
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

        attach_exact_perform_types(&mut ctx, module);
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
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref attributes {{tribute.calling_convention = 1}} {{
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
    fn call_requires_declared_convention_and_canonical_evidence_slot() {
        for (attributes, params) in [
            ("", "%ev: !Evidence"),
            (
                "attributes {tribute.calling_convention = 0}",
                "%ev: !Evidence",
            ),
            (
                "attributes {tribute.calling_convention = 1}",
                "%value: core.i32, %ev: !Evidence",
            ),
            (
                "attributes {tribute.calling_convention = 2}",
                "%value: core.i32, %ev: !Evidence",
            ),
        ] {
            let mut ctx = IrContext::new();
            let evidence = evidence_type_str();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !Evidence = {evidence}
  func.func @test_fn({params}) -> core.i32 {attributes} {{
    %result = ability.call {{ability_ref = core.ability_ref() {{name = @Counter}}, op_name = @next}} : core.i32
    func.return %result
  }}
}}"#
                ),
            );
            let before = print_module(&ctx, module.op());
            lower_ability_perform(&mut ctx, module);
            assert_eq!(print_module(&ctx, module.op()), before);
            assert!(crate::lower_handle_dispatch::lower_handle_dispatch(&mut ctx, module).is_err());
        }
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
  func.func @test_fn(%ev: {ev_ty}) -> core.i32 attributes {{tribute.calling_convention = 1}} {{
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
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref attributes {{tribute.calling_convention = 1}} {{
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
