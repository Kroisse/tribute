//! Lower `ability.call` operations to the effect ABI.
//!
//! ```text
//! // Input:
//! %result = ability.call [%args...]
//!   { ability: @State, op_name: "get" }
//!
//! // Output:
//! %payload = pack %args into the canonical operation product
//! %erased = effect.dispatch_tail %evidence, %payload
//!   { ability: @State, op_name: "get" }
//! ```
//!
//! `lower_continuation_frames` lowers `ability.perform` and `ability.abort`
//! with the same payload packing.
//!
//! Uses `PatternApplicator` for declarative op-level rewriting. This is an
//! intermediate best-effort pass: the final `ability-lowered` boundary is
//! established by `LowerHandleDispatch` after evidence resolution.

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    OpRewritePattern, PatternApplicator, PatternRewriter, RewriteScope, TypeConverter,
};

use crate::effect_dispatch::pack_payload;
use tribute_ir::calling_convention::DynConventionBearing;
use tribute_ir::dialect::ability;
use tribute_ir::dialect::effect;
use tribute_ir::dialect::tribute_rt;

/// Cached common type references used by the call lowering pattern.
#[derive(Clone, Copy)]
struct CommonTypes {
    anyref: TypeRef,
}

impl CommonTypes {
    fn new(ctx: &mut IrContext) -> Self {
        Self {
            anyref: tribute_rt::Anyref::params().intern(ctx).as_type_ref(),
        }
    }
}

/// Lower all currently legalizable `ability.call` ops.
///
/// Residual ability operations are allowed here and rejected at the final
/// `ability-lowered` boundary.
pub(crate) fn lower_ability_call<S: RewriteScope>(ctx: &mut IrContext, scope: S) {
    let types = CommonTypes::new(ctx);
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(LowerCallPattern { types });
    applicator.apply_partial(ctx, scope);
}

/// Lowers `ability.call` operations to `effect.dispatch_tail` with a packed
/// operation payload.
pub struct LowerAbilityCall;

impl Pass for LowerAbilityCall {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-ability-call"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: func::Func,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_ability_call(ctx, target);
        Ok(())
    }
}

/// Pattern: final `ability.call` → `effect.dispatch_tail`.
struct LowerCallPattern {
    types: CommonTypes,
}

impl OpRewritePattern for LowerCallPattern {
    type Op = ability::Call;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        call: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = call.op_ref();
        let location = ctx.op(op).location;
        let ability = ctx.op_attrs(op).get_symbol_ref("ability").cloned().unwrap();
        let op_name = ctx.op_attrs(op).get_string_ref("op_name").unwrap();
        let result_types = ctx.op_result_types(op).to_vec();
        let [result_type] = result_types.as_array()?;

        // Operands: [...values]
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let value_operands = &operands[..];

        let t = &self.types;

        // Consume the enclosing callable's exact hidden-parameter contract.
        let Some(evidence_val) = enclosing_callable_evidence(ctx, op) else {
            // Leave malformed input for the final ability boundary to reject.
            return None;
        };

        // === 2. Build the canonical payload product. ===
        let shift_value_val = pack_payload(
            ctx,
            rewriter,
            location,
            &ability,
            op_name,
            value_operands,
            t.anyref,
        );

        // === 3. Dispatch through target-independent effect ABI ===
        let dispatch_op = effect::DispatchTail::operands(evidence_val, shift_value_val)
            .ability(ability)
            .op_name(op_name)
            .build(ctx, location);
        rewriter.insert_op(dispatch_op.op_ref());

        // The target-independent dispatch ABI erases the operation result.
        // Restore its exact source type before replacing the typed call result;
        // later CPS continuations still consume that logical value directly.
        let typed_result =
            tribute_rt::convert_value(ctx, location, dispatch_op.result(ctx), *result_type);
        rewriter.insert_op(typed_result);

        // === 4. Erase ability.call, mapping its result to the typed dispatch result ===
        rewriter.erase_op(vec![ctx.op_result(typed_result, 0)]);

        Some(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Read the canonical evidence slot of the nearest callable with a declared ABI.
///
/// A convention that carries evidence places it first, ahead of any
/// environment.
fn enclosing_callable_evidence(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    let mut current = op;
    loop {
        let block = ctx.op(current).parent_block?;
        let region = ctx.block(block).parent_region?;
        let parent = ctx.region(region).parent_op?;
        if func::Func::matches(ctx, parent) {
            if !DynConventionBearing::of(ctx, parent)?.needs_evidence() {
                return None;
            }
            let entry = *ctx.region(ctx.op_region(parent, 0)?).blocks.first()?;
            let &evidence = ctx.block_args(entry).first()?;
            return ability::is_evidence_type_ref(ctx, ctx.value_ty(evidence)).then_some(evidence);
        }
        current = parent;
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
        "adt.array<adt.typeref<{name = \"std::runtime::evidence::Marker\"}>>"
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
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref attributes {{tribute.calling_convention = "evidence_direct"}} {{
    %msg = arith.const {{value = 1}} : tribute_rt.anyref
    %result = ability.call %msg {{ability = @Console, op_name = "print"}} : tribute_rt.anyref
    func.return %result
  }}
}}"#
            ),
        );

        lower_ability_call(&mut ctx, module);

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
                "attributes {tribute.calling_convention = \"direct\"}",
                "%ev: !Evidence",
            ),
            (
                "attributes {tribute.calling_convention = \"evidence_direct\"}",
                "%value: core.i32, %ev: !Evidence",
            ),
            (
                "attributes {tribute.calling_convention = \"cps\"}",
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
    %result = ability.call {{ability = @Counter, op_name = "next"}} : core.i32
    func.return %result
  }}
}}"#
                ),
            );
            let before = print_module(&ctx, module.op());
            lower_ability_call(&mut ctx, module);
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
  func.func @test_fn(%ev: {ev_ty}) -> core.i32 attributes {{tribute.calling_convention = "evidence_direct"}} {{
    %result = ability.call {{ability = @Counter, op_name = "next"}} : core.i32
    func.return %result
  }}
}}"#
            ),
        );

        lower_ability_call(&mut ctx, module);

        let ir = print_module(&ctx, module.op());
        assert!(!ir.contains("ability.call"), "{ir}");
        assert!(ir.contains("effect.dispatch_tail"), "{ir}");
        assert!(ir.contains("core.i32"), "{ir}");
        assert!(ir.contains("tribute_rt.unbox_int"), "{ir}");
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &ir);
    }
}
