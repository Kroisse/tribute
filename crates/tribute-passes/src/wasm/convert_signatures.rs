//! Convert function signatures to their Wasm types before target lowering.
//!
//! `convert-builtin-layouts` already replaced the `tribute_rt` primitives
//! everywhere, so no primitive type is left to normalize here. This pass
//! applies the remaining rules of `wasm_type_converter()` to the places that
//! declare a signature:
//!
//! - `func.func` and `wasm.func` signatures (parameter and result types)
//! - the exact `signature` of indirect calls, so each input slot follows the
//!   converted type of its argument

use tracing::debug;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::op_interface::DynIndirectCallLike;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    ConversionTarget, FuncSignatureConversionPattern, Module, PatternApplicator, PatternRewriter,
    RewritePattern,
};
use trunk_ir_wasm_backend::passes::signature_conversion::WasmFuncSignatureConversionPattern;

/// Convert function and indirect-call signatures to their Wasm types.
///
/// This pass should run early in the WASM pipeline, before target operation lowering.
pub(crate) fn lower(ctx: &mut IrContext, module: Module) {
    let type_converter = crate::wasm::type_converter::wasm_type_converter(ctx);

    // The operations whose signatures this pass converts follow their
    // converted operands. A retyped parameter that an operation not yet
    // lowered uses, such as an `adt.array_get` of an array parameter, is cast
    // back to the type that operation's schema requires; target lowering
    // consumes the cast.
    let applicator = PatternApplicator::new(type_converter)
        .with_target(
            ConversionTarget::new()
                .illegal_op("func.func")
                .illegal_op("wasm.func")
                .illegal_op("func.call_indirect")
                .illegal_op("func.tail_call_indirect"),
        )
        .with_auto_type_conversion(true)
        .add_pattern(FuncSignatureConversionPattern)
        .add_pattern(WasmFuncSignatureConversionPattern)
        .add_pattern(IndirectCallSignaturePattern);
    applicator.apply_partial(ctx, module);
}

// ============================================================================
// Patterns
// ============================================================================

/// Convert the exact signature of indirect calls.
///
/// `func.call_indirect` and `func.tail_call_indirect` declare their operand
/// and result types through the `signature` attribute, including for
/// resultless tail calls. Function parameters are retyped by the signature
/// patterns. An input slot therefore follows its argument only once the
/// argument holds a conversion of the declared type, which keeps the signature
/// equal to the argument types whatever order the patterns apply in.
struct IndirectCallSignaturePattern;

impl RewritePattern for IndirectCallSignaturePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Some(call) = DynIndirectCallLike::get(ctx, op) else {
            return false;
        };
        let Some(signature_ty) = call.exact_signature(ctx) else {
            return false;
        };
        let Some(signature) = func::FuncSig::from_type_ref(ctx, signature_ty) else {
            return false;
        };
        let declared = signature.inputs(ctx).to_vec();
        let results = signature.results(ctx).to_vec();
        let Some(arguments) = call.indirect_arguments(ctx) else {
            return false;
        };
        let arguments: Vec<_> = arguments.iter().map(|&arg| ctx.value_ty(arg)).collect();
        if arguments.len() != declared.len() {
            return false;
        }
        let mut inputs = Vec::with_capacity(declared.len());
        for (declared, actual) in declared.into_iter().zip(arguments) {
            let converted = actual != declared
                && rewriter.type_converter().convert_type(ctx, declared) == Some(actual);
            inputs.push(if converted { actual } else { declared });
        }
        let mut attrs = ctx.get_type(signature_ty).attrs.clone();
        func::FuncSig::remove_reserved_attrs(ctx, &mut attrs);
        let new_signature = func::func_sig_with_attrs(ctx, inputs, results, attrs).as_type_ref();

        if new_signature == signature_ty {
            return false;
        }
        let result_types = ctx.op_result_types(op).to_vec();

        let data = ctx.op(op);
        debug!(
            "convert_signatures: {} signature converted",
            data.name.display(ctx)
        );
        let mut builder = trunk_ir::context::OperationDataBuilder::new(data.location, data.name)
            .operands(ctx.op_operands(op).to_vec())
            .results(result_types);
        for (key, value) in data.attributes.clone() {
            builder = builder.attr(key, value);
        }
        let op_data = builder.build(ctx);
        let new_op = ctx.create_op(op_data);
        assert!(
            DynIndirectCallLike::get(ctx, new_op)
                .is_some_and(|call| call.set_exact_signature(ctx, new_signature)),
            "a rebuilt indirect call keeps its signature slot"
        );
        rewriter.replace_op(new_op);
        true
    }

    fn name(&self) -> &'static str {
        "IndirectCallSignaturePattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::validation::validate_op_schemas;

    #[test]
    fn resultless_indirect_tail_call_signature_follows_normalized_arguments() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @transfer(%callee: core.i32, %value: tribute_rt.anyref) {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(tribute_rt.anyref) -> ()>}
  }
}"#,
        );
        lower(&mut ctx, module);

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("tribute_rt.anyref"), "{output}");
        let schemas = validate_op_schemas(&ctx, module.op());
        assert!(schemas.is_ok(), "{schemas}\n{output}");
    }

    #[test]
    fn indirect_call_signature_keeps_slots_whose_argument_is_unconverted() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @transfer(%callee: core.i32, %value: core.i32) {
    func.tail_call_indirect %callee, %value {signature = func.func_sig<(core.i32) -> ()>}
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        lower(&mut ctx, module);

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
