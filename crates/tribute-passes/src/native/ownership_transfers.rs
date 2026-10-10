//! Lower the ownership transfer operations to representation conversions.
//!
//! `tribute_rt.into_raw` and `tribute_rt.from_raw` exist for ownership
//! planning: one consumes a managed value's unit, the other creates one. Once
//! the plan is materialized each is only a change of type between a managed
//! reference and `core.ptr`, which native type conversion later resolves to
//! an identity.
//!
//! This runs as the last step of ownership lowering, while managed types are
//! still distinct from `core.ptr`, so the operations satisfy their declared
//! types for as long as they exist.

use tribute_ir::dialect::tribute_rt;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

/// Replace every ownership transfer in `module` by an unrealized cast.
pub(super) fn lower(ctx: &mut IrContext, module: Module) {
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(TransferToCast)
        .apply_partial(ctx, module);
}

struct TransferToCast;

impl RewritePattern for TransferToCast {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let source = if let Ok(into_raw) = tribute_rt::IntoRaw::from_op(ctx, op) {
            into_raw.value(ctx)
        } else if let Ok(from_raw) = tribute_rt::FromRaw::from_op(ctx, op) {
            from_raw.ptr(ctx)
        } else {
            return false;
        };
        let cast = core::UnrealizedConversionCast::operands(source)
            .results(ctx.op_result_types(op)[0])
            .build(ctx, ctx.op(op).location);
        rewriter.replace_op(cast.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "TransferToCast"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::validation::validate_op_schemas;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(&mut ctx, module);
        print_module(&ctx, module.op())
    }

    #[test]
    fn the_transfers_declare_a_managed_reference_and_a_raw_pointer() {
        let accepted = |op: &str, from: &str, to: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @f(%value: {from}) -> {to} {{
    %result = tribute_rt.{op} %value : {to}
    func.return %result
  }}
}}"#
                ),
            );
            validate_op_schemas(&ctx, module.op()).is_ok()
        };
        let closure = "closure.closure<func.func_sig<() -> core.nil>>";
        for managed in ["core.bytes", "tribute_rt.anyref", closure] {
            assert!(accepted("into_raw", managed, "core.ptr"), "{managed}");
            assert!(accepted("from_raw", "core.ptr", managed), "{managed}");
        }
        // Neither side accepts the other's type or a scalar.
        assert!(!accepted("into_raw", "core.ptr", "core.ptr"));
        assert!(!accepted("into_raw", "core.i32", "core.ptr"));
        assert!(!accepted("into_raw", "core.bytes", "core.bytes"));
        assert!(!accepted("from_raw", "core.bytes", "core.bytes"));
        assert!(!accepted("from_raw", "core.ptr", "core.ptr"));
        assert!(!accepted("from_raw", "core.ptr", "core.i32"));
    }

    #[test]
    fn into_raw_becomes_a_conversion_to_a_raw_pointer() {
        let output = run_pass(
            r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  func.func @f(%closure: !_closure) -> core.ptr {
    %raw = tribute_rt.into_raw %closure : core.ptr
    func.return %raw
  }
}"#,
        );
        assert!(!output.contains("tribute_rt.into_raw"), "{output}");
        assert!(
            output.contains("core.unrealized_conversion_cast %0 : core.ptr"),
            "{output}"
        );
    }

    #[test]
    fn from_raw_becomes_a_conversion_to_its_managed_type() {
        let output = run_pass(
            r#"core.module @test {
  func.func @f(%raw: core.ptr) -> core.bytes {
    %bytes = tribute_rt.from_raw %raw : core.bytes
    func.return %bytes
  }
}"#,
        );
        assert!(!output.contains("tribute_rt.from_raw"), "{output}");
        assert!(
            output.contains("core.unrealized_conversion_cast %0 : core.bytes"),
            "{output}"
        );
    }
}
