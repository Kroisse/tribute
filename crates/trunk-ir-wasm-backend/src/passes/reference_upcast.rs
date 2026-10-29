//! Write WasmGC reference upcasts left as `core.unrealized_conversion_cast`
//! as `wasm.ref_upcast`.
//!
//! WasmGC accepts a reference of a subtype where a supertype is declared and
//! has no upcast instruction. The shared cast legalization never forwards a
//! value of another type, so a cast from a concrete reference to an abstract
//! GC reference stays after it. Operation schemas compare types exactly, so
//! the cast cannot simply be dropped; this target pattern replaces it with
//! `wasm.ref_upcast`, which emission writes as no instruction. Only the
//! widenings of the backend's physical assignability rule
//! ([`is_wasm_physical_argument_assignable`]) qualify: those emission performs
//! without a runtime cast.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::{OpRewritePattern, PatternRewriter};

use crate::is_wasm_physical_argument_assignable;

/// Replace a cast from a concrete WasmGC reference to an abstract supertype
/// with `wasm.ref_upcast`.
///
/// Matches only a cast whose result already has its converted type and is
/// `wasm.anyref`, `wasm.structref`, or `wasm.arrayref`. Register it after the
/// cast legalization pattern so that result types are converted first.
pub struct ReferenceUpcastPattern;

impl OpRewritePattern for ReferenceUpcastPattern {
    type Op = core::UnrealizedConversionCast;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        unrealized_conversion_cast: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = unrealized_conversion_cast.op_ref();
        let (&[input], &[result_ty]) = (
            ctx.op_operands(op).as_array()?,
            ctx.op_result_types(op).as_array()?,
        );
        if rewriter
            .type_converter()
            .convert_type_or_identity(ctx, result_ty)
            != result_ty
            || !is_abstract_gc_reference(ctx, result_ty)
        {
            return None;
        }
        let input_ty = ctx.value_ty(input);
        if input_ty == result_ty || !is_wasm_physical_argument_assignable(ctx, input_ty, result_ty)
        {
            return None;
        }
        let location = ctx.op(op).location;
        let upcast = wasm::RefUpcast::operands(input)
            .results(result_ty)
            .build(ctx, location);
        rewriter.replace_op(upcast.op_ref());
        Some(())
    }
}

fn is_abstract_gc_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    use trunk_ir::dialect::wasm::{Anyref, Arrayref, Structref};
    use trunk_ir::ops::DialectType;
    [Anyref::NAME, Structref::NAME, Arrayref::NAME].contains(&ctx.get_type(ty).name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::{Module, PatternApplicator, TypeConverter};

    fn upcast(ctx: &mut IrContext, module: Module, tc: TypeConverter) {
        let result = PatternApplicator::new(tc)
            .add_pattern(ReferenceUpcastPattern)
            .apply_partial(ctx, module);
        assert!(result.reached_fixpoint);
    }

    #[track_caller]
    fn assert_ir(ctx: &IrContext, module: Module, expected: &str) {
        let mut expected_ctx = IrContext::new();
        let expected_module = parse_test_module(&mut expected_ctx, expected);
        assert_eq!(
            print_module(ctx, module.op()),
            print_module(&expected_ctx, expected_module.op())
        );
    }

    #[test]
    fn writes_registered_and_abstract_upcasts() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Closure = test.layout<core.i32, wasm.anyref, {layout = "closure"}>
  !Ref = wasm_gc.struct<core.i32>
  func.func @f(%c: !Closure, %r: !Ref, %s: wasm.structref, %a: wasm_gc.array<core.i32>) {
    %c_struct = core.unrealized_conversion_cast %c : wasm.structref
    %c_any = core.unrealized_conversion_cast %c : wasm.anyref
    %r_struct = core.unrealized_conversion_cast %r : wasm.structref
    %s_any = core.unrealized_conversion_cast %s : wasm.anyref
    %a_array = core.unrealized_conversion_cast %a : wasm.arrayref
    func.call %c_struct, %c_any, %r_struct, %s_any, %a_array {callee = @use}
    func.return
  }
}"#,
        );

        upcast(&mut ctx, module, TypeConverter::new());

        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  !Closure = test.layout<core.i32, wasm.anyref, {layout = "closure"}>
  !Ref = wasm_gc.struct<core.i32>
  func.func @f(%c: !Closure, %r: !Ref, %s: wasm.structref, %a: wasm_gc.array<core.i32>) {
    %c_struct = wasm.ref_upcast %c : wasm.structref
    %c_any = wasm.ref_upcast %c : wasm.anyref
    %r_struct = wasm.ref_upcast %r : wasm.structref
    %s_any = wasm.ref_upcast %s : wasm.anyref
    %a_array = wasm.ref_upcast %a : wasm.arrayref
    func.call %c_struct, %c_any, %r_struct, %s_any, %a_array {callee = @use}
    func.return
  }
}"#,
        );
    }

    #[test]
    fn keeps_downcasts_and_unregistered_struct_to_structref() {
        let input = r#"core.module @test {
  !Plain = test.opaque<core.i32>
  func.func @f(%any: wasm.anyref, %plain: !Plain) {
    %down = core.unrealized_conversion_cast %any : wasm.structref
    %plain_struct = core.unrealized_conversion_cast %plain : wasm.structref
    func.call %down, %plain_struct {callee = @use}
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        upcast(&mut ctx, module, TypeConverter::new());

        assert_ir(&ctx, module, input);
    }

    #[test]
    fn waits_for_the_result_type_to_be_converted() {
        let input = r#"core.module @test {
  !Closure = test.layout<core.i32, wasm.anyref, {layout = "closure"}>
  func.func @f(%c: !Closure) {
    %r = core.unrealized_conversion_cast %c : wasm.structref
    func.call %r {callee = @use}
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        // An upcast whose declared result still converts is left to the cast
        // legalization pattern.
        let structref = trunk_ir::dialect::wasm::Structref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let anyref = trunk_ir::dialect::wasm::Anyref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == structref).then_some(anyref));

        upcast(&mut ctx, module, tc);

        assert_ir(&ctx, module, input);
    }
}
