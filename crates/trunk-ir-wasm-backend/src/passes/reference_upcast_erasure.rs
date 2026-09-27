//! Erase WasmGC reference upcasts left as `core.unrealized_conversion_cast`.
//!
//! WasmGC accepts a reference of a subtype where a supertype is declared and
//! has no upcast instruction. The shared cast legalization never forwards a
//! value of another type, so a cast from a concrete reference to an abstract
//! GC reference stays after it. This target pattern erases such a cast and
//! uses its source directly, following the backend's physical assignability
//! rule ([`is_wasm_physical_argument_assignable`]): only widenings the
//! emission performs without a runtime cast qualify.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};

use crate::is_wasm_physical_argument_assignable;

/// Erase a cast from a concrete WasmGC reference to an abstract supertype.
///
/// Matches only a cast whose result already has its converted type and is
/// `wasm.anyref`, `wasm.structref`, or `wasm.arrayref`. Register it after the
/// cast legalization pattern so that result types are converted first.
pub struct ReferenceUpcastErasurePattern;

impl RewritePattern for ReferenceUpcastErasurePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !core::UnrealizedConversionCast::matches(ctx, op) {
            return false;
        }
        let (&[input], &[result_ty]) = (ctx.op_operands(op), ctx.op_result_types(op)) else {
            return false;
        };
        if rewriter
            .type_converter()
            .convert_type_or_identity(ctx, result_ty)
            != result_ty
            || !is_abstract_gc_reference(ctx, result_ty)
        {
            return false;
        }
        let input_ty = ctx.value_ty(input);
        if input_ty == result_ty || !is_wasm_physical_argument_assignable(ctx, input_ty, result_ty)
        {
            return false;
        }
        rewriter.erase_op(vec![input]);
        true
    }
}

fn is_abstract_gc_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.get_type(ty);
    data.dialect == Symbol::new("wasm")
        && [
            Symbol::new("anyref"),
            Symbol::new("structref"),
            Symbol::new("arrayref"),
        ]
        .contains(&data.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::{Module, PatternApplicator, TypeConverter};

    fn erase(ctx: &mut IrContext, module: Module, tc: TypeConverter) {
        let result = PatternApplicator::new(tc)
            .add_pattern(ReferenceUpcastErasurePattern)
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
    fn erases_registered_and_abstract_upcasts() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, wasm.anyref]]}
  !Ref = adt.typeref() {name = @Node}
  func.func @f(%c: !Closure, %r: !Ref, %s: wasm.structref, %a: core.array(core.i32)) {
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

        erase(&mut ctx, module, TypeConverter::new());

        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  !Closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, wasm.anyref]]}
  !Ref = adt.typeref() {name = @Node}
  func.func @f(%c: !Closure, %r: !Ref, %s: wasm.structref, %a: core.array(core.i32)) {
    func.call %c, %c, %r, %s, %a {callee = @use}
    func.return
  }
}"#,
        );
    }

    #[test]
    fn keeps_downcasts_and_unregistered_struct_to_structref() {
        let input = r#"core.module @test {
  !Plain = adt.struct() {name = @Plain, fields = [[@x, core.i32]]}
  func.func @f(%any: wasm.anyref, %plain: !Plain) {
    %down = core.unrealized_conversion_cast %any : wasm.structref
    %plain_struct = core.unrealized_conversion_cast %plain : wasm.structref
    func.call %down, %plain_struct {callee = @use}
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        erase(&mut ctx, module, TypeConverter::new());

        assert_ir(&ctx, module, input);
    }

    #[test]
    fn waits_for_the_result_type_to_be_converted() {
        let input = r#"core.module @test {
  !Closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, wasm.anyref]]}
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
        let new_type = |ctx: &mut IrContext, name| {
            ctx.intern_type(trunk_ir::types::TypeDataBuilder::new("wasm", name).build())
        };
        let structref = new_type(&mut ctx, "structref");
        let anyref = new_type(&mut ctx, "anyref");
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == structref).then_some(anyref));

        erase(&mut ctx, module, tc);

        assert_ir(&ctx, module, input);
    }
}
