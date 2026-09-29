//! Lower the bytes compiler intrinsic inside the native representation/ABI
//! boundary.
//!
//! Calls to a declaration whose verified compiler intrinsic identity is
//! `__bytes_get_or_panic` become shared `mem`/`arith` operations on the
//! native TributeBytes layout: `{ ptr: *const u8, len: u64 }`. The pass is the
//! last reader of that identity and removes the declaration.
//!
//! The WASM backend has its own lowering in `intrinsic_to_wasm.rs`.

use std::collections::HashSet;
use std::rc::Rc;

use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, func, mem};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, LegalityDecision, Module, PatternApplicator,
    PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;

/// Canonical identity of the bytes element read intrinsic.
const BYTES_GET_OR_PANIC: &str = "__bytes_get_or_panic";

/// Lower calls to the bytes intrinsic to native `mem` operations and remove
/// its declaration.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    // Select declarations through their verified identity, never by name or
    // `abi` string.
    let eligible: Rc<HashSet<Symbol>> = Rc::new(
        module
            .ops(ctx)
            .into_iter()
            .filter_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                (ctx.op(op).attributes.get_symbol(COMPILER_INTRINSIC_ATTR)
                    == Some(Symbol::new(BYTES_GET_OR_PANIC)))
                .then(|| function.sym_name(ctx))
            })
            .collect(),
    );
    if eligible.is_empty() {
        return Ok(());
    }

    let call_eligible = Rc::clone(&eligible);
    let decl_eligible = Rc::clone(&eligible);
    let target = ConversionTarget::new()
        .dynamic_op("func", "call", move |ctx, op| {
            if func::Call::from_op(ctx, op)
                .is_ok_and(|call| call_eligible.contains(&call.callee(ctx)))
            {
                LegalityDecision::Illegal
            } else {
                LegalityDecision::Defer
            }
        })
        .dynamic_op("func", "func", move |ctx, op| {
            if func::Func::from_op(ctx, op)
                .is_ok_and(|function| decl_eligible.contains(&function.sym_name(ctx)))
            {
                LegalityDecision::Illegal
            } else {
                LegalityDecision::Defer
            }
        });

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(BytesGetOrPanicPattern {
            eligible: Rc::clone(&eligible),
        })
        .add_pattern(BytesIntrinsicDeclPattern { eligible })
        .with_target(target)
        .apply_partial_conversion(ctx, module, "intrinsic-to-native")?;
    Ok(())
}

/// Pattern that lowers a bytes element read to loads on the TributeBytes
/// layout.
///
/// TributeBytes native layout (payload pointer points here):
///   offset 0: ptr (*const u8) - 8 bytes
///   offset 8: len (u64)       - 8 bytes
///
/// Emits:
///   %data_ptr = mem.load %bytes {offset = 0} : core.ptr
///   %offset   = arith.extend %index : core.i64
///   %addr     = mem.ptr_add %data_ptr, %offset : core.ptr
///   %byte     = mem.load %addr {offset = 0} : core.i8
///   %result   = arith.extend %byte : result_ty
struct BytesGetOrPanicPattern {
    eligible: Rc<HashSet<Symbol>>,
}

impl RewritePattern for BytesGetOrPanicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = func::Call::from_op(ctx, op) else {
            return false;
        };
        if !self.eligible.contains(&call_op.callee(ctx)) {
            return false;
        }
        let operands = ctx.op_operands(op).to_vec();
        let ([bytes, index], [result_ty]) = (operands.as_slice(), ctx.op_result_types(op)) else {
            return false;
        };
        let (bytes, index, result_ty) = (*bytes, *index, *result_ty);

        let loc = ctx.op(op).location;
        let ptr_ty = ctx.intern_type(TypeDataBuilder::new("core", "ptr").build());
        let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
        let i8_ty = ctx.intern_type(TypeDataBuilder::new("core", "i8").build());

        let data_ptr = mem::Load::operands(bytes)
            .offset(0)
            .results(ptr_ty)
            .build(ctx, loc);
        rewriter.insert_op(data_ptr.op_ref());

        // Widen the index (Nat) to a pointer-width byte offset.
        let offset = arith::Extend::operands(index)
            .results(i64_ty)
            .build(ctx, loc);
        rewriter.insert_op(offset.op_ref());

        let addr = mem::PtrAdd::operands(data_ptr.result(ctx), offset.result(ctx)).build(ctx, loc);
        rewriter.insert_op(addr.op_ref());

        let byte = mem::Load::operands(addr.result(ctx))
            .offset(0)
            .results(i8_ty)
            .build(ctx, loc);
        rewriter.insert_op(byte.op_ref());

        // Zero-extend the byte to the result type (Nat).
        let extended = arith::Extend::operands(byte.result(ctx))
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(extended.op_ref());
        true
    }
}

/// Pattern that removes the bytes intrinsic declaration, consuming its
/// identity.
struct BytesIntrinsicDeclPattern {
    eligible: Rc<HashSet<Symbol>>,
}

impl RewritePattern for BytesIntrinsicDeclPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !func::Func::from_op(ctx, op)
            .is_ok_and(|function| self.eligible.contains(&function.sym_name(ctx)))
        {
            return false;
        }
        rewriter.erase_op(vec![]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn lower_text(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(&mut ctx, module).expect("bytes intrinsic lowering");
        print_module(&ctx, module.op())
    }

    #[test]
    fn verified_identity_selects_calls_and_consumes_the_declaration() {
        let printed = lower_text(
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = @__bytes_get_or_panic}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @read} : core.i32
    func.return %byte
  }
}"#,
        );

        assert!(!printed.contains("func.call"), "{printed}");
        assert!(!printed.contains("@read"), "{printed}");
        assert!(!printed.contains("tribute.compiler_intrinsic"), "{printed}");
        assert!(printed.contains("mem.ptr_add"), "{printed}");
        assert_eq!(printed.matches("mem.load").count(), 2, "{printed}");
        assert!(!printed.contains("clif."), "{printed}");
    }

    #[test]
    fn name_and_abi_alone_do_not_select_the_intrinsic() {
        let input = r#"core.module @test {
  func.func @__bytes_get_or_panic(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic"}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @__bytes_get_or_panic} : core.i32
    func.return %byte
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        lower(&mut ctx, module).expect("nothing to lower");

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
