//! Lower the bytes compiler intrinsic inside the native representation/ABI
//! boundary.
//!
//! Calls to the verified intrinsic `__bytes_get_or_panic`, selected and
//! validated by [`crate::bytes_intrinsic`], become shared `mem`/`arith`
//! operations on the native TributeBytes layout: `{ ptr: *const u8, len: u64 }`.
//!
//! The Wasm lowering lives in `wasm/bytes.rs`.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, mem};
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::Module;
use trunk_ir::types::TypeDataBuilder;

use crate::bytes_intrinsic::{self, BytesIntrinsicError};

/// Lower calls to the bytes intrinsic to native `mem` operations.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), BytesIntrinsicError> {
    bytes_intrinsic::lower_get_or_panic(ctx, module, lower_call)
}

/// Rewrite one bytes element read into loads on the TributeBytes layout.
///
/// TributeBytes native layout (payload pointer points here):
///   offset 0: ptr (*const u8) - 8 bytes
///   offset 8: len (u64)       - 8 bytes
///
/// Emits:
///   %payload  = core.unrealized_conversion_cast %bytes : core.ptr
///   %data_ptr = mem.load %payload {offset = 0} : core.ptr
///   %offset   = arith.extui %index : core.i64
///   %addr     = mem.ptr_add %data_ptr, %offset : core.ptr
///   %byte     = mem.load %addr {offset = 0} : core.i8
///   %result   = arith.extui %byte : core.i32
fn lower_call(ctx: &mut IrContext, call: OpRef) {
    let [bytes, index] = ctx.op_operands(call) else {
        unreachable!("call arity is validated before lowering")
    };
    let (bytes, index) = (*bytes, *index);
    let result_ty = ctx.op_result_types(call)[0];
    let loc = ctx.op(call).location;
    let ptr_ty = core::ptr(ctx).as_type_ref();
    let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
    let i8_ty = ctx.intern_type(TypeDataBuilder::new("core", "i8").build());

    // The Bytes payload is read in place: view the borrowed reference as its
    // native payload pointer. Native type conversion maps both to `core.ptr`,
    // so the cast folds away.
    let payload = core::UnrealizedConversionCast::operands(bytes)
        .results(ptr_ty)
        .build(ctx, loc);
    let data_ptr = mem::Load::operands(payload.result(ctx))
        .offset(0)
        .results(ptr_ty)
        .build(ctx, loc);
    // The index is an unsigned Nat and a byte is 0..=255: both widen with
    // zero extension.
    let offset = arith::Extui::operands(index)
        .results(i64_ty)
        .build(ctx, loc);
    let addr = mem::PtrAdd::operands(data_ptr.result(ctx), offset.result(ctx)).build(ctx, loc);
    let byte = mem::Load::operands(addr.result(ctx))
        .offset(0)
        .results(i8_ty)
        .build(ctx, loc);
    let value = arith::Extui::operands(byte.result(ctx))
        .results(result_ty)
        .build(ctx, loc);
    bytes_intrinsic::replace_call(
        ctx,
        call,
        &[
            payload.op_ref(),
            data_ptr.op_ref(),
            offset.op_ref(),
            addr.op_ref(),
            byte.op_ref(),
            value.op_ref(),
        ],
        value.result(ctx),
    );
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
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = @"std::__bytes_get_or_panic"}
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
}
