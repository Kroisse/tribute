//! Lower the bytes compiler intrinsic inside the native representation/ABI
//! boundary.
//!
//! Calls to the verified intrinsic `__bytes_get_or_panic`, selected and
//! validated by [`crate::bytes_intrinsic`], become calls to the runtime's
//! `__tribute_bytes_get_or_panic`. The runtime borrows the `Bytes` for the
//! call, so the value stays live while its byte is read.
//!
//! The Wasm lowering lives in `wasm/bytes.rs`.

use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::Module;
use trunk_ir::types::TypeDataBuilder;

use crate::bytes_intrinsic::{self, BytesIntrinsicError};

/// Runtime function that reads one byte of a `Bytes`; see `tribute-runtime`.
const GET_OR_PANIC_FN: &str = "__tribute_bytes_get_or_panic";

/// Lower calls to the bytes intrinsic to calls to the native runtime.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), BytesIntrinsicError> {
    let mut lowered = false;
    bytes_intrinsic::lower_get_or_panic(ctx, module, |ctx, call| {
        lowered = true;
        lower_call(ctx, call);
    })?;
    if lowered {
        ensure_runtime_declaration(ctx, module);
    }
    Ok(())
}

/// Rewrite one bytes element read into a call to the runtime function, which
/// has the intrinsic's signature.
fn lower_call(ctx: &mut IrContext, call: OpRef) {
    let [bytes, index] = ctx.op_operands(call) else {
        unreachable!("call arity is validated before lowering")
    };
    let (bytes, index) = (*bytes, *index);
    let result_ty = ctx.op_result_types(call)[0];
    let loc = ctx.op(call).location;
    let read = func::Call::operands([bytes, index])
        .callee(SymbolPath::from(GET_OR_PANIC_FN))
        .results([result_ty])
        .build(ctx, loc);
    let value = read.results(ctx)[0];
    bytes_intrinsic::replace_call(ctx, call, &[read.op_ref()], value);
}

fn ensure_runtime_declaration(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    let declared = ctx.block(block).ops.iter().any(|&op| {
        func::Func::from_op(ctx, op).is_ok_and(|function| function.sym_name(ctx) == GET_OR_PANIC_FN)
    });
    if declared {
        return;
    }
    let loc = ctx.op(module.op()).location;
    let bytes_ty = core::bytes(ctx).as_type_ref();
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let declaration =
        super::build_extern_func(ctx, loc, GET_OR_PANIC_FN, &[bytes_ty, i32_ty], i32_ty);
    match ctx.block(block).ops.first().copied() {
        Some(first) => ctx.insert_op_before(block, first, declaration),
        None => ctx.push_op(block, declaration),
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
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::__bytes_get_or_panic"}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @read} : core.i32
    func.return %byte
  }
}"#,
        );

        assert!(!printed.contains("@read"), "{printed}");
        assert!(!printed.contains("tribute.compiler_intrinsic"), "{printed}");
        assert!(
            printed.contains(
                r#"func.func @__tribute_bytes_get_or_panic(%arg0: core.bytes, %arg1: core.i32) -> core.i32 attributes {abi = "C"}"#
            ),
            "{printed}"
        );
        assert!(
            printed
                .contains("func.call %0, %1 {callee = @__tribute_bytes_get_or_panic} : core.i32"),
            "{printed}"
        );
        assert!(!printed.contains("mem."), "{printed}");
    }
}
