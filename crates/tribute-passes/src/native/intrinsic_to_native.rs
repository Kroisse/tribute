//! Lower the bytes compiler intrinsic inside the native representation/ABI
//! boundary.
//!
//! Calls whose callee resolves to a declaration carrying the verified compiler
//! intrinsic identity `__bytes_get_or_panic` become shared `mem`/`arith`
//! operations on the native TributeBytes layout: `{ ptr: *const u8, len: u64 }`.
//! The pass is the last reader of that identity.
//!
//! The WASM backend has its own lowering in `intrinsic_to_wasm.rs`.

use std::ops::ControlFlow;

use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, func, mem};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::{Attribute, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

/// Canonical identity of the bytes element read intrinsic.
const BYTES_GET_OR_PANIC: &str = "__bytes_get_or_panic";

/// A bytes intrinsic declaration or use the native lowering cannot honor.
#[derive(Debug, derive_more::Display, derive_more::Error)]
#[display("native bytes intrinsic lowering: {message}")]
pub struct BytesIntrinsicError {
    message: String,
}

impl BytesIntrinsicError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Lower calls to the bytes intrinsic to native `mem` operations.
///
/// Each declaration with the intrinsic identity must have the exact signature
/// `(core.bytes, core.i32) -> core.i32`. Every call resolving to one is
/// rewritten; the identity is then consumed, and the declaration is removed
/// because nothing else may reference it.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), BytesIntrinsicError> {
    let declarations: Vec<OpRef> = module
        .ops(ctx)
        .into_iter()
        .filter(|&op| {
            func::Func::matches(ctx, op)
                && ctx.op(op).attributes.get_symbol(COMPILER_INTRINSIC_ATTR)
                    == Some(Symbol::new(BYTES_GET_OR_PANIC))
        })
        .collect();
    if declarations.is_empty() {
        return Ok(());
    }
    let expected = exact_signature(ctx);
    for &declaration in &declarations {
        let function = func::Func::from_op(ctx, declaration).expect("filtered func.func");
        if function.r#type(ctx) != expected {
            return Err(BytesIntrinsicError::new(format!(
                "`{}` must have the exact signature {}",
                function.sym_name(ctx),
                trunk_ir::printer::print_type(ctx, expected),
            )));
        }
    }

    // Select calls through the declaration they resolve to, never by callee
    // spelling or `abi` string.
    let symbols = SymbolTable::collect(ctx, module);
    let mut calls = Vec::new();
    let mut other_references = Vec::new();
    let names: Vec<Symbol> = declarations
        .iter()
        .map(|&op| {
            func::Func::from_op(ctx, op)
                .expect("func.func")
                .sym_name(ctx)
        })
        .collect();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(call) = func::Call::from_op(ctx, op)
            && symbols
                .resolve(call.callee(ctx))
                .is_some_and(|callee| declarations.contains(&callee))
        {
            calls.push(op);
        } else if !declarations.contains(&op) && references_any(ctx, op, &names) {
            other_references.push(op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    if let Some(&op) = other_references.first() {
        return Err(BytesIntrinsicError::new(format!(
            "the intrinsic is referenced by {}.{}, not only called",
            ctx.op(op).dialect,
            ctx.op(op).name,
        )));
    }
    // Validate every call's shape before rewriting any of them.
    if let Some(&call) = calls
        .iter()
        .find(|&&call| ctx.op_operands(call).len() != 2 || ctx.op_results(call).len() != 1)
    {
        return Err(BytesIntrinsicError::new(format!(
            "a call to the intrinsic has {} operand(s) and {} result(s), expected 2 and 1",
            ctx.op_operands(call).len(),
            ctx.op_results(call).len(),
        )));
    }

    for call in calls {
        lower_call(ctx, call);
    }
    // Every use was rewritten against the verified identity, so this pass is
    // its last reader and the declaration has no remaining references.
    for declaration in declarations {
        ctx.detach_op(declaration);
        ctx.remove_op(declaration);
    }
    Ok(())
}

/// The intrinsic's exact signature: `(core.bytes, core.i32) -> core.i32`.
fn exact_signature(ctx: &mut IrContext) -> TypeRef {
    let bytes = core::bytes(ctx).as_type_ref();
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    func::func_sig(ctx, [bytes, i32_ty], [i32_ty]).as_type_ref()
}

/// Whether any attribute of `op` names one of `names`.
fn references_any(ctx: &IrContext, op: OpRef, names: &[Symbol]) -> bool {
    ctx.op(op)
        .attributes
        .values()
        .any(|attribute| matches!(attribute, Attribute::Symbol(symbol) if names.contains(symbol)))
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
///   %offset   = arith.extend %index : core.i64
///   %addr     = mem.ptr_add %data_ptr, %offset : core.ptr
///   %byte     = mem.load %addr {offset = 0} : core.i8
///   %result   = arith.extend %byte : core.i32
fn lower_call(ctx: &mut IrContext, call: OpRef) {
    let [bytes, index] = ctx.op_operands(call) else {
        unreachable!("call arity is validated before lowering")
    };
    let (bytes, index) = (*bytes, *index);
    let result_ty = ctx.op_result_types(call)[0];
    let block = ctx.op(call).parent_block.expect("call in a block");
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
    // Widen the index (Nat) to a pointer-width byte offset.
    let offset = arith::Extend::operands(index)
        .results(i64_ty)
        .build(ctx, loc);
    let addr = mem::PtrAdd::operands(data_ptr.result(ctx), offset.result(ctx)).build(ctx, loc);
    let byte = mem::Load::operands(addr.result(ctx))
        .offset(0)
        .results(i8_ty)
        .build(ctx, loc);
    // Zero-extend the byte to the result type (Nat).
    let extended = arith::Extend::operands(byte.result(ctx))
        .results(result_ty)
        .build(ctx, loc);
    for op in [
        payload.op_ref(),
        data_ptr.op_ref(),
        offset.op_ref(),
        addr.op_ref(),
        byte.op_ref(),
        extended.op_ref(),
    ] {
        ctx.insert_op_before(block, call, op);
    }
    let old = ctx.op_results(call)[0];
    ctx.replace_all_uses(old, extended.result(ctx));
    ctx.detach_op(call);
    ctx.remove_op(call);
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
    fn declaration_with_another_signature_is_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i64) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = @__bytes_get_or_panic}
}"#,
        );

        let error = lower(&mut ctx, module).expect_err("the signature must be exact");

        assert!(error.to_string().contains("exact signature"), "{error}");
    }

    #[test]
    fn non_call_reference_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = @__bytes_get_or_panic}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @read} : core.i32
    %f = func.constant {func_ref = @read} : func.func_sig<(core.bytes, core.i32) -> core.i32>
    func.return %byte
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower(&mut ctx, module).expect_err("a first-class use cannot be lowered");

        assert!(error.to_string().contains("func.constant"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn malformed_call_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = @__bytes_get_or_panic}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %ok = func.call %bytes, %index {callee = @read} : core.i32
    %bad = func.call %bytes {callee = @read} : core.i32
    func.return %ok
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower(&mut ctx, module).expect_err("a call must have two operands");

        assert!(error.to_string().contains("expected 2 and 1"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
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
