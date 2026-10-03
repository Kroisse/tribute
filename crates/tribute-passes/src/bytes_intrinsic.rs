//! Selection and validation shared by the target lowerings of the bytes
//! compiler intrinsic.
//!
//! Calls whose callee resolves to a declaration carrying the verified compiler
//! intrinsic identity `std::__bytes_get_or_panic` are handed to a target's
//! rewrite of one call. The lowering is the last reader of that identity: it
//! consumes the identity and removes the declaration.

use std::ops::ControlFlow;

use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::{Attribute, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

/// Canonical identity of the bytes element read intrinsic.
const BYTES_GET_OR_PANIC: &str = "std::__bytes_get_or_panic";

/// A bytes intrinsic declaration or use a target lowering cannot honor.
#[derive(Debug, derive_more::Display, derive_more::Error)]
#[display("bytes intrinsic lowering: {message}")]
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

/// Lower every call to the bytes element read intrinsic with `lower_call`.
///
/// Each declaration with the intrinsic identity must have the exact signature
/// `(core.bytes, core.i32) -> core.i32`, and it may only be called, each call
/// with two operands and one result. Everything is validated before any call
/// is rewritten. `lower_call` rewrites one call; afterwards the declarations
/// are removed because nothing references them.
pub(crate) fn lower_get_or_panic(
    ctx: &mut IrContext,
    module: Module,
    mut lower_call: impl FnMut(&mut IrContext, OpRef),
) -> Result<(), BytesIntrinsicError> {
    let declarations: Vec<OpRef> = module
        .ops(ctx)
        .iter()
        .copied()
        .filter(|&op| {
            func::Func::matches(ctx, op)
                && ctx.op(op).attributes.get_str(ctx, COMPILER_INTRINSIC_ATTR)
                    == Some(BYTES_GET_OR_PANIC)
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
            Symbol::from_dynamic(
                func::Func::from_op(ctx, op)
                    .expect("func.func")
                    .sym_name(ctx),
            )
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

/// Replace `call` by `ops`, inserted before it, whose last result `value`
/// takes over the call's result.
pub(crate) fn replace_call(ctx: &mut IrContext, call: OpRef, ops: &[OpRef], value: ValueRef) {
    let block = ctx.op(call).parent_block.expect("call in a block");
    for &op in ops {
        ctx.insert_op_before(block, call, op);
    }
    let old = ctx.op_results(call)[0];
    ctx.replace_all_uses(old, value);
    ctx.detach_op(call);
    ctx.remove_op(call);
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    /// Run the shared validation with a rewrite that must not be reached.
    fn validate(ctx: &mut IrContext, module: Module) -> Result<(), BytesIntrinsicError> {
        lower_get_or_panic(ctx, module, |_, _| {
            unreachable!("invalid input must be rejected before any rewrite")
        })
    }

    #[test]
    fn declaration_with_another_signature_is_rejected() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i64) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::__bytes_get_or_panic"}
}"#,
        );

        let error = validate(&mut ctx, module).expect_err("the signature must be exact");

        assert!(error.to_string().contains("exact signature"), "{error}");
    }

    #[test]
    fn non_call_reference_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::__bytes_get_or_panic"}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @read} : core.i32
    %f = func.constant {func_ref = @read} : func.func_sig<(core.bytes, core.i32) -> core.i32>
    func.return %byte
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = validate(&mut ctx, module).expect_err("a first-class use cannot be lowered");

        assert!(error.to_string().contains("func.constant"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn malformed_call_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @read(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic", tribute.compiler_intrinsic = "std::__bytes_get_or_panic"}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %ok = func.call %bytes, %index {callee = @read} : core.i32
    %bad = func.call %bytes {callee = @read} : core.i32
    func.return %ok
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = validate(&mut ctx, module).expect_err("a call must have two operands");

        assert!(error.to_string().contains("expected 2 and 1"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn name_and_abi_alone_do_not_select_the_intrinsic() {
        let input = r#"core.module @test {
  func.func @"std::__bytes_get_or_panic"(%bytes: core.bytes, %index: core.i32) -> core.i32 attributes {abi = "intrinsic"}
  func.func @user(%bytes: core.bytes, %index: core.i32) -> core.i32 {
    %byte = func.call %bytes, %index {callee = @"std::__bytes_get_or_panic"} : core.i32
    func.return %byte
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        validate(&mut ctx, module).expect("nothing to lower");

        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
