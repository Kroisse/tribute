//! Lower bytes intrinsic calls to native IR operations.
//!
//! This native-specific pass converts `func.call @__bytes_get_or_panic(bytes, index)`
//! to a sequence of `mem.load` operations that directly access the TributeBytes
//! memory layout: `{ ptr: *const u8, len: u64 }`.
//!
//! The WASM backend has its own lowering in `intrinsic_to_wasm.rs`.

use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::dialect::mem;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, LegalityDecision, Module, PatternApplicator,
    PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::TypeDataBuilder;

/// Lower bytes intrinsic calls to native mem operations.
///
/// Also removes `func.func` declarations for bytes intrinsics.
pub fn lower(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    let intrinsic_names: Rc<HashSet<Symbol>> =
        Rc::new([Symbol::new("__bytes_get_or_panic")].into());

    let mut applicator = PatternApplicator::new(TypeConverter::new());
    applicator =
        applicator
            .add_pattern(BytesGetOrPanicPattern)
            .add_pattern(BytesIntrinsicFuncDeclPattern {
                intrinsic_names: Rc::clone(&intrinsic_names),
            });

    let call_intrinsic_names = Rc::clone(&intrinsic_names);
    let func_intrinsic_names = Rc::clone(&intrinsic_names);
    let target = ConversionTarget::new()
        .dynamic_op("func", "call", move |ctx, op| {
            if let Ok(call_op) = func::Call::from_op(ctx, op)
                && call_intrinsic_names.contains(&call_op.callee(ctx))
            {
                return LegalityDecision::Illegal;
            }

            LegalityDecision::Defer
        })
        .dynamic_op("func", "func", move |ctx, op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op) {
                let attrs = &ctx.op(op).attributes;
                let is_intrinsic = attrs.get_str("abi") == Some("intrinsic");
                if is_intrinsic && func_intrinsic_names.contains(&func_op.sym_name(ctx)) {
                    return LegalityDecision::Illegal;
                }
            }

            LegalityDecision::Defer
        });

    applicator
        .with_target(target)
        .apply_partial_conversion(ctx, module, "intrinsic-to-native")?;
    Ok(())
}

/// Pattern that lowers `func.call @__bytes_get_or_panic(bytes, index)` to
/// mem.load operations on the TributeBytes layout.
///
/// TributeBytes native layout (payload pointer points here):
///   offset 0: ptr (*const u8) - 8 bytes
///   offset 8: len (u64)       - 8 bytes
///
/// Emits:
///   %data_ptr = mem.load %bytes, offset=0 : core.ptr
///   %addr = arith.addi %data_ptr, %index : core.ptr
///   %byte = mem.load %addr, offset=0 : core.i8
///   %result = arith.extend %byte : result_ty (Nat/i32)
struct BytesGetOrPanicPattern;

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
        if call_op.callee(ctx) != "__bytes_get_or_panic" {
            return false;
        }

        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let operands = ctx.op_operands(op).to_vec();
        let bytes = operands[0];
        let index = operands[1];

        let ptr_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());

        // Load data pointer from TributeBytes (offset 0)
        let data_ptr = mem::load(ctx, loc, bytes, ptr_ty, 0);
        rewriter.insert_op(data_ptr.op_ref());

        // Extend index (Nat = i32) to pointer width (i64)
        let index_ext = trunk_ir::dialect::arith::extend(ctx, loc, index, ptr_ty);
        rewriter.insert_op(index_ext.op_ref());

        // Compute address: data_ptr + index
        let addr = trunk_ir::dialect::arith::addi(
            ctx,
            loc,
            data_ptr.result(ctx),
            index_ext.result(ctx),
            ptr_ty,
        );
        rewriter.insert_op(addr.op_ref());

        // Load byte (i8) from computed address
        let byte_val = mem::load(ctx, loc, addr.result(ctx), i8_ty, 0);
        rewriter.insert_op(byte_val.op_ref());

        // Zero-extend i8 → result type (Nat = i32)
        let extended = trunk_ir::dialect::arith::extend(ctx, loc, byte_val.result(ctx), result_ty);
        rewriter.replace_op(extended.op_ref());

        true
    }
}

/// Pattern that removes `func.func` declarations for bytes intrinsics.
struct BytesIntrinsicFuncDeclPattern {
    intrinsic_names: Rc<HashSet<Symbol>>,
}

impl RewritePattern for BytesIntrinsicFuncDeclPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = func::Func::from_op(ctx, op) else {
            return false;
        };

        let attrs = &ctx.op(op).attributes;
        let is_intrinsic = attrs.get_str("abi") == Some("intrinsic");
        if !is_intrinsic {
            return false;
        }

        let sym_name = func_op.sym_name(ctx);
        if !self.intrinsic_names.contains(&sym_name) {
            return false;
        }

        rewriter.erase_op(vec![]);
        true
    }
}
