//! Arena-based clif dialect.

use crate::op_interface::{IndirectCallLikeModel, IndirectCallLikeOps};
use crate::ops::{DialectOp, DialectType};

#[trunk_ir::dialect]
mod clif {
    // Module
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)]
        {}
    }

    #[attr(callee: Symbol)]
    fn call(#[rest] args: ()) -> result {}

    #[attr(sig: Type)]
    fn call_indirect(callee: (), #[rest] args: ()) -> result {}

    fn r#return(#[rest] values: ()) {}

    // Constants
    #[attr(value: i64)]
    fn iconst() -> result {}

    #[attr(value: f32)]
    fn f32const() -> result {}

    #[attr(value: f64)]
    fn f64const() -> result {}

    // Integer arithmetic
    fn iadd(lhs: (), rhs: ()) -> result {}
    fn isub(lhs: (), rhs: ()) -> result {}
    fn imul(lhs: (), rhs: ()) -> result {}
    fn sdiv(lhs: (), rhs: ()) -> result {}
    fn udiv(lhs: (), rhs: ()) -> result {}
    fn srem(lhs: (), rhs: ()) -> result {}
    fn urem(lhs: (), rhs: ()) -> result {}
    fn ineg(operand: ()) -> result {}

    // Float arithmetic
    fn fadd(lhs: (), rhs: ()) -> result {}
    fn fsub(lhs: (), rhs: ()) -> result {}
    fn fmul(lhs: (), rhs: ()) -> result {}
    fn fdiv(lhs: (), rhs: ()) -> result {}
    fn fneg(operand: ()) -> result {}

    // Comparisons
    #[attr(cond: Symbol)]
    fn icmp(lhs: (), rhs: ()) -> result {}

    #[attr(cond: Symbol)]
    fn fcmp(lhs: (), rhs: ()) -> result {}

    // Bitwise
    fn band(lhs: (), rhs: ()) -> result {}
    fn bor(lhs: (), rhs: ()) -> result {}
    fn bxor(lhs: (), rhs: ()) -> result {}
    fn ishl(lhs: (), rhs: ()) -> result {}
    fn sshr(lhs: (), rhs: ()) -> result {}
    fn ushr(lhs: (), rhs: ()) -> result {}

    // Control flow
    fn brif(cond: ()) {
        #[successor(then_dest)]
        {}
        #[successor(else_dest)]
        {}
    }

    fn jump(#[rest] args: ()) {
        #[successor(dest)]
        {}
    }

    #[attr(table: any)]
    fn br_table(index: ()) {}

    #[attr(code: Symbol)]
    fn trap() {}

    #[attr(callee: Symbol)]
    fn return_call(#[rest] args: ()) {}

    #[attr(sig: Type)]
    fn return_call_indirect(callee: (), #[rest] args: ()) {}

    // Memory
    #[attr(offset: i32)]
    fn load(addr: ()) -> result {}

    #[attr(offset: i32)]
    fn store(value: (), addr: ()) {}

    #[attr(op: Symbol, offset: i32)]
    fn atomic_rmw(addr: (), value: ()) -> result {}

    #[attr(size: u32, align: u32)]
    fn stack_slot() -> result {}

    fn stack_addr(slot: ()) -> result {}

    #[attr(sym: Symbol)]
    fn symbol_addr() -> result {}

    // Type conversions
    fn ireduce(operand: ()) -> result {}
    fn uextend(operand: ()) -> result {}
    fn sextend(operand: ()) -> result {}
    fn fpromote(operand: ()) -> result {}
    fn fdemote(operand: ()) -> result {}
    fn fcvt_to_sint(operand: ()) -> result {}
    fn fcvt_from_sint(operand: ()) -> result {}
    fn fcvt_to_uint(operand: ()) -> result {}
    fn fcvt_from_uint(operand: ()) -> result {}
}

const INDIRECT_CALL_SIGNATURE_ATTR: &str = "sig";

impl IndirectCallLikeModel for CallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        Some(self.sig(ctx))
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl IndirectCallLikeModel for ReturnCallIndirect {
    fn exact_signature(self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        Some(self.sig(ctx))
    }

    fn set_exact_signature(self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

inventory::submit! {
    IndirectCallLikeOps::register::<CallIndirect>()
}

inventory::submit! {
    IndirectCallLikeOps::register::<ReturnCallIndirect>()
}

/// Attach the required exact callable contract to a `clif` indirect transfer.
///
/// Returns `false` without mutation when the operation is not a `clif`
/// indirect call or the supplied type is not a `core.func` contract.
pub fn set_indirect_call_signature(
    ctx: &mut crate::IrContext,
    op: crate::OpRef,
    signature: crate::TypeRef,
) -> bool {
    if crate::dialect::core::Func::from_type_ref(ctx, signature).is_none()
        || (CallIndirect::from_op(ctx, op).is_err()
            && ReturnCallIndirect::from_op(ctx, op).is_err())
    {
        return false;
    }
    set_indirect_call_signature_attribute(&mut ctx.op_mut(op).attributes, signature);
    true
}

fn set_indirect_call_signature_attribute(
    attributes: &mut crate::AttributeMap,
    signature: crate::TypeRef,
) {
    attributes.insert(
        crate::Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR),
        crate::Attribute::Type(signature),
    );
}

#[cfg(test)]
mod tests {
    use crate::op_interface::IndirectCallLikeOps;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

    #[test]
    fn indirect_call_interface_uses_clif_owned_sig() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @ordinary(%callee: core.ptr, %value: core.i32) -> core.i32 {
    %result = clif.call_indirect %callee, %value {sig = core.func(core.i32, core.i32)} : core.i32
    clif.return %result
  }
  clif.func @tail(%callee: core.ptr, %value: core.i32) -> core.nil {
    clif.return_call_indirect %callee, %value {sig = core.func(core.nil, core.i32)}
  }
  clif.func @direct() -> core.nil {
    clif.return
  }
}"#,
        );
        let functions = module.ops(&ctx);
        let body_op = |index| {
            let body = ctx.op(functions[index]).regions[0];
            ctx.block(ctx.region(body).blocks[0]).ops[0]
        };

        let ordinary = body_op(0);
        let tail = body_op(1);
        let direct = body_op(2);
        assert!(IndirectCallLikeOps::exact_signature(&ctx, ordinary).is_some());
        assert!(IndirectCallLikeOps::exact_signature(&ctx, tail).is_some());
        for op in [ordinary, tail] {
            let operands = ctx.op_operands(op);
            assert_eq!(IndirectCallLikeOps::callee(&ctx, op), Some(operands[0]));
            assert_eq!(
                IndirectCallLikeOps::arguments(&ctx, op),
                Some(&operands[1..])
            );
        }
        assert!(IndirectCallLikeOps::get(&ctx, direct).is_none());
        assert_eq!(IndirectCallLikeOps::callee(&ctx, direct), None);
        assert_eq!(IndirectCallLikeOps::arguments(&ctx, direct), None);

        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("clif.call_indirect"));
        assert!(printed.contains("sig = core.func(core.i32, core.i32)"));
    }
}
