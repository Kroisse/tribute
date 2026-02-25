//! Arena-based clif dialect.

crate::arena_dialect_internal! {
    mod clif {
        // Module
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() {
            #[region(body)] {}
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
            #[successor(then_dest)] {}
            #[successor(else_dest)] {}
        }

        fn jump(#[rest] args: ()) {
            #[successor(dest)] {}
        }

        #[attr(table: any)]
        fn br_table(index: ()) {}

        #[attr(code: Symbol)]
        fn trap() {}

        #[attr(callee: Symbol)]
        fn return_call(#[rest] args: ()) {}

        // Memory
        #[attr(offset: i32)]
        fn load(addr: ()) -> result {}

        #[attr(offset: i32)]
        fn store(value: (), addr: ()) {}

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
}
