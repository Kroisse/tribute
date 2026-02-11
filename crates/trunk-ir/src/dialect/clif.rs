//! Cranelift dialect operations.
//!
//! This dialect represents Cranelift IR operations for native code generation.
//! These are low-level, target-specific operations that map 1:1 to Cranelift IR.
//! Symmetric to the `wasm` dialect for the WebAssembly target.

use crate::dialect;

dialect! {
    mod clif {
        // === Module-level Definitions ===

        /// `clif.func` operation: define a function.
        #[attr(sym_name: QualifiedName, r#type: Type)]
        fn func() {
            #[region(body)] {}
        };

        /// `clif.call` operation: direct call to a function symbol.
        #[attr(callee: Symbol)]
        fn call(#[rest] args) -> result;

        /// `clif.call_indirect` operation: indirect call via function pointer.
        #[attr(sig: Type)]
        fn call_indirect(callee, #[rest] args) -> result;

        /// `clif.return` operation: return from function.
        fn r#return(#[rest] values);

        // === Constants ===

        /// `clif.iconst` operation: integer constant.
        #[attr(value: i64)]
        fn iconst() -> result;

        /// `clif.f32const` operation: f32 constant.
        #[attr(value: f32)]
        fn f32const() -> result;

        /// `clif.f64const` operation: f64 constant.
        #[attr(value: f64)]
        fn f64const() -> result;

        // === Integer Arithmetic ===

        /// `clif.iadd` operation: integer addition.
        fn iadd(lhs, rhs) -> result;

        /// `clif.isub` operation: integer subtraction.
        fn isub(lhs, rhs) -> result;

        /// `clif.imul` operation: integer multiplication.
        fn imul(lhs, rhs) -> result;

        /// `clif.sdiv` operation: signed integer division.
        fn sdiv(lhs, rhs) -> result;

        /// `clif.udiv` operation: unsigned integer division.
        fn udiv(lhs, rhs) -> result;

        /// `clif.srem` operation: signed integer remainder.
        fn srem(lhs, rhs) -> result;

        /// `clif.urem` operation: unsigned integer remainder.
        fn urem(lhs, rhs) -> result;

        /// `clif.ineg` operation: integer negation.
        fn ineg(operand) -> result;

        // === Floating Point Arithmetic ===

        /// `clif.fadd` operation: floating point addition.
        fn fadd(lhs, rhs) -> result;

        /// `clif.fsub` operation: floating point subtraction.
        fn fsub(lhs, rhs) -> result;

        /// `clif.fmul` operation: floating point multiplication.
        fn fmul(lhs, rhs) -> result;

        /// `clif.fdiv` operation: floating point division.
        fn fdiv(lhs, rhs) -> result;

        /// `clif.fneg` operation: floating point negation.
        fn fneg(operand) -> result;

        // === Comparisons ===

        /// `clif.icmp` operation: integer comparison.
        /// cond: "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"
        #[attr(cond: Symbol)]
        fn icmp(lhs, rhs) -> result;

        /// `clif.fcmp` operation: floating point comparison.
        /// cond: "eq", "ne", "lt", "le", "gt", "ge"
        #[attr(cond: Symbol)]
        fn fcmp(lhs, rhs) -> result;

        // === Bitwise Operations ===

        /// `clif.band` operation: bitwise AND.
        fn band(lhs, rhs) -> result;

        /// `clif.bor` operation: bitwise OR.
        fn bor(lhs, rhs) -> result;

        /// `clif.bxor` operation: bitwise XOR.
        fn bxor(lhs, rhs) -> result;

        /// `clif.ishl` operation: shift left.
        fn ishl(lhs, rhs) -> result;

        /// `clif.sshr` operation: signed shift right.
        fn sshr(lhs, rhs) -> result;

        /// `clif.ushr` operation: unsigned shift right.
        fn ushr(lhs, rhs) -> result;

        /// `clif.trap` operation: trigger a trap (unreachable code).
        /// `code` specifies the trap reason (e.g., "unreachable").
        #[attr(code: Symbol)]
        fn trap();

        /// `clif.return_call` operation: tail call (returns callee's result directly).
        #[attr(callee: Symbol)]
        fn return_call(#[rest] args);

        // === Control Flow ===

        /// `clif.brif` operation: conditional branch.
        /// Branches to `then_dest` if cond is nonzero, else to `else_dest`.
        fn brif(cond) {
            #[successor(then_dest)]
            #[successor(else_dest)]
        };

        /// `clif.jump` operation: unconditional jump to a block.
        fn jump(#[rest] args) {
            #[successor(dest)]
        };

        /// `clif.br_table` operation: table-based branch.
        #[attr(table: any)]
        fn br_table(index);

        // === Memory ===

        /// `clif.load` operation: load value from memory address.
        #[attr(offset: i32)]
        fn load(addr) -> result;

        /// `clif.store` operation: store value to memory address.
        #[attr(offset: i32)]
        fn store(value, addr);

        /// `clif.stack_slot` operation: allocate a stack slot.
        #[attr(size: u32, align: u32)]
        fn stack_slot() -> result;

        /// `clif.stack_addr` operation: get address of a stack slot.
        fn stack_addr(slot) -> result;

        /// `clif.symbol_addr` operation: get address of a symbol.
        #[attr(sym: Symbol)]
        fn symbol_addr() -> result;

        // === Type Conversions (Integer) ===

        /// `clif.ireduce` operation: reduce integer width (e.g. i64 → i32).
        fn ireduce(operand) -> result;

        /// `clif.uextend` operation: zero-extend integer (e.g. i32 → i64).
        fn uextend(operand) -> result;

        /// `clif.sextend` operation: sign-extend integer (e.g. i32 → i64).
        fn sextend(operand) -> result;

        // === Type Conversions (Float) ===

        /// `clif.fpromote` operation: promote float (f32 → f64).
        fn fpromote(operand) -> result;

        /// `clif.fdemote` operation: demote float (f64 → f32).
        fn fdemote(operand) -> result;

        // === Type Conversions (Int ↔ Float) ===

        /// `clif.fcvt_to_sint` operation: float to signed integer.
        fn fcvt_to_sint(operand) -> result;

        /// `clif.fcvt_from_sint` operation: signed integer to float.
        fn fcvt_from_sint(operand) -> result;

        /// `clif.fcvt_to_uint` operation: float to unsigned integer.
        fn fcvt_to_uint(operand) -> result;

        /// `clif.fcvt_from_uint` operation: unsigned integer to float.
        fn fcvt_from_uint(operand) -> result;

        // Note: Types (i8, i16, i32, i64, f32, f64) are provided by
        // core::I8, core::I16, core::I32, core::I64, core::F32, core::F64.
        // No clif-specific types needed.
    }
}
