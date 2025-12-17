//! Wasm dialect operations.
//!
//! This dialect represents Wasm 3.0 + WasmGC operations.
//! These are low-level, target-specific operations for WebAssembly code generation.

use crate::dialect;

dialect! {
    wasm {
        // === Control Flow ===

        /// `wasm.block` operation: structured block with a label.
        op block[label]() -> result @body {};

        /// `wasm.loop` operation: loop construct.
        op r#loop[label]() -> result @body {};

        /// `wasm.if` operation: conditional branch with then/else bodies.
        op r#if(cond) -> result @then {} @r#else {};

        /// `wasm.br` operation: unconditional branch to a label.
        op br[target]();

        /// `wasm.br_if` operation: conditional branch to a label.
        op br_if[target](cond);

        /// `wasm.return` operation: return from function.
        op r#return(..values);

        /// `wasm.return_call` operation: tail call (return_call).
        op return_call[callee](..args);

        /// `wasm.call` operation: function call.
        op call[callee](..args) -> result;

        /// `wasm.call_indirect` operation: indirect function call.
        op call_indirect[type_idx, table](..args) -> result;

        // === Integer Arithmetic (i32) ===

        /// `wasm.i32_const` operation: i32 constant.
        op i32_const[value]() -> result;

        /// `wasm.i32_add` operation: i32 addition.
        op i32_add(lhs, rhs) -> result;

        /// `wasm.i32_sub` operation: i32 subtraction.
        op i32_sub(lhs, rhs) -> result;

        /// `wasm.i32_mul` operation: i32 multiplication.
        op i32_mul(lhs, rhs) -> result;

        /// `wasm.i32_div_s` operation: i32 signed division.
        op i32_div_s(lhs, rhs) -> result;

        /// `wasm.i32_div_u` operation: i32 unsigned division.
        op i32_div_u(lhs, rhs) -> result;

        /// `wasm.i32_rem_s` operation: i32 signed remainder.
        op i32_rem_s(lhs, rhs) -> result;

        /// `wasm.i32_rem_u` operation: i32 unsigned remainder.
        op i32_rem_u(lhs, rhs) -> result;

        // === Integer Comparison (i32) ===

        /// `wasm.i32_eq` operation: i32 equality.
        op i32_eq(lhs, rhs) -> result;

        /// `wasm.i32_ne` operation: i32 inequality.
        op i32_ne(lhs, rhs) -> result;

        /// `wasm.i32_lt_s` operation: i32 signed less than.
        op i32_lt_s(lhs, rhs) -> result;

        /// `wasm.i32_lt_u` operation: i32 unsigned less than.
        op i32_lt_u(lhs, rhs) -> result;

        /// `wasm.i32_le_s` operation: i32 signed less or equal.
        op i32_le_s(lhs, rhs) -> result;

        /// `wasm.i32_le_u` operation: i32 unsigned less or equal.
        op i32_le_u(lhs, rhs) -> result;

        /// `wasm.i32_gt_s` operation: i32 signed greater than.
        op i32_gt_s(lhs, rhs) -> result;

        /// `wasm.i32_gt_u` operation: i32 unsigned greater than.
        op i32_gt_u(lhs, rhs) -> result;

        /// `wasm.i32_ge_s` operation: i32 signed greater or equal.
        op i32_ge_s(lhs, rhs) -> result;

        /// `wasm.i32_ge_u` operation: i32 unsigned greater or equal.
        op i32_ge_u(lhs, rhs) -> result;

        // === Integer Arithmetic (i64) ===

        /// `wasm.i64_const` operation: i64 constant.
        op i64_const[value]() -> result;

        /// `wasm.i64_add` operation: i64 addition.
        op i64_add(lhs, rhs) -> result;

        /// `wasm.i64_sub` operation: i64 subtraction.
        op i64_sub(lhs, rhs) -> result;

        /// `wasm.i64_mul` operation: i64 multiplication.
        op i64_mul(lhs, rhs) -> result;

        // === Floating Point (f32) ===

        /// `wasm.f32_const` operation: f32 constant.
        op f32_const[value]() -> result;

        /// `wasm.f32_add` operation: f32 addition.
        op f32_add(lhs, rhs) -> result;

        /// `wasm.f32_sub` operation: f32 subtraction.
        op f32_sub(lhs, rhs) -> result;

        /// `wasm.f32_mul` operation: f32 multiplication.
        op f32_mul(lhs, rhs) -> result;

        /// `wasm.f32_div` operation: f32 division.
        op f32_div(lhs, rhs) -> result;

        // === Floating Point (f64) ===

        /// `wasm.f64_const` operation: f64 constant.
        op f64_const[value]() -> result;

        /// `wasm.f64_add` operation: f64 addition.
        op f64_add(lhs, rhs) -> result;

        /// `wasm.f64_sub` operation: f64 subtraction.
        op f64_sub(lhs, rhs) -> result;

        /// `wasm.f64_mul` operation: f64 multiplication.
        op f64_mul(lhs, rhs) -> result;

        /// `wasm.f64_div` operation: f64 division.
        op f64_div(lhs, rhs) -> result;

        // === Local Variables ===

        /// `wasm.local_get` operation: get local variable.
        op local_get[index]() -> result;

        /// `wasm.local_set` operation: set local variable.
        op local_set[index](value);

        /// `wasm.local_tee` operation: set local and return value.
        op local_tee[index](value) -> result;

        // === WasmGC: Structs ===

        /// `wasm.struct_new` operation: create a new struct instance.
        op struct_new[type_idx](..fields) -> result;

        /// `wasm.struct_get` operation: get a field from a struct.
        op struct_get[type_idx, field_idx](r#ref) -> result;

        /// `wasm.struct_set` operation: set a field in a struct.
        op struct_set[type_idx, field_idx](r#ref, value);

        // === WasmGC: Arrays ===

        /// `wasm.array_new` operation: create a new array.
        op array_new[type_idx](size, init) -> result;

        /// `wasm.array_new_default` operation: create array with default values.
        op array_new_default[type_idx](size) -> result;

        /// `wasm.array_get` operation: get element from array.
        op array_get[type_idx](r#ref, index) -> result;

        /// `wasm.array_set` operation: set element in array.
        op array_set[type_idx](r#ref, index, value);

        /// `wasm.array_len` operation: get array length.
        op array_len(r#ref) -> result;

        // === WasmGC: References ===

        /// `wasm.ref_null` operation: create null reference.
        op ref_null[heap_type]() -> result;

        /// `wasm.ref_is_null` operation: check if reference is null.
        op ref_is_null(r#ref) -> result;

        /// `wasm.ref_cast` operation: cast reference type.
        op ref_cast[target_type](r#ref) -> result;

        /// `wasm.ref_test` operation: test reference type.
        op ref_test[target_type](r#ref) -> result;
    }
}
