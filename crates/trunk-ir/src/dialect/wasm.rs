//! Wasm dialect operations.
//!
//! This dialect represents Wasm 3.0 + WasmGC operations.
//! These are low-level, target-specific operations for WebAssembly code generation.

use crate::dialect;

dialect! {
    mod wasm {
        // === Control Flow ===

        /// `wasm.block` operation: structured block with a label.
        #[attr(label)]
        fn block() -> result {
            #[region(body)] {}
        };

        /// `wasm.loop` operation: loop construct.
        #[attr(label)]
        fn r#loop() -> result {
            #[region(body)] {}
        };

        /// `wasm.if` operation: conditional branch with then/else bodies.
        fn r#if(cond) -> result {
            #[region(then)] {}
            #[region(r#else)] {}
        };

        /// `wasm.br` operation: unconditional branch to a label.
        #[attr(target)]
        fn br();

        /// `wasm.br_if` operation: conditional branch to a label.
        #[attr(target)]
        fn br_if(cond);

        /// `wasm.return` operation: return from function.
        fn r#return(#[rest] values);

        /// `wasm.drop` operation: drop a value on the stack.
        fn drop(value);

        /// `wasm.return_call` operation: tail call (return_call).
        #[attr(callee)]
        fn return_call(#[rest] args);

        /// `wasm.call` operation: function call.
        #[attr(callee)]
        fn call(#[rest] args) -> result;

        /// `wasm.call_indirect` operation: indirect function call.
        #[attr(type_idx, table)]
        fn call_indirect(#[rest] args) -> result;

        // === Module-level Definitions ===

        /// `wasm.import_func` operation: import a function.
        #[attr(module, name, sym_name, r#type)]
        fn import_func();

        /// `wasm.export_func` operation: export a function by symbol.
        #[attr(name, func)]
        fn export_func();

        /// `wasm.export_memory` operation: export a memory by index.
        #[attr(name, index)]
        fn export_memory();

        /// `wasm.memory` operation: define linear memory.
        #[attr(min, max, shared, memory64)]
        fn memory();

        /// `wasm.data` operation: define a data segment.
        #[attr(offset, bytes)]
        fn data();

        // === Integer Arithmetic (i32) ===

        /// `wasm.i32_const` operation: i32 constant.
        #[attr(value)]
        fn i32_const() -> result;

        /// `wasm.i32_add` operation: i32 addition.
        fn i32_add(lhs, rhs) -> result;

        /// `wasm.i32_sub` operation: i32 subtraction.
        fn i32_sub(lhs, rhs) -> result;

        /// `wasm.i32_mul` operation: i32 multiplication.
        fn i32_mul(lhs, rhs) -> result;

        /// `wasm.i32_div_s` operation: i32 signed division.
        fn i32_div_s(lhs, rhs) -> result;

        /// `wasm.i32_div_u` operation: i32 unsigned division.
        fn i32_div_u(lhs, rhs) -> result;

        /// `wasm.i32_rem_s` operation: i32 signed remainder.
        fn i32_rem_s(lhs, rhs) -> result;

        /// `wasm.i32_rem_u` operation: i32 unsigned remainder.
        fn i32_rem_u(lhs, rhs) -> result;

        // === Integer Comparison (i32) ===

        /// `wasm.i32_eq` operation: i32 equality.
        fn i32_eq(lhs, rhs) -> result;

        /// `wasm.i32_ne` operation: i32 inequality.
        fn i32_ne(lhs, rhs) -> result;

        /// `wasm.i32_lt_s` operation: i32 signed less than.
        fn i32_lt_s(lhs, rhs) -> result;

        /// `wasm.i32_lt_u` operation: i32 unsigned less than.
        fn i32_lt_u(lhs, rhs) -> result;

        /// `wasm.i32_le_s` operation: i32 signed less or equal.
        fn i32_le_s(lhs, rhs) -> result;

        /// `wasm.i32_le_u` operation: i32 unsigned less or equal.
        fn i32_le_u(lhs, rhs) -> result;

        /// `wasm.i32_gt_s` operation: i32 signed greater than.
        fn i32_gt_s(lhs, rhs) -> result;

        /// `wasm.i32_gt_u` operation: i32 unsigned greater than.
        fn i32_gt_u(lhs, rhs) -> result;

        /// `wasm.i32_ge_s` operation: i32 signed greater or equal.
        fn i32_ge_s(lhs, rhs) -> result;

        /// `wasm.i32_ge_u` operation: i32 unsigned greater or equal.
        fn i32_ge_u(lhs, rhs) -> result;

        // === Integer Arithmetic (i64) ===

        /// `wasm.i64_const` operation: i64 constant.
        #[attr(value)]
        fn i64_const() -> result;

        /// `wasm.i64_add` operation: i64 addition.
        fn i64_add(lhs, rhs) -> result;

        /// `wasm.i64_sub` operation: i64 subtraction.
        fn i64_sub(lhs, rhs) -> result;

        /// `wasm.i64_mul` operation: i64 multiplication.
        fn i64_mul(lhs, rhs) -> result;

        // === Floating Point (f32) ===

        /// `wasm.f32_const` operation: f32 constant.
        #[attr(value)]
        fn f32_const() -> result;

        /// `wasm.f32_add` operation: f32 addition.
        fn f32_add(lhs, rhs) -> result;

        /// `wasm.f32_sub` operation: f32 subtraction.
        fn f32_sub(lhs, rhs) -> result;

        /// `wasm.f32_mul` operation: f32 multiplication.
        fn f32_mul(lhs, rhs) -> result;

        /// `wasm.f32_div` operation: f32 division.
        fn f32_div(lhs, rhs) -> result;

        // === Floating Point (f64) ===

        /// `wasm.f64_const` operation: f64 constant.
        #[attr(value)]
        fn f64_const() -> result;

        /// `wasm.f64_add` operation: f64 addition.
        fn f64_add(lhs, rhs) -> result;

        /// `wasm.f64_sub` operation: f64 subtraction.
        fn f64_sub(lhs, rhs) -> result;

        /// `wasm.f64_mul` operation: f64 multiplication.
        fn f64_mul(lhs, rhs) -> result;

        /// `wasm.f64_div` operation: f64 division.
        fn f64_div(lhs, rhs) -> result;

        // === Local Variables ===

        /// `wasm.local_get` operation: get local variable.
        #[attr(index)]
        fn local_get() -> result;

        /// `wasm.local_set` operation: set local variable.
        #[attr(index)]
        fn local_set(value);

        /// `wasm.local_tee` operation: set local and return value.
        #[attr(index)]
        fn local_tee(value) -> result;

        // === WasmGC: Structs ===

        /// `wasm.struct_new` operation: create a new struct instance.
        #[attr(type_idx)]
        fn struct_new(#[rest] fields) -> result;

        /// `wasm.struct_get` operation: get a field from a struct.
        #[attr(type_idx, field_idx)]
        fn struct_get(r#ref) -> result;

        /// `wasm.struct_set` operation: set a field in a struct.
        #[attr(type_idx, field_idx)]
        fn struct_set(r#ref, value);

        // === WasmGC: Arrays ===

        /// `wasm.array_new` operation: create a new array.
        #[attr(type_idx)]
        fn array_new(size, init) -> result;

        /// `wasm.array_new_default` operation: create array with default values.
        #[attr(type_idx)]
        fn array_new_default(size) -> result;

        /// `wasm.array_get` operation: get element from array.
        #[attr(type_idx)]
        fn array_get(r#ref, index) -> result;

        /// `wasm.array_set` operation: set element in array.
        #[attr(type_idx)]
        fn array_set(r#ref, index, value);

        /// `wasm.array_len` operation: get array length.
        fn array_len(r#ref) -> result;

        // === WasmGC: References ===

        /// `wasm.ref_null` operation: create null reference.
        #[attr(heap_type)]
        fn ref_null() -> result;

        /// `wasm.ref_is_null` operation: check if reference is null.
        fn ref_is_null(r#ref) -> result;

        /// `wasm.ref_cast` operation: cast reference type.
        #[attr(target_type)]
        fn ref_cast(r#ref) -> result;

        /// `wasm.ref_test` operation: test reference type.
        #[attr(target_type)]
        fn ref_test(r#ref) -> result;
    }
}
