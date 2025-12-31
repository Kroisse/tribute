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
        fn call(#[rest] args) -> #[rest] results;

        /// `wasm.call_indirect` operation: indirect function call.
        #[attr(type_idx, table)]
        fn call_indirect(#[rest] args) -> #[rest] results;

        /// `wasm.unreachable` operation: trap / unreachable code.
        fn unreachable();

        /// `wasm.nop` operation: no-op placeholder for nil constants.
        /// Preserves SSA form without runtime effect.
        fn nop() -> result;

        // === Module-level Definitions ===

        /// `wasm.func` operation: define a function.
        #[attr(sym_name: QualifiedName, r#type: Type)]
        fn func() {
            #[region(body)] {}
        };

        /// `wasm.import_func` operation: import a function.
        #[attr(module: Symbol, name: Symbol, sym_name: QualifiedName, r#type: Type)]
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
        /// For active segments: offset is the linear memory offset.
        /// For passive segments: set passive=true (used with array.new_data).
        #[attr(offset: i32, bytes: any, passive: bool)]
        fn data();

        // === Tables and Element Segments ===

        /// `wasm.table` operation: define a function table.
        /// reftype: "funcref" or "externref"
        #[attr(reftype: Symbol, min: u32, max?: u32)]
        fn table();

        /// `wasm.elem` operation: define an active element segment.
        /// The funcs region contains func.constant operations for each function reference.
        #[attr(table?: u32, offset?: i32)]
        fn elem() {
            #[region(funcs)] {}
        };

        // === Global Variables ===

        /// `wasm.global` operation: define a global variable.
        /// valtype: "i32", "i64", "f32", "f64", "funcref", "externref", etc.
        /// mutable: whether the global can be modified
        /// init: initial value (i64 for integers, f64 for floats)
        #[attr(valtype: Symbol, mutable: bool, init: i64)]
        fn global();

        /// `wasm.global_get` operation: get global variable value.
        #[attr(index: u32)]
        fn global_get() -> result;

        /// `wasm.global_set` operation: set global variable value.
        #[attr(index: u32)]
        fn global_set(value);

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

        /// `wasm.i64_div_s` operation: i64 signed division.
        fn i64_div_s(lhs, rhs) -> result;

        /// `wasm.i64_div_u` operation: i64 unsigned division.
        fn i64_div_u(lhs, rhs) -> result;

        /// `wasm.i64_rem_s` operation: i64 signed remainder.
        fn i64_rem_s(lhs, rhs) -> result;

        /// `wasm.i64_rem_u` operation: i64 unsigned remainder.
        fn i64_rem_u(lhs, rhs) -> result;

        // === Integer Comparison (i64) ===

        /// `wasm.i64_eq` operation: i64 equality.
        fn i64_eq(lhs, rhs) -> result;

        /// `wasm.i64_ne` operation: i64 inequality.
        fn i64_ne(lhs, rhs) -> result;

        /// `wasm.i64_lt_s` operation: i64 signed less than.
        fn i64_lt_s(lhs, rhs) -> result;

        /// `wasm.i64_lt_u` operation: i64 unsigned less than.
        fn i64_lt_u(lhs, rhs) -> result;

        /// `wasm.i64_le_s` operation: i64 signed less or equal.
        fn i64_le_s(lhs, rhs) -> result;

        /// `wasm.i64_le_u` operation: i64 unsigned less or equal.
        fn i64_le_u(lhs, rhs) -> result;

        /// `wasm.i64_gt_s` operation: i64 signed greater than.
        fn i64_gt_s(lhs, rhs) -> result;

        /// `wasm.i64_gt_u` operation: i64 unsigned greater than.
        fn i64_gt_u(lhs, rhs) -> result;

        /// `wasm.i64_ge_s` operation: i64 signed greater or equal.
        fn i64_ge_s(lhs, rhs) -> result;

        /// `wasm.i64_ge_u` operation: i64 unsigned greater or equal.
        fn i64_ge_u(lhs, rhs) -> result;

        // === Integer Bitwise (i32) ===

        /// `wasm.i32_and` operation: i32 bitwise AND.
        fn i32_and(lhs, rhs) -> result;

        /// `wasm.i32_or` operation: i32 bitwise OR.
        fn i32_or(lhs, rhs) -> result;

        /// `wasm.i32_xor` operation: i32 bitwise XOR.
        fn i32_xor(lhs, rhs) -> result;

        /// `wasm.i32_shl` operation: i32 shift left.
        fn i32_shl(lhs, rhs) -> result;

        /// `wasm.i32_shr_s` operation: i32 signed shift right.
        fn i32_shr_s(lhs, rhs) -> result;

        /// `wasm.i32_shr_u` operation: i32 unsigned shift right.
        fn i32_shr_u(lhs, rhs) -> result;

        // === Integer Bitwise (i64) ===

        /// `wasm.i64_and` operation: i64 bitwise AND.
        fn i64_and(lhs, rhs) -> result;

        /// `wasm.i64_or` operation: i64 bitwise OR.
        fn i64_or(lhs, rhs) -> result;

        /// `wasm.i64_xor` operation: i64 bitwise XOR.
        fn i64_xor(lhs, rhs) -> result;

        /// `wasm.i64_shl` operation: i64 shift left.
        fn i64_shl(lhs, rhs) -> result;

        /// `wasm.i64_shr_s` operation: i64 signed shift right.
        fn i64_shr_s(lhs, rhs) -> result;

        /// `wasm.i64_shr_u` operation: i64 unsigned shift right.
        fn i64_shr_u(lhs, rhs) -> result;

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

        /// `wasm.f32_neg` operation: f32 negation.
        fn f32_neg(operand) -> result;

        // === Floating Point Comparison (f32) ===

        /// `wasm.f32_eq` operation: f32 equality.
        fn f32_eq(lhs, rhs) -> result;

        /// `wasm.f32_ne` operation: f32 inequality.
        fn f32_ne(lhs, rhs) -> result;

        /// `wasm.f32_lt` operation: f32 less than.
        fn f32_lt(lhs, rhs) -> result;

        /// `wasm.f32_le` operation: f32 less or equal.
        fn f32_le(lhs, rhs) -> result;

        /// `wasm.f32_gt` operation: f32 greater than.
        fn f32_gt(lhs, rhs) -> result;

        /// `wasm.f32_ge` operation: f32 greater or equal.
        fn f32_ge(lhs, rhs) -> result;

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

        /// `wasm.f64_neg` operation: f64 negation.
        fn f64_neg(operand) -> result;

        // === Floating Point Comparison (f64) ===

        /// `wasm.f64_eq` operation: f64 equality.
        fn f64_eq(lhs, rhs) -> result;

        /// `wasm.f64_ne` operation: f64 inequality.
        fn f64_ne(lhs, rhs) -> result;

        /// `wasm.f64_lt` operation: f64 less than.
        fn f64_lt(lhs, rhs) -> result;

        /// `wasm.f64_le` operation: f64 less or equal.
        fn f64_le(lhs, rhs) -> result;

        /// `wasm.f64_gt` operation: f64 greater than.
        fn f64_gt(lhs, rhs) -> result;

        /// `wasm.f64_ge` operation: f64 greater or equal.
        fn f64_ge(lhs, rhs) -> result;

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

        /// `wasm.array_new_data` operation: create array from data segment.
        /// Operands: offset (i32), size (i32) - offset and length within data segment.
        #[attr(type_idx, data_idx)]
        fn array_new_data(offset, size) -> result;

        /// `wasm.bytes_from_data` operation: create Bytes struct from data segment.
        /// This is a compound operation that emits:
        /// - array.new_data to create the backing array
        /// - struct.new to create the Bytes struct (array_ref, offset=0, len)
        #[attr(data_idx: u32, offset: u32, len: u32)]
        fn bytes_from_data() -> result;

        /// `wasm.array_get` operation: get element from array.
        #[attr(type_idx)]
        fn array_get(r#ref, index) -> result;

        /// `wasm.array_get_s` operation: get packed element with sign extension.
        #[attr(type_idx)]
        fn array_get_s(r#ref, index) -> result;

        /// `wasm.array_get_u` operation: get packed element with zero extension.
        #[attr(type_idx)]
        fn array_get_u(r#ref, index) -> result;

        /// `wasm.array_set` operation: set element in array.
        #[attr(type_idx)]
        fn array_set(r#ref, index, value);

        /// `wasm.array_len` operation: get array length.
        fn array_len(r#ref) -> result;

        /// `wasm.array_copy` operation: copy elements between arrays.
        /// Operands: dst_ref, dst_offset, src_ref, src_offset, len
        #[attr(dst_type_idx, src_type_idx)]
        fn array_copy(dst, dst_offset, src, src_offset, len);

        // === WasmGC: References ===

        /// `wasm.ref_null` operation: create null reference.
        #[attr(heap_type)]
        fn ref_null() -> result;

        /// `wasm.ref_func` operation: create funcref from function name.
        /// The func_name is resolved to a function index at emit time.
        #[attr(func_name: QualifiedName)]
        fn ref_func() -> result;

        /// `wasm.ref_is_null` operation: check if reference is null.
        fn ref_is_null(r#ref) -> result;

        /// `wasm.ref_cast` operation: cast reference type.
        #[attr(target_type)]
        fn ref_cast(r#ref) -> result;

        /// `wasm.ref_test` operation: test reference type.
        #[attr(target_type)]
        fn ref_test(r#ref) -> result;

        // === WasmGC: i31ref (Fixnum) ===

        /// `wasm.ref_i31` operation: create i31ref from i32.
        /// The i32 value must fit in 31 bits (range: [-2^30, 2^30)).
        fn ref_i31(value) -> result;

        /// `wasm.i31_get_s` operation: extract signed i32 from i31ref.
        fn i31_get_s(r#ref) -> result;

        /// `wasm.i31_get_u` operation: extract unsigned i32 from i31ref.
        fn i31_get_u(r#ref) -> result;

        // === Type Conversions (Integer) ===

        /// `wasm.i32_wrap_i64` operation: wrap i64 to i32 (truncate).
        fn i32_wrap_i64(operand) -> result;

        /// `wasm.i64_extend_i32_s` operation: sign-extend i32 to i64.
        fn i64_extend_i32_s(operand) -> result;

        /// `wasm.i64_extend_i32_u` operation: zero-extend i32 to i64.
        fn i64_extend_i32_u(operand) -> result;

        // === Type Conversions (Float to Int) ===

        /// `wasm.i32_trunc_f32_s` operation: truncate f32 to i32 (signed).
        fn i32_trunc_f32_s(operand) -> result;

        /// `wasm.i32_trunc_f32_u` operation: truncate f32 to i32 (unsigned).
        fn i32_trunc_f32_u(operand) -> result;

        /// `wasm.i32_trunc_f64_s` operation: truncate f64 to i32 (signed).
        fn i32_trunc_f64_s(operand) -> result;

        /// `wasm.i32_trunc_f64_u` operation: truncate f64 to i32 (unsigned).
        fn i32_trunc_f64_u(operand) -> result;

        /// `wasm.i64_trunc_f32_s` operation: truncate f32 to i64 (signed).
        fn i64_trunc_f32_s(operand) -> result;

        /// `wasm.i64_trunc_f32_u` operation: truncate f32 to i64 (unsigned).
        fn i64_trunc_f32_u(operand) -> result;

        /// `wasm.i64_trunc_f64_s` operation: truncate f64 to i64 (signed).
        fn i64_trunc_f64_s(operand) -> result;

        /// `wasm.i64_trunc_f64_u` operation: truncate f64 to i64 (unsigned).
        fn i64_trunc_f64_u(operand) -> result;

        // === Type Conversions (Int to Float) ===

        /// `wasm.f32_convert_i32_s` operation: convert i32 to f32 (signed).
        fn f32_convert_i32_s(operand) -> result;

        /// `wasm.f32_convert_i32_u` operation: convert i32 to f32 (unsigned).
        fn f32_convert_i32_u(operand) -> result;

        /// `wasm.f32_convert_i64_s` operation: convert i64 to f32 (signed).
        fn f32_convert_i64_s(operand) -> result;

        /// `wasm.f32_convert_i64_u` operation: convert i64 to f32 (unsigned).
        fn f32_convert_i64_u(operand) -> result;

        /// `wasm.f64_convert_i32_s` operation: convert i32 to f64 (signed).
        fn f64_convert_i32_s(operand) -> result;

        /// `wasm.f64_convert_i32_u` operation: convert i32 to f64 (unsigned).
        fn f64_convert_i32_u(operand) -> result;

        /// `wasm.f64_convert_i64_s` operation: convert i64 to f64 (signed).
        fn f64_convert_i64_s(operand) -> result;

        /// `wasm.f64_convert_i64_u` operation: convert i64 to f64 (unsigned).
        fn f64_convert_i64_u(operand) -> result;

        // === Type Conversions (Float to Float) ===

        /// `wasm.f32_demote_f64` operation: demote f64 to f32.
        fn f32_demote_f64(operand) -> result;

        /// `wasm.f64_promote_f32` operation: promote f32 to f64.
        fn f64_promote_f32(operand) -> result;

        // === Reinterpretations (Bitcast) ===

        /// `wasm.i32_reinterpret_f32` operation: reinterpret f32 bits as i32.
        fn i32_reinterpret_f32(operand) -> result;

        /// `wasm.i64_reinterpret_f64` operation: reinterpret f64 bits as i64.
        fn i64_reinterpret_f64(operand) -> result;

        /// `wasm.f32_reinterpret_i32` operation: reinterpret i32 bits as f32.
        fn f32_reinterpret_i32(operand) -> result;

        /// `wasm.f64_reinterpret_i64` operation: reinterpret i64 bits as f64.
        fn f64_reinterpret_i64(operand) -> result;

        // === WasmGC Types ===

        /// `wasm.anyref` type: top reference type (all GC references).
        type anyref;

        /// `wasm.eqref` type: reference types supporting equality comparison.
        type eqref;

        /// `wasm.i31ref` type: 31-bit unboxed integer reference (fixnum).
        type i31ref;

        /// `wasm.structref` type: reference to any struct.
        type structref;

        /// `wasm.arrayref` type: reference to any array.
        type arrayref;

        /// `wasm.funcref` type: reference to any function.
        type funcref;

        /// `wasm.externref` type: external reference (host objects).
        type externref;
    }
}
