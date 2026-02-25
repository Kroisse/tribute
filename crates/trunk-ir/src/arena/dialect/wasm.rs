//! Arena-based wasm dialect.

crate::arena_dialect_internal! {
    mod wasm {
        // Control flow
        fn block() -> result {
            #[region(body)] {}
        };

        fn r#loop(#[rest] init) -> result {
            #[region(body)] {}
        };

        fn r#if(cond) -> result {
            #[region(then_region)] {}
            #[region(else_region)] {}
        };

        #[attr(target: u32)]
        fn br();

        #[attr(target: u32)]
        fn br_if(cond);

        fn r#return(#[rest] values);
        fn r#yield(value);
        fn drop(value);

        // Functions
        #[attr(callee: Symbol)]
        fn call(#[rest] args) -> #[rest] results;

        #[attr(type_idx: u32, table: u32)]
        fn call_indirect(#[rest] args) -> #[rest] results;

        #[attr(callee: Symbol)]
        fn return_call(#[rest] args);

        fn unreachable();
        fn nop() -> result;

        // Module
        #[attr(sym_name: Symbol, r#type: Type)]
        fn func() {
            #[region(body)] {}
        };

        #[attr(module: String, name: String, sym_name: Symbol, r#type: Type)]
        fn import_func();

        #[attr(name: String, func: Symbol)]
        fn export_func();

        #[attr(name: String, index: u32)]
        fn export_memory();

        #[attr(min: u32, max: u32, shared: bool, memory64: bool)]
        fn memory();

        #[attr(offset: u32, bytes: any, passive: bool)]
        fn data();

        #[attr(reftype: Symbol, min: u32, max?: u32)]
        fn table();

        #[attr(table?: u32, offset?: u32)]
        fn elem() {
            #[region(funcs)] {}
        };

        #[attr(valtype: Symbol, mutable: bool, init: any)]
        fn global();

        #[attr(index: u32)]
        fn global_get() -> result;

        #[attr(index: u32)]
        fn global_set(value);

        // i32
        #[attr(value: i32)]
        fn i32_const() -> result;

        fn i32_add(lhs, rhs) -> result;
        fn i32_sub(lhs, rhs) -> result;
        fn i32_mul(lhs, rhs) -> result;
        fn i32_div_s(lhs, rhs) -> result;
        fn i32_div_u(lhs, rhs) -> result;
        fn i32_rem_s(lhs, rhs) -> result;
        fn i32_rem_u(lhs, rhs) -> result;

        fn i32_eq(lhs, rhs) -> result;
        fn i32_ne(lhs, rhs) -> result;
        fn i32_lt_s(lhs, rhs) -> result;
        fn i32_lt_u(lhs, rhs) -> result;
        fn i32_le_s(lhs, rhs) -> result;
        fn i32_le_u(lhs, rhs) -> result;
        fn i32_gt_s(lhs, rhs) -> result;
        fn i32_gt_u(lhs, rhs) -> result;
        fn i32_ge_s(lhs, rhs) -> result;
        fn i32_ge_u(lhs, rhs) -> result;

        fn i32_and(lhs, rhs) -> result;
        fn i32_or(lhs, rhs) -> result;
        fn i32_xor(lhs, rhs) -> result;
        fn i32_shl(lhs, rhs) -> result;
        fn i32_shr_s(lhs, rhs) -> result;
        fn i32_shr_u(lhs, rhs) -> result;

        // i64
        #[attr(value: i64)]
        fn i64_const() -> result;

        fn i64_add(lhs, rhs) -> result;
        fn i64_sub(lhs, rhs) -> result;
        fn i64_mul(lhs, rhs) -> result;
        fn i64_div_s(lhs, rhs) -> result;
        fn i64_div_u(lhs, rhs) -> result;
        fn i64_rem_s(lhs, rhs) -> result;
        fn i64_rem_u(lhs, rhs) -> result;

        fn i64_eq(lhs, rhs) -> result;
        fn i64_ne(lhs, rhs) -> result;
        fn i64_lt_s(lhs, rhs) -> result;
        fn i64_lt_u(lhs, rhs) -> result;
        fn i64_le_s(lhs, rhs) -> result;
        fn i64_le_u(lhs, rhs) -> result;
        fn i64_gt_s(lhs, rhs) -> result;
        fn i64_gt_u(lhs, rhs) -> result;
        fn i64_ge_s(lhs, rhs) -> result;
        fn i64_ge_u(lhs, rhs) -> result;

        fn i64_and(lhs, rhs) -> result;
        fn i64_or(lhs, rhs) -> result;
        fn i64_xor(lhs, rhs) -> result;
        fn i64_shl(lhs, rhs) -> result;
        fn i64_shr_s(lhs, rhs) -> result;
        fn i64_shr_u(lhs, rhs) -> result;

        // f32
        #[attr(value: f32)]
        fn f32_const() -> result;

        fn f32_add(lhs, rhs) -> result;
        fn f32_sub(lhs, rhs) -> result;
        fn f32_mul(lhs, rhs) -> result;
        fn f32_div(lhs, rhs) -> result;
        fn f32_neg(operand) -> result;

        fn f32_eq(lhs, rhs) -> result;
        fn f32_ne(lhs, rhs) -> result;
        fn f32_lt(lhs, rhs) -> result;
        fn f32_le(lhs, rhs) -> result;
        fn f32_gt(lhs, rhs) -> result;
        fn f32_ge(lhs, rhs) -> result;

        // f64
        #[attr(value: f64)]
        fn f64_const() -> result;

        fn f64_add(lhs, rhs) -> result;
        fn f64_sub(lhs, rhs) -> result;
        fn f64_mul(lhs, rhs) -> result;
        fn f64_div(lhs, rhs) -> result;
        fn f64_neg(operand) -> result;

        fn f64_eq(lhs, rhs) -> result;
        fn f64_ne(lhs, rhs) -> result;
        fn f64_lt(lhs, rhs) -> result;
        fn f64_le(lhs, rhs) -> result;
        fn f64_gt(lhs, rhs) -> result;
        fn f64_ge(lhs, rhs) -> result;

        // Local variables
        #[attr(index: u32)]
        fn local_get() -> result;

        #[attr(index: u32)]
        fn local_set(value);

        #[attr(index: u32)]
        fn local_tee(value) -> result;

        // GC structs
        #[attr(type_idx: u32)]
        fn struct_new(#[rest] fields) -> result;

        #[attr(type_idx: u32, field_idx: u32)]
        fn struct_get(r#ref) -> result;

        #[attr(type_idx: u32, field_idx: u32)]
        fn struct_set(r#ref, value);

        // GC arrays
        #[attr(type_idx: u32)]
        fn array_new(size, init) -> result;

        #[attr(type_idx: u32)]
        fn array_new_default(size) -> result;

        #[attr(type_idx: u32, data_idx: u32)]
        fn array_new_data(offset, size) -> result;

        #[attr(data_idx: u32, offset: u32, len: u32)]
        fn bytes_from_data() -> result;

        #[attr(type_idx: u32)]
        fn array_get(r#ref, index) -> result;

        #[attr(type_idx: u32)]
        fn array_get_s(r#ref, index) -> result;

        #[attr(type_idx: u32)]
        fn array_get_u(r#ref, index) -> result;

        #[attr(type_idx: u32)]
        fn array_set(r#ref, index, value);

        fn array_len(r#ref) -> result;

        #[attr(dst_type_idx: u32, src_type_idx: u32)]
        fn array_copy(dst, dst_offset, src, src_offset, len);

        // References
        #[attr(heap_type: Symbol, type_idx?: u32)]
        fn ref_null() -> result;

        #[attr(func_name: Symbol)]
        fn ref_func() -> result;

        fn ref_is_null(r#ref) -> result;

        #[attr(target_type: Symbol, type_idx?: u32)]
        fn ref_cast(r#ref) -> result;

        #[attr(target_type: Symbol, type_idx?: u32)]
        fn ref_test(r#ref) -> result;

        // i31ref
        fn ref_i31(value) -> result;
        fn i31_get_s(r#ref) -> result;
        fn i31_get_u(r#ref) -> result;

        // Type conversions (integer)
        fn i32_wrap_i64(operand) -> result;
        fn i64_extend_i32_s(operand) -> result;
        fn i64_extend_i32_u(operand) -> result;

        // Type conversions (float to int)
        fn i32_trunc_f32_s(operand) -> result;
        fn i32_trunc_f32_u(operand) -> result;
        fn i32_trunc_f64_s(operand) -> result;
        fn i32_trunc_f64_u(operand) -> result;
        fn i64_trunc_f32_s(operand) -> result;
        fn i64_trunc_f32_u(operand) -> result;
        fn i64_trunc_f64_s(operand) -> result;
        fn i64_trunc_f64_u(operand) -> result;

        // Type conversions (int to float)
        fn f32_convert_i32_s(operand) -> result;
        fn f32_convert_i32_u(operand) -> result;
        fn f32_convert_i64_s(operand) -> result;
        fn f32_convert_i64_u(operand) -> result;
        fn f64_convert_i32_s(operand) -> result;
        fn f64_convert_i32_u(operand) -> result;
        fn f64_convert_i64_s(operand) -> result;
        fn f64_convert_i64_u(operand) -> result;

        // Float conversions
        fn f32_demote_f64(operand) -> result;
        fn f64_promote_f32(operand) -> result;

        // Bitcast
        fn i32_reinterpret_f32(operand) -> result;
        fn i64_reinterpret_f64(operand) -> result;
        fn f32_reinterpret_i32(operand) -> result;
        fn f64_reinterpret_i64(operand) -> result;

        // Linear memory
        #[attr(memory: u32)]
        fn memory_size() -> result;

        #[attr(memory: u32)]
        fn memory_grow(delta) -> result;

        // Memory loads (full width)
        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_load(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn f32_load(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn f64_load(addr) -> result;

        // Memory loads (partial width i32)
        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_load8_s(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_load8_u(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_load16_s(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_load16_u(addr) -> result;

        // Memory loads (partial width i64)
        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load8_s(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load8_u(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load16_s(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load16_u(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load32_s(addr) -> result;

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_load32_u(addr) -> result;

        // Memory stores (full width)
        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_store(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_store(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn f32_store(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn f64_store(addr, value);

        // Memory stores (partial width)
        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_store8(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i32_store16(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_store8(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_store16(addr, value);

        #[attr(offset: u32, align: u32, memory: u32)]
        fn i64_store32(addr, value);
    }
}
