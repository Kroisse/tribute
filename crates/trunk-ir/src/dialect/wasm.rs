//! Arena-based wasm dialect.

use crate::attr_kind::SymbolRef;
use crate::attr_kind::{Bytes, Type};
use crate::dialect::core::I32;
use crate::op_flag::{IsolatedFromAbove, NoTerminator, Terminator};
use crate::op_interface::{CallLike, IndirectCallLike};
use crate::ops::{DialectOp, DialectType};
use crate::refs::{OpRef, ValueRef};
use crate::{Attribute, AttributeMap, IrContext, SymbolPath, TypeDataBuilder, TypeRef};
use smallvec::SmallVec;
use trunk_ir_macros::op_interface;

#[trunk_ir::dialect]
mod wasm {
    // Types (abstract heap types)
    struct Anyref;
    struct Eqref;
    struct I31ref;
    struct Structref;
    struct Arrayref;
    struct Funcref;
    struct Externref;

    // Control flow
    /// ```text
    /// %r = wasm.block -> (core.i32) {
    ///   wasm.br {target = 0}
    /// }
    /// ```
    #[flags(NoTerminator)]
    #[asm_format(("->" "(" types(results)^ ")")? attr_dict region(body))]
    fn block() -> Variadic<_> {
        #[region(body)]
        {}
    }

    /// ```text
    /// wasm.loop(%init) {
    ///   ^body(%acc: core.i32):
    ///     wasm.br {target = 0}
    /// }
    /// ```
    ///
    /// Each `init` value initializes the body block argument at its
    /// position, and a `wasm.yield` before a `wasm.br` to the loop's label
    /// sets every body block argument to the yielded value. Emission's
    /// `ExplicitLocals` pass writes both as `wasm.local_set`s to the
    /// arguments' locals, removes `init`, and reads the arguments from
    /// their locals, so the body block arguments have no uses and the Wasm
    /// `loop` takes no parameters (`design/wasm-backend.md`, "명시적 지역
    /// 변수와 stack 계약").
    #[flags(NoTerminator)]
    #[asm_format("(" init ")" ("->" "(" types(results)^ ")")? attr_dict region(body))]
    #[implements(StackOperandOrder)]
    fn r#loop(init: Variadic<_>) -> Variadic<_> {
        #[region(body)]
        {}
    }

    /// ```text
    /// %r = wasm.if %cond -> (core.i32) {
    ///   wasm.yield %a
    /// } else {
    ///   wasm.yield %b
    /// }
    /// ```
    #[flags(NoTerminator)]
    #[asm_format(cond ("->" "(" types(results)^ ")")? attr_dict region(then_region) "else" region(else_region))]
    fn r#if(cond: Value<_>) -> Variadic<_> {
        #[region(then_region)]
        {}
        #[region(else_region)]
        {}
    }

    #[flags(Terminator)]
    #[asm_format(attr_dict_bare)]
    fn br(target: Attr<u32>) {}

    #[asm_format(cond attr_dict_bare)]
    fn br_if(target: Attr<u32>, cond: Value<_>) {}

    /// Branch to the label that `index` selects, as Wasm's `br_table`.
    ///
    /// `targets[i]` is the relative label depth taken when `index` is `i`;
    /// `default` is taken when `index`, read as unsigned, is at least the
    /// number of targets. Depths count enclosing `wasm.block`, `wasm.loop`,
    /// and `wasm.if` labels as `wasm.br` does. Like `wasm.br`, it ends its
    /// block: nothing after it runs. The branch carries no values, so every
    /// label it names takes no branch operands.
    ///
    /// ```text
    /// wasm.br_table %index, default 3 [0, 2, 1]
    /// ```
    #[flags(Terminator)]
    #[asm_format(index "," "default" default targets attr_dict_bare)]
    fn br_table(index: Value<I32>, targets: Attr<[u32]>, default: Attr<u32>) {}

    #[flags(Terminator)]
    #[asm_format(values attr_dict_bare)]
    fn r#return(values: Variadic<_>) {}
    /// Pass `values` to the enclosing structured operation.
    ///
    /// ```text
    /// wasm.yield %a, %b
    /// ```
    ///
    /// - **Meaning:** as the last operation of a region whose parent has
    ///   results, `values` are that region's results in order. Followed by
    ///   a `wasm.br`, `values` are the branch's values, or, for a branch to
    ///   a `wasm.loop`, the loop's next block arguments.
    /// - **Value flow:** after `ExplicitLocals`, `values` are the values
    ///   the operations before it left on top of the stack in order, the
    ///   last of them possibly function parameters, which emission reads
    ///   from their locals right before the yield. Emission writes no
    ///   instruction for the yield itself. A yield that passes no value, or
    ///   that precedes a branch to a loop, does not remain
    ///   (`design/wasm-backend.md`, "명시적 지역 변수와 stack 계약").
    #[asm_format(values attr_dict_bare)]
    fn r#yield(values: Variadic<_>) {}
    #[asm_format(value attr_dict_bare)]
    fn drop(value: Value<_>) {}

    // Functions
    #[implements(CallLike)]
    #[asm_format(callee "(" args ")" attr_dict ":" functional_type(args, results))]
    fn call(callee: Attr<SymbolRef>, args: Variadic<_>) -> Variadic<_> {}

    /// Call the function at a table index through its exact signature.
    ///
    /// ```text
    /// %result = wasm.call_indirect %index(%arg0, ...) table 1 : (A, ...) -> R
    /// ```
    ///
    /// - **Shape:** `callee` is the `core.i32` index into the function table
    ///   `table`, table 0 when absent; `args` are the call's arguments, and
    ///   `signature` is the callee's `wasm.func_sig`.
    /// - **Meaning:** Wasm `call_indirect` with the type `signature`, which
    ///   traps when the table entry has another type. The instruction takes
    ///   the arguments, then the index, which is its `StackOperandOrder`.
    /// - **Verification:** emission checks that each argument is physically
    ///   assignable to its parameter and that the results are the
    ///   signature's, or none for a signature whose one result is
    ///   `core.nil`.
    /// - **Ownership and value flow:** the index and arguments are ordinary
    ///   SSA uses.
    #[asm_format(callee "(" args ")" ("table" table^)? attr_dict ":" signature(signature))]
    #[implements(IndirectCallLike, StackOperandOrder)]
    fn call_indirect<S: FuncSig>(
        #[default(0)] table: Attr<u32>,
        signature: Attr<TypeOf<S>>,
        callee: Value<I32>,
        args: Variadic<_>,
    ) -> Variadic<_> {
    }

    #[flags(Terminator)]
    #[implements(CallLike)]
    #[asm_format(callee "(" args ")" attr_dict ":" functional_type(args))]
    fn return_call(callee: Attr<SymbolRef>, args: Variadic<_>) {}

    /// A tail call of the function at a table index through its exact
    /// signature.
    ///
    /// ```text
    /// wasm.return_call_indirect %index(%arg0, ...) : (A, ...) -> R
    /// ```
    ///
    /// - **Shape:** as `wasm.call_indirect`, without results.
    /// - **Meaning:** Wasm `return_call_indirect`, which takes the arguments,
    ///   then the index, as `wasm.call_indirect` does.
    /// - **Verification:** as for `wasm.call_indirect`'s arguments; the
    ///   signature's results are the enclosing function's.
    /// - **Ownership and value flow:** as `wasm.call_indirect`.
    #[asm_format(callee "(" args ")" ("table" table^)? attr_dict ":" signature(signature))]
    #[flags(Terminator)]
    #[implements(IndirectCallLike, StackOperandOrder)]
    fn return_call_indirect<S: FuncSig>(
        #[default(0)] table: Attr<u32>,
        signature: Attr<TypeOf<S>>,
        callee: Value<I32>,
        args: Variadic<_>,
    ) {
    }

    #[flags(Terminator)]
    #[asm_format(attr_dict_bare)]
    fn unreachable() {}
    #[asm_format(attr_dict_bare ":" type(result))]
    fn nop() -> Value<_> {}

    // Module
    /// A Wasm function, or a declaration when it has no body.
    ///
    /// ```text
    /// wasm.func @f(%x: core.i32) -> core.i32 attributes {locals = [core.i64]} {
    ///   ...
    /// }
    /// ```
    ///
    /// - **Shape:** `type` is the function's `wasm.func_sig`; the body's
    ///   entry block arguments are its parameters.
    /// - **Locals:** the parameters are locals `0..n`. `locals` lists the IR
    ///   types of the further locals, local `n + i` having `locals[i]`;
    ///   absent, the function has no further locals. Emission's
    ///   `ExplicitLocals` pass sets it when it assigns the locals, and the
    ///   emitter declares each listed type as its Wasm value type
    ///   (`design/wasm-backend.md`, "명시적 지역 변수와 stack 계약").
    #[asm_format(symbol(sym_name) signature(r#type, body) attr_dict region(body))]
    #[flags(IsolatedFromAbove, NoTerminator)]
    fn func<S: FuncSig>(
        sym_name: Attr<String>,
        r#type: Attr<TypeOf<S>>,
        locals: Option<Attr<[Type]>>,
    ) {
        #[region(body?)]
        {}
    }

    #[asm_format(symbol(sym_name) "from" module name attr_dict ":" signature(r#type))]
    fn import_func<S: FuncSig>(
        module: Attr<String>,
        name: Attr<String>,
        sym_name: Attr<String>,
        r#type: Attr<TypeOf<S>>,
    ) {
    }

    #[asm_format(func "as" name attr_dict)]
    fn export_func(name: Attr<String>, func: Attr<SymbolRef>) {}

    #[asm_format(attr_dict_bare)]
    fn export_memory(name: Attr<String>, index: Attr<u32>) {}

    #[asm_format(attr_dict_bare)]
    fn memory(min: Attr<u32>, max: Attr<u32>, shared: Attr<bool>, memory64: Attr<bool>) {}

    /// A data segment holding `bytes`. An active segment is copied to
    /// `offset` of the memory at instantiation; a passive one is not.
    ///
    /// ```text
    /// wasm.data(b"active") offset 16
    /// wasm.data(b"kept") offset 0 passive
    /// ```
    #[asm_format("(" bytes ")" "offset" offset ("passive" passive^)? attr_dict)]
    fn data(offset: Attr<u32>, bytes: Attr<Bytes>, #[default(false)] passive: Attr<bool>) {}

    #[asm_format(attr_dict_bare)]
    fn table(reftype: Attr<String>, min: Attr<u32>, max: Option<Attr<u32>>) {}

    /// An active element segment that writes the functions its `funcs` region
    /// names, in order, to `table` from `offset` at instantiation.
    ///
    /// ```text
    /// wasm.elem {
    ///   wasm.ref_func @f : wasm.funcref
    /// }
    /// wasm.elem table 1 offset 4 {
    ///   wasm.ref_func @g : wasm.funcref
    /// }
    /// ```
    #[flags(NoTerminator)]
    #[asm_format(("table" table^)? ("offset" offset^)? attr_dict region(funcs))]
    fn elem(#[default(0)] table: Attr<u32>, #[default(0)] offset: Attr<u32>) {
        #[region(funcs)]
        {}
    }

    #[asm_format(attr_dict_bare)]
    fn global(valtype: Attr<String>, mutable: Attr<bool>, init: Attr<_>) {}

    #[asm_format(attr_dict_bare ":" type(result))]
    fn global_get(index: Attr<u32>) -> Value<_> {}

    #[asm_format(value attr_dict_bare)]
    fn global_set(index: Attr<u32>, value: Value<_>) {}

    // i32
    #[asm_format(attr_dict_bare value ":" type(result))]
    fn i32_const(value: Attr<i32>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_add(lhs: Value<I32>, rhs: Value<I32>) -> Value<I32> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_div_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_div_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_rem_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_rem_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_lt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_lt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_le_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_le_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_gt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_gt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_ge_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_ge_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_and(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_or(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_xor(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_shl(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_shr_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i32_shr_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // i64
    #[asm_format(attr_dict_bare value ":" type(result))]
    fn i64_const(value: Attr<i64>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_div_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_div_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_rem_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_rem_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_lt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_lt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_le_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_le_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_gt_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_gt_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_ge_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_ge_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_and(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_or(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_xor(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_shl(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_shr_s(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn i64_shr_u(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // f32
    #[asm_format(attr_dict_bare value ":" type(result))]
    fn f32_const(value: Attr<f32>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_div(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_neg(operand: Value<_>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_lt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_le(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_gt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f32_ge(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // f64
    #[asm_format(attr_dict_bare value ":" type(result))]
    fn f64_const(value: Attr<f64>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_mul(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_div(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_neg(operand: Value<_>) -> Value<_> {}

    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_eq(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_ne(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_lt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_le(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_gt(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    #[asm_format(lhs "," rhs attr_dict_bare ":" type(result))]
    fn f64_ge(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

    // Local variables
    /// Read local `index` of the enclosing `wasm.func`.
    ///
    /// ```text
    /// %v = wasm.local_get {index = 2} : core.i32
    /// ```
    ///
    /// `index` names a parameter or an entry of the function's `locals`,
    /// whose type is the result's type. A local nothing set reads as its
    /// type's default value.
    #[asm_format(attr_dict_bare ":" type(result))]
    fn local_get(index: Attr<u32>) -> Value<_> {}

    /// Write `value` to local `index` of the enclosing `wasm.func`.
    ///
    /// ```text
    /// wasm.local_set %v {index = 2}
    /// ```
    ///
    /// `value`'s type is the local's type.
    #[asm_format(value attr_dict_bare)]
    fn local_set(index: Attr<u32>, value: Value<_>) {}

    /// Write `value` to local `index` and pass it on as the result.
    ///
    /// ```text
    /// %r = wasm.local_tee %v {index = 2} : core.i32
    /// ```
    #[asm_format(value attr_dict_bare ":" type(result))]
    fn local_tee(index: Attr<u32>, value: Value<_>) -> Value<_> {}

    // GC structs
    #[asm_format(fields attr_dict_bare ":" type(result))]
    fn struct_new(type_idx: Attr<u32>, fields: Variadic<_>) -> Value<_> {}

    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn struct_get(type_idx: Attr<u32>, field_idx: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    #[asm_format(r#ref "," value attr_dict_bare)]
    fn struct_set(type_idx: Attr<u32>, field_idx: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    // GC arrays
    /// Wasm `array.new`, which takes `init`, then `size`, as its
    /// `StackOperandOrder` states.
    ///
    /// ```text
    /// %a = wasm.array_new %size, %init {type_idx = 3} : core.ref
    /// ```
    #[asm_format(size "," init attr_dict_bare ":" type(result))]
    #[implements(StackOperandOrder)]
    fn array_new(type_idx: Attr<u32>, size: Value<I32>, init: Value<_>) -> Value<_> {}

    #[asm_format(size attr_dict_bare ":" type(result))]
    fn array_new_default(type_idx: Attr<u32>, size: Value<I32>) -> Value<_> {}

    #[asm_format(elements attr_dict_bare ":" type(result))]
    fn array_new_fixed(type_idx: Attr<u32>, elements: Variadic<_>) -> Value<_> {}

    #[asm_format(offset "," size attr_dict_bare ":" type(result))]
    fn array_new_data(
        type_idx: Attr<u32>,
        data_idx: Attr<u32>,
        offset: Value<I32>,
        size: Value<I32>,
    ) -> Value<_> {
    }

    #[asm_format(attr_dict_bare ":" type(result))]
    fn bytes_from_data(data_idx: Attr<u32>, offset: Attr<u32>, len: Attr<u32>) -> Value<_> {}

    #[asm_format(r#ref "," index attr_dict_bare ":" type(result))]
    fn array_get(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<I32>) -> Value<_> {}

    #[asm_format(r#ref "," index attr_dict_bare ":" type(result))]
    fn array_get_s(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<I32>) -> Value<_> {}

    #[asm_format(r#ref "," index attr_dict_bare ":" type(result))]
    fn array_get_u(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<I32>) -> Value<_> {}

    #[asm_format(r#ref "," index "," value attr_dict_bare)]
    fn array_set(type_idx: Attr<u32>, r#ref: Value<_>, index: Value<I32>, value: Value<_>) {}

    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn array_len(r#ref: Value<_>) -> Value<I32> {}

    #[asm_format(dst "," dst_offset "," src "," src_offset "," len attr_dict_bare)]
    fn array_copy(
        dst_type_idx: Attr<u32>,
        src_type_idx: Attr<u32>,
        dst: Value<_>,
        dst_offset: Value<I32>,
        src: Value<_>,
        src_offset: Value<I32>,
        len: Value<I32>,
    ) {
    }

    // References
    #[asm_format(attr_dict_bare ":" type(result))]
    fn ref_null(heap_type: Attr<String>, type_idx: Option<Attr<u32>>) -> Value<_> {}

    #[asm_format(func_name attr_dict ":" type(result))]
    fn ref_func(func_name: Attr<SymbolRef>) -> Value<_> {}

    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn ref_is_null(r#ref: Value<_>) -> Value<_> {}

    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn ref_cast(target_type: Attr<Type>, type_idx: Option<Attr<u32>>, r#ref: Value<_>) -> Value<_> {
    }

    /// Widen a GC reference to the abstract heap type `wasm.anyref`,
    /// `wasm.structref`, or `wasm.arrayref` that its result declares.
    /// WasmGC accepts a subtype reference where a supertype is declared and
    /// has no upcast instruction, so emission produces no instruction for it:
    /// the result is the operand's reference at the wider type. Operation
    /// schemas compare types exactly, so a widening the target relies on is
    /// written with this operation rather than by using the narrower value
    /// directly.
    ///
    /// Verification checks that the result type is one of the three abstract
    /// heap types. Whether the operand widens to it without a runtime cast is
    /// the target's physical assignability rule, which the Wasm backend checks
    /// before emission.
    ///
    /// ```text
    /// %wide = wasm.ref_upcast %narrow : wasm.structref
    /// ```
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn ref_upcast(r#ref: Value<_>) -> Value<_> {}

    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn ref_test(target_type: Attr<Type>, type_idx: Option<Attr<u32>>, r#ref: Value<_>) -> Value<_> {
    }

    // i31ref
    #[asm_format(value attr_dict_bare ":" type(result))]
    fn ref_i31(value: Value<_>) -> Value<_> {}
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn i31_get_s(r#ref: Value<_>) -> Value<_> {}
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn i31_get_u(r#ref: Value<_>) -> Value<_> {}

    // Type conversions (integer)
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_wrap_i64(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_extend_i32_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_extend_i32_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_extend8_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_extend16_s(operand: Value<_>) -> Value<_> {}

    // Type conversions (float to int)
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_trunc_f32_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_trunc_f32_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_trunc_f64_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_trunc_f64_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_trunc_f32_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_trunc_f32_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_trunc_f64_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_trunc_f64_u(operand: Value<_>) -> Value<_> {}

    // Type conversions (int to float)
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_convert_i32_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_convert_i32_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_convert_i64_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_convert_i64_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_convert_i32_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_convert_i32_u(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_convert_i64_s(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_convert_i64_u(operand: Value<_>) -> Value<_> {}

    // Float conversions
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_demote_f64(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_promote_f32(operand: Value<_>) -> Value<_> {}

    // Bitcast
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i32_reinterpret_f32(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn i64_reinterpret_f64(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f32_reinterpret_i32(operand: Value<_>) -> Value<_> {}
    #[asm_format(operand attr_dict_bare ":" type(result))]
    fn f64_reinterpret_i64(operand: Value<_>) -> Value<_> {}

    // Linear memory
    #[asm_format(attr_dict_bare ":" type(result))]
    fn memory_size(memory: Attr<u32>) -> Value<_> {}

    #[asm_format(delta attr_dict_bare ":" type(result))]
    fn memory_grow(memory: Attr<u32>, delta: Value<_>) -> Value<_> {}

    // Memory loads (full width)
    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i32_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn f32_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn f64_load(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    // Memory loads (partial width i32)
    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i32_load8_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i32_load8_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i32_load16_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i32_load16_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    // Memory loads (partial width i64)
    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load8_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load8_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load16_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load16_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load32_s(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    #[asm_format(addr attr_dict_bare ":" type(result))]
    fn i64_load32_u(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
    ) -> Value<_> {
    }

    // Memory stores (full width)
    #[asm_format(addr "," value attr_dict_bare)]
    fn i32_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn i64_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn f32_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn f64_store(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    // Memory stores (partial width)
    #[asm_format(addr "," value attr_dict_bare)]
    fn i32_store8(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn i32_store16(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn i64_store8(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn i64_store16(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }

    #[asm_format(addr "," value attr_dict_bare)]
    fn i64_store32(
        offset: Attr<u32>,
        align: Attr<u32>,
        memory: Attr<u32>,
        addr: Value<_>,
        value: Value<_>,
    ) {
    }
}

/// Reserved delimiter attribute for the number of Wasm signature inputs.
pub const NUM_INPUTS_ATTR: &str = "num_inputs";
/// Reserved delimiter attribute for the number of Wasm signature results.
pub const NUM_RESULTS_ATTR: &str = "num_results";

/// Why a name-matching `wasm.func_sig` does not satisfy its storage invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuncSigTypeError {
    MissingCount(&'static str),
    InvalidCount(&'static str),
    CountOverflow,
    LengthMismatch {
        num_inputs: u32,
        num_results: u32,
        params: usize,
    },
}

impl std::fmt::Display for FuncSigTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCount(name) => write!(f, "missing required `{name}` u32 attribute"),
            Self::InvalidCount(name) => write!(f, "`{name}` must be a u32 attribute"),
            Self::CountOverflow => write!(f, "input and result counts overflow u32"),
            Self::LengthMismatch {
                num_inputs,
                num_results,
                params,
            } => write!(
                f,
                "num_inputs ({num_inputs}) + num_results ({num_results}) must equal params length ({params})",
            ),
        }
    }
}

impl std::error::Error for FuncSigTypeError {}

/// Validated wrapper for an input-first, zero-or-more-result `wasm.func_sig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncSig(TypeRef);

crate::impl_func_sig_constraint!(FuncSig, "wasm.func_sig");

impl crate::type_verifier::VerifyType for FuncSig {
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, crate::ops::VerificationError> {
        let data = ctx.get_type(ty);
        debug_assert!(data.name == <Self as DialectType>::NAME);
        let num_inputs = read_count(ctx, &data.attrs, NUM_INPUTS_ATTR)?;
        let num_results = read_count(ctx, &data.attrs, NUM_RESULTS_ATTR)?;
        let total = num_inputs
            .checked_add(num_results)
            .ok_or(FuncSigTypeError::CountOverflow)?;
        if usize::try_from(total).ok() != Some(data.params.len()) {
            return Err(FuncSigTypeError::LengthMismatch {
                num_inputs,
                num_results,
                params: data.params.len(),
            }
            .into());
        }
        Ok(Self(ty))
    }
}

impl FuncSig {
    fn counts(self, ctx: &IrContext) -> (usize, usize) {
        let data = ctx.get_type(self.0);
        let inputs = usize::try_from(
            read_count(ctx, &data.attrs, NUM_INPUTS_ATTR)
                .expect("validated wasm.func_sig must retain num_inputs"),
        )
        .expect("u32 must fit usize");
        let results = usize::try_from(
            read_count(ctx, &data.attrs, NUM_RESULTS_ATTR)
                .expect("validated wasm.func_sig must retain num_results"),
        )
        .expect("u32 must fit usize");
        (inputs, results)
    }

    pub fn inputs(self, ctx: &IrContext) -> &[TypeRef] {
        let (inputs, _) = self.counts(ctx);
        &ctx.get_type(self.0).params[..inputs]
    }

    pub fn results(self, ctx: &IrContext) -> &[TypeRef] {
        let (inputs, results) = self.counts(ctx);
        &ctx.get_type(self.0).params[inputs..inputs + results]
    }

    pub fn is_resultless(self, ctx: &IrContext) -> bool {
        self.results(ctx).is_empty()
    }

    pub fn single_result(self, ctx: &IrContext) -> Option<TypeRef> {
        let results = self.results(ctx);
        (results.len() == 1).then(|| results[0])
    }

    pub fn non_reserved_attrs(
        self,
        ctx: &IrContext,
    ) -> impl Iterator<Item = (crate::StringRef, &Attribute)> {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|&(key, _)| ![NUM_INPUTS_ATTR, NUM_RESULTS_ATTR].contains(&ctx.str(key)))
    }

    pub fn remove_reserved_attrs(ctx: &mut crate::IrContext, attrs: &mut AttributeMap) {
        let mut attrs = attrs.view_mut(ctx);
        attrs.remove(NUM_INPUTS_ATTR);
        attrs.remove(NUM_RESULTS_ATTR);
    }
}

static FUNC_SIG_DEF: crate::type_def::TypeDef = crate::type_def::TypeDef::new("wasm.func_sig")
    .with_verifier::<FuncSig>()
    .with_asm_format::<FuncSig>();

crate::register_type_def!(&FUNC_SIG_DEF);

/// The function syntax `func.func_sig` also uses, `wasm.func_sig<(A) -> R>`,
/// with any number of results.
impl crate::asm_format::TypeAsmFormat for FuncSig {
    fn print(&self, p: &mut crate::printer::AsmPrinter<'_>) -> std::fmt::Result {
        super::func::print_signature_type(p, self.0)
    }

    fn parse(
        p: &mut crate::parser::AsmParser<'_>,
        name: &str,
    ) -> crate::parser::ParseResult<Option<crate::TypeRef>> {
        super::func::parse_signature_type(p, name, None)
    }
}

impl crate::asm_format::SignatureCodec for FuncSig {
    fn decompose(
        ctx: &crate::IrContext,
        ty: crate::TypeRef,
    ) -> Option<crate::asm_format::SignatureParts<'_>> {
        Self::from_type_ref(ctx, ty)?;
        super::func::signature_codec_parts(ctx, ty)
    }

    fn build(
        p: &mut crate::parser::AsmParser<'_>,
        inputs: Vec<(crate::TypeRef, crate::AttributeMap)>,
        results: Vec<(crate::TypeRef, crate::AttributeMap)>,
        tail: crate::AttributeMap,
    ) -> crate::parser::ParseResult<crate::TypeRef> {
        super::func::signature_from_parts(p, "wasm.func_sig", inputs, results, tail, None)
    }
}

impl DialectType for FuncSig {
    const DEF: &'static crate::type_def::TypeDef = &FUNC_SIG_DEF;

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        Self::matches(ctx, ty)
            .then(|| <Self as crate::type_verifier::VerifyType>::verify(ctx, ty).ok())
            .flatten()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

impl From<FuncSig> for TypeRef {
    fn from(value: FuncSig) -> Self {
        value.0
    }
}

fn read_count(
    ctx: &crate::IrContext,
    attrs: &AttributeMap,
    name: &'static str,
) -> Result<u32, FuncSigTypeError> {
    match attrs.view(ctx).get(name) {
        None => Err(FuncSigTypeError::MissingCount(name)),
        Some(Attribute::Int(value)) => {
            u32::try_from(*value).map_err(|_| FuncSigTypeError::InvalidCount(name))
        }
        Some(_) => Err(FuncSigTypeError::InvalidCount(name)),
    }
}

/// Whether `ty` is one of WasmGC's abstract heap types (`anyref`, `eqref`,
/// `i31ref`, `structref`, `arrayref`, `funcref`, `externref`).
pub fn is_abstract_heap_type(ctx: &IrContext, ty: TypeRef) -> bool {
    [
        Anyref::NAME,
        Eqref::NAME,
        I31ref::NAME,
        Structref::NAME,
        Arrayref::NAME,
        Funcref::NAME,
        Externref::NAME,
    ]
    .contains(&ctx.get_type(ty).name)
}

/// Construct a canonical target-owned `wasm.func_sig`.
pub fn func_sig(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
) -> FuncSig {
    func_sig_with_attrs(ctx, inputs, results, AttributeMap::new())
}

/// Construct a canonical target signature while preserving non-reserved attributes.
pub fn func_sig_with_attrs(
    ctx: &mut IrContext,
    inputs: impl IntoIterator<Item = TypeRef>,
    results: impl IntoIterator<Item = TypeRef>,
    attrs: AttributeMap,
) -> FuncSig {
    assert!(
        !attrs.view(ctx).contains_key(NUM_INPUTS_ATTR)
            && !attrs.view(ctx).contains_key(NUM_RESULTS_ATTR),
        "wasm.func_sig count attributes are reserved"
    );
    let inputs: Vec<_> = inputs.into_iter().collect();
    let results: Vec<_> = results.into_iter().collect();
    let num_inputs = u32::try_from(inputs.len()).expect("wasm.func_sig input count exceeds u32");
    let num_results = u32::try_from(results.len()).expect("wasm.func_sig result count exceeds u32");
    let mut builder = TypeDataBuilder::new(<FuncSig as DialectType>::NAME)
        .params(inputs)
        .params(results);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let ty = builder
        .attr(NUM_INPUTS_ATTR, Attribute::from(num_inputs))
        .attr(NUM_RESULTS_ATTR, Attribute::from(num_results))
        .intern(ctx);
    FuncSig(ty)
}

const INDIRECT_CALL_SIGNATURE_ATTR: &str = "signature";

/// The `wasm.func_sig` an indirect transfer declares, read without assuming
/// its schema holds.
fn exact_indirect_signature(ctx: &crate::IrContext, op: crate::OpRef) -> Option<crate::TypeRef> {
    ctx.op_attrs(op)
        .get_type(INDIRECT_CALL_SIGNATURE_ATTR)
        .filter(|&ty| FuncSig::from_type_ref(ctx, ty).is_some())
}

impl IndirectCallLike for CallIndirect {
    fn exact_signature(&self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        exact_indirect_signature(ctx, self.op_ref())
    }

    fn set_exact_signature(&self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

impl IndirectCallLike for ReturnCallIndirect {
    fn exact_signature(&self, ctx: &crate::IrContext) -> Option<crate::TypeRef> {
        exact_indirect_signature(ctx, self.op_ref())
    }

    fn set_exact_signature(&self, ctx: &mut crate::IrContext, signature: crate::TypeRef) -> bool {
        set_indirect_call_signature(ctx, self.op_ref(), signature)
    }
}

/// The operands a Wasm instruction takes from the stack, in the order they
/// were pushed.
pub type StackOperands = SmallVec<[ValueRef; 4]>;

/// The order in which an operation's instruction takes its operands from the
/// stack.
///
/// The last operand is on top of the stack when the instruction runs. An
/// operation whose instruction takes its operands in operand order does not
/// implement it; [`DynStackOperandOrder::of`] answers operand order for it.
#[op_interface]
pub trait StackOperandOrder: DialectOp {
    fn stack_operands(&self, ctx: &IrContext) -> StackOperands;
}

impl DynStackOperandOrder {
    /// The stack operands of `op`: its `StackOperandOrder`, or its operands
    /// in order when it does not implement the interface.
    pub fn of(ctx: &IrContext, op: OpRef) -> StackOperands {
        match Self::get(ctx, op) {
            Some(order) => order.stack_operands(ctx),
            None => ctx.op_operands(op).iter().copied().collect(),
        }
    }
}

impl StackOperandOrder for CallIndirect {
    fn stack_operands(&self, ctx: &IrContext) -> StackOperands {
        let mut operands: StackOperands = self.args(ctx).iter().copied().collect();
        operands.push(self.callee(ctx));
        operands
    }
}

impl StackOperandOrder for ReturnCallIndirect {
    fn stack_operands(&self, ctx: &IrContext) -> StackOperands {
        let mut operands: StackOperands = self.args(ctx).iter().copied().collect();
        operands.push(self.callee(ctx));
        operands
    }
}

impl StackOperandOrder for ArrayNew {
    fn stack_operands(&self, ctx: &IrContext) -> StackOperands {
        smallvec::smallvec![self.init(ctx), self.size(ctx)]
    }
}

/// The Wasm `loop` takes no parameters: `init` sets the body block
/// arguments' locals before it.
impl StackOperandOrder for Loop {
    fn stack_operands(&self, _ctx: &IrContext) -> StackOperands {
        StackOperands::new()
    }
}

impl CallLike for Call {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}
impl CallLike for ReturnCall {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}

/// Attach an exact callable contract to a `wasm` indirect transfer.
///
/// Returns `false` without mutation when the operation is not a `wasm`
/// indirect call or the supplied type is not a `wasm.func_sig` contract.
pub fn set_indirect_call_signature(
    ctx: &mut crate::IrContext,
    op: crate::OpRef,
    signature: crate::TypeRef,
) -> bool {
    if FuncSig::from_type_ref(ctx, signature).is_none()
        || (CallIndirect::from_op(ctx, op).is_err()
            && ReturnCallIndirect::from_op(ctx, op).is_err())
    {
        return false;
    }
    ctx.op_attrs_mut(op).insert(
        INDIRECT_CALL_SIGNATURE_ATTR,
        crate::Attribute::Type(signature),
    );
    true
}

impl crate::ops::Verify for RefUpcast {
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        let result = ctx.get_type(self.result_ty(ctx)).name;
        if [Anyref::NAME, Structref::NAME, Arrayref::NAME].contains(&result) {
            Ok(())
        } else {
            Err(format!(
                "wasm.ref_upcast result must be wasm.anyref, wasm.structref, or wasm.arrayref, found {}",
                result.display(ctx)
            )
            .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_interface::DynIndirectCallLike;
    use crate::parser::parse_test_module;

    #[test]
    fn structured_control_writes_its_results_only_when_present() {
        let text = r#"core.module @test {
  wasm.func @f(%0: core.i32) -> core.i32 {
      %1 = wasm.if %0 -> (core.i32) {
          wasm.yield %0
      } else {
          wasm.yield %0
      }
      wasm.if %0 {
          wasm.unreachable
      } else {
          wasm.unreachable
      }
      %2 = wasm.block -> (core.i32) {
          wasm.yield %1
      }
      wasm.loop(%2) {
        ^bb6(%3: core.i32):
          wasm.br {target = 0}
      }
      wasm.return %2
  }
}
"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
    }

    #[test]
    fn br_table_writes_its_default_before_the_table() {
        let text = r#"core.module @test {
  wasm.func @f(%0: core.i32) {
      wasm.block {
          wasm.block {
              wasm.br_table %0, default 1 [0, 1, 0]
          }
          wasm.br {target = 0}
      }
      wasm.return
  }
}
"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
        let std::ops::ControlFlow::Break(table) = crate::walk::walk_op(
            &ctx,
            module.op(),
            &mut |op| match BrTable::from_op(&ctx, op) {
                Ok(table) => std::ops::ControlFlow::Break(table),
                Err(_) => std::ops::ControlFlow::Continue(crate::walk::WalkAction::Advance),
            },
        ) else {
            panic!("the module has a br_table");
        };
        assert_eq!(table.default(&ctx), 1);
        assert_eq!(table.targets(&ctx).collect::<Vec<_>>(), [0, 1, 0]);
    }

    #[test]
    fn elem_writes_only_a_non_default_table_and_offset() {
        let text = r#"core.module @test {
  wasm.elem {
      %0 = wasm.ref_func @f : wasm.funcref
  }
  wasm.elem table 1 offset 4 {
      %0 = wasm.ref_func @g : wasm.funcref
  }
}
"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
        let elems: Vec<_> = module
            .ops(&ctx)
            .iter()
            .map(|&op| Elem::from_op(&ctx, op).unwrap())
            .collect();
        assert_eq!((elems[0].table(&ctx), elems[0].offset(&ctx)), (0, 0));
        assert_eq!((elems[1].table(&ctx), elems[1].offset(&ctx)), (1, 4));
    }

    #[test]
    fn ref_upcast_result_is_an_abstract_gc_reference() {
        let verifies = |result: &str| {
            let mut ctx = crate::IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{\n  func.func @f(%s: wasm.structref) -> core.nil {{\n    %w = wasm.ref_upcast %s : {result}\n    func.return\n  }}\n}}"
                ),
            );
            crate::validation::validate_operation_verifiers(&ctx, module).is_ok()
        };
        assert!(verifies("wasm.anyref"));
        assert!(verifies("wasm.structref"));
        assert!(!verifies("core.i32"));
    }

    #[test]
    fn gc_array_indices_and_lengths_are_i32() {
        let schema_ok = |body: &str| {
            let mut ctx = crate::IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{\n  func.func @f(%a: wasm.arrayref, %n: core.i32, %w: core.i64) -> core.nil {{\n    {body}\n    func.return\n  }}\n}}"
                ),
            );
            crate::validation::validate_op_schemas(&ctx, module.op()).is_ok()
        };
        assert!(schema_ok(
            "%e = wasm.array_get %a, %n {type_idx = 0} : core.i32"
        ));
        assert!(!schema_ok(
            "%e = wasm.array_get %a, %w {type_idx = 0} : core.i32"
        ));
        assert!(schema_ok("%l = wasm.array_len %a : core.i32"));
        assert!(!schema_ok("%l = wasm.array_len %a : core.i64"));
        assert!(!schema_ok(
            "wasm.array_copy %a, %n, %a, %n, %w {dst_type_idx = 0, src_type_idx = 0}"
        ));
    }

    #[test]
    fn indirect_call_interface_uses_wasm_owned_signature() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @ordinary(%table: core.i32, %value: core.i32) -> core.i32 {
    %result = wasm.call_indirect %table(%value) : (core.i32) -> core.i32
    wasm.return %result
  }
  wasm.func @plain(%table: core.i32, %value: core.i32) -> core.i32 {
    %result = "wasm.call_indirect"(%table, %value) : core.i32
    wasm.return %result
  }
  wasm.func @tail(%table: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table(%value) : (core.i32) -> core.nil
  }
  wasm.func @direct() -> core.nil {
    wasm.return
  }
}"#,
        );
        let functions = module.ops(&ctx);
        let body_op = |index| {
            let body = ctx.op_region(functions[index], 0).unwrap();
            ctx.block(ctx.region(body).blocks[0]).ops[0]
        };
        let ordinary = body_op(0);
        let plain = body_op(1);
        let tail = body_op(2);
        let direct = body_op(3);

        for op in [ordinary, tail] {
            let call = DynIndirectCallLike::get(&ctx, op).expect("indirect call");
            let operands = ctx.op_operands(op);
            assert!(call.exact_signature(&ctx).is_some());
            assert_eq!(call.indirect_callee(&ctx), Some(operands[0]));
            assert_eq!(call.indirect_arguments(&ctx), Some(&operands[1..]));
        }
        let plain_call = DynIndirectCallLike::get(&ctx, plain).expect("plain indirect call");
        assert_eq!(plain_call.exact_signature(&ctx), None);
        assert!(plain_call.indirect_callee(&ctx).is_some());
        assert!(plain_call.indirect_arguments(&ctx).is_some());
        assert!(DynIndirectCallLike::get(&ctx, direct).is_none());

        let printed = module.op().display(&ctx).to_string();
        assert!(
            printed.contains("wasm.call_indirect %0(%1) : (core.i32) -> core.i32"),
            "{printed}"
        );
    }

    #[test]
    fn indirect_calls_write_the_index_as_their_callee() {
        let text = r#"core.module @test {
  wasm.func @calls(%0: core.i32, %1: core.i64) -> core.i32 {
      %2 = wasm.call_indirect %0(%1) : (core.i64) -> core.i32
      %3, %4 = wasm.call_indirect %0() table 1 : () -> (core.i32, core.i64)
      wasm.return_call_indirect %0(%1, %2) table 2 attributes {extra = 1} : (core.i64, core.i32) -> core.i32
  }
}
"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
    }

    #[test]
    #[should_panic(expected = "count attributes are reserved")]
    fn func_sig_constructor_rejects_reserved_count_attributes() {
        let mut ctx = crate::IrContext::new();
        let mut attrs = crate::AttributeMap::new();
        attrs
            .view_mut(&mut ctx)
            .insert(NUM_INPUTS_ATTR, crate::Attribute::Int(0));
        let _ = func_sig_with_attrs(&mut ctx, [], [], attrs);
    }

    #[test]
    fn func_writes_its_signature_in_the_function_form() {
        let source = r#"core.module @test {
  wasm.func @external(%arg0: core.i32) -> core.i32 attributes {abi = "C"}
  wasm.func @pair(%0: core.i32) -> (core.i32, core.i64) {
      %1 = wasm.i64_const 1 : core.i64
      wasm.return %0, %1
  }
}
"#;
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(&mut ctx, source);
        assert_eq!(module.op().display(&ctx).to_string(), source);
    }
}
