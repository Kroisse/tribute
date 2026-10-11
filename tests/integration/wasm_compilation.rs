//! End-to-end WebAssembly compilation tests.
//!
//! These tests validate the full source code → WASM compilation pipeline.
//! Programs whose output is checked on both the native and Wasm targets live
//! with the native end-to-end tests and use
//! `common::assert_output_on_both_targets`.
//!
//! ## Current Status
//!
//! Most basic compilation scenarios work:
//! - Simple literals and arithmetic expressions
//! - Functions with parameters
//! - Local variables (let bindings)
//! - Intrinsics like print_line
//!
//! ## Remaining Work
//!
//! The following features need additional lowering passes:
//! - `tribute.block` → block expressions in case branches

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;

use crate::common::{assert_wasm_output, run_wasm, run_wasm_invoking};

fn expect_wasm_compilation_success<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    message: &str,
) -> &'db [u8] {
    compile_to_wasm_binary(db, source)
        .unwrap_or_else(|diagnostics| panic!("{message}: {diagnostics:?}"))
}

#[salsa_test]
fn test_compile_failure_returns_diagnostics(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "invalid.trb", "fn main() -> Int { true }");
    let Err(diagnostics) = compile_to_wasm_binary(db, source) else {
        panic!("invalid source should fail WebAssembly compilation");
    };
    assert!(!diagnostics.is_empty());
}

#[salsa_test]
fn test_compile_source_list_prepend_is_an_ordinary_function(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "source_list_prepend.trb",
        r#"
use std::io::{Io, print_line}
pub mod List {
    pub fn prepend(value: Nat, tail: Nat) -> Nat { value + tail }
}
fn main() ->{Io} Nil {
    print_line(case List::prepend(20, 22) {
        42 -> "ok"
        _ -> "bad"
    })
}
"#,
    );
    expect_wasm_compilation_success(db, source, "Source List::prepend must remain ordinary");
}

#[salsa_test]
fn test_compile_unsupported_wasm_read_line(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "wasm_read_line.trb",
        r#"use abilities::Throw
use std::io::{Error as IoError, Io, print_line, read_line}

fn next_line() ->{Io} String {
    handle read_line() {
        do line { line }
        op Throw::throw(error) {
            case error {
                IoError::EndOfFile -> ""
                IoError::InvalidEncoding -> ""
                IoError::System(_) -> ""
            }
        }
    }
}

fn main() ->{Io} Nil {
    print_line(next_line())
}
"#,
    );
    let Err(diagnostics) = compile_to_wasm_binary(db, source) else {
        panic!("Wasm read_line must remain explicitly unsupported");
    };
    let messages: Vec<String> = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("io-to-wasm")
                && message.contains("tribute_io.read_line")),
        "read_line must fail at the explicit io-to-wasm boundary, not as a missing body: {messages:?}"
    );
}

// =============================================================================
// Passing end-to-end tests
// =============================================================================

#[salsa_test]
fn test_compile_builtin_io_entrypoint(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "builtin_io_main.trb",
        r#"use std::io::Io

fn touch_world() ->{Io} Nil { Nil }

fn main() ->{Io} Nil {
    touch_world()
}
"#,
    );
    let bytes = expect_wasm_compilation_success(db, source, "Should compile builtin Io main");
    assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
}

#[salsa_test]
fn test_compile_simple_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "literal.trb",
        r#"
fn main() ->{std::io::Io} Nil {
    case 42 {
        42 -> std::io::print_line("ok")
        _ -> std::io::print_line("unexpected")
    }
}
"#,
    );
    let bytes = expect_wasm_compilation_success(db, source, "Should compile literal return");
    assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
}

#[salsa_test]
fn test_compile_arithmetic_expr(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "arith.trb",
        r#"
fn main() ->{std::io::Io} Nil {
    case 1 + 2 * 3 {
        7 -> std::io::print_line("ok")
        _ -> std::io::print_line("unexpected")
    }
}
"#,
    );
    expect_wasm_compilation_success(db, source, "Should compile arithmetic expression");
}

#[salsa_test]
fn test_compile_function_with_params(db: &salsa::DatabaseImpl) {
    let code = r#"
fn add(a: Nat, b: Nat) -> Nat { a + b }
fn main() ->{std::io::Io} Nil {
    case add(1, 2) {
        3 -> std::io::print_line("ok")
        _ -> std::io::print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "params.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile function with params");
}

/// The target-independent root delimiter keeps an open-callback `main` on the
/// Direct/EvidenceDirect entry ABI, so Wasm never receives a CPS entrypoint.
#[salsa_test]
fn test_compile_open_callback_root_main(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "open_callback_root_main.trb",
        r#"
fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int {
    f(x)
}

fn main() -> Nil {
    let _ = apply(fn(value) { value + +1 }, +41)
}
"#,
    );
    let binary = expect_wasm_compilation_success(
        db,
        source,
        "Should compile a root main that closes an open callback worker",
    );
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(binary)
        .expect("compiled source must produce a valid Wasm binary");
}

#[salsa_test]
fn test_compile_print_line(db: &salsa::DatabaseImpl) {
    let code = r#"
fn main() ->{std::io::Io} Nil {
    let left = b"Hello, "
    let right = b"World!"
    std::io::print_line(String::from_bytes(left <> right))
}
"#;
    let source = SourceCst::from_source_str(db, "hello.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile dynamic print_line");
}

#[test]
fn test_execute_dynamic_bytes_write_boundary() {
    let mut ctx = trunk_ir::IrContext::new();
    let left = "x".repeat(35_000);
    let right = "y".repeat(35_000);
    let ir = format!(
        r#"core.module @test {{
  func.func @__tribute_bytes_concat(%left: core.bytes, %right: core.bytes) -> core.bytes attributes {{abi = "C"}}
  func.func @main() -> core.nil {{
    %left = adt.bytes_const {{value = b"{left}"}} : core.bytes
    %right = adt.bytes_const {{value = b"{right}"}} : core.bytes
    %joined = func.call %left, %right {{callee = @__tribute_bytes_concat}} : core.bytes
    %newline = arith.const {{value = 1}} : core.i1
    %result = tribute_io.write %joined, %newline : core.nil
    func.return
  }}
}}"#
    );
    let module = trunk_ir::parser::parse_test_module(&mut ctx, &ir);
    tribute_passes::wasm::lower::lower_to_wasm(&mut ctx, module, &mut Default::default())
        .expect("lower dynamic output to Wasm");
    tribute_passes::wasm::lower::finalize_wasm_gc_types(&mut ctx, module)
        .expect("finalize semantic WasmGC types");
    let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module)
        .expect("emit dynamic output Wasm");
    let mut expected = left.into_bytes();
    expected.extend_from_slice(right.as_bytes());
    expected.push(b'\n');
    assert_wasm_output(&binary.bytes, expected);
}

#[salsa_test]
fn test_execute_string_literals_and_dynamic_bytes(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "string_literals.trb",
        r#"
fn main() ->{std::io::Io} Nil {
    std::io::print_line("before")
    std::io::print_line("")
    std::io::print_line("안녕")
    let dynamic = b"dynamic bytes"
    std::io::print_line(String::from_bytes(dynamic))
    let shared = b"shared"
    std::io::print_line("shared")
    std::io::print_line(String::from_bytes(shared <> b" bytes"))
    std::io::print_line("rope " <> "branch")
    std::io::print_line("after")
}
"#,
    );
    let binary = expect_wasm_compilation_success(db, source, "Should compile String literals");
    assert_wasm_output(
        binary,
        "before\n\n안녕\ndynamic bytes\nshared\nshared bytes\nrope branch\nafter\n".as_bytes(),
    );
}

#[salsa_test]
fn test_compile_local_variables(db: &salsa::DatabaseImpl) {
    let code = r#"
fn test_ops() -> Nat {
    let a = 10
    let b = 3
    a + b
}
fn main() ->{std::io::Io} Nil {
    case test_ops() {
        13 -> std::io::print_line("ok")
        _ -> std::io::print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "locals.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile local variables");
}

// =============================================================================
// Tests requiring additional lowering passes
// =============================================================================

// Note: Tribute does not have if-else expressions; control flow uses
// pattern matching (case) and algebraic effects.

#[salsa_test]
fn test_compile_case_expression(db: &salsa::DatabaseImpl) {
    let code = r#"
fn classify(n: Nat) -> String {
    case n {
        0 -> "zero"
        1 -> "one"
        _ -> "other"
    }
}
fn main() ->{std::io::Io} Nil { std::io::print_line(classify(1)) }
"#;
    let source = SourceCst::from_source_str(db, "case_expr.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile case expression");
}

#[salsa_test]
fn test_execute_tail_dispatch_ability(db: &salsa::DatabaseImpl) {
    let code = r#"
ability Console {
    fn read() -> Int
    fn print(value: Int) -> Nil
}

fn use_console() ->{Console} Int {
    let n = Console::read()
    Console::print(n)
    n
}

fn run() -> Int {
    handle use_console() {
        do result { result }
        fn Console::read() { +41 }
        fn Console::print(value) { Nil }
    }
}

fn main() ->{std::io::Io} Nil {
    case run() {
        +41 -> std::io::print_line("ok")
        _ -> std::io::print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "tail_dispatch_ability.trb", code);
    let binary = expect_wasm_compilation_success(
        db,
        source,
        "Should compile tail-dispatch ability through wasm effect ABI lowering",
    );
    assert_wasm_output(binary, b"ok\n");
}

#[salsa_test]
fn test_execute_cps_dispatch_ability(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn bump() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + +1)
    n
}

fn run_state() -> Int {
    handle bump() {
        do result { result }
        op State::get() { resume +41 }
        op State::set(value) { resume Nil }
    }
}

fn main() ->{std::io::Io} Nil {
    case run_state() {
        +41 -> std::io::print_line("ok")
        _ -> std::io::print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "cps_dispatch_ability.trb", code);
    let binary = expect_wasm_compilation_success(
        db,
        source,
        "Should compile CPS ability dispatch through wasm effect ABI lowering",
    );
    assert_wasm_output(binary, b"ok\n");
}

#[salsa_test]
fn test_execute_handler_that_drops_its_continuation(db: &salsa::DatabaseImpl) {
    let code = r#"
ability Abort {
    op abort(code: Nat) -> Never
}

fn checked(value: Nat) ->{Abort} Nat {
    case value {
        0 -> Abort::abort(7)
        _ -> value + 100
    }
}

fn run(value: Nat) -> Nat {
    handle checked(value) {
        do result { result }
        op Abort::abort(code) { code }
    }
}

fn check(ok: Bool) ->{std::io::Io} Nil {
    case ok {
        True -> std::io::print_line("ok")
        False -> std::io::print_line("unexpected")
    }
}

fn main() ->{std::io::Io} Nil {
    check(run(0) == 7)
    check(run(1) == 101)
}
"#;
    let source = SourceCst::from_source_str(db, "aborting_handler.trb", code);
    let binary = expect_wasm_compilation_success(
        db,
        source,
        "Should compile a handler arm that drops its continuation",
    );
    assert_wasm_output(binary, b"ok\nok\n");
}

const BYTES_SLICES: &str = r#"
fn main() ->{std::io::Io} Nil {
    let bytes = b"<hello world>"
    let inner = bytes.slice_or_panic(1, 12)
    std::io::print_line(String::from_bytes(inner.slice_or_panic(6, 11) <> b" " <> inner.slice_or_panic(0, 5)))
    std::io::print_line(String::from_bytes(bytes.slice_or_panic(3, 3)))
}
"#;

#[salsa_test]
fn test_execute_bytes_slice_or_panic_shares_the_backing_array(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "bytes_slice.trb", BYTES_SLICES);
    let binary = expect_wasm_compilation_success(db, source, "Should compile bytes slices");
    assert_wasm_output(binary, b"world hello\n\n");
}

#[salsa_test]
fn test_execute_bytes_slice_or_panic_traps_out_of_range(db: &salsa::DatabaseImpl) {
    for (name, range) in [("past_end", "(1, 14)"), ("reversed", "(5, 4)")] {
        let code = BYTES_SLICES.replace("(1, 12)", range);
        let source = SourceCst::from_source_str(db, &format!("bytes_slice_{name}.trb"), &code);
        let binary = expect_wasm_compilation_success(db, source, "Should compile bytes slices");
        let output = run_wasm(binary);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{name} must trap");
        assert!(stderr.contains("unreachable"), "{name}: {stderr}");
    }
}

#[salsa_test]
fn test_execute_string_from_case_produced_bytes(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "case_bytes.trb",
        r#"
fn pick(x: Bytes, y: Bytes, f: Bool) -> Bytes {
    case f {
        True -> x
        False -> y
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(String::from_bytes(pick(b"ab", b"cd", True)))
    std::io::print_line(String::from_bytes(pick(b"ab", b"cd", False)))
    std::io::print_line(String::from_bytes(b"<hello>".slice(1, 3)))
    std::io::print_line(String::from_bytes(b"<hello>".slice(2, 100)))
}
"#,
    );
    let binary = expect_wasm_compilation_success(db, source, "Should compile case-produced Bytes");
    assert_wasm_output(binary, b"ab\ncd\nhe\nello>\n");
}

#[salsa_test]
fn test_variants_are_told_apart_by_described_descriptor(db: &salsa::DatabaseImpl) {
    use trunk_ir_wasm_backend::gc_types::{DESCRIBED_IDX, FIRST_USER_TYPE_IDX};
    use wasmparser::{CompositeInnerType, Operator, Payload};

    let source = SourceCst::from_source_str(
        db,
        "variant_descriptor.trb",
        r#"
enum Shape {
    Circle(Int),
    Square(Int),
    Empty,
}

fn area(shape: Shape) -> Int {
    case shape {
        Circle(r) -> r * r * +3
        Square(s) -> s * s
        Empty -> +0
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(Int::to_string(area(Circle(+2)) + area(Square(+3)) + area(Empty)))
}
"#,
    );
    let binary = expect_wasm_compilation_success(db, source, "Should compile enum matching");

    let mut struct_types = Vec::new();
    let mut ref_tests = 0;
    for payload in wasmparser::Parser::new(0).parse_all(binary) {
        match payload.expect("Wasm payload") {
            Payload::TypeSection(types) => {
                for group in types {
                    for sub_type in group.expect("rec group").into_types() {
                        struct_types.push(match &sub_type.composite_type.inner {
                            CompositeInnerType::Struct(layout) => Some((
                                layout.fields.to_vec(),
                                sub_type.is_final,
                                sub_type.supertype_idx.map(|index| {
                                    index.as_module_index().expect("module type index")
                                }),
                            )),
                            _ => None,
                        });
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                for operator in body.get_operators_reader().expect("operators") {
                    if matches!(
                        operator.expect("operator"),
                        Operator::RefTestNonNull { .. } | Operator::RefTestNullable { .. }
                    ) {
                        ref_tests += 1;
                    }
                }
            }
            _ => {}
        }
    }

    let (described_fields, described_is_final, described_supertype) = struct_types
        [DESCRIBED_IDX as usize]
        .clone()
        .expect("the Described type is a struct");
    let [descriptor] = described_fields[..] else {
        panic!("the Described type holds the descriptor field alone")
    };
    assert_eq!(
        descriptor.element_type,
        wasmparser::StorageType::Val(wasmparser::ValType::I32)
    );
    assert!(!described_is_final);
    assert_eq!(described_supertype, None);

    let user_structs: Vec<_> = struct_types[FIRST_USER_TYPE_IDX as usize..]
        .iter()
        .flatten()
        .collect();
    // `Circle(Int)` and `Square(Int)` have equal fields, so they share one
    // structural type and only their descriptors tell them apart.
    for (index, (fields, ..)) in user_structs.iter().enumerate() {
        assert!(
            user_structs[..index]
                .iter()
                .all(|(earlier, ..)| earlier != fields),
            "struct types with equal fields are one type"
        );
    }
    let int_payload = [descriptor, descriptor];
    assert!(
        user_structs
            .iter()
            .any(|(fields, ..)| fields[..] == int_payload),
        "the shared Circle and Square type"
    );
    for (fields, is_final, supertype) in user_structs {
        assert_eq!(fields.first(), Some(&descriptor));
        assert!(is_final);
        assert_eq!(*supertype, Some(DESCRIBED_IDX));
    }
    assert_eq!(ref_tests, 0, "variants are not tested by GC type");

    assert_wasm_output(binary, b"21\n");
}

#[test]
fn test_validate_fixed_wasm_dispatch_abis() {
    let mut ctx = trunk_ir::IrContext::new();
    let module = trunk_ir::parser::parse_test_module(
        &mut ctx,
        r#"core.module @test {
        !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
        !Closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
        func.func @tail(%ev: !Evidence, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
            %result = effect.dispatch_tail %ev, %payload {ability_ref = @Console, op_name = "read"} : tribute_rt.anyref
            func.return %result
        }
        func.func @cps(%ev: !Evidence, %dispatch: !Closure, %resume: !Closure, %payload: tribute_rt.anyref) {
            effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = @State, op_name = "get", answer_type = core.i32}
        }
        func.func @install(%ev: !Evidence, %prompt: core.i32, %tr: !Closure) -> !Evidence {
            %extended = effect.extend %ev, %prompt, %tr, %ev {ability_ref = @State} : !Evidence
            func.return %extended
        }
    }"#,
    );
    // Boundary evidence lowering declares the helper ABI; Wasm lowering past
    // the exit binds it to the target's GC-array implementation.
    tribute_passes::wasm::evidence_to_wasm::prepare_wasm_evidence_runtime(&mut ctx, module);
    for op in module.ops_snapshot(&ctx) {
        if let Ok(function) =
            <trunk_ir::dialect::func::Func as trunk_ir::ops::DialectOp>::from_op(&ctx, op)
        {
            tribute_passes::wasm::evidence_to_wasm::lower_evidence_to_wasm_func(&mut ctx, function)
                .unwrap();
        }
    }
    let lowered = trunk_ir::printer::print_module(&ctx, module.op());
    assert!(!lowered.contains("effect."), "{lowered}");
    assert!(!lowered.contains("wasm."), "{lowered}");
    assert!(!lowered.contains("tribute.calling_convention"), "{lowered}");
    tribute_passes::wasm::lower::lower_to_wasm(&mut ctx, module, &mut Default::default()).unwrap();
    tribute_passes::wasm::lower::finalize_wasm_gc_types(&mut ctx, module).unwrap();
    let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module).unwrap();
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&binary.bytes)
        .expect("both fixed dispatch ABIs must encode valid function types");
}

/// The Wasm evidence helpers keep one marker stack per ability: `extend` and
/// `dup` push, `mask` pops, and popping the last marker removes the slot.
#[test]
fn test_execute_wasm_evidence_marker_stacks() {
    let mut ctx = trunk_ir::IrContext::new();
    let module = trunk_ir::parser::parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
  func.func @__tribute_evidence_lookup(%ev: !Evidence, %id: core.i32) -> core.i32 attributes {abi = "C"}
  func.func @__tribute_evidence_extend(%ev: !Evidence, %id: core.i32, %prompt: core.i32, %tr: wasm.anyref, %outer: !Evidence) -> !Evidence attributes {abi = "C"}
  func.func @__tribute_evidence_outer(%ev: !Evidence, %id: core.i32) -> !Evidence attributes {abi = "C"}
  func.func @__tribute_evidence_mask(%ev: !Evidence, %id: core.i32) -> !Evidence attributes {abi = "C"}
  func.func @__tribute_evidence_dup(%ev: !Evidence, %id: core.i32) -> !Evidence attributes {abi = "C"}
  wasm.func @check() -> core.i32 {
    %zero = wasm.i32_const {value = 0} : core.i32
    %one = wasm.i32_const {value = 1} : core.i32
    %two = wasm.i32_const {value = 2} : core.i32
    %nine = wasm.i32_const {value = 9} : core.i32
    %ten = wasm.i32_const {value = 10} : core.i32
    %state = wasm.i32_const {value = 10} : core.i32
    %console = wasm.i32_const {value = 20} : core.i32
    %null = wasm.ref_null {heap_type = "any"} : wasm.anyref
    %empty = wasm.array_new_default %zero {type_idx = 5} : !Evidence
    %with_console = wasm.call %empty, %console, %nine, %null, %empty {callee = @__tribute_evidence_extend} : !Evidence
    %outer = wasm.call %with_console, %state, %one, %null, %with_console {callee = @__tribute_evidence_extend} : !Evidence
    %inner = wasm.call %outer, %state, %two, %null, %outer {callee = @__tribute_evidence_extend} : !Evidence
    %inner_tag = wasm.call %inner, %state {callee = @__tribute_evidence_lookup} : core.i32
    %masked = wasm.call %inner, %state {callee = @__tribute_evidence_mask} : !Evidence
    %masked_tag = wasm.call %masked, %state {callee = @__tribute_evidence_lookup} : core.i32
    %dup = wasm.call %inner, %state {callee = @__tribute_evidence_dup} : !Evidence
    %dup_once = wasm.call %dup, %state {callee = @__tribute_evidence_mask} : !Evidence
    %dup_once_tag = wasm.call %dup_once, %state {callee = @__tribute_evidence_lookup} : core.i32
    %dup_twice = wasm.call %dup_once, %state {callee = @__tribute_evidence_mask} : !Evidence
    %dup_twice_tag = wasm.call %dup_twice, %state {callee = @__tribute_evidence_lookup} : core.i32
    %removed = wasm.call %masked, %state {callee = @__tribute_evidence_mask} : !Evidence
    %removed_console_tag = wasm.call %removed, %console {callee = @__tribute_evidence_lookup} : core.i32
    %removed_len = wasm.array_len %removed : core.i32
    %inner_console_tag = wasm.call %inner, %console {callee = @__tribute_evidence_lookup} : core.i32
    %inner_len = wasm.array_len %inner : core.i32
    %d0 = wasm.i32_mul %inner_tag, %ten : core.i32
    %d1 = wasm.i32_add %d0, %masked_tag : core.i32
    %d2 = wasm.i32_mul %d1, %ten : core.i32
    %d3 = wasm.i32_add %d2, %dup_once_tag : core.i32
    %d4 = wasm.i32_mul %d3, %ten : core.i32
    %d5 = wasm.i32_add %d4, %dup_twice_tag : core.i32
    %d6 = wasm.i32_mul %d5, %ten : core.i32
    %d7 = wasm.i32_add %d6, %removed_console_tag : core.i32
    %d8 = wasm.i32_mul %d7, %ten : core.i32
    %d9 = wasm.i32_add %d8, %removed_len : core.i32
    %d10 = wasm.i32_mul %d9, %ten : core.i32
    %d11 = wasm.i32_add %d10, %inner_console_tag : core.i32
    %d12 = wasm.i32_mul %d11, %ten : core.i32
    %d13 = wasm.i32_add %d12, %inner_len : core.i32
    %installed_on = wasm.call %dup, %state {callee = @__tribute_evidence_outer} : !Evidence
    %installed_on_tag = wasm.call %installed_on, %state {callee = @__tribute_evidence_lookup} : core.i32
    %d14 = wasm.i32_mul %d13, %ten : core.i32
    %d15 = wasm.i32_add %d14, %installed_on_tag : core.i32
    wasm.return %d15
  }
  wasm.export_func {name = "check", func = @check}
}"#,
    );
    tribute_passes::wasm::evidence_to_wasm::bind_wasm_evidence_runtime(&mut ctx, module);
    tribute_passes::wasm::lower::finalize_wasm_gc_types(&mut ctx, module).unwrap();
    let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module).unwrap();
    let output = run_wasm_invoking(&binary.bytes, "check");
    assert!(
        output.status.success(),
        "wasmtime failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // inner top, masked top, dup masked once and twice, then the other
    // ability's tag and the slot count after and before removing a slot,
    // then the tag in the evidence the copied inner handler was installed on.
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "212191921");
}
