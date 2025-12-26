//! End-to-end tests for compilation and execution with wasmtime.

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::stage_lower_to_wasm;
use tribute_front::SourceCst;

/// Helper to create a wasmtime engine with GC support
fn create_gc_engine() -> wasmtime::Engine {
    let mut config = wasmtime::Config::new();
    config.wasm_gc(true);
    wasmtime::Engine::new(&config).expect("Failed to create engine")
}

#[test]
fn test_add_compiles_and_runs() {
    use tribute::database::parse_with_thread_local;

    let source = include_str!("../lang-examples/add.trb");
    let source_code = Rope::from_str(source);

    TributeDatabaseImpl::default().attach(|db| {
        // Parse and create source file
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "add.trb", source_code.clone(), tree);

        // Run full compilation pipeline including WASM lowering
        let wasm_binary = stage_lower_to_wasm(db, source_file).expect("WASM compilation failed");

        // Execute with wasmtime (no WASI needed for add.trb)
        let engine = create_gc_engine();
        let module = wasmtime::Module::new(&engine, wasm_binary.bytes(db)).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance =
            wasmtime::Instance::new(&mut store, &module, &[]).expect("Failed to instantiate");

        // Call main directly (add.trb doesn't use print_line, so no _start is generated)
        // Note: Tribute's Int type compiles to i64 in WASM
        let main = instance
            .get_typed_func::<(), i64>(&mut store, "main")
            .expect("main function not found");

        let result = main.call(&mut store, ()).expect("Execution failed");

        // Verify the result: add(40, 2) = 42
        assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test that Int boxing/unboxing works correctly in polymorphic contexts.
/// This verifies PR #61 (uniform representation for generics).
#[test]
fn test_generic_int_identity() {
    use tribute::database::parse_with_thread_local;

    // Generic identity function that boxes Int to i31ref and unboxes back
    let source_code = Rope::from_str(
        r#"
fn identity(x: a) -> a { x }
fn main() -> Int { identity(42) }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "int_identity.trb", source_code.clone(), tree);

        let wasm_binary = stage_lower_to_wasm(db, source_file).expect("WASM compilation failed");

        let engine = create_gc_engine();
        let module = wasmtime::Module::new(&engine, wasm_binary.bytes(db)).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance =
            wasmtime::Instance::new(&mut store, &module, &[]).expect("Failed to instantiate");

        // Int compiles to i64 in WASM
        let main = instance
            .get_typed_func::<(), i64>(&mut store, "main")
            .expect("main function not found");

        let result = main.call(&mut store, ()).expect("Execution failed");

        // Verify the result is 42 (Int identity should preserve the value)
        assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test that Float boxing/unboxing works correctly in polymorphic contexts.
/// This verifies issue #52 (Float boxing for generic type parameters).
#[test]
fn test_generic_float_identity() {
    use tribute::database::parse_with_thread_local;

    // Generic identity function that boxes Float to anyref and unboxes back
    let source_code = Rope::from_str(
        r#"
fn identity(x: a) -> a { x }
fn main() -> Float { identity(3.125) }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_identity.trb", source_code.clone(), tree);

        let wasm_binary = stage_lower_to_wasm(db, source_file).expect("WASM compilation failed");

        let engine = create_gc_engine();
        let module = wasmtime::Module::new(&engine, wasm_binary.bytes(db)).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance =
            wasmtime::Instance::new(&mut store, &module, &[]).expect("Failed to instantiate");

        // Float compiles to f64 in WASM
        let main = instance
            .get_typed_func::<(), f64>(&mut store, "main")
            .expect("main function not found");

        let result = main.call(&mut store, ()).expect("Execution failed");

        // Verify the result (Float identity should preserve the value)
        assert!(
            (result - 3.125).abs() < 0.0001,
            "Expected main to return 3.125, got {}",
            result
        );
    });
}
