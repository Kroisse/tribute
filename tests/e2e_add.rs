//! End-to-end test for add.trb compilation and execution with wasmtime.

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::stage_lower_to_wasm;
use tribute_front::SourceCst;

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
        let wasm_binary = stage_lower_to_wasm(db, source_file)
            .expect("WASM compilation failed");

        // Execute with wasmtime (no WASI needed for add.trb)
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, wasm_binary.bytes(db)).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[]).expect("Failed to instantiate");

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
