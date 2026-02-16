//! End-to-end tests for compilation and execution with wasmtime CLI.

mod common;

// TODO: Re-enable once print_line is fixed for wasmtime output
#[allow(unused_imports)]
use common::run_wasm;
use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;
use tribute_ir::ModulePathExt as _;
use trunk_ir::DialectOp;

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
        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 42, "Expected main to return 42, got {}", result);
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
fn identity(x: a) ->{} a { x }
fn compute() ->{} Int { identity(42) }
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "int_identity.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test that struct construction works (without accessor).
#[test]
fn test_struct_construction() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
struct Point { x: Int, y: Int }

fn compute() ->{} Int {
    let p = Point { x: 10, y: 20 }
    42
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "struct_construction.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test that struct accessor works.
#[test]
fn test_struct_accessor() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
struct Point { x: Int, y: Int }

fn compute() ->{} Int {
    let p = Point { x: 10, y: 20 }
    p.x()
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "struct_accessor.trb", source_code.clone(), tree);

        let _wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "WASM compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 10, "Expected main to return 10, got {}", result);
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
fn identity(x: a) ->{} a { x }
fn compute() ->{} Float { identity(3.125) }
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_identity.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<f64>(wasm_binary.bytes(db));
        // assert!((result - 3.125).abs() < 0.0001, "Expected main to return 3.125, got {}", result);
    });
}

/// Test that struct can be passed to generic function.
/// Structs should upcast to anyref without additional wrapping.
#[test]
fn test_generic_struct_argument() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
struct Point { x: Int, y: Int }

fn identity(x: a) ->{} a { x }

fn compute() ->{} Int {
    let p = Point { x: 10, y: 20 }
    let p2 = identity(p)
    p2.x()
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "generic_struct.trb", source_code.clone(), tree);

        let _wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "WASM compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 10, "Expected main to return 10, got {}", result);
    });
}

/// Test multiple generic calls with different types in same function.
#[test]
fn test_generic_multiple_types() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn identity(x: a) ->{} a { x }

fn compute() ->{} Int {
    let i = identity(42)
    let f = identity(3.14)
    i
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_multiple.trb", source_code.clone(), tree);

        let _wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "WASM compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test generic function with two type parameters.
#[test]
fn test_generic_two_params() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn first(x: a, y: b) ->{} a { x }

fn compute() ->{} Int {
    first(10, 3.14)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_two_params.trb", source_code.clone(), tree);

        let _wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "WASM compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 10, "Expected main to return 10, got {}", result);
    });
}

/// Test nested generic calls.
#[test]
fn test_generic_nested_calls() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn identity(x: a) ->{} a { x }

fn compute() ->{} Int {
    identity(identity(identity(42)))
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "generic_nested.trb", source_code.clone(), tree);

        let _wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "WASM compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_binary.bytes(db));
        // assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test generic instantiation in indirect function calls.
/// When a closure with generic type is called, the type parameter
/// should be properly instantiated at the call site.
///
/// Note: This test verifies typeck only. Full WASM execution requires
/// closure support in the WASM backend (not yet implemented).
#[test]
fn test_generic_indirect_call() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Int {
    let f = fn(x) { x }
    f(42)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_indirect.trb", source_code.clone(), tree);

        // Run typecheck stage - this should succeed with generic instantiation
        let _module = parse_and_lower_ast(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test function type syntax in parameter annotations.
/// Higher-order function with explicit function type: `fn(Int) -> Int`
///
/// Note: This test verifies typeck only. Full WASM execution requires
/// closure support in the WASM backend (not yet implemented).
#[test]
fn test_function_type_parameter() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn double(n: Int) -> Int {
    n + n
}

fn compute() ->{} Int {
    apply(double, +21)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "function_type.trb", source_code.clone(), tree);

        // Run typecheck stage
        let _module = parse_and_lower_ast(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test nested function types.
/// Function that takes a function returning a function.
#[test]
fn test_nested_function_type() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compose(f: fn(Int) -> Int, g: fn(Int) -> Int, x: Int) -> Int {
    f(g(x))
}

fn inc(n: Int) -> Int { n + +1 }
fn double(n: Int) -> Int { n + n }

fn compute() ->{} Int {
    compose(inc, double, +10)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "nested_function_type.trb", source_code.clone(), tree);

        // Run typecheck stage
        let _module = parse_and_lower_ast(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test generic function type parameters.
/// Function type with type variables: `fn(a) -> b`
#[test]
fn test_generic_function_type() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn apply_generic(f: fn(a) -> b, x: a) ->{} b {
    f(x)
}

fn to_float(n: Int) -> Float {
    3.14
}

fn compute() ->{} Float {
    apply_generic(to_float, +42)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_function_type.trb", source_code.clone(), tree);

        // Run typecheck stage
        let _module = parse_and_lower_ast(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test AST-based calculator with enum, pattern matching, and recursion.
/// This is a milestone test for calc.trb functionality.
///
/// Note: Currently only tests compilation. WASM execution is disabled due to
/// wasmtime invocation issues (see test infrastructure).
#[test]
#[ignore = "WASM backend type inference issue: core.i64 vs wasm.anyref disagreement"]
fn test_calc_eval() {
    use tribute::database::parse_with_thread_local;

    let source = include_str!("../lang-examples/calc.trb");
    let source_code = Rope::from_str(source);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "calc.trb", source_code.clone(), tree);

        let _wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "WASM compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // TODO: Enable WASM execution test once wasmtime invocation is fixed
        // Expected result: (1 + 2) * (10 - 4) / 2 = 9
    });
}

// ============================================================================
// Lambda Lifting Tests (Issue #93)
// ============================================================================

/// Test simple identity lambda (no captures).
#[test]
fn test_lambda_identity() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::run_lambda_lift;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Int {
    let f = fn(x) { x }
    f(42)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "lambda_identity.trb", source_code.clone(), tree);

        // Run lambda lifting stage
        let module = run_lambda_lift(db, source_file);

        // Verify no diagnostics
        let diagnostics: Vec<_> =
            run_lambda_lift::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            diagnostics.len()
        );

        // Verify the module has a lifted function (name starts with __lambda_)
        let body = module.body(db);
        let blocks = body.blocks(db);
        let ops = blocks[0].operations(db);

        let has_lifted = ops.iter().any(|op| {
            if let Ok(f) = trunk_ir::dialect::func::Func::from_operation(db, *op) {
                f.sym_name(db)
                    .last_segment()
                    .with_str(|s: &str| s.starts_with("__lambda_"))
            } else {
                false
            }
        });

        assert!(
            has_lifted,
            "Expected a lifted lambda function in the module"
        );
    });
}

/// Test lambda with capture - verifies closure.new is created.
#[test]
fn test_lambda_with_capture() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::run_lambda_lift;

    let source_code = Rope::from_str(
        r#"
fn test_capture() ->{} Int {
    let a = 10
    let f = fn(x) { x + a }
    f(32)
}

fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "lambda_capture.trb", source_code.clone(), tree);

        // Run lambda lifting stage
        let module = run_lambda_lift(db, source_file);

        // Verify no diagnostics
        let diagnostics: Vec<_> =
            run_lambda_lift::accumulated::<tribute::Diagnostic>(db, source_file);

        assert!(
            diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            diagnostics.len()
        );

        // Check for closure.new in the output
        let body = module.body(db);
        let has_closure_new = check_for_closure_new(db, &body);

        assert!(
            has_closure_new,
            "Expected closure.new operation for captured lambda"
        );
    });
}

/// Helper to recursively check for closure.new in a region
fn check_for_closure_new(db: &dyn salsa::Database, region: &trunk_ir::Region<'_>) -> bool {
    use tribute_ir::dialect::closure;

    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == closure::DIALECT_NAME() && op.name(db) == closure::NEW() {
                return true;
            }
            // Recurse into nested regions
            for nested in op.regions(db).iter() {
                if check_for_closure_new(db, nested) {
                    return true;
                }
            }
        }
    }
    false
}

// ============================================================================
// Indirect Function Call Tests (Issue #94)
// ============================================================================

/// Test that func.call_indirect is generated for function-typed variables.
#[test]
fn test_indirect_call_ir_generation() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::run_lambda_lift;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Int {
    let f = fn(x) { x }
    f(42)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "indirect_call.trb", source_code.clone(), tree);

        let module = run_lambda_lift(db, source_file);

        // Verify no diagnostics
        let diagnostics: Vec<_> =
            run_lambda_lift::accumulated::<tribute::Diagnostic>(db, source_file);

        assert!(
            diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            diagnostics.len()
        );

        // Check for func.call_indirect in main function
        let body = module.body(db);
        let has_call_indirect = check_for_call_indirect(db, &body);

        assert!(
            has_call_indirect,
            "Expected func.call_indirect for indirect function call"
        );
    });
}

/// Test higher-order function with function parameter.
#[test]
fn test_higher_order_function_ir() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::run_lambda_lift;

    let source_code = Rope::from_str(
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn compute() ->{} Int {
    apply(fn(n) { n + +1 }, +41)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "higher_order.trb", source_code.clone(), tree);

        let module = run_lambda_lift(db, source_file);

        // Verify no diagnostics
        let diagnostics: Vec<_> =
            run_lambda_lift::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            diagnostics.len()
        );

        // Check that apply function has func.call_indirect
        let body = module.body(db);
        let has_call_indirect = check_for_call_indirect(db, &body);

        assert!(
            has_call_indirect,
            "Expected func.call_indirect in apply function"
        );
    });
}

/// Helper to recursively check for func.call_indirect in a region
fn check_for_call_indirect(db: &dyn salsa::Database, region: &trunk_ir::Region<'_>) -> bool {
    use trunk_ir::dialect::func;

    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if op.dialect(db) == func::DIALECT_NAME() && op.name(db) == func::CALL_INDIRECT() {
                return true;
            }
            // Recurse into nested regions
            for nested in op.regions(db).iter() {
                if check_for_call_indirect(db, nested) {
                    return true;
                }
            }
        }
    }
    false
}

/// Test that closure operations are properly lowered after closure lowering.
///
/// After lowering:
/// - `closure.new` → `func.constant` + `adt.struct_new`
/// - `closure.func` → `adt.struct_get` (field 0)
/// - `closure.env` → `adt.struct_get` (field 1)
#[test]
fn test_closure_lowering() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::run_closure_lower;

    let source_code = Rope::from_str(
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn compute() ->{} Int {
    apply(fn(n) { n + +1 }, +41)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "closure_lower.trb", source_code.clone(), tree);

        let module = run_closure_lower(db, source_file);

        // Verify no diagnostics
        let diagnostics: Vec<_> =
            run_closure_lower::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            diagnostics.len()
        );

        // Check that closure operations are lowered
        let body = module.body(db);
        let lowered_ops = check_for_lowered_closure_ops(db, &body);

        // Verify closure.func/closure.env are lowered to adt.struct_get
        assert!(
            lowered_ops.has_struct_get,
            "Expected adt.struct_get after closure lowering (from closure.func/closure.env)"
        );
    });
}

/// Tracks what lowered closure operations are present
struct LoweredClosureOps {
    has_func_constant: bool,
    has_struct_new: bool,
    has_struct_get: bool,
}

/// Helper to check for lowered closure operations
fn check_for_lowered_closure_ops(
    db: &dyn salsa::Database,
    region: &trunk_ir::Region<'_>,
) -> LoweredClosureOps {
    use trunk_ir::dialect::{adt, func};

    let mut result = LoweredClosureOps {
        has_func_constant: false,
        has_struct_new: false,
        has_struct_get: false,
    };

    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            let dialect = op.dialect(db);
            let op_name = op.name(db);

            // Check for func.constant (from closure.new lowering)
            if dialect == func::DIALECT_NAME() && op_name == func::CONSTANT() {
                result.has_func_constant = true;
            }
            // Check for adt.struct_new (from closure.new lowering)
            if dialect == adt::DIALECT_NAME() && op_name == adt::STRUCT_NEW() {
                result.has_struct_new = true;
            }
            // Check for adt.struct_get (from closure.func/closure.env lowering)
            if dialect == adt::DIALECT_NAME() && op_name == adt::STRUCT_GET() {
                result.has_struct_get = true;
            }

            // Recurse into nested regions
            for nested in op.regions(db).iter() {
                let nested_result = check_for_lowered_closure_ops(db, nested);
                result.has_func_constant =
                    result.has_func_constant || nested_result.has_func_constant;
                result.has_struct_new = result.has_struct_new || nested_result.has_struct_new;
                result.has_struct_get = result.has_struct_get || nested_result.has_struct_get;
            }
        }
    }

    result
}

// ============================================================================
// Closure Execution Tests (verifies function table based closure implementation)
// ============================================================================

/// Test simple lambda (no capture) compiles correctly.
#[test]
fn test_closure_execution_simple() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Int {
    let f = fn(x) { x + 1 }
    f(41)
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "closure_exec_simple.trb", source_code.clone(), tree);

        use tribute::pipeline::compile_to_wasm_binary;
        let diagnostics: Vec<_> =
            compile_to_wasm_binary::accumulated::<tribute::Diagnostic>(db, source_file);
        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        let wasm_binary = compile_to_wasm_binary(db, source_file)
            .expect("WASM compilation failed (returned None)");

        let _wasm_bytes = wasm_binary.bytes(db);

        // TODO: Re-enable once print_line is fixed for wasmtime output
        // let result = run_wasm::<i32>(wasm_bytes);
        // assert_eq!(result, 42, "Expected f(41) = 42, got {}", result);
    });
}

// =============================================================================
// Type Error Tests
// =============================================================================

/// Test that binary operations with mismatched types produce a type error.
/// Int + Nat should fail because they are different types.
/// Note: +1 is Int (signed), 2 is Nat (unsigned)
#[test]
fn test_binop_type_mismatch_int_nat() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    // +1 is Int (explicit sign), 2 is Nat (no sign)
    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Int {
    +1 + 2
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "binop_type_mismatch.trb", source_code.clone(), tree);

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        assert!(
            !diagnostics.is_empty(),
            "Expected type error for Int + Nat, but got no diagnostics"
        );
    });
}

/// Test that comparison operations with mismatched types produce a type error.
/// Note: +1 is Int, 2.0 is Float
#[test]
fn test_binop_comparison_type_mismatch() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Bool {
    +1 < 2.0
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(
            db,
            "comparison_type_mismatch.trb",
            source_code.clone(),
            tree,
        );

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        assert!(
            !diagnostics.is_empty(),
            "Expected type error for Int < Float, but got no diagnostics"
        );
    });
}

/// Test that boolean operations require Bool operands.
/// Note: +1 and +2 are Int, but && requires Bool
#[test]
fn test_binop_boolean_requires_bool() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Bool {
    +1 && +2
}
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "boolean_requires_bool.trb", source_code.clone(), tree);

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        assert!(
            !diagnostics.is_empty(),
            "Expected type error for Int && Int, but got no diagnostics"
        );
    });
}

/// Test that valid binary operations with matching types succeed.
/// Note: +1 and +2 are both Int
#[test]
fn test_binop_matching_types_succeed() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Int { +1 + +2 }
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "matching_types.trb", source_code.clone(), tree);

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors for Int + Int, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test that Nat + Nat succeeds (both operands are unsigned).
#[test]
fn test_binop_nat_plus_nat_succeeds() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Nat { 1 + 2 }
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "nat_plus_nat.trb", source_code.clone(), tree);

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors for Nat + Nat, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test that Float + Float succeeds.
#[test]
fn test_binop_float_plus_float_succeeds() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Float { 1.5 + 2.5 }
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "float_plus_float.trb", source_code.clone(), tree);

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors for Float + Float, got {} diagnostics",
            diagnostics.len()
        );
    });
}

/// Test that comparison with Bool operands produces a type error.
/// Bool == Bool should work, but Bool < Bool should fail (or at least be questionable)
/// Actually: We just test that True && False works (valid Bool operands)
#[test]
fn test_binop_bool_and_bool_succeeds() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::parse_and_lower_ast;

    let source_code = Rope::from_str(
        r#"
fn compute() ->{} Bool { True && False }
fn main() { }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "bool_and_bool.trb", source_code.clone(), tree);

        let _module = parse_and_lower_ast(db, source_file);

        let diagnostics: Vec<_> =
            parse_and_lower_ast::accumulated::<tribute::Diagnostic>(db, source_file);

        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            diagnostics.is_empty(),
            "Expected no type errors for Bool && Bool, got {} diagnostics",
            diagnostics.len()
        );
    });
}
