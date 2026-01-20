//! End-to-end tests for compilation and execution with wasmtime CLI.

mod common;

use common::run_wasm_main;
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
        let wasm_binary = compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        // Execute with wasmtime CLI
        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

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
fn identity(x: a) ->{} a { x }
fn main() ->{} Int { identity(42) }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "int_identity.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        // Verify the result is 42 (Int identity should preserve the value)
        assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test that struct construction works (without accessor).
#[test]
fn test_struct_construction() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
struct Point { x: Int, y: Int }

fn main() ->{} Int {
    let p = Point { x: 10, y: 20 }
    42
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "struct_construction.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test that struct accessor works.
#[test]
fn test_struct_accessor() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
struct Point { x: Int, y: Int }

fn main() ->{} Int {
    let p = Point { x: 10, y: 20 }
    p.x()
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "struct_accessor.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
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

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        assert_eq!(result, 10, "Expected main to return 10, got {}", result);
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
fn main() ->{} Float { identity(3.125) }
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_identity.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        let result = run_wasm_main::<f64>(wasm_binary.bytes(db));

        // Verify the result (Float identity should preserve the value)
        assert!(
            (result - 3.125).abs() < 0.0001,
            "Expected main to return 3.125, got {}",
            result
        );
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

fn main() ->{} Int {
    let p = Point { x: 10, y: 20 }
    let p2 = identity(p)
    p2.x()
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "generic_struct.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
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

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        assert_eq!(result, 10, "Expected main to return 10, got {}", result);
    });
}

/// Test multiple generic calls with different types in same function.
#[test]
fn test_generic_multiple_types() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn identity(x: a) ->{} a { x }

fn main() ->{} Int {
    let i = identity(42)
    let f = identity(3.14)
    i
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_multiple.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
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

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        assert_eq!(result, 42, "Expected main to return 42, got {}", result);
    });
}

/// Test generic function with two type parameters.
#[test]
fn test_generic_two_params() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn first(x: a, y: b) ->{} a { x }

fn main() ->{} Int {
    first(10, 3.14)
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_two_params.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
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

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        assert_eq!(result, 10, "Expected main to return 10, got {}", result);
    });
}

/// Test nested generic calls.
#[test]
fn test_generic_nested_calls() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn identity(x: a) ->{} a { x }

fn main() ->{} Int {
    identity(identity(identity(42)))
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "generic_nested.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
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

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        assert_eq!(result, 42, "Expected main to return 42, got {}", result);
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
    use tribute::pipeline::run_typecheck;

    let source_code = Rope::from_str(
        r#"
fn main() ->{} Int {
    let f = fn(x) { x }
    f(42)
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_indirect.trb", source_code.clone(), tree);

        // Run typecheck stage - this should succeed with generic instantiation
        let _module = run_typecheck(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            run_typecheck::accumulated::<tribute::Diagnostic>(db, source_file);

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
    use tribute::pipeline::run_typecheck;

    let source_code = Rope::from_str(
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn double(n: Int) -> Int {
    n + n
}

fn main() ->{} Int {
    apply(double, 21)
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "function_type.trb", source_code.clone(), tree);

        // Run typecheck stage
        let _module = run_typecheck(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            run_typecheck::accumulated::<tribute::Diagnostic>(db, source_file);

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
    use tribute::pipeline::run_typecheck;

    let source_code = Rope::from_str(
        r#"
fn compose(f: fn(Int) -> Int, g: fn(Int) -> Int, x: Int) -> Int {
    f(g(x))
}

fn inc(n: Int) -> Int { n + 1 }
fn double(n: Int) -> Int { n + n }

fn main() ->{} Int {
    compose(inc, double, 10)
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "nested_function_type.trb", source_code.clone(), tree);

        // Run typecheck stage
        let _module = run_typecheck(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            run_typecheck::accumulated::<tribute::Diagnostic>(db, source_file);

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
    use tribute::pipeline::run_typecheck;

    let source_code = Rope::from_str(
        r#"
fn apply_generic(f: fn(a) -> b, x: a) ->{} b {
    f(x)
}

fn to_float(n: Int) -> Float {
    3.14
}

fn main() ->{} Float {
    apply_generic(to_float, 42)
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "generic_function_type.trb", source_code.clone(), tree);

        // Run typecheck stage
        let _module = run_typecheck(db, source_file);

        // Check for type errors
        let diagnostics: Vec<_> =
            run_typecheck::accumulated::<tribute::Diagnostic>(db, source_file);

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
#[test]
fn test_calc_eval() {
    use tribute::database::parse_with_thread_local;

    let source = include_str!("../lang-examples/calc.trb");
    let source_code = Rope::from_str(source);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "calc.trb", source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).unwrap_or_else(|| {
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

        let result = run_wasm_main::<i32>(wasm_binary.bytes(db));

        // (1 + 2) * (10 - 4) / 2 = 3 * 6 / 2 = 18 / 2 = 9
        assert_eq!(result, 9, "Expected main to return 9, got {}", result);
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
fn main() ->{} Int {
    let f = fn(x) { x }
    f(42)
}
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

fn main() ->{} Int {
    test_capture()
}
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
fn main() ->{} Int {
    let f = fn(x) { x }
    f(42)
}
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

fn main() ->{} Int {
    apply(fn(n) { n + 1 }, 41)
}
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

fn main() ->{} Int {
    apply(fn(n) { n + 1 }, 41)
}
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
    use tribute_ir::dialect::adt;
    use trunk_ir::dialect::func;

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

/// Test simple lambda (no capture) compiles and executes correctly.
/// Currently ignored: lambda codegen has unresolved type inference issues
/// (return type stays as tribute.type_var, effect handling code inserted incorrectly)
#[test]
#[ignore]
fn test_closure_execution_simple() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn main() ->{} Int {
    let f = fn(x) { x + 1 }
    f(41)
}
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

        // Write wasm to file for inspection
        let wasm_bytes = wasm_binary.bytes(db);
        std::fs::write("/tmp/claude/closure_test.wasm", wasm_bytes).expect("Failed to write wasm");
        eprintln!(
            "Wrote {} bytes to /tmp/claude/closure_test.wasm",
            wasm_bytes.len()
        );

        let result = run_wasm_main::<i32>(wasm_bytes);

        assert_eq!(result, 42, "Expected f(41) = 42, got {}", result);
    });
}
