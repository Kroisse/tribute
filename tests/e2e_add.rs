//! End-to-end tests for compilation and execution.

mod common;

use common::{assert_native_output, compile_native_or_panic};
use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute_front::SourceCst;
use tribute_ir::ModulePathExt as _;
use trunk_ir::Symbol;
use trunk_ir::{IrContext, Module, RegionRef};

#[test]
fn test_add_compiles_and_runs() {
    assert_native_output(
        "add.trb",
        r#"
fn add(x: Nat, y: Nat) -> Nat { x + y }

fn main() {
    __tribute_print_nat(add(40, 2))
}
"#,
        "42",
    );
}

/// Test that Int boxing/unboxing works correctly in polymorphic contexts.
/// This verifies PR #61 (uniform representation for generics).
#[test]
fn test_generic_int_identity() {
    assert_native_output(
        "int_identity.trb",
        r#"
fn identity(x: a) ->{} a { x }

fn main() {
    __tribute_print_nat(identity(42))
}
"#,
        "42",
    );
}

/// Test that struct construction works (without accessor).
#[test]
fn test_struct_construction() {
    assert_native_output(
        "struct_construction.trb",
        r#"
struct Point { x: Nat, y: Nat }

fn main() {
    let p = Point { x: 10, y: 20 }
    __tribute_print_nat(42)
}
"#,
        "42",
    );
}

/// Test that struct accessor works.
#[test]
fn test_struct_accessor() {
    assert_native_output(
        "struct_accessor.trb",
        r#"
struct Point { x: Nat, y: Nat }

fn main() {
    let p = Point { x: 10, y: 20 }
    __tribute_print_nat(p.x())
}
"#,
        "10",
    );
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

        let _native_binary = compile_native_or_panic(db, source_file);
    });
}

/// Test that struct can be passed to generic function.
/// Structs should upcast to anyref without additional wrapping.
#[test]
fn test_generic_struct_argument() {
    assert_native_output(
        "generic_struct.trb",
        r#"
struct Point { x: Nat, y: Nat }

fn identity(x: a) ->{} a { x }

fn main() {
    let p = Point { x: 10, y: 20 }
    let p2 = identity(p)
    __tribute_print_nat(p2.x())
}
"#,
        "10",
    );
}

/// Test multiple generic calls with different types in same function.
#[test]
fn test_generic_multiple_types() {
    assert_native_output(
        "generic_multiple.trb",
        r#"
fn identity(x: a) ->{} a { x }

fn main() {
    let i = identity(42)
    let _ = identity(3.14)
    __tribute_print_nat(i)
}
"#,
        "42",
    );
}

/// Test generic function with two type parameters.
#[test]
fn test_generic_two_params() {
    assert_native_output(
        "generic_two_params.trb",
        r#"
fn first(x: a, y: b) ->{} a { x }

fn main() {
    __tribute_print_nat(first(10, 3.14))
}
"#,
        "10",
    );
}

/// Test nested generic calls.
#[test]
fn test_generic_nested_calls() {
    assert_native_output(
        "generic_nested.trb",
        r#"
fn identity(x: a) ->{} a { x }

fn main() {
    __tribute_print_nat(identity(identity(identity(42))))
}
"#,
        "42",
    );
}

/// Test generic instantiation in indirect function calls.
/// When a closure with generic type is called, the type parameter
/// should be properly instantiated at the call site.
#[test]
fn test_generic_indirect_call() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::{compile_with_diagnostics, run_through_evidence_params};

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

        // Collect diagnostics
        let result = compile_with_diagnostics(db, source_file);

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );

        // Run through evidence params to get arena IR
        let (ctx, m) = run_through_evidence_params(db, source_file)
            .expect("run_through_evidence_params should succeed");

        // Check for func.call_indirect in the module
        let has_call_indirect = check_for_call_indirect_in_module(&ctx, m);

        assert!(
            has_call_indirect,
            "Expected func.call_indirect for indirect function call"
        );
    });
}

/// Test function type syntax in parameter annotations.
/// Higher-order function with explicit function type: `fn(Int) -> Int`
#[test]
fn test_function_type_parameter() {
    assert_native_output(
        "function_type.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn double(n: Int) -> Int {
    n + n
}

fn main() {
    __tribute_print_int(apply(double, +21))
}
"#,
        "42",
    );
}

/// Test nested function types.
/// Function that takes a function returning a function.
#[test]
fn test_nested_function_type() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::compile_with_diagnostics;

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

        let result = compile_with_diagnostics(db, source_file);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no type errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test generic function type parameters.
/// Function type with type variables: `fn(a) -> b`
#[test]
fn test_generic_function_type() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::compile_with_diagnostics;

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

        let result = compile_with_diagnostics(db, source_file);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no type errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test AST-based calculator with enum, pattern matching, and recursion.
/// This is a milestone test for calc.trb functionality.
#[test]
fn test_calc_eval() {
    assert_native_output(
        "calc.trb",
        r#"
enum Expr {
    Num(Int),
    Add(Expr, Expr),
    Sub(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
}

fn eval(e: Expr) -> Int {
    case e {
        Num(n) -> n,
        Add(l, r) -> eval(l) + eval(r),
        Sub(l, r) -> eval(l) - eval(r),
        Mul(l, r) -> eval(l) * eval(r),
        Div(l, r) -> eval(l) / eval(r),
    }
}

fn main() {
    let expr = Div(
        Mul(
            Add(Num(+1), Num(+2)),
            Sub(Num(+10), Num(+4))
        ),
        Num(+2)
    )
    __tribute_print_int(eval(expr))
}
"#,
        "9",
    );
}

// ============================================================================
// Lambda Lifting Tests (Issue #93)
// ============================================================================

/// Test simple identity lambda (no captures).
#[test]
fn test_lambda_identity() {
    use tribute::database::parse_with_thread_local;
    use tribute::pipeline::{compile_with_diagnostics, run_through_evidence_params};

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

        // Collect diagnostics via compile_with_diagnostics
        let result = compile_with_diagnostics(db, source_file);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );

        // Run through evidence params to get arena IR for structural checks
        let (ctx, m) = run_through_evidence_params(db, source_file)
            .expect("run_through_evidence_params should succeed");

        // Verify the module has a lifted function (name starts with __lambda_)
        let func_dialect = Symbol::new("func");
        let func_name = Symbol::new("func");

        let has_lifted = m.ops(&ctx).iter().any(|&op_ref| {
            let op_data = ctx.op(op_ref);
            if op_data.dialect == func_dialect && op_data.name == func_name {
                if let Some(trunk_ir::Attribute::Symbol(name)) =
                    op_data.attributes.get(&Symbol::new("sym_name"))
                {
                    name.last_segment()
                        .with_str(|s: &str| s.starts_with("__clam_"))
                } else {
                    false
                }
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
    use tribute::pipeline::{compile_with_diagnostics, run_through_evidence_params};

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

        // Collect diagnostics
        let result = compile_with_diagnostics(db, source_file);

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );

        // Run through evidence params to get arena IR
        let (ctx, m) = run_through_evidence_params(db, source_file)
            .expect("run_through_evidence_params should succeed");

        // Check for closure.new in the output
        let has_closure_new = check_for_closure_new_in_module(&ctx, m);

        assert!(
            has_closure_new,
            "Expected closure.new operation for captured lambda"
        );
    });
}

/// Helper to check for closure.new in a module (arena version)
fn check_for_closure_new_in_module(ctx: &IrContext, m: Module) -> bool {
    for &op_ref in &m.ops(ctx) {
        let op_data = ctx.op(op_ref);
        for &region_ref in &op_data.regions {
            if check_for_closure_new_in_region(ctx, region_ref) {
                return true;
            }
        }
    }
    false
}

/// Helper to recursively check for closure.new in a region (arena version)
fn check_for_closure_new_in_region(ctx: &IrContext, region_ref: RegionRef) -> bool {
    let closure_dialect = Symbol::new("closure");
    let closure_new_name = Symbol::new("new");

    let region = ctx.region(region_ref);
    for &block_ref in &region.blocks {
        let block = ctx.block(block_ref);
        for &op_ref in &block.ops {
            let op_data = ctx.op(op_ref);
            if op_data.dialect == closure_dialect && op_data.name == closure_new_name {
                return true;
            }
            // Recurse into nested regions
            for &nested_region in &op_data.regions {
                if check_for_closure_new_in_region(ctx, nested_region) {
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
    use tribute::pipeline::{compile_with_diagnostics, run_through_evidence_params};

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

        // Collect diagnostics
        let result = compile_with_diagnostics(db, source_file);

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );

        // Run through evidence params to get arena IR
        let (ctx, m) = run_through_evidence_params(db, source_file)
            .expect("run_through_evidence_params should succeed");

        // Check for func.call_indirect in the module
        let has_call_indirect = check_for_call_indirect_in_module(&ctx, m);

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
    use tribute::pipeline::{compile_with_diagnostics, run_through_evidence_params};

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

        // Collect diagnostics
        let result = compile_with_diagnostics(db, source_file);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );

        // Run through evidence params to get arena IR
        let (ctx, m) = run_through_evidence_params(db, source_file)
            .expect("run_through_evidence_params should succeed");

        // Check that apply function has func.call_indirect
        let has_call_indirect = check_for_call_indirect_in_module(&ctx, m);

        assert!(
            has_call_indirect,
            "Expected func.call_indirect in apply function"
        );
    });
}

/// Helper to check for func.call_indirect in a module (arena version)
fn check_for_call_indirect_in_module(ctx: &IrContext, m: Module) -> bool {
    for &op_ref in &m.ops(ctx) {
        let op_data = ctx.op(op_ref);
        for &region_ref in &op_data.regions {
            if check_for_call_indirect_in_region(ctx, region_ref) {
                return true;
            }
        }
    }
    false
}

/// Helper to recursively check for func.call_indirect in a region (arena version)
fn check_for_call_indirect_in_region(ctx: &IrContext, region_ref: RegionRef) -> bool {
    let func_dialect = Symbol::new("func");
    let call_indirect_name = Symbol::new("call_indirect");

    let region = ctx.region(region_ref);
    for &block_ref in &region.blocks {
        let block = ctx.block(block_ref);
        for &op_ref in &block.ops {
            let op_data = ctx.op(op_ref);
            if op_data.dialect == func_dialect && op_data.name == call_indirect_name {
                return true;
            }
            // Recurse into nested regions
            for &nested_region in &op_data.regions {
                if check_for_call_indirect_in_region(ctx, nested_region) {
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
    use tribute::pipeline::{compile_with_diagnostics, run_through_closure_lower};

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

        // Collect diagnostics
        let result = compile_with_diagnostics(db, source_file);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );

        // Run through closure lower to get arena IR
        let (ctx, m) = run_through_closure_lower(db, source_file)
            .expect("run_through_closure_lower should succeed");

        // Check that closure operations are lowered
        let lowered_ops = check_for_lowered_closure_ops_in_module(&ctx, m);

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

/// Helper to check for lowered closure operations in a module (arena version)
fn check_for_lowered_closure_ops_in_module(ctx: &IrContext, m: Module) -> LoweredClosureOps {
    let mut result = LoweredClosureOps {
        has_func_constant: false,
        has_struct_new: false,
        has_struct_get: false,
    };

    for &op_ref in &m.ops(ctx) {
        let op_data = ctx.op(op_ref);
        for &region_ref in &op_data.regions {
            let nested_result = check_for_lowered_closure_ops_in_region(ctx, region_ref);
            result.has_func_constant = result.has_func_constant || nested_result.has_func_constant;
            result.has_struct_new = result.has_struct_new || nested_result.has_struct_new;
            result.has_struct_get = result.has_struct_get || nested_result.has_struct_get;
        }
    }

    result
}

/// Helper to check for lowered closure operations in a region (arena version)
fn check_for_lowered_closure_ops_in_region(
    ctx: &IrContext,
    region_ref: RegionRef,
) -> LoweredClosureOps {
    let func_dialect = Symbol::new("func");
    let func_constant_name = Symbol::new("constant");
    let adt_dialect = Symbol::new("adt");
    let adt_struct_new_name = Symbol::new("struct_new");
    let adt_struct_get_name = Symbol::new("struct_get");

    let mut result = LoweredClosureOps {
        has_func_constant: false,
        has_struct_new: false,
        has_struct_get: false,
    };

    let region = ctx.region(region_ref);
    for &block_ref in &region.blocks {
        let block = ctx.block(block_ref);
        for &op_ref in &block.ops {
            let op_data = ctx.op(op_ref);
            let dialect = op_data.dialect;
            let op_name = op_data.name;

            // Check for func.constant (from closure.new lowering)
            if dialect == func_dialect && op_name == func_constant_name {
                result.has_func_constant = true;
            }
            // Check for adt.struct_new (from closure.new lowering)
            if dialect == adt_dialect && op_name == adt_struct_new_name {
                result.has_struct_new = true;
            }
            // Check for adt.struct_get (from closure.func/closure.env lowering)
            if dialect == adt_dialect && op_name == adt_struct_get_name {
                result.has_struct_get = true;
            }

            // Recurse into nested regions
            for &nested_region in &op_data.regions {
                let nested_result = check_for_lowered_closure_ops_in_region(ctx, nested_region);
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
#[test]
fn test_closure_execution_simple() {
    assert_native_output(
        "closure_exec_simple.trb",
        r#"
fn main() {
    let f = fn(x) { x + 1 }
    __tribute_print_nat(f(41))
}
"#,
        "42",
    );
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
