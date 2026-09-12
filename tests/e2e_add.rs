//! End-to-end tests for compilation and execution.

mod common;

use common::{assert_native_output, compile_native_or_panic};
use ropey::Rope;
use salsa::Database;
use std::ops::ControlFlow;
use tribute::TributeDatabaseImpl;
use tribute_front::SourceCst;
use tribute_ir::dialect::closure;
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::walk::{WalkAction, walk_region};
use trunk_ir::{Attribute, IrContext, Module, OpRef, ValueDef, ValueRef};

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
    let code = r#"
fn apply_generic(f: fn(a) -> a, x: a) ->{} a { f(x) }
fn compute_int() ->{} Int { apply_generic(fn(x) { x }, +42) }
fn compute_float() ->{} Float { apply_generic(fn(x) { x }, 3.5) }
fn main() { }
"#;
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, "generic_indirect.trb", code);
        let (ctx, module) = tribute::pipeline::run_through_evidence_params(db, source)
            .expect("CPS and lambda lifting").expect("frontend module");
        let mut specializations = Vec::new();
        for (name, scalar) in [("compute_int", "i32"), ("compute_float", "f64")] {
            let caller = named_function(&ctx, module, name);
            let calls = function_ops::<func::TailCall>(&ctx, caller);
            let call = calls.into_iter().find(|call| call.args(&ctx).iter().any(|&arg| {
                matches!(ctx.value_def(arg), ValueDef::OpResult(op, _) if closure::New::matches(&ctx, op))
            })).expect("source call must pass a closure to its specialization");
            let target = named_function(&ctx, module, &call.callee(&ctx).to_string());
            let indirect = only_indirect_call(&ctx, target);
            assert_indirect_signature(&ctx, target, indirect);
            let args = &ctx.op_operands(indirect)[1..];
            assert_eq!(ctx.types.get(ctx.value_ty(*args.last().unwrap())).name, scalar);
            specializations.push(target.op_ref());
        }
        assert_ne!(specializations[0], specializations[1], "Int and Float must instantiate distinct callable contracts");
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

/// Root `main` closes the CPS implementation convention of an open callback
/// worker while preserving the backend's Direct entry ABI.
#[test]
fn test_open_callback_root_main_executes() {
    assert_native_output(
        "open_callback_root_main.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int {
    f(x)
}

fn main() {
    __tribute_print_int(apply(fn(value) { value + +1 }, +41))
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

/// Locate the exact source definition or a definition reached by its symbol reference.
fn named_function(ctx: &IrContext, module: Module, name: &str) -> func::Func {
    let functions: Vec<_> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| {
            func::Func::from_op(ctx, op)
                .ok()
                .filter(|f| f.sym_name(ctx) == name)
        })
        .collect();
    assert_eq!(functions.len(), 1, "expected one physical function {name}");
    functions[0]
}

fn function_ops<T: DialectOp>(ctx: &IrContext, function: func::Func) -> Vec<T> {
    let mut found = Vec::new();
    let _ = walk_region::<()>(ctx, function.body(ctx), &mut |op| {
        if let Ok(typed) = T::from_op(ctx, op) {
            found.push(typed);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    found
}

fn defining_op(ctx: &IrContext, value: ValueRef) -> OpRef {
    let ValueDef::OpResult(op, _) = ctx.value_def(value) else {
        panic!("expected operation result");
    };
    op
}

fn only_indirect_call(ctx: &IrContext, function: func::Func) -> OpRef {
    let mut calls: Vec<_> = function_ops::<func::CallIndirect>(ctx, function)
        .into_iter()
        .map(|op| op.op_ref())
        .collect();
    calls.extend(
        function_ops::<func::TailCallIndirect>(ctx, function)
            .into_iter()
            .map(|op| op.op_ref()),
    );
    assert_eq!(calls.len(), 1, "expected source-linked indirect transfer");
    calls[0]
}

/// Check the complete physical transfer contract, including resultless CPS tails.
fn assert_indirect_signature(ctx: &IrContext, owner: func::Func, call: OpRef) {
    let signature = ctx
        .op(call)
        .attributes
        .get_type("signature")
        .expect("exact indirect signature");
    let signature = func::FuncSig::from_type_ref(ctx, signature).expect("complete signature");
    let operands = ctx.op_operands(call);
    assert_eq!(
        operands[1..]
            .iter()
            .map(|&arg| ctx.value_ty(arg))
            .collect::<Vec<_>>(),
        signature.inputs(ctx)
    );
    if func::TailCallIndirect::matches(ctx, call) {
        assert!(ctx.op_results(call).is_empty());
        let owner_signature = func::FuncSig::from_type_ref(ctx, owner.r#type(ctx)).unwrap();
        assert_eq!(signature.results(ctx), owner_signature.results(ctx));
        assert_eq!(
            tribute_core::get_calling_convention(ctx, call),
            Some(tribute_core::CallingConvention::Cps)
        );
    } else {
        assert_eq!(ctx.op_result_types(call), signature.results(ctx));
    }
}

fn source_closure(ctx: &IrContext, module: Module, name: &str) -> (closure::New, func::Func) {
    let owner = named_function(ctx, module, name);
    let call = only_indirect_call(ctx, owner);
    assert_indirect_signature(ctx, owner, call);
    let closure = closure::New::from_op(ctx, defining_op(ctx, ctx.op_operands(call)[0]))
        .expect("indirect callee must be the source lambda");
    let lifted = named_function(ctx, module, &closure.func_ref(ctx).to_string());
    let closure_ty = ctx.value_ty(closure.result(ctx));
    let callable = closure::Closure::from_type_ref(ctx, closure_ty).expect("typed closure");
    let signature = func::FuncSig::from_type_ref(ctx, callable.func_type(ctx)).unwrap();
    assert_eq!(
        signature.as_type_ref(),
        ctx.op(call).attributes.get_type("signature").unwrap()
    );
    let environment_index =
        tribute_core::calling_convention::get_physical_closure_environment_index(ctx, closure_ty)
            .expect("environment placement");
    let lifted_signature = func::FuncSig::from_type_ref(ctx, lifted.r#type(ctx)).unwrap();
    let entry = ctx.region(lifted.body(ctx)).blocks[0];
    let entry_types: Vec<_> = ctx
        .block_args(entry)
        .iter()
        .map(|&arg| ctx.value_ty(arg))
        .collect();
    assert_eq!(entry_types, lifted_signature.inputs(ctx));
    assert_eq!(
        ctx.types
            .get(lifted_signature.inputs(ctx)[environment_index])
            .dialect,
        "tribute_rt"
    );
    assert_eq!(
        ctx.types
            .get(lifted_signature.inputs(ctx)[environment_index])
            .name,
        "anyref"
    );
    let mut expected = signature.inputs(ctx).to_vec();
    expected.insert(
        environment_index,
        lifted_signature.inputs(ctx)[environment_index],
    );
    assert_eq!(expected, lifted_signature.inputs(ctx));
    assert_eq!(signature.results(ctx), lifted_signature.results(ctx));
    assert_eq!(
        tribute_core::get_calling_convention(ctx, lifted.op_ref()),
        tribute_core::get_physical_closure_convention(ctx, closure_ty)
    );
    (closure, lifted)
}

/// Lambda lifting must preserve the identity result and empty environment.
#[test]
fn test_lambda_identity() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(
            db,
            "lambda_identity.trb",
            "fn compute() ->{} Int { let f = fn(x) { x } f(+42) } fn main() {}",
        );
        let (ctx, module) = tribute::pipeline::run_through_evidence_params(db, source)
            .unwrap()
            .unwrap();
        let (closure, lifted) = source_closure(&ctx, module, "compute");
        assert!(adt::RefNull::matches(
            &ctx,
            defining_op(&ctx, closure.env(&ctx))
        ));
        let call = only_indirect_call(&ctx, named_function(&ctx, module, "compute"));
        let boxed = *ctx.op_operands(call).last().unwrap();
        assert_eq!(ctx.types.get(ctx.value_ty(boxed)).name, "anyref");
        let cast = defining_op(&ctx, boxed);
        assert!(core::UnrealizedConversionCast::matches(&ctx, cast));
        let input = ctx.op_operands(cast);
        assert_eq!(input.len(), 1);
        assert_eq!(ctx.types.get(ctx.value_ty(input[0])).name, "i32");
        let argument = arith::Const::from_op(&ctx, defining_op(&ctx, input[0])).unwrap();
        assert_eq!(argument.value(&ctx), Attribute::Int(42));
        let entry = ctx.region(lifted.body(&ctx)).blocks[0];
        let value = *ctx.block_args(entry).last().unwrap();
        let transfer = only_indirect_call(&ctx, lifted);
        assert_indirect_signature(&ctx, lifted, transfer);
        assert_eq!(
            ctx.op_operands(transfer).last(),
            Some(&value),
            "identity lambda must forward its parameter"
        );
        assert!(function_ops::<closure::Lambda>(&ctx, lifted).is_empty());
    });
}

/// The capture value must be stored, recovered, added, and delivered to done.
#[test]
fn test_lambda_with_capture() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, "lambda_capture.trb", "fn test_capture() ->{} Int { let a = +10 let f = fn(x) { x + a } f(+32) } fn main() {}");
        let (ctx, module) = tribute::pipeline::run_through_evidence_params(db, source).unwrap().unwrap();
        let (closure, lifted) = source_closure(&ctx, module, "test_capture");
        let call = only_indirect_call(&ctx, named_function(&ctx, module, "test_capture"));
        let argument = arith::Const::from_op(&ctx, defining_op(&ctx, *ctx.op_operands(call).last().unwrap())).unwrap();
        assert_eq!(argument.value(&ctx), Attribute::Int(32));
        let environment = adt::StructNew::from_op(&ctx, defining_op(&ctx, closure.env(&ctx))).expect("capture environment");
        assert_eq!(environment.fields(&ctx).len(), 1);
        let capture = arith::Const::from_op(&ctx, defining_op(&ctx, environment.fields(&ctx)[0])).unwrap();
        assert_eq!(capture.value(&ctx), Attribute::Int(10));
        let additions = function_ops::<arith::Addi>(&ctx, lifted);
        assert_eq!(additions.len(), 1);
        let addition = additions[0];
        let recovered = adt::StructGet::from_op(&ctx, defining_op(&ctx, addition.rhs(&ctx))).expect("captured addend");
        assert_eq!(recovered.field(&ctx), 0);
        assert_eq!(recovered.r#type(&ctx), environment.r#type(&ctx));
        let recovery = adt::RefCast::from_op(&ctx, defining_op(&ctx, recovered.r#ref(&ctx))).unwrap();
        let closure_ty = ctx.value_ty(closure.result(&ctx));
        let env_index = tribute_core::calling_convention::get_physical_closure_environment_index(&ctx, closure_ty).unwrap();
        let entry = ctx.region(lifted.body(&ctx)).blocks[0];
        assert_eq!(recovery.r#ref(&ctx), ctx.block_args(entry)[env_index]);
        assert_eq!(addition.lhs(&ctx), *ctx.block_args(entry).last().unwrap());
        let transfer = only_indirect_call(&ctx, lifted);
        assert_indirect_signature(&ctx, lifted, transfer);
        assert_eq!(ctx.op_operands(transfer).last(), Some(&addition.result(&ctx)));
    });
}

/// An ordinary monomorphic callback must lower to a typed indirect transfer.
#[test]
fn test_indirect_call_ir_generation() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(
            db,
            "indirect_call.trb",
            "fn invoke(f: fn(Int) -> Int, x: Int) -> Int { f(x) } fn main() {}",
        );
        let (ctx, module) = tribute::pipeline::run_through_evidence_params(db, source)
            .unwrap()
            .unwrap();
        let owner = named_function(&ctx, module, "invoke");
        let call = only_indirect_call(&ctx, owner);
        assert_indirect_signature(&ctx, owner, call);
        let entry = ctx.region(owner.body(&ctx)).blocks[0];
        assert!(
            ctx.block_args(entry).contains(&ctx.op_operands(call)[0]),
            "callee must be invoke's callback parameter"
        );
        assert_eq!(ctx.op_operands(call).last(), ctx.block_args(entry).last());
    });
}

/// Follow the source call's closure argument into the lifted implementation.
#[test]
fn test_higher_order_function_ir() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, "higher_order.trb", "fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) } fn compute() ->{} Int { apply(fn(n) { n + +1 }, +41) } fn main() {}");
        let (ctx, module) = tribute::pipeline::run_through_evidence_params(db, source).unwrap().unwrap();
        let apply = named_function(&ctx, module, "apply");
        assert_indirect_signature(&ctx, apply, only_indirect_call(&ctx, apply));
        let compute = named_function(&ctx, module, "compute");
        let call = function_ops::<func::TailCall>(&ctx, compute).into_iter().find(|call| call.callee(&ctx) == "apply").expect("source apply call");
        let argument = call.args(&ctx).iter().find_map(|&value| match ctx.value_def(value) { ValueDef::OpResult(op, _) => closure::New::from_op(&ctx, op).ok(), _ => None }).expect("source lambda argument");
        let lifted = named_function(&ctx, module, &argument.func_ref(&ctx).to_string());
        let additions = function_ops::<arith::Addi>(&ctx, lifted);
        assert_eq!(additions.len(), 1);
        let constant = arith::Const::from_op(&ctx, defining_op(&ctx, additions[0].rhs(&ctx))).unwrap();
        assert_eq!(constant.value(&ctx), Attribute::Int(1));
    });
}

/// Closure lowering must project both code and environment from the same callback
/// and insert that environment according to its retained callable metadata.
#[test]
fn test_closure_lowering() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, "closure_lower.trb", "fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) } fn compute() ->{} Int { let a = +1 apply(fn(n) { n + a }, +41) } fn main() {}");
        let (ctx, module) = tribute::pipeline::run_through_closure_lower(db, source).unwrap().unwrap();
        let apply = named_function(&ctx, module, "apply");
        let call = only_indirect_call(&ctx, apply);
        assert_indirect_signature(&ctx, apply, call);
        let code = adt::StructGet::from_op(&ctx, defining_op(&ctx, ctx.op_operands(call)[0])).expect("projected callback code");
        assert_eq!(code.field(&ctx), 0);
        let callback = code.r#ref(&ctx);
        let closure_ty = ctx.value_ty(callback);
        let callable = closure::Closure::from_type_ref(&ctx, closure_ty).expect("retained callback contract");
        let env_index = tribute_core::calling_convention::get_physical_closure_environment_index(&ctx, closure_ty).unwrap();
        let operands = &ctx.op_operands(call)[1..];
        let env = adt::StructGet::from_op(&ctx, defining_op(&ctx, operands[env_index])).expect("projected callback environment");
        assert_eq!(env.field(&ctx), 1);
        assert_eq!(env.r#ref(&ctx), callback);
        let signature = func::FuncSig::from_type_ref(&ctx, callable.func_type(&ctx)).unwrap();
        let mut expected = signature.inputs(&ctx).to_vec();
        expected.insert(env_index, ctx.value_ty(env.result(&ctx)));
        assert_eq!(expected, operands.iter().map(|&value| ctx.value_ty(value)).collect::<Vec<_>>());
        let entry = ctx.region(apply.body(&ctx)).blocks[0];
        assert!(ctx.block_args(entry).contains(&callback));
        assert_eq!(operands.last(), ctx.block_args(entry).last());
        assert!(function_ops::<closure::New>(&ctx, apply).is_empty());
        assert!(function_ops::<closure::Func>(&ctx, apply).is_empty());
        assert!(function_ops::<closure::Env>(&ctx, apply).is_empty());
        let compute = named_function(&ctx, module, "compute");
        let source_call = function_ops::<func::TailCall>(&ctx, compute).into_iter().find(|call| call.callee(&ctx) == "apply").unwrap();
        let pack = source_call.args(&ctx).iter().find_map(|&value| match ctx.value_def(value) {
            ValueDef::OpResult(op, _) if tribute_core::get_closure_callable_type(&ctx, op) == Some(closure_ty) => adt::StructNew::from_op(&ctx, op).ok(), _ => None,
        }).expect("source callback storage with exact callable contract");
        let fields = pack.fields(&ctx);
        let function = func::Constant::from_op(&ctx, defining_op(&ctx, fields[0])).unwrap();
        let lifted = named_function(&ctx, module, &function.func_ref(&ctx).to_string());
        let environment = adt::StructNew::from_op(&ctx, defining_op(&ctx, fields[1])).unwrap();
        assert_eq!(environment.fields(&ctx).len(), 1);
        assert_eq!(arith::Const::from_op(&ctx, defining_op(&ctx, environment.fields(&ctx)[0])).unwrap().value(&ctx), Attribute::Int(1));
        let additions = function_ops::<arith::Addi>(&ctx, lifted);
        assert_eq!(additions.len(), 1);
        let capture = adt::StructGet::from_op(&ctx, defining_op(&ctx, additions[0].rhs(&ctx))).unwrap();
        assert_eq!(capture.r#type(&ctx), environment.r#type(&ctx));
        assert_eq!(capture.field(&ctx), 0);
        let transfer = only_indirect_call(&ctx, lifted);
        assert_indirect_signature(&ctx, lifted, transfer);
        assert_eq!(ctx.op_operands(transfer).last(), Some(&additions[0].result(&ctx)));
    });
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
