//! A generic callback selects the instance of its convention class and keeps
//! the root export contract.

use crate::common;

use salsa_test_macros::salsa_test;
use tribute::pipeline::{compile_frontend, compile_to_wasm_binary};
use tribute_front::SourceCst;
use tribute_ir::dialect::tribute_control;
use trunk_ir::ops::DialectOp;

// Keep the original generic frontend regression on the production path, which
// specializes its callback before lowering source-logical IR.
const SOURCE: &str = r#"
fn apply_open(value: a, callback: fn(a) ->{e} b) ->{e} b {
    callback(value)
}

fn main() ->{std::io::Io} Nil {
    let _ = apply_open(+41, fn(value) { value })
    Nil
}
"#;

#[salsa_test]
fn generic_callback_selects_a_direct_instance_and_executes_wasm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "test.trb", SOURCE);
    let (ctx, module) = compile_frontend(db, source).expect("production frontend should lower");
    let convention = |named: &dyn Fn(&str) -> bool| {
        module.ops(&ctx).iter().copied().find_map(|op| {
            let function = tribute_control::Func::from_op(&ctx, op).ok()?;
            named(function.sym_name(&ctx))
                .then(|| tribute_control::func_sig_convention(&ctx, function.r#type(&ctx)))
        })
    };
    assert_eq!(
        convention(&|name| name == "main"),
        Some(Some(tribute_control::CallingConvention::EvidenceDirect)),
        "a pure callback leaves the Io root as it is"
    );
    assert_eq!(
        convention(&|name| name.starts_with("apply_open$") && name.ends_with("$9D")),
        Some(Some(tribute_control::CallingConvention::Direct)),
        "a pure callback selects the Direct instance"
    );

    let binary = compile_to_wasm_binary(db, source).expect("generic root should compile to Wasm");
    let output = common::run_wasm(binary);
    assert!(
        output.status.success(),
        "Wasm root execution failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
}

/// A pure callback in a pure function selects the `Direct` instance of a
/// prelude function.
#[salsa_test]
fn a_pure_prelude_callback_selects_the_direct_instance(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pure() ->{} Option(Int) { Option::map(Some(+1), fn(n) n + +1) }

fn main() -> Nil {
    let _ = pure()
    Nil
}
"#,
    );
    let (ctx, module) = compile_frontend(db, source).expect("production frontend should lower");
    let instances: Vec<_> = module
        .ops(&ctx)
        .iter()
        .copied()
        .filter_map(|op| {
            let function = tribute_control::Func::from_op(&ctx, op).ok()?;
            let name = function.sym_name(&ctx);
            let rest = name.strip_prefix("std::Option::map$")?;
            let class = rest.rsplit_once("$9").map_or("", |(_, class)| class);
            let convention = tribute_control::func_sig_convention(&ctx, function.r#type(&ctx));
            Some((class.to_owned(), convention))
        })
        .collect();
    assert_eq!(
        instances,
        [(
            "D".to_owned(),
            Some(tribute_control::CallingConvention::Direct)
        )]
    );
}

/// Each of several row-polymorphic calls in one body selects the instance
/// for its own callback, and the `Io` root stays as it is.
#[salsa_test]
fn calls_in_one_body_select_instances_independently(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    let _ = Option::map(Some(+1), fn(n) n + +1)
    let _ = Option::map(Some("io"), fn(s) print_line(s))
    Nil
}
"#,
    );
    let (ctx, module) = compile_frontend(db, source).expect("production frontend should lower");
    let mut functions: Vec<_> = module
        .ops(&ctx)
        .iter()
        .copied()
        .filter_map(|op| {
            let function = tribute_control::Func::from_op(&ctx, op).ok()?;
            let name = function.sym_name(&ctx);
            let name = if name == "main" {
                name
            } else {
                let rest = name.strip_prefix("std::Option::map$")?;
                rest.rsplit_once("$9").map_or("", |(_, class)| class)
            };
            let convention = tribute_control::func_sig_convention(&ctx, function.r#type(&ctx));
            Some((name.to_owned(), convention))
        })
        .collect();
    functions.sort();
    use tribute_control::CallingConvention::{Direct, EvidenceDirect};
    assert_eq!(
        functions,
        [
            ("D".to_owned(), Some(Direct)),
            ("E".to_owned(), Some(EvidenceDirect)),
            ("main".to_owned(), Some(EvidenceDirect)),
        ]
    );
}

/// Root `main` is the instance of its own row with an empty tail, so a call
/// it makes with a pure callback selects the `Direct` instance even when its
/// effect annotation is omitted.
#[salsa_test]
fn root_main_selects_instances_for_an_empty_tail(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn apply(f: fn(Nat) ->{e} Nat, value: Nat) ->{e} Nat { f(value) }

fn main() -> Nil {
    let _ = apply(fn(x) { x + 1 }, 41)
    Nil
}
"#,
    );
    let (ctx, module) = compile_frontend(db, source).expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let main = ir
        .split("tribute_control.func @main(")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("root main");
    assert!(
        main.starts_with(") -> core.nil convention(direct)"),
        "{main}"
    );
    assert!(main.contains("callee = @\"apply$9D\""), "{main}");
    assert!(
        main.contains("tribute_control.lambda") && !main.contains("convention(cps)"),
        "{main}"
    );
}

/// A let-bound lambda used at two classes is emitted once for each, and one
/// used at an empty row is `Direct`.
#[salsa_test]
fn let_bound_lambdas_are_emitted_per_class_in_use(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
use std::io::{Io, print_line}

fn app(x: Int, f: fn(Int) ->{e} Int) ->{e} Int { f(x) }

fn each(n: Int) -> Int {
    let step = fn(m: Int) -> Int { m + +1 }
    step(n)
}

fn main() ->{Io} Nil {
    let both = fn(x: Int, f: fn(Int) ->{e} Int) app(x, f)
    let _ = both(+1, fn(n) n + +1)
    let _ = both(+2, fn(n) { print_line("io") n })
    let _ = each(+3)
    Nil
}
"#,
    );
    let (ctx, module) = compile_frontend(db, source).expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let main = ir
        .split("tribute_control.func @main(")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("root main");
    // `both` is used with a pure and with an `Io` callback.
    assert!(main.contains("callee = @\"app$9D\""), "{main}");
    assert!(main.contains("callee = @\"app$9E\""), "{main}");
    assert!(!main.contains("callee = @app}"), "{main}");
    // `step` is used at an empty row, so it and `each` need no Cps control.
    let each = ir
        .split("tribute_control.func @each(")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("each");
    assert!(
        each.starts_with("%0: core.i32) -> core.i32 convention(direct)"),
        "{each}"
    );
    assert!(
        each.contains("tribute_control.lambda") && !each.contains("convention(cps)"),
        "{each}"
    );
}

#[test]
fn generic_callback_evidence_root_executes_native() {
    let output = common::compile_and_run_native("test.trb", SOURCE);
    assert!(
        output.status.success(),
        "native root execution failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
}
