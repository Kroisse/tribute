//! A generic callback selects the instance of its convention class and keeps
//! the root export contract.

use crate::common;

use salsa_test_macros::salsa_test;
use tribute::pipeline::{compile_specialized_frontend, compile_to_wasm_binary};
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
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let convention = |named: &dyn Fn(&str) -> bool| {
        module.ops(&ctx).iter().copied().find_map(|op| {
            let function = tribute_control::Func::from_op(&ctx, op).ok()?;
            named(function.sym_name(&ctx))
                .then(|| tribute_control::func_sig_convention(&ctx, function.r#type(&ctx)))
        })
    };
    assert_eq!(
        convention(&|name| name == "pkg::main"),
        Some(Some(
            tribute_ir::calling_convention::CallingConvention::EvidenceDirect
        )),
        "a pure callback leaves the Io root as it is"
    );
    assert_eq!(
        convention(&|name| name.starts_with("pkg::apply_open$") && name.ends_with("$9D")),
        Some(Some(
            tribute_ir::calling_convention::CallingConvention::Direct
        )),
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
fn pure() ->{} Option(Int) { Option::map(.Some(+1), fn(n) n + +1) }

fn main() -> Nil {
    let _ = pure()
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
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
            Some(tribute_ir::calling_convention::CallingConvention::Direct)
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
    let _ = Option::map(.Some(+1), fn(n) n + +1)
    let _ = Option::map(.Some("io"), fn(s) print_line(s))
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let mut functions: Vec<_> = module
        .ops(&ctx)
        .iter()
        .copied()
        .filter_map(|op| {
            let function = tribute_control::Func::from_op(&ctx, op).ok()?;
            let name = function.sym_name(&ctx);
            let name = if name == "pkg::main" {
                "main"
            } else {
                let rest = name.strip_prefix("std::Option::map$")?;
                rest.rsplit_once("$9").map_or("", |(_, class)| class)
            };
            let convention = tribute_control::func_sig_convention(&ctx, function.r#type(&ctx));
            Some((name.to_owned(), convention))
        })
        .collect();
    functions.sort();
    use tribute_ir::calling_convention::CallingConvention::{Direct, EvidenceDirect};
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
/// ability annotation is omitted.
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
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let main = ir
        .split("tribute_control.func @\"pkg::main\"")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("root main");
    assert!(
        main.starts_with("['a0]() -> core.nil row(| 'a0) inferred convention(direct)"),
        "{main}"
    );
    assert!(main.contains("callee = @\"pkg::apply$9D\""), "{main}");
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
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let main = ir
        .split("tribute_control.func @\"pkg::main\"")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("root main");
    // `both` is used with a pure and with an `Io` callback.
    assert!(main.contains("callee = @\"pkg::app$9D\""), "{main}");
    assert!(main.contains("callee = @\"pkg::app$9E\""), "{main}");
    assert!(!main.contains("callee = @\"pkg::app\"}"), "{main}");
    // `step` is used at an empty row, so it and `each` need no Cps control.
    let each = ir
        .split("tribute_control.func @\"pkg::each\"")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("each");
    assert!(
        each.starts_with("['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct)"),
        "{each}"
    );
    assert!(
        each.contains("tribute_control.lambda") && !each.contains("convention(cps)"),
        "{each}"
    );
}

/// A row a definition passes only to a lambda it binds is a class variable
/// of the definition: a caller at an empty row selects the `Direct` instance,
/// whose lambda is `Direct` too.
#[salsa_test]
fn a_row_passed_to_a_let_bound_lambda_is_a_class_variable(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn countdown(n: Int) -> Int {
    let step = fn(m: Int) -> Int {
        case m == +0 {
            True -> m
            False -> become countdown(m - +1)
        }
    }
    become step(n)
}

fn main() -> Nil {
    let _ = countdown(+3)
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    assert!(ir.contains("callee = @\"pkg::countdown$9D\""), "{ir}");
    let instance = ir
        .split("tribute_control.func @\"pkg::countdown$9D\"")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("the Direct instance");
    assert!(
        instance.starts_with(
            "['a0, local 'a1](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct)"
        ),
        "{instance}"
    );
    assert!(
        instance.contains("tribute_control.lambda") && !instance.contains("convention(cps)"),
        "{instance}"
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

/// A function inside a module is the instance its let-bound lambdas and
/// references are inside, as a root function is.
#[salsa_test]
fn let_bound_lambdas_in_a_module_are_emitted_per_class_in_use(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
use std::io::Io

mod m {
    use std::io::{Io, print_line}

    pub fn app(x: Int, f: fn(Int) ->{e} Int) ->{e} Int { f(x) }

    pub fn each(n: Int) -> Int {
        let step = fn(m: Int) -> Int { m + +1 }
        step(n)
    }

    pub fn run() ->{Io} Nil {
        let both = fn(x: Int, f: fn(Int) ->{e} Int) app(x, f)
        let _ = both(+1, fn(n) n + +1)
        let _ = both(+2, fn(n) { print_line("io") n })
        let _ = each(+3)
        Nil
    }
}

fn main() ->{Io} Nil {
    m::run()
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let function = |name: &str| {
        ir.split(&format!("tribute_control.func @\"{name}\""))
            .nth(1)
            .and_then(|rest| rest.split("\n  }\n").next())
            .unwrap_or_else(|| panic!("{name} in\n{ir}"))
    };
    let run = function("pkg::m::run");
    assert!(run.contains("callee = @\"pkg::m::app$9D\""), "{run}");
    assert!(run.contains("callee = @\"pkg::m::app$9E\""), "{run}");
    assert!(!run.contains("callee = @\"pkg::m::app\"}"), "{run}");
    let each = function("pkg::m::each");
    assert!(
        each.starts_with("['a0](%0: core.i32) -> core.i32 row(| 'a0) inferred convention(direct)"),
        "{each}"
    );
    assert!(
        each.contains("tribute_control.lambda") && !each.contains("convention(cps)"),
        "{each}"
    );
}

/// A let-bound lambda quantifies the rows its type spells, and each use
/// instantiates it with the rows it passes, so an argument is lowered at the
/// instantiated type.
#[salsa_test]
fn uses_of_let_bound_lambdas_instantiate_them(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
use std::io::{Io, print_line}

fn app(x: Int, f: fn(Int) ->{e} Int) ->{e} Int { f(x) }

fn countdown(n: Int) -> Int {
    let step = fn(m: Int) -> Int {
        case m == +0 {
            True -> m
            False -> become countdown(m - +1)
        }
    }
    become step(n)
}

fn main() ->{Io} Nil {
    let both = fn(x: Int, f: fn(Int) ->{e} Int) app(x, f)
    let _ = both(+1, fn(n) n + +1)
    let _ = both(+2, fn(n) { print_line("io") n })
    let _ = countdown(+3)
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let function = |name: &str| {
        ir.split(&format!("tribute_control.func {name}["))
            .nth(1)
            .and_then(|rest| rest.split("\n  }\n").next())
            .unwrap_or_else(|| panic!("{name} in\n{ir}"))
    };
    let countdown = function("@\"pkg::countdown$9D\"");
    assert!(countdown.starts_with("'a0, local 'a1]"), "{countdown}");
    assert!(
        countdown.contains("tribute_control.lambda['a1](")
            && countdown.contains("tribute_control.instantiate %1 {row_args = [{tail = 0}]}"),
        "{countdown}"
    );
    let main = function("@\"pkg::main\"");
    assert!(main.starts_with("local 'a0, local 'a1]"), "{main}");
    assert!(
        main.contains("%0 = tribute_control.lambda['a0](")
            && main.contains("%4 = tribute_control.lambda['a1](")
            && main.contains("tribute_control.instantiate %0 {row_args = [{}]}")
            && main.contains(
                "tribute_control.instantiate %4 {row_args = [{instances = [@\"ability_1_11_std::io::Io\"]}]}"
            ),
        "{main}"
    );
    // The pure callback is lowered at the instantiated, closed row.
    assert!(
        main.contains("tribute_control.lambda(%11: core.i32) -> core.i32 convention(direct)"),
        "{main}"
    );
}

/// An alias of a let-bound lambda is a value that is not a lambda: the alias
/// instantiates the lambda once, and its own rows are body rows.
#[salsa_test]
fn an_alias_of_a_let_bound_lambda_instantiates_it_once(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
use std::io::{Io, print_line}

fn app(x: Int, f: fn(Int) ->{e} Int) ->{e} Int { f(x) }

fn main() ->{Io} Nil {
    let both = fn(x: Int, f: fn(Int) ->{e} Int) app(x, f)
    let h = both
    let _ = h(+1, fn(n) n + +1)
    let _ = h(+2, fn(n) { print_line("io") n })
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let main = ir
        .split("tribute_control.func @\"pkg::main\"[")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("root main");
    assert!(main.starts_with("local 'a0, body 'a1]"), "{main}");
    assert_eq!(
        main.matches("tribute_control.instantiate").count(),
        1,
        "{main}"
    );
    assert!(
        main.contains("tribute_control.instantiate %0 {row_args = [{tail = 1}]}"),
        "{main}"
    );
}

/// A let-bound lambda lowered once for the type each use needs carries only
/// instantiated rows, so it quantifies nothing and its rows are body rows.
#[salsa_test]
fn a_lambda_materialized_per_use_quantifies_no_rows(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn run() ->{} Int {
    let action = fn(x: Int) x
    let first = pure(action, +3)
    open(action, first)
}
fn main() -> Nil {
    let _ = run()
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let run = ir
        .split("tribute_control.func @\"pkg::run\"")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .expect("run");
    assert!(run.contains("tribute_control.lambda("), "{run}");
    assert!(
        !run.contains("lambda['") && !run.contains("instantiate") && !run.contains("local '"),
        "{run}"
    );
}

/// A use inside another lambda instantiates the captured lambda in that
/// lambda's body.
#[salsa_test]
fn a_use_inside_another_lambda_instantiates_the_captured_lambda(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn app(x: Int, f: fn(Int) ->{e} Int) ->{e} Int { f(x) }

fn run() -> Int {
    let both = fn(x: Int, f: fn(Int) ->{e} Int) app(x, f)
    let wrap = fn(x: Int) both(x, fn(n) n)
    wrap(+1)
}

fn main() -> Nil {
    let _ = run()
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let run = ir
        .split("tribute_control.func @\"pkg::run$9D\"")
        .nth(1)
        .or_else(|| ir.split("tribute_control.func @\"pkg::run\"").nth(1))
        .and_then(|rest| rest.split("\n  }\n").next())
        .unwrap_or_else(|| panic!("run in\n{ir}"));
    let inner = run
        .split("captures [%0] {")
        .nth(1)
        .unwrap_or_else(|| panic!("a lambda capturing `both` in\n{run}"));
    assert!(
        inner.contains("tribute_control.instantiate %0 {row_args = ["),
        "{run}"
    );
}

/// The disagreements the convention-class check finds in `source`. This
/// reads the frontend's output, before unreferenced definitions are removed;
/// the check never requires an unsuffixed base definition, so it agrees with
/// the pipeline's debug check.
fn convention_class_disagreements(
    db: &salsa::DatabaseImpl,
    source: &str,
) -> Vec<(&'static str, String)> {
    let source = SourceCst::from_source_str(db, "test.trb", source);
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    tribute_passes::check_convention_classes::check(&ctx, module)
        .into_iter()
        .map(|disagreement| (disagreement.rule, disagreement.message))
        .collect()
}

/// Representative row-polymorphic programs agree with the classes their
/// rows imply.
#[salsa_test]
fn specialized_classes_agree_with_their_rows(db: &salsa::DatabaseImpl) {
    for source in [
        SOURCE,
        r#"
use std::io::{Io, print_line}

fn app(x: Int, f: fn(Int) ->{e} Int) ->{e} Int { f(x) }

fn countdown(n: Int) -> Int {
    let step = fn(m: Int) -> Int {
        case m == +0 {
            True -> m
            False -> become countdown(m - +1)
        }
    }
    become step(n)
}

fn main() ->{Io} Nil {
    let both = fn(x: Int, f: fn(Int) ->{e} Int) app(x, f)
    let _ = both(+1, fn(n) n + +1)
    let _ = both(+2, fn(n) { print_line("io") n })
    let nested = fn() both(+3, fn(n) n)
    let _ = nested()
    let _ = countdown(+3)
    Nil
}
"#,
    ] {
        assert_eq!(convention_class_disagreements(db, source), []);
    }
}

/// A let-bound lambda copied into a class instance reads its body at the
/// copy's class, including a reference its source body already named by a
/// type instance.
#[salsa_test]
fn a_let_bound_lambda_copy_reads_its_body_at_its_class(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn app(x: a, f: fn(a) ->{e} a) ->{e} a { f(x) }

fn before() -> String {
    let call = fn(s: String) after(s)
    call("order")
}

fn after(s: String) -> String {
    let inner = fn(t: String) app(t, fn(u) u <> "!")
    inner(s)
}

fn main() -> Nil {
    let _ = before()
    Nil
}
"#,
    );
    let (ctx, module) = compile_specialized_frontend(db, source)
        .expect("class specialization accepts frontend output")
        .expect("production frontend should lower");
    assert_eq!(
        tribute_passes::check_convention_classes::check(&ctx, module),
        []
    );
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    let after = ir
        .split("tribute_control.func @\"pkg::after$9D\"")
        .nth(1)
        .and_then(|rest| rest.split("\n  }\n").next())
        .unwrap_or_else(|| panic!("after$9D in\n{ir}"));
    assert!(
        after.contains("callee = @\"pkg::app$std::String$9D\"")
            && !after.contains("convention(cps)"),
        "{after}"
    );
}
