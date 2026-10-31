//! Class specialization of source programs: the conventions and class
//! instances it assigns to the frontend's source-logical IR. The frontend
//! assigns none, so these run the production frontend, then the class pass.

use crate::common;

use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// Whether `line` is the header of the source function `name`.
fn is_function_header(line: &str, name: &str) -> bool {
    let line = line.trim_start();
    [
        format!("tribute_cps.func @{name}"),
        format!("tribute_cps.func @\"{name}\""),
        format!("tribute_cps.func @\"pkg::{name}\""),
    ]
    .iter()
    .any(|prefix| {
        line.strip_prefix(prefix.as_str())
            .is_some_and(|rest| rest.starts_with(['(', '[']))
    })
}

/// Source-logical IR: control callables, none of their physical lowering.
fn assert_logical_boundary(ir: &str) {
    assert!(
        ir.contains("tribute_cps.func"),
        "source-logical lowering must define control callables:\n{ir}"
    );
    for forbidden in [
        "func.func",
        "func.call",
        "func.return",
        "func.unreachable",
        "closure.",
        "ability.legacy",
        "ability.handle",
        "effect.",
        "__tribute_cps_control",
        "func.func_sig",
    ] {
        assert!(
            !ir.contains(forbidden),
            "source-logical lowering must not contain `{forbidden}`:\n{ir}"
        );
    }
}

fn checked_logical_function<'a>(ir: &'a str, name: &str) -> &'a str {
    assert_logical_boundary(ir);
    logical_function(ir, name)
}

/// The printed source function `name`, up to the next one.
fn logical_function<'a>(ir: &'a str, name: &str) -> &'a str {
    let start = ir
        .lines()
        .find(|line| is_function_header(line, name))
        .map(|line| {
            line.as_ptr() as usize - ir.as_ptr() as usize + (line.len() - line.trim_start().len())
        })
        .unwrap_or_else(|| panic!("missing logical function `{name}` in canonical IR:\n{ir}"));
    let function_and_following = &ir[start..];
    let end = function_and_following[1..]
        .find("\n  tribute_cps.func ")
        .map_or(function_and_following.len(), |offset| offset + 1);
    &function_and_following[..end]
}

fn named_function_header<'a>(ir: &'a str, name: &str) -> &'a str {
    ir.lines()
        .find(|line| {
            line.trim_start()
                .strip_prefix(&format!("tribute_cps.func @\"pkg::{name}\""))
                .or_else(|| {
                    line.trim_start()
                        .strip_prefix(&format!("tribute_cps.func @{name}"))
                })
                .is_some_and(|rest| rest.starts_with(['(', '[']))
        })
        .unwrap_or_else(|| panic!("missing lowered function @{name}:\n{ir}"))
}

fn value_name(line: &str) -> &str {
    line.trim_start()
        .split_once(" = ")
        .map(|(value, _)| value)
        .unwrap_or_else(|| panic!("missing result value in {line}"))
}

fn type_after_colon(line: &str) -> &str {
    line.rsplit_once(": ")
        .map(|(_, ty)| ty.trim())
        .unwrap_or_else(|| panic!("missing type in {line}"))
}

/// `callable_ty` without its ability row, which names the binders of the
/// definition it is written in.
fn without_row(callable_ty: &str) -> String {
    match callable_ty.split_once("ability_row = {") {
        Some((head, tail)) => {
            let rest = tail.split_once("}, ").map_or("", |(_, rest)| rest);
            format!("{head}{rest}")
        }
        None => callable_ty.to_owned(),
    }
}

/// The convention a printed `tribute_cps` operation declares, quoted.
/// The conventions of the entries of the function table named `name`.
fn table_conventions(ir: &str, name: &str) -> Vec<String> {
    let header = format!("tribute_cps.func_table @\"{name}\" = {{");
    let line = ir
        .lines()
        .find_map(|line| line.trim().strip_prefix(&header))
        .unwrap_or_else(|| panic!("missing function table {name}:\n{ir}"));
    line.trim_end_matches('}')
        .split(", ")
        .filter_map(|entry| entry.split(" = ").next())
        .map(str::to_owned)
        .collect()
}

/// Each printed lambda's header line, followed by the line that opens its
/// convention entry.
fn lambda_headers(ir: &str) -> Vec<String> {
    let lines: Vec<&str> = ir.lines().collect();
    lines
        .windows(2)
        .filter(|pair| pair[0].contains("tribute_cps.lambda"))
        .map(|pair| format!("{} {}", pair[0].trim(), pair[1].trim()))
        .collect()
}

fn named_ref_lines<'a>(ir: &'a str, name: &str) -> Vec<&'a str> {
    ir.lines()
        .filter(|line| line.contains(&format!("func_ref @\"pkg::{name}$8\"")))
        .collect()
}

fn assert_not_cast(ir: &str, value: &str) {
    assert!(
        !ir.lines().any(|line| {
            line.contains("core.unrealized_conversion_cast")
                && line
                    .split("core.unrealized_conversion_cast")
                    .nth(1)
                    .is_some_and(|suffix| suffix.trim_start().starts_with(value))
        }),
        "the callable value {value} must not be adapted by a raw conversion cast:\n{ir}"
    );
}

#[salsa_test]
fn named_function_value_uses_its_callback_contract(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callback_adaptation.trb",
        r#"
fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn double(value: Int) -> Int { value + value }
fn forward(x: Int) ->{e} Int { apply(double, x) }
fn main() -> Nil {}
"#,
    );

    let ir = common::specialized_frontend_ir(db, source);
    let refs = named_ref_lines(&ir, "double");
    assert_eq!(
        refs.len(),
        1,
        "expected one named callback reference:\n{ir}"
    );
    let double_ref = refs[0];
    let callback_ty = type_after_colon(double_ref);
    assert!(
        without_row(named_function_header(&ir, "apply"))
            .contains(&format!("(%0: {}, ", without_row(callback_ty))),
        "the fresh function reference must have apply's callback type, up to its row:\n{ir}"
    );
    // The value enters `double` directly; an adapter fills the `Cps` slot
    // that `apply` calls.
    assert_eq!(table_conventions(&ir, "pkg::double$8"), ["direct"]);
    assert!(
        named_function_header(&ir, "double").contains("convention(direct)"),
        "the original named worker must remain Direct:\n{ir}"
    );
    assert_not_cast(logical_function(&ir, "forward"), value_name(double_ref));
}

#[salsa_test]
fn named_value_uses_distinct_contracts(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callback_distinct_contracts.trb",
        r#"
fn apply_direct(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn apply_open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn double(value: Int) -> Int { value + value }
fn forward(x: Int) ->{e} Int { apply_open(double, apply_direct(double, x)) }
fn main() -> Nil {}
"#,
    );

    let ir = common::specialized_frontend_ir(db, source);
    let refs = named_ref_lines(&ir, "double");
    assert_eq!(
        refs.len(),
        2,
        "each use needs its own function reference:\n{ir}"
    );
    let reference_values = refs.iter().map(|line| value_name(line)).collect::<Vec<_>>();
    assert_ne!(reference_values[0], reference_values[1]);
    // Both references share the table of the one instance they name.
    assert_eq!(table_conventions(&ir, "pkg::double$8"), ["direct"]);
    for value in reference_values {
        assert_not_cast(logical_function(&ir, "forward"), value);
    }
}

#[salsa_test]
fn lambda_value_uses_an_exact_compatible_callback_contract(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_lambda_callback_adaptation.trb",
        r#"
fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn forward(x: Int) ->{e} Int { apply(fn(value) { value + value }, x) }
fn main() -> Nil {}
"#,
    );

    let ir = common::specialized_frontend_ir(db, source);
    let lambda = lambda_headers(logical_function(&ir, "forward"))
        .into_iter()
        .next()
        .expect("logical lowering must materialize the callback lambda");
    assert!(
        lambda.ends_with(" cps {"),
        "the exact-compatible lambda must be emitted at the CPS callback convention:\n{ir}"
    );
    assert_not_cast(logical_function(&ir, "forward"), value_name(&lambda));
}

/// A named function bound by `let` and passed on at another callable type
/// leaves no reference unused.
#[salsa_test]
fn every_named_function_reference_is_used(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callback_unused_references.trb",
        r#"
fn identity(x: Int) ->{} Int { x }
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn entry() ->{} Int {
    let named = identity
    let alias = named
    pure(fn(x: Int) open(alias, x), +3)
}
fn main() -> Nil {}
"#,
    );

    let ir = common::specialized_frontend_ir(db, source);
    let mut ctx = trunk_ir::context::IrContext::new();
    let module = trunk_ir::parser::parse_test_module(&mut ctx, &ir);
    let mut references = 0;
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        if <tribute_ir::dialect::tribute_cps::FuncRef as trunk_ir::ops::DialectOp>::matches(
            &ctx, op,
        ) {
            references += 1;
            assert!(
                ctx.has_uses(ctx.op_results(op)[0]),
                "unused reference:\n{ir}"
            );
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    assert!(references > 0, "{ir}");
}

// Root and worker conventions.

#[salsa_test]
fn logical_root_main_preserves_pure_and_io_conventions(db: &salsa::DatabaseImpl) {
    let pure = SourceCst::from_source_str(db, "pure.trb", "fn main() -> Nil { }");
    let pure_ir = common::specialized_frontend_ir(db, pure);
    assert!(
        pure_ir.contains(
            "tribute_cps.func @\"pkg::main\"['a0]() -> core.nil row(| 'a0) inferred convention(direct)"
        ),
        "pure root main must be direct:\n{pure_ir}"
    );
    let io = SourceCst::from_source_str(db, "io.trb", "fn main() ->{std::io::Io} Nil { Nil }");
    let io_ir = common::specialized_frontend_ir(db, io);
    assert!(
        io_ir.contains("tribute_cps.func @\"pkg::main\"() -> core.nil row(@\"ability_1_11_std::io::Io\") inferred convention(evidence_direct)"),
        "Io root main must preserve EvidenceDirect:\n{io_ir}"
    );
}

#[salsa_test]
fn nested_main_is_not_a_root_convention_special_case(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State {
    op get() -> Int
}
mod Nested {
    use super::State

    fn main() ->{State} Int { State::get() }
}
fn main() -> Nil { }
"#,
    );
    let ir_text = common::specialized_frontend_ir(db, source);
    assert!(
        ir_text.contains(
            "tribute_cps.func @\"pkg::Nested::main\"() -> core.i32 row(@\"ability_1_10_pkg::State\") convention(cps)"
        ),
        "nested main must preserve its ordinary checker-selected CPS convention:\n{ir_text}"
    );
}

/// Worker conventions start with concrete effects, then monotonically promote
/// bodies whose open callback evaluation requires CPS.  `forward_open` appears
/// before `apply_open` so this also proves named-call propagation needs a
/// fixed-point pass rather than a declaration-order scan.
#[salsa_test]
fn test_open_callback_workers_are_cps_and_their_caller_is_not(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn forward_open(value: a, callback: fn(a) ->{e} b) ->{e} b {
    apply_open(value, callback)
}

fn apply_open(value: a, callback: fn(a) ->{e} b) ->{e} b {
    callback(value)
}

fn pure(value: Int) ->{} Int { value }

fn apply_closed(value: Int, callback: fn(Int) ->{} Int) ->{} Int {
    callback(value)
}

fn evidence() ->{std::io::Io} Nil { Nil }

struct Box(a) { value: a }
fn unwrap(value: Box(a)) ->{} a { value.value }

mod Nested {
    struct Box(a) { value: a }
    fn unwrap(value: Box(a)) ->{} a { value.value }
    pub fn use_nested() ->{} Int { unwrap(Box { value: +41 }) }
}

fn main() -> Nil {
    let _ = forward_open(+41, fn(value) { value })
    let _ = unwrap(Box { value: +1 })
    let _ = Nested::use_nested()
}
"#,
    );

    let ir_text = common::specialized_frontend_ir(db, source);
    for name in ["forward_open$Int$Int", "apply_open$Int$Int"] {
        let header = ir_text
            .lines()
            .find(|line| is_function_header(line, name))
            .unwrap_or_else(|| panic!("missing lowered worker {name}"));
        assert!(
            header.contains("convention(cps)"),
            "{name} must be promoted to Cps:\n{header}"
        );
    }
    let pure_header = ir_text
        .lines()
        .find(|line| is_function_header(line, "pure"))
        .expect("missing lowered pure worker");
    assert!(
        pure_header.contains("convention(direct)"),
        "pure worker must remain Direct:\n{pure_header}"
    );
    let apply_closed_header = ir_text
        .lines()
        .find(|line| is_function_header(line, "apply_closed"))
        .expect("missing lowered apply_closed worker");
    assert!(
        apply_closed_header.contains("convention(direct)"),
        "apply_closed must stay Direct:\n{apply_closed_header}"
    );
    let evidence_header = ir_text
        .lines()
        .find(|line| is_function_header(line, "evidence"))
        .expect("missing lowered evidence worker");
    assert!(
        evidence_header.contains("convention(evidence_direct)"),
        "evidence worker must stay EvidenceDirect:\n{evidence_header}"
    );
    let main = checked_logical_function(&ir_text, "main");
    let main_header = main.lines().next().expect("logical function has a header");
    assert!(
        main_header.contains("convention(direct)"),
        "a call to a Cps worker leaves the root main as its row has it:\n{main_header}"
    );
    assert!(
        main.contains("tribute_control.call"),
        "root main must contain the direct worker call:\n{main}"
    );
    assert!(
        main.contains("callee = @\"pkg::forward_open$Int$Int$9D\""),
        "root main must call the Direct instance of the monomorphized forward_open worker:\n{main}"
    );
    assert!(
        !ir_text.contains("Nested::Nested::"),
        "nested workers must not be requalified:\n{ir_text}"
    );
    assert!(
        logical_function(&ir_text, "pkg::Nested::unwrap$Int")
            .contains("callee = @\"pkg::Nested::Box$Int::value\""),
        "nested worker must call the exact qualified accessor:\n{ir_text}"
    );
    assert!(
        ir_text.lines().any(|line| line
            .trim_start()
            .starts_with("tribute_cps.func @\"pkg::Box$Int::value\"")),
        "specialized accessor must keep its exact root identity:\n{ir_text}"
    );
}

/// An `Io` root that calls an open-callback worker keeps the convention of
/// its row.
#[salsa_test]
fn test_open_callback_leaves_the_evidence_root_as_it_is(db: &salsa::DatabaseImpl) {
    // The generic fixture, which also runs on Wasm, is in
    // `open_callback_evidence_root.rs`; this one is monomorphic.
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn apply_open(value: Int, callback: fn(Int) ->{e} Int) ->{e} Int {
    callback(value)
}

fn main() ->{std::io::Io} Nil {
    let _ = apply_open(+41, fn(value) { value })
    Nil
}
"#,
    );

    let ir_text = common::specialized_frontend_ir(db, source);
    let main_header = ir_text
        .lines()
        .find(|line| is_function_header(line, "main"))
        .expect("missing lowered root main");
    assert!(
        main_header.contains("convention(evidence_direct)"),
        "the Io root keeps the convention of its row:\n{main_header}"
    );
}

/// A nested `main` that calls an open-callback worker is an ordinary function:
/// its row's tail is a class variable, so the definition itself is the
/// instance whose classes are all `Cps`, unlike root `main`.
#[salsa_test]
fn test_nested_open_callback_main_keeps_its_convention(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
mod Nested {
    fn apply_open(value: a, callback: fn(a) ->{e} b) ->{e} b {
        callback(value)
    }

    fn main() -> Int {
        apply_open(+41, fn(value) { value })
    }
}

fn main() -> Nil { }
"#,
    );

    let ir_text = common::specialized_frontend_ir(db, source);
    let nested_main = ir_text
        .lines()
        .find(|line| is_function_header(line, "Nested::main"))
        .expect("missing lowered Nested::main worker");
    assert!(
        nested_main.contains("convention(cps)"),
        "nested open-callback main is not a root main:\n{nested_main}"
    );
}

// Local callable instances.

#[salsa_test]
fn named_operator_preserves_target_and_consumer_conventions(db: &salsa::DatabaseImpl) {
    use std::ops::ControlFlow;
    use tribute_ir::calling_convention::CallingConvention;
    use tribute_ir::dialect::{tribute_control, tribute_cps};
    use trunk_ir::SymbolPath;
    use trunk_ir::context::IrContext;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::refs::ValueDef;
    use trunk_ir::walk::{WalkAction, walk_op};

    for argument in ["(Int::+)", "alias"] {
        let code = format!(
            r#"
fn pure(f: fn(Int, Int) ->{{}} Int) ->{{}} Int {{ f(+3, +4) }}
fn open(f: fn(Int, Int) ->{{e}} Int, x: Int, y: Int) ->{{e}} Int {{ f(x, y) }}
fn main() -> Nil {{
    let add = (Int::+)
    let alias = add
    let first = pure({argument})
    open({argument}, first, +2)
    Nil
}}
"#
        );
        let source = SourceCst::from_source_str(db, "named_operator.trb", &code);
        let text = common::specialized_frontend_ir(db, source);
        assert!(
            !logical_function(&text, "main").contains("unrealized_conversion_cast"),
            "{text}"
        );
        let mut ir = IrContext::new();
        let module = parse_test_module(&mut ir, &text);
        let mut conventions = Vec::new();
        let _ = walk_op::<()>(&ir, module.op(), &mut |op| {
            if let Ok(call) = tribute_control::Call::from_op(&ir, op)
                && ["pkg::pure", "pkg::open$9D"]
                    .iter()
                    .any(|name| call.callee(&ir) == [*name])
            {
                let callback = ir.op_operands(op)[0];
                let ValueDef::OpResult(producer, _) = ir.value_def(callback) else {
                    panic!("expected named callback: {text}");
                };
                let table = tribute_cps::func_ref_table(&ir, producer).unwrap();
                let entries = tribute_cps::table_entries(&ir, table).unwrap();
                let [(convention, target)] = &entries[..] else {
                    panic!("expected one table entry: {text}");
                };
                assert_eq!(*target, SymbolPath::new(["std::Int::+"]));
                conventions.push(*convention);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(
            conventions,
            [CallingConvention::Direct, CallingConvention::Direct],
            "{argument}: {text}"
        );
    }
}

#[salsa_test]
fn fixed_local_identity_reaches_pure_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_identity.trb",
        r#"
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn entry() ->{} Int {
    let identity = fn(x: Int) x
    apply(identity, +3)
}
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    let entry = logical_function(&ir, "entry");
    assert!(entry.contains("tribute_cps.lambda"));
    assert!(
        lambda_headers(entry)
            .iter()
            .any(|line| line.contains("tribute_cps.lambda(") && line.ends_with(" direct {")),
        "a pure lambda passed to an effect-free consumer must be Direct:\n{ir}"
    );
    assert!(!entry.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn escaping_lambda_retains_open_callable_contract(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "escaping_lambda.trb",
        r#"
fn make() -> fn(Int) -> Int { fn(x) x }
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    assert!(
        lambda_headers(logical_function(&ir, "make"))
            .iter()
            .any(|line| line.contains("tribute_cps.lambda(") && line.ends_with(" cps {")),
        "a returned lambda must retain the open callable contract:\n{ir}"
    );
}

#[salsa_test]
fn open_effect_consumer_keeps_lambda_cps(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "open_lambda_consumer.trb",
        r#"
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn forward(x: Int) ->{e} Int { open(fn(y) y, x) }
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    assert!(
        lambda_headers(logical_function(&ir, "forward"))
            .iter()
            .any(|line| line.contains("tribute_cps.lambda(") && line.ends_with(" cps {")),
        "an open-effect consumer must receive a CPS lambda:\n{ir}"
    );
}

#[salsa_test]
fn effectful_lambda_keeps_effect_convention(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "effectful_lambda_convention.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn run(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        op State::get() { resume +0 }
    }
}

fn entry() -> Int { run(fn() State::get()) }
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    assert!(
        lambda_headers(logical_function(&ir, "entry")).iter().any(|line| line.contains(
            "tribute_cps.lambda() -> core.i32 row(@\"ability_2_10_pkg::State_3_int\") captures [] entries { cps {"
        )),
        "a lambda that performs an effect must remain CPS:\n{ir}"
    );
}

#[salsa_test]
fn fixed_local_serves_pure_and_open_with_one_surface(db: &salsa::DatabaseImpl) {
    for calls in [
        "let a = pure(action, +3)\nopen(action, a)",
        "let a = open(action, +3)\npure(action, a)",
    ] {
        let code = format!(
            r#"
fn pure(f: fn(Int) ->{{}} Int, x: Int) ->{{}} Int {{ f(x) }}
fn open(f: fn(Int) ->{{e}} Int, x: Int) ->{{e}} Int {{ f(x) }}
fn entry() ->{{}} Int {{
    let action = fn(x: Int) x
    {calls}
}}
fn main() -> Nil {{}}
"#
        );
        let source = SourceCst::from_source_str(db, "multi_surface.trb", &code);
        let ir = common::specialized_frontend_ir(db, source);
        // `entry` calls the `Direct` instance of `open`, whose callback
        // type is the one `pure` takes.
        let entry = logical_function(&ir, "entry");
        assert_eq!(entry.matches("tribute_cps.lambda").count(), 1, "{ir}");
        assert!(entry.contains("callee = @\"pkg::open$9D\""), "{ir}");
        assert!(!entry.contains("unrealized_conversion_cast"), "{ir}");
    }
}

#[salsa_test]
fn caller_of_a_cps_worker_keeps_the_convention_of_its_row(db: &salsa::DatabaseImpl) {
    // `g` quantifies its row, and `entry` instantiates it only at the pure
    // row, so `g` and the callback it passes to `app$9D` are Direct.
    let source = SourceCst::from_source_str(
        db,
        "original_cps_lambda.trb",
        r#"
fn app(x: a, f: fn(a) ->{e} a) ->{e} a { f(x) }
fn entry() ->{} Int {
    let g = fn(n: Int) app(n, fn(m) m)
    g(+1)
}
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    let entry = logical_function(&ir, "entry");
    let header = entry.lines().next().expect("a function has a header");
    let lambdas = lambda_headers(entry);
    assert!(!header.contains("convention(cps)"), "{ir}");
    assert_eq!(lambdas.len(), 2, "{ir}");
    assert!(
        lambdas.iter().all(|line| line.ends_with(" direct {")),
        "{ir}"
    );
}

#[salsa_test]
fn local_named_consumer_retains_its_worker_parameter_convention(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_consumer.trb",
        r#"
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn entry() ->{} Int {
    let consume = open
    let alias = consume
    let action = fn(x: Int) x
    alias(action, +3)
}
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    assert!(
        !logical_function(&ir, "entry").contains("unrealized_conversion_cast"),
        "{ir}"
    );
}

/// Contextual generic rows must retain one solved logical signature regardless
/// of harmless body wrapping.  In particular, lowering must not recover a
/// lambda result by recognizing a perform-shaped AST body.
#[salsa_test]
fn test_generic_effect_lambda_wrappers_share_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn with_state(f: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle f() {
        do result { result }
        op State::get() { with_state(fn() { resume init }, init) }
    }
}

fn total() -> Nat {
    let direct = with_state(fn() { State::get() }, 42)
    let empty_block = with_state(fn() { { State::get() } }, 42)
    let let_wrapped = with_state(fn() {
        let value = State::get()
        value
    }, 42)
    let case_wrapped = with_state(fn() {
        case State::get() {
            value -> value
        }
    }, 42)
    direct + empty_block + let_wrapped + case_wrapped
}
fn main() -> Nil {}
"#,
    );

    let ir_text = common::specialized_frontend_ir(db, source);
    let total = logical_function(&ir_text, "total");
    assert_eq!(
        lambda_headers(total)
            .iter()
            .filter(|line| line.contains(
                "tribute_cps.lambda() -> core.i32 row(@\"ability_2_10_pkg::State_3_nat\" | 'a0) captures [] entries { cps {"
            ))
            .count(),
        4,
        "every direct or wrapped generic State operation lambda must retain the same solved signature:\n{ir_text}"
    );
    assert!(
        !total.contains("tribute_cps.lambda() -> tribute_rt.anyref"),
        "the four lambdas of total must not erase their result:\n{total}"
    );
}

#[salsa_test]
fn returned_callback_keeps_its_cps_convention_after_pure_let(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "returned_callback_preserves_cps_convention_after_pure_let.trb",
        r#"
fn consume(value: Nat) -> Nil { Nil }

fn identity(f: fn(Nat) ->{e} Nat) -> fn(Nat) ->{e} Nat { f }

fn main() -> Nil {
    let f = identity(fn(x: Nat) { x + 10 })
    let result = f(5)
    consume(result)
}
"#,
    );

    let ir = common::specialized_frontend_ir(db, source);
    let main = ir
        .lines()
        .find(|line| line.contains("func @\"pkg::main\"(") || line.contains("func @\"pkg::main\"["))
        .expect("IR must contain main");
    assert!(
        main.contains("convention(direct)"),
        "main keeps the convention of its row around the returned callback call:\n{ir}"
    );
    // The lambda's row is a body binder of `main` that its closed row does not
    // fix, so its class stays `Cps` while `main` itself is `Direct`.
    let main_body = logical_function(&ir, "main");
    let lambda = lambda_headers(main_body)
        .into_iter()
        .find(|line| line.contains("tribute_cps.lambda("))
        .expect("IR must contain the callback lambda");
    assert!(
        lambda.ends_with(" cps {"),
        "the callback lambda must retain its CPS convention:\n{ir}"
    );
    assert!(
        main_body
            .lines()
            .any(|line| line.contains("tribute_cps.call_indirect")),
        "the returned callback must use an indirect call:\n{ir}"
    );
}

/// A local lambda in a monomorphized instance takes the convention of its
/// closed row.
#[salsa_test]
fn outer_generic_local_lambdas_are_direct(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "outer_local_signature.trb",
        r#"
fn pure(f: fn(a) ->{} a, value: a) ->{} a { f(value) }
fn apply(value: a) ->{} a {
    let identity = fn(x: a) x
    let alias = identity
    pure(alias, value)
}
fn use_int() ->{} Int { apply(+3) }
fn use_bool() ->{} Bool { apply(True) }
fn main() -> Nil {}
"#,
    );
    let ir = common::specialized_frontend_ir(db, source);
    for instance in ["apply$Int", "apply$Bool"] {
        let lambdas = lambda_headers(logical_function(&ir, instance));
        assert_eq!(lambdas.len(), 1, "{instance}:\n{ir}");
        assert!(lambdas[0].ends_with(" direct {"), "{instance}:\n{ir}");
    }
}

/// Uses of a local helper that pass classes of their own for its binder
/// keep it one value; the closure its body makes reads the binder as their
/// upper bound and has a body for each class up to it.
#[salsa_test]
fn a_local_helper_used_at_two_classes_stays_one_value(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_upper_bound.trb",
        r#"
fn both(k1: fn() ->{} Int, k2: fn() ->{e} Int) ->{e} Int {
    let wrap = fn(k: fn() ->{r} Int) { fn() { k() } }
    let a = wrap(k1)
    let b = wrap(k2)
    a() + b()
}
fn main() -> Nil {}
"#,
    );

    let ir = common::specialized_frontend_ir(db, source);
    let both = logical_function(&ir, "both");
    assert_eq!(
        both.matches("tribute_cps.lambda[").count(),
        1,
        "the helper is one value:\n{both}"
    );
    let closure = &both[both
        .find("tribute_cps.lambda() ")
        .unwrap_or_else(|| panic!("the helper makes a closure:\n{both}"))..];
    for body in ["direct {", "evidence_direct {", "cps {"] {
        assert!(closure.contains(body), "missing body `{body}`:\n{both}");
    }
}
