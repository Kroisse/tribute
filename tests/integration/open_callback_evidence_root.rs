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
            let class = name.strip_prefix("std::Option::map$")?.rsplit_once("$9")?.1;
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
