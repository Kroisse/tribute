//! Generic callbacks must retain the root export contract after specialization.

mod common;

use std::io::Write as _;
use std::process::Command;

use salsa_test_macros::salsa_test;
use tribute::pipeline::{compile_frontend, compile_to_wasm_binary};
use tribute_front::SourceCst;
use tribute_ir::dialect::tribute_control;
use trunk_ir::Attribute;
use trunk_ir::ops::DialectOp;

// Keep the original generic frontend regression on the production path, which
// specializes its callback before lowering source-logical IR.
const SOURCE: &str = r#"
fn apply_open(value: a, callback: fn(a) -> b) -> b {
    callback(value)
}

fn main() ->{std::io::Io} Nil {
    let _ = apply_open(+41, fn(value) { value })
    Nil
}
"#;

#[salsa_test]
fn generic_callback_preserves_evidence_root_and_executes_wasm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "test.trb", SOURCE);
    let (ctx, module) = compile_frontend(db, source).expect("production frontend should lower");
    let main = module
        .ops(&ctx)
        .into_iter()
        .find_map(|op| {
            let function = tribute_control::Func::from_op(&ctx, op).ok()?;
            (function.sym_name(&ctx) == "main").then_some(function)
        })
        .expect("source main");
    assert_eq!(
        tribute_control::func_sig_convention(&ctx, main.r#type(&ctx)),
        Some(tribute_control::CallingConvention::Cps),
        "the open callback requires a CPS worker"
    );
    assert_eq!(
        ctx.op(main.op_ref())
            .attributes
            .get("tribute.root_export_convention"),
        Some(&Attribute::Int(1)),
        "the Io root must retain its EvidenceDirect export"
    );

    let binary = compile_to_wasm_binary(db, source).expect("generic root should compile to Wasm");
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&binary)
        .expect("generic root must produce a valid Wasm binary");
    let mut file = tempfile::NamedTempFile::new().expect("temporary Wasm file");
    file.write_all(&binary).expect("write Wasm binary");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(file.path())
        .output()
        .expect("run Wasm module with wasmtime");
    assert!(
        output.status.success(),
        "Wasm root execution failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
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
