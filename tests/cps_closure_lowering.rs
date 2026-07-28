//! Downstream closure-lowering compatibility for CPS frontend output.

use ropey::Rope;
use salsa_test_macros::salsa_test;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{compile_with_diagnostics, run_through_closure_lower};
use tribute_front::SourceCst;
use trunk_ir::printer::print_module;
use trunk_ir::validation::validate_value_integrity;

fn source_from_code(db: &dyn salsa::Database, name: &str, code: &str) -> SourceCst {
    let code = Rope::from_str(code);
    let tree = parse_with_thread_local(&code, None);
    SourceCst::from_path(db, name, code, tree)
}

fn assert_cps_closures_lower(db: &dyn salsa::Database, source: SourceCst) {
    let diagnostics = compile_with_diagnostics(db, source);
    assert!(
        diagnostics.diagnostics.is_empty(),
        "frontend must accept CPS closure source: {:?}",
        diagnostics.diagnostics
    );
    let (ctx, module) = run_through_closure_lower(db, source)
        .expect("closure lowering should accept frontend CPS output")
        .expect("frontend should produce a module");
    let validation = validate_value_integrity(&ctx, module);
    assert!(
        validation.is_ok(),
        "closure lowering must not leave cross-function SSA references:\n{validation}"
    );

    let ir = print_module(&ctx, module.op());
    assert!(
        ir.contains("run::__clam_"),
        "CPS source must produce a continuation closure for lowering:\n{ir}"
    );
    assert!(
        !ir.contains("closure.lambda"),
        "closure lowering must consume frontend-generated closures:\n{ir}"
    );
}

/// A CPS continuation must preserve a source lambda capture used only as a
/// record spread base through downstream closure lowering.
#[salsa_test]
fn record_spread_capture_survives_closure_lowering(db: &salsa::DatabaseImpl) {
    let source = source_from_code(
        db,
        "record_spread_capture.trb",
        r#"
ability Trace {
    op before() -> Nil
}

struct Point { x: Int, y: Int }

fn run(point: Point) ->{Trace} fn() -> Point {
    Trace::before()
    fn() { Point { x: +1, ..point } }
}

fn main() { }
"#,
    );

    assert_cps_closures_lower(db, source);
}

/// Continuations produced while evaluating tuple, record, and constructor
/// strict children must remain valid after closure lowering.
#[salsa_test]
fn mixed_strict_aggregates_survive_closure_lowering(db: &salsa::DatabaseImpl) {
    let source = source_from_code(
        db,
        "mixed_strict_aggregates.trb",
        r#"
ability State(s) {
    op get() -> s
}

struct Point { x: Int, y: Int }

enum Boxed(a) {
    Boxed(a)
}

fn read() ->{State(Int)} Int { State::get() }

fn run() ->{State(Int)} Int {
    let base = Point { x: +0, y: +0 }
    let pair = #(read(), read())
    let point = Point { x: read(), ..base }
    let boxed = Boxed(read())
    +0
}

fn main() { }
"#,
    );

    assert_cps_closures_lower(db, source);
}
