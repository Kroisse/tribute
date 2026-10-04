//! Typechecked evidence selections reach source-logical IR and survive CPS
//! legalization on the operations that pass the evidence.

use salsa_test_macros::salsa_test;
use tribute::pipeline::{compile_frontend, run_through_cps_lowering};
use tribute_front::SourceCst;
use trunk_ir::printer::print_module;

const SOURCE: &str = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn both(g: fn() ->{e} Nil) ->{e, State(Nat)} Nil {
    g()
    State::set(1)
}

fn caller() ->{State(Nat)} Nil {
    both(fn() { State::set(2) })
}

fn counter(comp: fn() ->{e, State(Nat)} Nat) ->{e, State(Nat)} Nat {
    handle comp() {
        do result { result }
        op State::get() { resume State::get() }
        op State::set(v) { resume Nil }
    }
}

fn main() -> Nil {
    let _ = handle counter(fn() {
        caller()
        State::get()
    }) {
        do result { result }
        op State::get() { resume 0 }
        op State::set(v) { resume Nil }
    }
    Nil
}
"#;

/// The lines of `ir` that carry an evidence selection.
fn planned_lines(ir: &str) -> Vec<&str> {
    ir.lines()
        .map(str::trim)
        .filter(|line| line.contains("evidence_plan"))
        .collect()
}

fn assert_line(lines: &[&str], op: &str, plan: &str, ir: &str) {
    assert!(
        lines
            .iter()
            .any(|line| line.contains(op) && line.contains(plan)),
        "expected `{op}` with `{plan}`:\n{ir}"
    );
}

#[salsa_test]
fn evidence_plans_reach_source_logical_ir(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "plans.trb", SOURCE);
    let (ctx, module) = compile_frontend(db, source).expect("frontend should lower");
    let ir = print_module(&ctx, module.op());
    let lines = planned_lines(&ir);
    // `g()` reaches `State` only through its tail; `both` takes the caller's
    // handler twice; `counter` hides the handler its row names.
    assert_line(&lines, "tribute_control.call_indirect", "[{mask = ", &ir);
    assert_line(&lines, "tribute_control.call ", "[{dup = ", &ir);
    assert_line(&lines, "tribute_control.handle ", "[{mask = ", &ir);
    assert_eq!(lines.len(), 3, "{ir}");
}

#[salsa_test]
fn cps_legalization_carries_evidence_plans(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "plans.trb", SOURCE);
    let (ctx, module) = run_through_cps_lowering(db, source)
        .expect("CPS legalization should accept frontend output")
        .expect("frontend should lower");
    let ir = print_module(&ctx, module.op());
    let lines = planned_lines(&ir);
    assert_line(&lines, "call_indirect", "[{mask = ", &ir);
    assert_line(&lines, "func.tail_call ", "[{dup = ", &ir);
    assert_line(&lines, "ability.handle_dispatch", "[{mask = ", &ir);
    // Each selection also appears wherever a resumed continuation rebuilds
    // the layer that passes the evidence: the frame of the call, and each
    // installation of the handle.
    let count = |op: &str, plan: &str| {
        lines
            .iter()
            .filter(|line| line.contains(op) && line.contains(plan))
            .count()
    };
    let call_masks = count("call_indirect", "[{mask = ");
    let call_dups = count("call", "[{dup = ");
    let handle_masks = count("ability.handle_dispatch", "[{mask = ");
    assert_eq!(call_masks, 2, "{ir}");
    assert_eq!(call_dups, 2, "{ir}");
    assert!(handle_masks > 1, "{ir}");
    assert_eq!(lines.len(), call_masks + call_dups + handle_masks, "{ir}");
}
