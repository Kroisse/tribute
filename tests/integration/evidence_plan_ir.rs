//! Typechecked evidence selections reach source-logical IR and survive CPS
//! legalization on the operations that pass the evidence.

use salsa_test_macros::salsa_test;
use tribute::pipeline::{compile_frontend, run_through_cps_lowering};
use tribute_front::SourceCst;

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

fn run_state(comp: fn() ->{State(Nat)} Nat) -> Nat {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume 0 }) }
        op State::set(v) { run_state(fn() { resume Nil }) }
    }
}

fn main() -> Nil {
    let _ = run_state(fn() {
        counter(fn() {
            caller()
            State::get()
        })
    })
    Nil
}
"#;

/// The lines of `ir` that carry an evidence selection. A generic operation
/// with regions writes its attributes after them, so that line is prefixed
/// with the name the operation opened its regions with.
fn planned_lines(ir: &str) -> Vec<String> {
    let mut open = Vec::new();
    let mut lines = Vec::new();
    for line in ir.lines().map(str::trim) {
        let mut planned = line.to_owned();
        if line.starts_with("})") {
            let name: &str = open
                .pop()
                .expect("a generic region list closes what it opened");
            planned = format!("{name} {line}");
        }
        if line.ends_with("({") {
            open.push(line.split('"').nth(1).expect("a generic operation name"));
        }
        if planned.contains("evidence_plan") {
            lines.push(planned);
        }
    }
    lines
}

fn assert_line(lines: &[String], op: &str, plan: &str, ir: &str) {
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
    let ir = module.op().display(&ctx).to_string();
    // An instance repeats the lines of its definition.
    let mut lines = planned_lines(&ir);
    lines.sort_unstable();
    lines.dedup();
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
    let ir = module.op().display(&ctx).to_string();
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

/// A program whose calls all pass their evidence unchanged carries no
/// selection, so evidence resolution adds no `effect.mask` or `effect.dup`.
#[salsa_test]
fn identity_selections_leave_no_evidence_plan(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn bump() ->{State(Nat)} Nat {
    State::set(State::get() + 1)
    State::get()
}

fn main() -> Nil {
    let _ = run_state(fn() { bump() }, 0)
    Nil
}
"#;
    let source = SourceCst::from_source_str(db, "identity.trb", code);
    let (ctx, module) = run_through_cps_lowering(db, source)
        .expect("CPS legalization should accept frontend output")
        .expect("frontend should lower");
    let ir = module.op().display(&ctx).to_string();
    assert!(planned_lines(&ir).is_empty(), "{ir}");
}
