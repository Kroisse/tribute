//! Typechecking selects each call's evidence by the callee's row position.
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::{
    SourceCst,
    typeck::{EvidenceStep, TypeCheckOutput},
};

#[salsa::tracked(returns(copy))]
fn checked(db: &dyn salsa::Database, source: SourceCst) -> TypeCheckOutput<'_> {
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let spans = parsed.span_map(db).clone();
    let resolved = tribute_front::resolve::resolve_module(db, parsed.module(db), spans.clone());
    tribute_front::typeck::typecheck_module(db, &resolved, spans)
}

/// Each non-identity selection as `source text: steps`, in source order.
fn plans(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    let source = SourceCst::from_source_str(db, "plans.trb", text);
    let output = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let spans = output.span_map(db);
    let mut plans: Vec<_> = output
        .expression_types(db)
        .evidence_plans
        .iter()
        .map(|(node, plan)| {
            let span = spans.get_or_default(*node);
            let steps: Vec<_> = plan
                .iter()
                .map(|step| match step {
                    EvidenceStep::Mask(instance) => {
                        format!("mask {}", instance.ability_id.name(db))
                    }
                    EvidenceStep::Dup(instance) => format!("dup {}", instance.ability_id.name(db)),
                })
                .collect();
            (
                span.start,
                format!("{}: {}", &text[span.start..span.end], steps.join(", ")),
            )
        })
        .collect();
    plans.sort();
    plans.into_iter().map(|(_, plan)| plan).collect()
}

const STATE: &str = r#"
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
"#;

#[salsa_test]
fn tail_callback_masks_the_callers_explicit_handler(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn twice_counted(f: fn() ->{e} Nil) ->{e} Nat {
    run_state(fn() {
        f()
        State::set(State::get())
        f()
        State::get()
    }, 0)
}

fn entry() -> Nat {
    run_state(fn() {
        twice_counted(fn() { State::set(State::get()) })
    }, 0)
}
"#
    );
    // Resumes in the handler arms and in their lambdas keep the evidence.
    assert_eq!(plans(db, &source), ["f(): mask State", "f(): mask State"]);
}

#[salsa_test]
fn explicit_instance_shared_with_the_tail_is_duplicated(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn both(g: fn() ->{e} Nil) ->{e, State(Nat)} Nil {
    g()
    State::set(1)
}

fn caller() ->{State(Nat)} Nil {
    both(fn() { State::set(2) })
}
"#
    );
    assert_eq!(
        plans(db, &source),
        ["g(): mask State", "both(fn() { State::set(2) }): dup State"]
    );
}

#[salsa_test]
fn handle_hides_an_outer_explicit_handler_of_its_instance(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn counter(comp: fn() ->{e, State(Nat)} a) ->{e, State(Nat)} a {
    handle comp() {
        do result { result }
        op State::get() { resume State::get() }
        op State::set(v) { resume Nil }
    }
}

fn tail_only(comp: fn() ->{e, State(Nat)} a) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { resume 0 }
        op State::set(v) { resume Nil }
    }
}
"#
    );
    let plans = plans(db, &source);
    assert_eq!(plans.len(), 1, "{plans:?}");
    assert!(plans[0].starts_with("handle comp()"), "{plans:?}");
    assert!(plans[0].ends_with(": mask State"), "{plans:?}");
}

#[salsa_test]
fn calls_through_explicit_rows_keep_the_evidence(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn bump() ->{State(Nat)} Nil {
    State::set(State::get())
}

fn twice() ->{State(Nat)} Nil {
    bump()
    bump()
}

fn entry() -> Nat {
    run_state(fn() {
        twice()
        State::get()
    }, 0)
}
"#
    );
    assert!(plans(db, &source).is_empty());
}

#[salsa_test]
fn ambient_io_is_never_selected(db: &salsa::DatabaseImpl) {
    let source = r#"use std::io::Io

fn apply(callback: fn() ->{e} Nil) ->{e, Io} Nil {
    callback()
}
"#;
    assert!(plans(db, source).is_empty());
}

#[salsa_test]
fn unconstrained_callee_tails_keep_the_evidence(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
pub mod labels {
    pub fn label(value: Nat) -> Nat {
        value
    }
}

fn bump() ->{State(Nat)} Nat {
    case State::get() {
        0 -> labels::label(0)
        value -> labels::label(value)
    }
}
"#
    );
    assert!(plans(db, &source).is_empty());
}
