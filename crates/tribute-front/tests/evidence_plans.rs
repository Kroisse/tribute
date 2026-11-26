//! Typechecking selects each call's evidence by the callee's row position.
use itertools::Itertools;
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::analysis;
use tribute_front::{
    SourceCst,
    typeck::{EvidenceStep, TypeCheckOutput},
};

fn checked(db: &dyn salsa::Database, source: SourceCst) -> TypeCheckOutput<'_> {
    *analysis::analyze(db, source, *analysis::builtins(db)).unwrap()
}

fn diagnostics(db: &dyn salsa::Database, source: SourceCst) -> Vec<Diagnostic<'_>> {
    analysis::diagnostics(db, source, *analysis::builtins(db))
}

fn describe(db: &dyn salsa::Database, step: &EvidenceStep<'_>) -> String {
    match step {
        EvidenceStep::Mask(instance) => format!("mask {}", instance.ability_id.name(db)),
        EvidenceStep::Dup(instance) => format!("dup {}", instance.ability_id.name(db)),
        EvidenceStep::Push(instance) => format!("push {}", instance.ability_id.name(db)),
        EvidenceStep::Select(index) => format!("select {index}"),
        EvidenceStep::Fresh => "fresh".to_owned(),
        EvidenceStep::Tails(plans) => format!(
            "tails {}",
            plans.iter().format_with(" ", |plan, f| f(&format_args!(
                "[{}]",
                plan.iter().map(|step| describe(db, step)).format(", ")
            )))
        ),
    }
}

/// Each non-identity selection as `source text: steps`, in source order.
fn plans(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    let source = SourceCst::from_source_str(db, "plans.trb", text);
    let output = checked(db, source);
    let errors = diagnostics(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let spans = tribute_front::query::span_map(db, source).expect("parsed source");
    let mut plans: Vec<_> = output
        .expression_types(db)
        .evidence_plans
        .iter()
        .map(|(node, plan)| {
            let span = spans.get_or_default(*node);
            let steps = plan.iter().map(|step| describe(db, step)).join(", ");
            (span.start, format!("{}: {}", &text[span.range()], steps))
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
        op State::get() { become run_state(fn() { resume init }, init) }
        op State::set(v) { become run_state(fn() { resume Nil }, v) }
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
        op State::get() { become tail_only(fn() resume 0) }
        op State::set(v) { become tail_only(fn() resume Nil) }
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

#[salsa_test]
fn each_tail_of_a_union_takes_its_own_selection(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
pub mod Nat { pub extern "intrinsic" fn (+)(left: Nat, right: Nat) -> Nat }
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Nat {
    run_state(fn() {
        both(fn() { State::set(State::get() + 1) }, h)
        State::get()
    }, 0)
}
"#
    );
    let plans = plans(db, &source);
    assert!(plans.contains(&"f(): select 0".to_owned()), "{plans:#?}");
    assert!(plans.contains(&"g(): select 1".to_owned()), "{plans:#?}");
    assert!(
        plans
            .iter()
            .any(|plan| plan.starts_with("both(") && plan.ends_with(": tails [] [mask State]")),
        "{plans:#?}"
    );
}

#[salsa_test]
fn selected_tail_takes_the_explicit_handlers_a_callee_names(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn stateful(f: fn() ->{e1, State(Nat)} Nil, g: fn() ->{e2} Nil) ->{e1, e2, State(Nat)} Nil {
    f()
    g()
}
"#
    );
    let plans = plans(db, &source);
    assert_eq!(plans, ["f(): select 0, push State", "g(): select 1"]);
}

const PING: &str = r#"
ability Ping { op ping() -> Nat }
ability Other { op other() -> Nil }
"#;

/// A `resume` passes the evidence of the scope it is written in, wherever
/// that is: the resumed body takes its handlers from there.
#[salsa_test]
fn resume_selects_from_the_scope_it_is_written_in(db: &salsa::DatabaseImpl) {
    let resumes = [
        // The arm runs with the handle's outer evidence, which already holds
        // the handled label for the resumed body.
        ("resume 0", Vec::<&str>::new()),
        ("let k = fn() { resume 0 }\n        k()", vec![]),
        // A handler around the `resume` that the resumed body does not name
        // must not catch what reaches it through the body's row tail.
        (
            "handle { Other::other()\n resume 0 } { op Other::other() { 0 } }",
            vec!["resume 0: mask Other"],
        ),
        (
            "let k = fn() { Other::other()\n resume 0 }\n        handle k() { op Other::other() { 0 } }",
            vec!["resume 0: mask Other"],
        ),
    ];
    for (resume, expected) in resumes {
        let source = format!(
            r#"{PING}
fn handled(comp: fn() ->{{e, Ping}} Nat) ->{{e, Ping}} Bool {{
    handle comp() {{
        do n {{ False }}
        op Ping::ping() {{
            case {{ {resume} }} {{
                0 -> True
                _ -> False
            }}
        }}
    }}
}}
"#
        );
        let (handle, plans): (Vec<_>, Vec<_>) = plans(db, &source)
            .into_iter()
            .partition(|plan| plan.starts_with("handle comp()"));
        // The handled label is explicit in the outer row, so the handle
        // hides the outer handler from its body.
        assert!(
            matches!(handle.as_slice(), [plan] if plan.ends_with(": mask Ping")),
            "{resume}: {handle:?}"
        );
        assert_eq!(plans, expected, "{resume}");
    }
}

fn loop_plans(db: &dyn salsa::Database, both: &str) -> Vec<String> {
    let source = format!(
        "{STATE}{}",
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil, done: Bool) ->{e1, e2} Nil {
    f()
    case done {
        True -> Nil
        False -> BOTH
    }
}
"#
        .replace("BOTH", both),
    );
    plans(db, &source)
        .into_iter()
        .filter(|plan| plan.starts_with("both("))
        .collect()
}

#[salsa_test]
fn a_callee_taking_the_callers_tails_in_order_has_no_plan(db: &salsa::DatabaseImpl) {
    assert_eq!(loop_plans(db, "both(f, g, True)"), Vec::<String>::new());
}

#[salsa_test]
fn a_callee_taking_the_callers_tails_swapped_selects_them(db: &salsa::DatabaseImpl) {
    assert_eq!(
        loop_plans(db, "both(g, f, True)"),
        ["both(g, f, True): tails [select 1] [select 0]"]
    );
}

#[salsa_test]
fn a_let_bound_lambda_with_an_open_row_has_no_plan(db: &salsa::DatabaseImpl) {
    let source = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    let go = fn(c: fn() ->{e, State(s)} a, i: s) ->{e} a {
        handle c() {
            do result { result }
            op State::get() { run_state(fn() { resume i }, i) }
            op State::set(v) { run_state(fn() { resume Nil }, v) }
        }
    }
    go(comp, init)
}
"#;
    assert!(
        plans(db, source)
            .iter()
            .all(|plan| !plan.starts_with("go(")),
        "{:#?}",
        plans(db, source)
    );
}

/// A tail whose row is closed takes a fresh evidence with the handlers that
/// row names: the caller's evidence holds handlers it cannot reach, and the
/// handler the callee pushes on the selected tail would stack on one of them.
/// A caller whose closed row names the same instances passes its own.
#[salsa_test]
fn a_closed_tail_takes_a_fresh_evidence(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn two(f: fn(Int) ->{State(Int), e1} Int, g: fn() ->{e2} Nil, n: Int) ->{State(Int), e1, e2} Int {
    g()
    f(n)
}

fn closed(n: Int) ->{State(Int)} Int {
    two(closed, fn() { Nil }, n)
}

fn open(n: Int, h: fn() ->{r} Nil) ->{State(Int), r} Int {
    two(closed, h, n)
}

fn same(f: fn(Int) ->{e1} Int, g: fn() ->{e2} Nil, n: Int) ->{e1, e2} Int {
    g()
    f(n)
}

fn passes(n: Int) ->{State(Int)} Int {
    same(passes, fn() { State::set(n) }, n)
}
"#
    );
    let plans = plans(db, &source);
    assert!(
        plans.contains(&"f(n): select 0, push State".to_owned()),
        "{plans:#?}"
    );
    assert!(
        plans.contains(&"two(closed, fn() { Nil }, n): tails [fresh] [fresh]".to_owned()),
        "{plans:#?}"
    );
    assert!(
        plans.contains(&"two(closed, h, n): tails [fresh] [mask State]".to_owned()),
        "{plans:#?}"
    );
    assert!(
        plans.contains(&"same(passes, fn() { State::set(n) }, n): tails [] []".to_owned()),
        "{plans:#?}"
    );
}

/// A callee whose tails all take an empty selection keeps its tails when the
/// caller does not pass its own.
#[salsa_test]
fn a_callee_dropping_every_tail_keeps_its_tails(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn both2(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Int {
    handle {
        f()
        g()
        State::get()
    } {
        do r { r }
        op State::get() { +1000 }
        op State::set(v) { +2000 }
    }
}

fn caller() ->{State(Int)} Int {
    both2(fn() { State::set(State::get()) }, fn() { State::set(State::get()) })
}
"#
    );
    let plans = plans(db, &source);
    assert!(
        plans
            .iter()
            .any(|plan| plan.starts_with("both2(") && plan.ends_with(": tails [] []")),
        "{plans:#?}"
    );
}

/// A multi-tail function used as a value selects its tails for the value's
/// entry, with the value's row as the caller's: a call through the value
/// attaches none.
#[salsa_test]
fn a_multi_tail_function_value_selects_its_tails_from_its_own_row(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Int {
    f()
    g()
    +0
}

fn merged(h: fn(fn() ->{State(Int)} Nil, fn() ->{State(Int)} Nil) ->{State(Int)} Int) ->{State(Int)} Int {
    h(fn() { State::set(State::get()) }, fn() { State::set(State::get()) })
}

fn split(
    h: fn(fn() ->{State(Int), r} Nil, fn() ->{r} Nil) ->{State(Int), r} Int,
    k: fn() ->{r} Nil
) ->{State(Int), r} Int {
    h(fn() { State::set(State::get()) }, k)
}

fn caller(k: fn() ->{t} Nil) ->{State(Int), t} Int {
    merged(both)
    both(fn() { State::set(State::get()) }, k)
    split(both, k)
}
"#
    );
    let plans = plans(db, &source);
    let references: Vec<_> = plans
        .iter()
        .filter(|plan| plan.starts_with("both:"))
        .collect();
    assert_eq!(
        references,
        ["both: tails [] []", "both: tails [] [mask State]"],
        "{plans:#?}"
    );
    assert!(
        !plans.iter().any(|plan| plan.starts_with("h(")),
        "{plans:#?}"
    );
}

/// A value whose type keeps the multi-tail function's row is still entered
/// through the function's entry: the call through it selects for the
/// value's merged row and attaches no tails, so nothing is selected twice.
#[salsa_test]
fn a_call_through_a_multi_tail_value_selects_for_its_merged_row(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
fn stateful(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2, State(Int)} Nil {
    f()
    g()
    State::set(State::get())
}

fn count_calls(k: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        let h = { stateful }
        h(fn() { State::set(State::get()) }, k)
        State::get()
    }, +0)
}
"#
    );
    let plans = plans(db, &source);
    assert!(
        plans
            .iter()
            .any(|plan| plan == "stateful: dup State, tails [] [mask State]"),
        "{plans:#?}"
    );
    assert!(
        !plans.iter().any(|plan| plan.starts_with("h(")),
        "{plans:#?}"
    );
}

/// A name that several `use`s give different functions, used as a value,
/// selects the tails of the function its type selects.
#[salsa_test]
fn a_selected_multi_tail_function_value_selects_its_tails(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{STATE}{}",
        r#"
mod a {
    pub fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Int {
        f()
        g()
        +0
    }
}

mod b {
    pub fn both(n: Int) -> Int { n }
}

use a::both
use b::both

fn merged(h: fn(fn() ->{State(Int)} Nil, fn() ->{State(Int)} Nil) ->{State(Int)} Int) ->{State(Int)} Int {
    h(fn() { State::set(State::get()) }, fn() { State::set(State::get()) })
}

fn caller() ->{State(Int)} Int {
    merged(both)
}
"#
    );
    let plans = plans(db, &source);
    assert!(
        plans.iter().any(|plan| plan == "both: tails [] []"),
        "{plans:#?}"
    );
}
