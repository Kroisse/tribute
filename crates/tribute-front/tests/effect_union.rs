//! Semantic effect unions survive declaration collection and generalization.
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::{SourceCst, ast::TypeKind, typeck::TypeCheckOutput};

#[salsa::tracked(returns(copy))]
fn checked(db: &dyn salsa::Database, source: SourceCst) -> TypeCheckOutput<'_> {
    let parsed = tribute_front::query::parsed_ast(db, source).unwrap();
    let spans = parsed.span_map(db);
    let resolved = tribute_front::resolve::resolve_module(db, parsed.module(db), spans.clone());
    tribute_front::typeck::typecheck_module(db, &resolved, spans)
}

#[salsa_test]
fn named_rows_remain_distinct_and_union_survives_scheme(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "union.trb",
        r#"
fn combine(left: fn() ->{e1} Nil, right: fn() ->{e2} Nil) ->{e1, e2} Nil {
    left()
    right()
}
"#,
    );
    let output = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let scheme = output.function_types(db)[0].1;
    let TypeKind::Func { params, effect, .. } = scheme.body(db).kind(db) else {
        panic!()
    };
    let tail = |ty: tribute_front::ast::Type<'_>| match ty.kind(db) {
        TypeKind::Func { effect, .. } => effect.rest(db).unwrap(),
        _ => panic!(),
    };
    assert_ne!(tail(params[0]), tail(params[1]));
    assert!(!scheme.row_unions(db).is_empty());
    assert!(
        scheme
            .row_unions(db)
            .iter()
            .any(|union| union.result.rest(db) == effect.rest(db))
    );
}

#[salsa_test]
fn local_annotation_rows_share_signature_names_and_keep_unions(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_rows.trb",
        r#"
fn combine(left: fn() ->{e1} Nil, right: fn() ->{e2} Nil) ->{e1, e2} Nil {
    left()
    right()
}
fn local(left: fn() ->{e1} Nil, right: fn() ->{e2} Nil) ->{e1, e2} Nil {
    let action = fn(first: fn() ->{e1} Nil, second: fn() ->{e2} Nil) {
        combine(first, second)
    }
    action(left, right)
}
"#,
    );
    let output = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let scheme = output
        .function_types(db)
        .iter()
        .find(|(name, _)| *name == "local")
        .unwrap()
        .1;
    let TypeKind::Func { params, .. } = scheme.body(db).kind(db) else {
        panic!("function")
    };
    let tail = |ty: tribute_front::ast::Type<'_>| match ty.kind(db) {
        TypeKind::Func { effect, .. } => effect.rest(db).unwrap(),
        _ => panic!("callback"),
    };
    assert_ne!(tail(params[0]), tail(params[1]));
    assert!(!scheme.row_unions(db).is_empty());
    let TypeKind::Func {
        params: local_params,
        ..
    } = output.lambda_signatures(db)[0].1.function_type.kind(db)
    else {
        panic!("lambda")
    };
    assert_eq!(tail(local_params[0]), tail(params[0]));
    assert_eq!(tail(local_params[1]), tail(params[1]));
}

#[salsa_test]
fn local_row_name_refers_to_its_enclosing_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "shared_row.trb",
        r#"
fn pure(comp: fn() ->{} Nil) ->{} Nil { comp() }
fn shared(comp: fn() ->{e} Nil) ->{e} Nil {
    let unused = fn(other: fn() ->{e} Nil) { pure(other) }
    comp()
}
"#,
    );
    // The lambda's `e` is the signature's `e`, which is rigid in the body:
    // passing `other` where a pure callback is expected would close it.
    checked(db, source);
    let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert_eq!(
        errors,
        ["effect variable `e` in the signature of `shared` is closed in its body"],
    );
}

#[salsa_test]
fn handler_removes_effect_discovered_through_union_tails(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "handled_union.trb",
        r#"
ability Ping { op ping() -> Nil }
fn ping() ->{Ping} Nil { Ping::ping() }
fn combine(left: fn() ->{e1} Nil, right: fn() ->{e2} Nil) ->{e1, e2} Nil {
    left()
    right()
}
fn main() ->{} Nil {
    handle combine(ping, ping) {
        do value { value }
        op Ping::ping() { resume Nil }
    }
}
"#,
    );
    let _ = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
}

/// A handler removes only what the signature names in the handled row, so
/// the declared scheme needs no retained removal and every call checks
/// against it alone.
#[salsa_test]
fn handler_removal_is_declared_by_the_signature(db: &salsa::DatabaseImpl) {
    const ABILITIES: &str = r#"
ability Ping { op ping() -> Nil }
ability Other { op other() -> Nil }
fn ping() ->{Ping} Nil { Ping::ping() }
fn pure() ->{} Nil { Nil }
fn other() ->{Other} Nil { Other::other() }
"#;
    // A function value's row must match the parameter's exactly, so the
    // callbacks without `Ping` are lambdas.
    for (callback, valid) in [
        ("ping", true),
        ("fn() { pure() }", true),
        ("fn() { other() }", false),
    ] {
        let source = SourceCst::from_source_str(
            db,
            "removal_scheme.trb",
            &format!(
                r#"{ABILITIES}
fn handled(comp: fn() ->{{e, Ping}} Nil) ->{{e}} Nil {{
    handle comp() {{
        do value {{ value }}
        op Ping::ping() {{ resume Nil }}
    }}
}}
fn main() ->{{}} Nil {{ handled({callback}) }}
"#
            ),
        );
        let output = checked(db, source);
        let errors = checked::accumulated::<Diagnostic>(db, source);
        assert_eq!(errors.is_empty(), valid, "{callback}: {errors:?}");
        let scheme = output
            .function_types(db)
            .iter()
            .find(|(name, _)| *name == "handled")
            .unwrap()
            .1;
        assert!(scheme.row_removals(db).is_empty());
    }

    let undeclared = SourceCst::from_source_str(
        db,
        "removal_undeclared.trb",
        &format!(
            r#"{ABILITIES}
fn handled(comp: fn() ->{{e}} Nil) ->{{}} Nil {{
    handle comp() {{
        do value {{ value }}
        op Ping::ping() {{ resume Nil }}
    }}
}}
"#
        ),
    );
    let _ = checked(db, undeclared);
    let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, undeclared)
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert_eq!(
        errors,
        [
            "function 'handled' handles Ping from effect variable `e` without declaring it there; \
             add it to that effect row"
        ],
    );
}

/// Calls joined into a closed row are each checked against it: a second
/// call cannot hide the effects another call leaves unhandled.
#[salsa_test]
fn joined_calls_cannot_leak_effects_into_a_closed_row(db: &salsa::DatabaseImpl) {
    for calls in [
        "handled(ping)\n    handled(fn() { other() })",
        "handled(fn() { other() })\n    handled(ping)",
    ] {
        let source = SourceCst::from_source_str(
            db,
            "joined_calls.trb",
            &format!(
                r#"
ability Ping {{ op ping() -> Nil }}
ability Other {{ op other() -> Nil }}
fn ping() ->{{Ping}} Nil {{ Ping::ping() }}
fn other() ->{{Other}} Nil {{ Other::other() }}
fn handled(comp: fn() ->{{e, Ping}} Nil) ->{{e}} Nil {{
    handle comp() {{
        do value {{ value }}
        op Ping::ping() {{ resume Nil }}
    }}
}}
fn main() ->{{}} Nil {{
    {calls}
}}
"#
            ),
        );
        let _ = checked(db, source);
        let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, source)
            .into_iter()
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect();
        assert!(
            errors
                .iter()
                .any(|message| message.contains("expected `{}`, found `{Other")),
            "{calls}: {errors:?}"
        );
    }
}

#[salsa_test]
fn relay_preserves_writer_argument_through_nested_lambda(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "relay.trb",
        r#"
extern "C" fn print(value: Nat) -> Nil
ability Writer(w) { op tell(value: w) -> Nil }
fn use_writer() ->{Writer(Nat)} Nat { Writer::tell(5)
5 }
fn run_writer(comp: fn() ->{e, Writer(w)} a) ->{e} a {
 handle comp() { do result { result }
op Writer::tell(v) { run_writer(fn() { resume Nil }) } }
}
fn relay(comp: fn() ->{e} a) ->{e} a { comp() }
fn main() -> Nil { print(run_writer(fn() { relay(use_writer) })) }
"#,
    );
    let output = checked(db, source);
    let errors = checked::accumulated::<Diagnostic>(db, source);
    assert!(errors.is_empty(), "{errors:?}");
    let main = output
        .function_types(db)
        .iter()
        .find(|(name, _)| *name == "main")
        .unwrap()
        .1;
    assert!(
        main.type_params(db).is_empty(),
        "main must be concrete: {main:?}"
    );
    let references = &output.expression_types(db).function_instances;
    assert!(
        references
            .iter()
            .any(|(_, instance)| instance.function.qualified(db)
                == trunk_ir::Symbol::new("run_writer")
                && instance
                    .type_arguments
                    .iter()
                    .all(|ty| matches!(ty.kind(db), TypeKind::Nat))
                && instance.type_arguments.len() == 2),
        "{references:?}"
    );
}

/// A handler removes a label that one tail of a multi-tail callee supplies,
/// whatever the other tails are and wherever the call is made.
#[salsa_test]
fn handler_removes_a_label_supplied_through_one_tail(db: &salsa::DatabaseImpl) {
    const PRELUDE: &str = r#"
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
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
}
fn inc() ->{State(Int)} Nil { State::set(State::get() + +1) }
"#;
    for (name, body) in [
        (
            "label_then_tail",
            r#"
fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        both(fn() { State::set(State::get() + +1) }, h)
        State::get()
    }, +0)
}
"#,
        ),
        (
            "tail_then_label",
            r#"
fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        both(h, fn() { State::set(State::get() + +1) })
        State::get()
    }, +0)
}
"#,
        ),
        (
            "label_in_both_tails",
            r#"
fn count() ->{} Int {
    run_state(fn() {
        both(fn() { State::set(State::get() + +1) }, fn() { State::set(State::get() + +10) })
        State::get()
    }, +0)
}
"#,
        ),
        (
            "closed_row",
            r#"
fn twice() ->{State(Int)} Nil { both(inc, inc) }
fn once() ->{State(Int)} Nil { both(inc, fn() { Nil }) }
"#,
        ),
    ] {
        let source = SourceCst::from_source_str(db, name, &format!("{PRELUDE}{body}"));
        let _ = checked(db, source);
        let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, source)
            .into_iter()
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
    }
}

/// A label supplied through one tail of a multi-tail callee still reaches the
/// caller's row when nothing handles it.
#[salsa_test]
fn unhandled_label_of_one_tail_stays_in_the_caller_row(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "unhandled_tail_label.trb",
        r#"
ability Ping { op ping() -> Nil }
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
}
fn leak(h: fn() ->{t} Nil) ->{t} Nil {
    both(fn() { Ping::ping() }, h)
}
"#,
    );
    let _ = checked(db, source);
    let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert_eq!(errors, ["function 'leak' uses undeclared effects: Ping"]);
}

/// A function value's labels reach the row tail it is passed for before the
/// enclosing closed row is equated with the accumulated one.
#[salsa_test]
fn function_value_labels_reach_a_tail_in_a_closed_row(db: &salsa::DatabaseImpl) {
    const PRELUDE: &str = r#"
ability Ask { op ask() -> Int }
fn apply(x: a, f: fn(a) ->{eff} b) ->{eff} b { f(x) }
fn bump(n: Int) ->{Ask} Int { n + Ask::ask() }
"#;
    for (name, body) in [
        ("named", "fn run() ->{Ask} Int { apply(+1, bump) }"),
        (
            "local",
            "fn run() ->{Ask} Int {\n    let f = bump\n    apply(+1, f)\n}",
        ),
        (
            "repeated",
            "fn run() ->{Ask} Int {\n    let _ = apply(+1, bump)\n    apply(+2, bump)\n}",
        ),
        (
            "joined",
            "fn run(flag: Bool) ->{Ask} Int {\n    let f = case flag {\n        True -> bump\n        False -> fn(n) { Ask::ask() + n }\n    }\n    apply(+1, f)\n}",
        ),
        (
            "lambda body",
            "fn run() ->{Ask} Int {\n    let go = fn(n: Int) ->{Ask} Int { apply(n, bump) }\n    go(+1)\n}",
        ),
        (
            "typed only by its uses",
            "fn needs(f: fn(Int) ->{Ask} Int) ->{Ask} Int { f(+1) }\nfn run() ->{Ask} Int {\n    let fs = []\n    case fs {\n        [g, ..] -> {\n            let _ = apply(+1, g)\n            needs(g)\n        }\n        [] -> +0\n    }\n}",
        ),
        (
            "beside a signature tail",
            "fn run(f: fn(Int) ->{e} Int) ->{e, Ask} Int {\n    let _ = apply(+1, f)\n    apply(+2, bump)\n}",
        ),
    ] {
        let text = format!("{PRELUDE}{body}\n");
        let source = SourceCst::from_source_str(db, "function_value.trb", &text);
        let _ = checked(db, source);
        let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, source)
            .into_iter()
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
    }
}

/// A function value's labels that the enclosing closed row does not declare
/// are reported against that row.
#[salsa_test]
fn function_value_labels_outside_a_closed_row_are_rejected(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "function_value_outside.trb",
        r#"
ability Ask { op ask() -> Int }
ability Tell { op tell(n: Int) -> Nil }
fn apply(x: a, f: fn(a) ->{eff} b) ->{eff} b { f(x) }
fn bump(n: Int) ->{Ask} Int { n + Ask::ask() }
fn run() ->{Tell} Int { apply(+1, bump) }
"#,
    );
    let _ = checked(db, source);
    let errors: Vec<_> = checked::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert_eq!(
        errors,
        ["type error in function 'run': effect mismatch: expected `{Tell}`, found `{Ask, Tell}`"]
    );
}
