//! Module-level functions are checked against complete, final signatures.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "signatures.trb", text))
}

#[salsa_test]
fn parameter_and_return_types_are_required(db: &salsa::DatabaseImpl) {
    assert_eq!(
        errors(db, "fn add(x, y: Nat) { y }\n\nfn main() -> Nil { }\n"),
        [
            "function `add` needs a type annotation for parameter `x`",
            "function `add` needs a return type",
        ],
    );
    assert_eq!(
        errors(db, "fn main() { }\n"),
        ["function `main` needs a return type"],
    );
}

/// An argument lambda checked against the error type of an unannotated
/// parameter keeps its parameter types when revisited, so a capturing `let`
/// closure rebound by another `let` is reported, not a type checker panic.
#[salsa_test]
fn lambda_argument_to_an_unannotated_parameter_is_reported(db: &salsa::DatabaseImpl) {
    assert_eq!(
        errors(
            db,
            "fn beta(f) -> Nil { Nil }\n\n\
             fn main() -> Nil {\n    beta(fn(y) {\n        let bar = fn(z) { y }\n        \
             let qux = bar\n    })\n}\n"
        ),
        ["function `beta` needs a type annotation for parameter `f`"],
    );
}

/// A lambda argument to a parameter of a non-function type is checked again
/// after the mismatch is constrained. Its `let`s keep the schemes of their
/// first visit, so the mismatch is reported, not a type checker panic.
#[salsa_test]
fn lambda_argument_to_a_non_function_parameter_is_reported(db: &salsa::DatabaseImpl) {
    assert_eq!(
        errors(
            db,
            "fn beta(f: Int) -> Nil { Nil }\n\n\
             fn main() -> Nil {\n    beta(fn(y) {\n        let bar = fn(z) { y }\n        \
             let qux = bar\n    })\n}\n"
        ),
        ["type error at call site in function 'main': expected `Int`, found `fn(_) -> Nil`"],
    );
}

#[salsa_test]
fn signature_type_variables_are_rigid(db: &salsa::DatabaseImpl) {
    assert_eq!(
        errors(db, "fn first(x: a) -> a { 1 }\n\nfn main() -> Nil { }\n"),
        ["type variable `a` in the signature of `first` is `Nat` in its body"],
    );
    assert_eq!(
        errors(
            db,
            "fn same(x: a, y: b) -> a { y }\n\nfn main() -> Nil { }\n"
        ),
        ["type variables `a` and `b` in the signature of `same` are the same type in its body"],
    );
}

const ASK: &str = "ability Ask {\n    op ask() -> Nat\n}\n\n";

/// An omitted row is exactly a fresh `->{e}`, and an explicit open row
/// admits only its named effects; neither is widened by the body.
#[salsa_test]
fn effect_rows_are_not_widened_by_the_body(db: &salsa::DatabaseImpl) {
    for signature in [
        "fn f() -> Nat",
        "fn f() ->{e} Nat",
        "fn f() ->{State(Nat), e} Nat",
    ] {
        let source = format!(
            "{ASK}ability State(s) {{\n    op get() -> s\n}}\n\n\
             {signature} {{ Ask::ask() }}\n\nfn main() -> Nil {{ }}\n"
        );
        assert_eq!(
            errors(db, &source),
            ["function 'f' uses undeclared effects: Ask"],
            "{signature}"
        );
    }
    let declared =
        format!("{ASK}fn f() ->{{Ask}} Nat {{ Ask::ask() }}\n\nfn main() -> Nil {{ }}\n");
    assert!(errors(db, &declared).is_empty());
}

/// A call written with method syntax performs the callee's effects like the
/// qualified call, including those of a callback it is given.
#[salsa_test]
fn a_method_call_performs_the_effects_of_its_callee(db: &salsa::DatabaseImpl) {
    let declarations = format!(
        "{ASK}struct Box {{ value: Nat }}\n\n\
         fn ask(b: Box) ->{{Ask}} Nat {{ Ask::ask() }}\n\n\
         fn apply(b: Box, g: fn(Nat) ->{{e}} Nat) ->{{e}} Nat {{ g(b.value) }}\n\n"
    );
    let check = |row: &str, body: &str| {
        errors(
            db,
            &format!(
                "{declarations}fn f(b: Box) ->{row} Nat {{ {body} }}\n\nfn main() -> Nil {{ }}\n"
            ),
        )
    };
    for (method, qualified) in [
        ("b.ask()", "ask(b)"),
        (
            "b.apply(fn(n) Ask::ask() + n)",
            "apply(b, fn(n) Ask::ask() + n)",
        ),
        (
            "b.value::modify(fn(n) Ask::ask() + n).value",
            "Box::value::modify(b, fn(n) Ask::ask() + n).value",
        ),
    ] {
        // A call resolved after solving reports the effect against the
        // function's row instead of the call site.
        for call in [method, qualified] {
            let undeclared = check("{}", call);
            assert!(
                matches!(&undeclared[..], [error] if error.contains("Ask")),
                "{call}: {undeclared:?}"
            );
        }
        assert!(check("{Ask}", method).is_empty(), "{method}");
    }
    assert!(check("{}", "b.apply(fn(n) n)").is_empty());
}

/// Several row-polymorphic calls in a body with a closed row leave their
/// callbacks' rows as the callbacks wrote them; an effect the row does not
/// name is still reported.
#[salsa_test]
fn calls_into_a_closed_row_keep_their_callbacks_effects(db: &salsa::DatabaseImpl) {
    let check = |row: &str| {
        errors(
            db,
            &format!(
                "{ASK}ability Tell {{\n    op tell() -> Nat\n}}\n\n\
                 fn apply(x: Nat, g: fn(Nat) ->{{e}} Nat) ->{{e}} Nat {{ g(x) }}\n\n\
                 fn f() ->{row} Nat {{\n    \
                     let a = apply(1, fn(n) n)\n    \
                     let b = apply(a, fn(n) n + Tell::tell())\n    \
                     apply(b, fn(n) n + Ask::ask())\n}}\n\n\
                 fn main() -> Nil {{ }}\n"
            ),
        )
    };
    assert!(check("{Ask, Tell}").is_empty());
    let undeclared = check("{Tell}");
    assert!(
        matches!(&undeclared[..], [error] if error.contains("Ask")),
        "{undeclared:?}"
    );
}

/// A caller sees only the callee's declaration, so the order in which the
/// functions are declared does not change the diagnostics.
#[salsa_test]
fn declaration_order_does_not_change_the_result(db: &salsa::DatabaseImpl) {
    let callee = "fn f() -> Nat {\n    Ask::ask()\n}\n\n";
    let caller = "fn g() -> Nat {\n    f()\n}\n\n";
    let main = "fn main() -> Nil {\n    let _ = g()\n}\n";
    let expected = ["function 'f' uses undeclared effects: Ask"];
    assert_eq!(
        errors(db, &format!("{ASK}{callee}{caller}{main}")),
        expected
    );
    assert_eq!(
        errors(db, &format!("{ASK}{caller}{callee}{main}")),
        expected
    );
}

#[salsa_test]
fn self_recursion_uses_the_declared_signature(db: &salsa::DatabaseImpl) {
    let source = "fn want_bool(b: Bool) -> Nat { 0 }\n\n\
                  fn f(n: Nat) -> Nat {\n    case n {\n        0 -> 1\n        _ -> want_bool(f(n - 1))\n    }\n}\n\n\
                  fn main() -> Nil { }\n";
    let errors = errors(db, source);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("expected `Bool`, found `Nat`"),
        "{errors:?}"
    );
}

/// Omitted rows are distinct variables: calling a callback propagates its
/// effects only through a row variable the signature shares with it.
#[salsa_test]
fn callback_effects_need_a_shared_row_variable(db: &salsa::DatabaseImpl) {
    assert_eq!(
        errors(
            db,
            "fn apply(f: fn(Nat) -> Nat, x: Nat) -> Nat { f(x) }\n\nfn main() -> Nil { }\n"
        ),
        [
            "function 'apply' performs the effects of an omitted effect row without declaring them; \
             use the same effect variable in its own effect row"
        ],
    );
    let shared = format!(
        "{ASK}fn apply(f: fn(Nat) ->{{e}} Nat, x: Nat) ->{{e}} Nat {{ f(x) }}\n\n\
         fn main() -> Nil {{\n    let _ = handle apply(fn(x) Ask::ask() + x, 1) {{\n        do v {{ v }}\n        op Ask::ask() {{ resume 41 }}\n    }}\n}}\n"
    );
    assert!(errors(db, &shared).is_empty());
}

/// Nothing handles an effect that escapes root `main`; only the ambient
/// `Io` may be declared there.
#[salsa_test]
fn root_main_reports_unhandled_and_undeclared_effects(db: &salsa::DatabaseImpl) {
    let unhandled = format!("{ASK}fn main() -> Nil {{\n    let _ = Ask::ask()\n}}\n");
    assert_eq!(
        errors(db, &unhandled),
        ["function 'main' has unhandled effects: Ask"],
    );
    let step = "use std::io::Io\n\nfn step() ->{Io} Nil { step() }\n\n";
    assert_eq!(
        errors(db, &format!("{step}fn main() -> Nil {{\n    step()\n}}\n")),
        ["function 'main' uses undeclared effects: Io"],
    );
    assert!(
        errors(
            db,
            &format!("{step}fn main() ->{{Io}} Nil {{\n    step()\n}}\n")
        )
        .is_empty()
    );
}

const POLYMORPHIC: [&str; 3] = [
    "fn id(x: a) -> a { x }\n\n",
    "fn apply(f: fn(a) ->{e} b, x: a) ->{e} b { f(x) }\n\n",
    "fn twice(n: Nat) -> Nat {\n    apply(fn(m) { id(m) + m }, id(n))\n}\n\n",
];

/// The checked schemes and every call's instance, keyed by name.
fn checked_signatures(db: &dyn salsa::Database, text: &str) -> (Vec<String>, Vec<String>) {
    let source = SourceCst::from_source_str(db, "signatures.trb", text);
    assert_eq!(errors(db, text), Vec::<String>::new());
    let output = tribute_front::query::type_check_output(db, source).unwrap();
    let mut schemes: Vec<_> = output
        .function_types(db)
        .iter()
        .map(|(name, scheme)| format!("{name}: {scheme:?}"))
        .collect();
    schemes.sort();
    let mut instances: Vec<_> = output
        .expression_types(db)
        .function_instances
        .iter()
        .map(|(_, instance)| {
            format!(
                "{} {:?} {:?} {:?} {:?}",
                instance.function.qualified(db),
                instance.scheme,
                instance.type_arguments,
                instance.row_arguments,
                instance.callable,
            )
        })
        .collect();
    instances.sort();
    (schemes, instances)
}

/// Each function is checked against declarations alone, so the order of the
/// declarations changes neither a scheme nor a call's instance.
#[salsa_test]
fn declaration_order_does_not_change_schemes_or_instances(db: &salsa::DatabaseImpl) {
    let main = "fn main() -> Nil {\n    let _ = twice(1)\n}\n";
    let [id, apply, twice] = POLYMORPHIC;
    let forward = checked_signatures(db, &format!("{id}{apply}{twice}{main}"));
    let backward = checked_signatures(db, &format!("{main}{twice}{apply}{id}"));
    assert_eq!(forward, backward);
}

/// A call instantiates the callee's declared scheme, which checking the
/// callee's body never replaces.
#[salsa_test]
fn instances_use_the_declared_scheme(db: &salsa::DatabaseImpl) {
    let text = format!(
        "{}{}{}fn main() -> Nil {{\n    let _ = twice(1)\n}}\n",
        POLYMORPHIC[0], POLYMORPHIC[1], POLYMORPHIC[2]
    );
    let source = SourceCst::from_source_str(db, "signatures.trb", &text);
    assert_eq!(errors(db, &text), Vec::<String>::new());
    let output = tribute_front::query::type_check_output(db, source).unwrap();
    let instances = &output.expression_types(db).function_instances;
    assert!(instances.len() >= 4, "{instances:?}");
    for (_, instance) in instances {
        let name = instance.function.qualified(db);
        let (_, scheme) = output
            .function_types(db)
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .unwrap_or_else(|| panic!("no scheme for {name}"));
        assert_eq!(instance.scheme, *scheme, "{name}");
    }
}

/// A handler's relations over the signature rows only define body-local rows,
/// so they stay out of the scheme instead of quantifying those rows.
#[salsa_test]
fn body_relations_stay_out_of_the_scheme(db: &salsa::DatabaseImpl) {
    let text = r#"ability State(s) {
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

fn main() -> Nil { }
"#;
    let source = SourceCst::from_source_str(db, "signatures.trb", text);
    assert_eq!(errors(db, text), Vec::<String>::new());
    let output = tribute_front::query::type_check_output(db, source).unwrap();
    let (_, scheme) = output
        .function_types(db)
        .iter()
        .find(|(name, _)| *name == "run_state")
        .unwrap();
    assert_eq!(scheme.effect_params(db).len(), 1);
    assert!(scheme.row_unions(db).is_empty());
    assert!(scheme.row_removals(db).is_empty());
}

/// A field's getter takes no arguments, whether or not the receiver's type
/// is known when the call is inferred.
#[salsa_test]
fn a_field_read_with_arguments_is_reported(db: &salsa::DatabaseImpl) {
    let text = r#"struct Name { text: String }

fn id(x: a) -> a { x }

fn known(name: Name) -> String { name.text("a") }
fn solved_later(name: Name) -> String { id(name).text("a", "b") }
"#;
    assert_eq!(
        errors(db, text),
        [
            "call arity mismatch: expected 1 arguments, found 2",
            "call arity mismatch: expected 1 arguments, found 3",
        ]
    );
}

/// A method call whose receiver is typed only after solving performs its
/// callee's effects, and those of a callback it is given, in the callable
/// that makes the call.
#[salsa_test]
fn a_method_call_resolved_after_solving_performs_its_effects(db: &salsa::DatabaseImpl) {
    let declarations = format!(
        "{ASK}ability Tell {{\n    op tell() -> Nat\n}}\n\n\
         struct Box {{ value: Nat }}\n\n\
         fn bump(n: Nat) ->{{Ask}} Nat {{ n + Ask::ask() }}\n\n\
         fn asks(b: Box) ->{{Ask}} Nat {{ Ask::ask() }}\n\n\
         fn each(b: Box, g: fn(Nat) ->{{e}} Nat) ->{{e}} Nat {{ g(b.value) }}\n\n\
         fn twice(g: fn(Box) ->{{e}} Nat) ->{{e}} Nat {{\n    \
             g(Box {{ value: 1 }}) + g(Box {{ value: 2 }})\n}}\n\n"
    );
    let check = |function: &str| {
        errors(
            db,
            &format!("{declarations}{function}\n\nfn main() -> Nil {{ }}\n"),
        )
    };
    for accepted in [
        "fn f() ->{Ask} Nat {\n    let go = fn(c) { c.asks() }\n    go(Box { value: 1 })\n}",
        "fn f() ->{Ask} Nat {\n    let go = fn(c) { c.each(bump) }\n    go(Box { value: 1 })\n}",
        "fn f() ->{Ask} Nat {\n    let go = fn(c) {\n        let k = Ask::ask()\n        c.each(bump) + k\n    }\n    go(Box { value: 1 })\n}",
        "fn f() ->{Ask, Tell} Nat {\n    let go = fn(c) ->{Ask} Nat { c.asks() + Ask::ask() }\n    Tell::tell() + go(Box { value: 1 })\n}",
        "fn f() ->{Ask} Nat { twice(fn(c) { c.asks() }) }",
        "fn f() ->{} Nat {\n    let go = fn(c) { c.each(fn(n) { n + 1 }) }\n    go(Box { value: 1 })\n}",
    ] {
        let errors = check(accepted);
        assert!(errors.is_empty(), "{accepted}: {errors:?}");
    }
    for rejected in [
        "fn f() -> Nat {\n    let go = fn(c) { c.asks() }\n    go(Box { value: 1 })\n}",
        "fn f() ->{Tell} Nat {\n    let go = fn(c) ->{Tell} Nat { Tell::tell() + c.asks() }\n    go(Box { value: 1 })\n}",
        "fn f() ->{Tell} Nat { twice(fn(c) { c.each(bump) }) }",
    ] {
        let errors = check(rejected);
        assert!(
            !errors.is_empty() && errors.iter().all(|error| error.contains("Ask")),
            "{rejected}: {errors:?}"
        );
    }
}
