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
