//! Row parameters of struct and enum declarations and the row arguments of
//! their types (`new-plans/types.md`, "Row 매개변수").
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic};
use tribute_front::analysis;
use tribute_front::{SourceCst, ast::TypeKind, typeck::TypeCheckOutput};

fn checked(db: &dyn salsa::Database, source: SourceCst) -> TypeCheckOutput<'_> {
    *analysis::analyze(db, source, *analysis::builtins(db)).unwrap()
}

fn diagnostics<'db>(db: &'db dyn salsa::Database, text: &str) -> Vec<Diagnostic<'db>> {
    let source = SourceCst::from_source_str(db, "rows.trb", text);
    analysis::diagnostics(db, source, *analysis::builtins(db))
}

/// The messages of `text`'s diagnostics with the phase that reported them.
fn messages(db: &dyn salsa::Database, text: &str) -> Vec<(CompilationPhase, String)> {
    diagnostics(db, text)
        .into_iter()
        .map(|diagnostic| (diagnostic.phase, diagnostic.message.to_string()))
        .collect()
}

const CALLBACK: &str = r#"
ability Log { op log() -> Nil }
struct Callback({e}, a) { run: fn(a) ->{e} a }
enum Step({e}) { Next(fn() ->{e} Step({e})), Done }
struct Logged({e}) { run: fn() ->{Log, e} Nil }
"#;

#[salsa_test]
fn row_parameters_and_arguments_check(db: &salsa::DatabaseImpl) {
    let text = format!(
        "{CALLBACK}{}",
        r#"
fn run_twice(cb: Callback({e}, Nil), x: Nil) ->{e} Nil {
    let run = cb.run
    run(run(x))
}
fn pure(x: Nil) -> Callback({}, Nil) { Callback { run: fn(y) { y } } }
fn logging(x: Nil) -> Callback({Log}, Nil) { Callback { run: fn(y) { Log::log() } } }
fn run_logging(x: Nil) ->{Log} Nil { run_twice(logging(x), x) }
fn run_pure(x: Nil) ->{} Nil { run_twice(pure(x), x) }
fn step() -> Step({Log}) {
    .Next(fn() {
        Log::log()
        .Done
    })
}
fn logged(l: Logged({})) ->{Log} Nil {
    let run = l.run
    run()
}
"#
    );
    assert_eq!(messages(db, &text), vec![]);
}

#[salsa_test]
fn constructor_scheme_quantifies_row_parameters(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "rows.trb", CALLBACK);
    let output = checked(db, source);
    let (_, scheme) = output
        .declarations(db)
        .constructor_types
        .schemes
        .iter()
        .find(|(id, _)| id.name(db) == "Callback")
        .expect("Callback has a constructor");
    assert_eq!(scheme.type_params(db).len(), 1);
    let [binder] = scheme.effect_params(db).as_slice() else {
        panic!("Callback has one row binder");
    };
    let TypeKind::Func { params, result, .. } = scheme.body(db).kind(db) else {
        panic!("Callback's constructor takes its field");
    };
    let TypeKind::Named { row_args, args, .. } = result.kind(db) else {
        panic!("Callback's constructor returns its named type");
    };
    assert_eq!(args.len(), 1);
    let [row] = row_args.as_slice() else {
        panic!("Callback has one row argument");
    };
    assert!(row.instances(db).is_empty());
    assert_eq!(row.rest(db), Some(binder));
    let [run] = params.as_slice() else {
        panic!("Callback has one field");
    };
    let TypeKind::Func { effect, .. } = run.kind(db) else {
        panic!("Callback's field is a function");
    };
    assert_eq!(effect.rest(db), Some(binder));
}

#[salsa_test]
fn row_arguments_are_part_of_the_type(db: &salsa::DatabaseImpl) {
    let text = format!(
        "{CALLBACK}{}",
        r#"
fn widen(cb: Callback({}, Nil)) -> Callback({Log}, Nil) { cb }
"#
    );
    let errors = messages(db, &text);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].0, CompilationPhase::TypeChecking);
}

#[salsa_test]
fn field_rows_take_the_receivers_row_argument(db: &salsa::DatabaseImpl) {
    let text = format!(
        "{CALLBACK}{}",
        r#"
fn run_quietly(cb: Callback({Log}, Nil), x: Nil) ->{} Nil {
    let run = cb.run
    run(x)
}
fn run_twice(cb: Callback({e}, Nil), x: Nil) ->{e} Nil {
    let run = cb.run
    run(run(x))
}
fn logging(x: Nil) -> Callback({Log}, Nil) { Callback { run: fn(y) { Log::log() } } }
fn run_logging_quietly(x: Nil) ->{} Nil { run_twice(logging(x), x) }
"#
    );
    let errors = messages(db, &text);
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(
        errors
            .iter()
            .all(|(phase, message)| *phase == CompilationPhase::TypeChecking
                && message.contains("Log")),
        "{errors:?}"
    );
}

#[salsa_test]
fn row_argument_counts_are_checked(db: &salsa::DatabaseImpl) {
    let text = format!(
        "{CALLBACK}{}",
        r#"
fn missing_row(cb: Callback(Nil)) -> Nil { Nil }
fn bare(s: Step) -> Nil { Nil }
fn extra_row(s: Step({}, {})) -> Nil { Nil }
"#
    );
    assert_eq!(
        messages(db, &text),
        vec![
            (
                CompilationPhase::TypeChecking,
                "`Callback` takes 1 row argument, but 0 are written".to_string()
            ),
            (
                CompilationPhase::TypeChecking,
                "`Step` takes 1 row argument, but 0 are written".to_string()
            ),
            (
                CompilationPhase::TypeChecking,
                "`Step` takes 1 row argument, but 2 are written".to_string()
            ),
        ]
    );
}

#[salsa_test]
fn declaration_rows_end_in_one_row_parameter(db: &salsa::DatabaseImpl) {
    let text = r#"
struct Box(a) { value: a }
struct Pipe({e1}, {e2}) { both: fn() ->{e1, e2} Nil }
struct Boxed({e}) { inner: Box(fn() ->{e} Nil) }
struct Unbound({e}) { run: fn() ->{f} Nil }
"#;
    assert_eq!(
        messages(db, text),
        vec![
            (
                CompilationPhase::TypeChecking,
                "a row in a declaration has at most one row parameter, but `e2` is a second one"
                    .to_string()
            ),
            (
                CompilationPhase::TypeChecking,
                "row parameter `e` cannot be used inside an ability argument or a type argument"
                    .to_string()
            ),
            (
                CompilationPhase::TypeChecking,
                "row variable `f` is not bound in this declaration".to_string()
            ),
        ]
    );
}

#[salsa_test]
fn row_parameter_lists_are_checked_before_resolution(db: &salsa::DatabaseImpl) {
    let text = r#"
struct Late(a, {e}) { run: fn(a) ->{e} a }
struct Twice({a}, a) { value: a }
ability Rowed({e}) { op go() -> Nil }
struct Box(a) { value: a }
fn late_row(b: Box(Nil, {})) -> Nil { Nil }
"#;
    let errors: Vec<_> = messages(db, text)
        .into_iter()
        .filter(|(phase, _)| *phase == CompilationPhase::AstGeneration)
        .map(|(_, message)| message)
        .collect();
    assert_eq!(
        errors,
        vec![
            "row parameter `{e}` must come before every type parameter",
            "parameter `a` is declared twice",
            "an ability declaration has no row parameters",
            "a row argument must come before every type argument",
        ]
    );
}
