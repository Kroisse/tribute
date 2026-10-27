//! Tests that an edit reruns only the queries whose inputs it changed.

use tribute_front::SourceCst;
use tribute_front::query::{func_names, resolved_module};
use tribute_front::testing::{Executed, LoggingDb};

const PATH: &str = "reuse.trb";

const PROGRAM: &str = "\
fn f(x: Int) -> Int { x }
fn g(y: Int) -> Int { y }
";

/// A query that depends on the function names only.
#[salsa::tracked(returns(copy))]
fn function_count(db: &dyn salsa::Database, source: SourceCst) -> usize {
    func_names(db, source).len()
}

/// A database holding `PROGRAM` with the queries under test computed, the
/// source, and the record of that computation.
fn computed() -> (LoggingDb, SourceCst, Executed) {
    let db = LoggingDb::default();
    let source = db.source(PATH, PROGRAM);
    let baseline = db.record(|db| {
        function_count(db, source);
        resolved_module(db, source);
    });
    (db, source, baseline)
}

/// Move every declaration with a comment and blank lines, and `f`'s body
/// with a comment inside it.
fn move_code(db: &mut LoggingDb, source: SourceCst) {
    db.replace(
        source,
        "fn f(x: Int) -> Int { x }",
        "\
// A comment and blank lines move every declaration.

fn f(x: Int) -> Int {
    // and one inside a body
    x
}",
    );
}

#[test]
fn moving_code_without_changing_it_reuses_the_parsed_module() {
    let (mut db, source, baseline) = computed();
    move_code(&mut db, source);
    db.record(|db| {
        function_count(db, source);
    })
    .assert_reuse(
        &baseline,
        &[],
        &["parsed_ast", "func_names", "function_count"],
    );
}

/// A body edit changes the module, so `func_names` reruns, but its result
/// is unchanged, so what depends on it is reused.
#[test]
fn a_body_edit_keeps_the_function_names() {
    let (mut db, source, baseline) = computed();
    db.replace(source, "{ x }", "{ x + 1 }");
    db.record(|db| assert_eq!(function_count(db, source), 2))
        .assert_reuse(&baseline, &["func_names"], &["function_count"]);
}

#[test]
fn adding_a_function_reruns_what_depends_on_the_names() {
    let (mut db, source, baseline) = computed();
    db.replace(source, "{ y }\n", "{ y }\nfn h() -> Int { 1 }\n");
    db.record(|db| assert_eq!(function_count(db, source), 3))
        .assert_reuse(&baseline, &["func_names", "function_count"], &[]);
}

/// Name resolution reads the span map to report its diagnostics, so moving
/// code reruns it although no name changed. Positions relative to their item
/// would let it be reused; this test then expects `resolved_module` not to run.
#[test]
fn moving_code_reruns_name_resolution() {
    let (mut db, source, baseline) = computed();
    move_code(&mut db, source);
    db.record(|db| {
        resolved_module(db, source);
    })
    .assert_reuse(&baseline, &["resolved_module"], &[]);
}
