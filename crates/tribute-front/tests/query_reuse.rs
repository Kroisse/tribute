//! Tests that an edit reruns only the queries whose inputs it changed.

mod common;

use self::common::execution_log::LoggingDb;
use tribute_front::SourceCst;
use tribute_front::query::{func_names, resolved_module};

const PATH: &str = "reuse.trb";

const ORIGINAL: &str = "\
fn f(x: Int) -> Int { x }
fn g(y: Int) -> Int { y }
";

const MOVED: &str = "\
// A comment and blank lines move every declaration.

fn f(x: Int) -> Int {
    // and one inside a body
    x
}
fn g(y: Int) -> Int { y }
";

/// A query that depends on the function names only.
#[salsa::tracked(returns(copy))]
fn function_count(db: &dyn salsa::Database, source: SourceCst) -> usize {
    func_names(db, source).len()
}

/// A database holding `ORIGINAL` with the queries under test computed, and
/// the source.
fn computed() -> (LoggingDb, SourceCst) {
    let db = LoggingDb::default();
    let source = db.source(PATH, ORIGINAL);
    function_count(&db, source);
    resolved_module(&db, source);
    db.take_executed();
    (db, source)
}

#[test]
fn moving_code_without_changing_it_reuses_the_parsed_module() {
    let (mut db, source) = computed();
    db.edit(source, MOVED);
    function_count(&db, source);
    db.assert_not_executed(&["parsed_ast", "func_names", "function_count"]);
}

/// A body edit changes the module, so `func_names` reruns, but its result
/// is unchanged, so what depends on it is reused.
#[test]
fn a_body_edit_keeps_the_function_names() {
    let (mut db, source) = computed();
    db.edit(source, &ORIGINAL.replace("{ x }", "{ x + 1 }"));
    assert_eq!(function_count(&db, source), 2);
    let executed = db.take_executed();
    assert!(
        executed.iter().any(|name| name == "func_names"),
        "{executed:?}"
    );
    assert!(
        !executed.iter().any(|name| name == "function_count"),
        "{executed:?}"
    );
}

#[test]
fn adding_a_function_reruns_what_depends_on_the_names() {
    let (mut db, source) = computed();
    db.edit(source, &format!("{ORIGINAL}fn h() -> Int {{ 1 }}\n"));
    assert_eq!(function_count(&db, source), 3);
    db.assert_executed(&["func_names", "function_count"]);
}

/// Name resolution reads the span map to report its diagnostics, so moving
/// code reruns it although no name changed. Positions relative to their item
/// would let it be reused; this test then expects `resolved_module` not to run.
#[test]
fn moving_code_reruns_name_resolution() {
    let (mut db, source) = computed();
    db.edit(source, MOVED);
    resolved_module(&db, source);
    db.assert_executed(&["resolved_module"]);
}
