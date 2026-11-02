//! Tests that an edit rechecks only the functions whose declarations it
//! changed (`new-plans/implementation.md`, "함수 단위 type check 재사용").

use tribute_front::SourceCst;
use tribute_front::analysis::{self, builtins};
use tribute_front::testing::{Executed, LoggingDb};

const PATH: &str = "reuse.trb";

const PROGRAM: &str = "\
fn f(x: Int) -> Int { x }
fn g(y: Int) -> Nil { y }
fn h(z: Int) -> Int { f(z) }
";

/// Type check `source` and return the text its diagnostics point at.
fn check(db: &LoggingDb, source: SourceCst) -> Vec<String> {
    let table = *builtins(db);
    analysis::analyze(db, source, table).expect("the source parses");
    let text = source.text(db).to_string();
    analysis::diagnostics(db, source, table)
        .into_iter()
        .map(|diagnostic| {
            let span = diagnostic.inner.span;
            text[span.start as usize..span.end as usize].to_owned()
        })
        .collect()
}

/// A database holding `PROGRAM` type checked, the source, and the record of
/// that computation.
fn computed() -> (LoggingDb, SourceCst, Executed) {
    let db = LoggingDb::default();
    let source = db.source(PATH, PROGRAM);
    let baseline = db.record(|db| {
        check(db, source);
    });
    assert_eq!(baseline.count("check_function"), 3);
    (db, source, baseline)
}

#[test]
fn a_body_edit_rechecks_only_its_function() {
    let (mut db, source, baseline) = computed();
    db.replace(source, "{ x }", "{ 1 }");
    let executed = db.record(|db| {
        check(db, source);
    });
    executed.assert_reuse(&baseline, &["analyze", "module_declarations"], &[]);
    assert_eq!(executed.count("check_function"), 1);
}

#[test]
fn a_signature_edit_rechecks_every_function() {
    let (mut db, source, _) = computed();
    db.replace(source, "fn f(x: Int) -> Int", "fn f(x: Int, w: Int) -> Int");
    let executed = db.record(|db| {
        check(db, source);
    });
    assert_eq!(executed.count("check_function"), 3);
}

/// Moving every function reuses each check, and its diagnostic follows the
/// node it is positioned at.
#[test]
fn moving_code_reuses_checks_and_moves_their_diagnostics() {
    let (mut db, source, _) = computed();
    let before = check(&db, source);
    assert_eq!(before, ["{ y }"]);
    db.replace(
        source,
        "fn f(x: Int) -> Int { x }",
        "// A comment moves every function.\n\nfn f(x: Int) -> Int {\n    x\n}",
    );
    let mut after = Vec::new();
    let executed = db.record(|db| {
        after = check(db, source);
    });
    assert_eq!(executed.count("check_function"), 0);
    assert_eq!(after, before);
}
