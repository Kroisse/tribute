//! Tests that an edit reruns only the queries whose inputs it changed.

use tribute_front::SourceCst;
use tribute_front::ast::{ItemLayout, ItemSpans};
use tribute_front::query::{func_names, parsed_ast, parsed_module, resolved_module};
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

/// Name resolution reads the span map to report its diagnostics, and the span
/// map reads the item layout, so moving code reruns it although no name
/// changed. Once resolution reports diagnostics by `NodeId`, this test
/// expects `resolved_module` not to run.
#[test]
fn moving_code_reruns_name_resolution() {
    let (mut db, source, baseline) = computed();
    move_code(&mut db, source);
    db.record(|db| {
        resolved_module(db, source);
    })
    .assert_reuse(&baseline, &["resolved_module"], &[]);
}

/// A program with every kind of item, including one in an inline module.
const ITEMS: &str = "\
use std::io::{Io, print_line}

struct Point { x: Int }

ability Counter {
    op tick() -> Nil
}

mod inner {
    fn h() -> Int { 1 }
}

fn f(x: Int) -> Int { x }
";

/// A query that depends on the parse's relative spans only.
#[salsa::tracked(returns(copy))]
fn has_item_spans(db: &dyn salsa::Database, source: SourceCst) -> bool {
    parsed_ast(db, source).is_some_and(|parsed| *parsed.item_spans(db) != ItemSpans::default())
}

/// A query that depends on the parse's item layout only.
#[salsa::tracked(returns(copy))]
fn has_item_layout(db: &dyn salsa::Database, source: SourceCst) -> bool {
    parsed_ast(db, source).is_some_and(|parsed| *parsed.item_layout(db) != ItemLayout::default())
}

/// Moving every item without changing it changes only the item layout: the
/// module and every node's span relative to its item are reused.
#[test]
fn moving_items_changes_only_their_layout() {
    let db = LoggingDb::default();
    let source = db.source(PATH, ITEMS);
    let baseline = db.record(|db| {
        parsed_module(db, source);
        has_item_spans(db, source);
        has_item_layout(db, source);
    });
    let mut db = db;
    db.replace(source, "use ", "// A comment moves every item.\n\nuse ");
    db.record(|db| {
        parsed_module(db, source);
        assert!(has_item_spans(db, source));
        assert!(has_item_layout(db, source));
    })
    .assert_reuse(
        &baseline,
        &[
            "lower_source_to_parsed_ast_with_module_path",
            "has_item_layout",
        ],
        &["parsed_ast", "has_item_spans"],
    );
}
