//! Tests that an edit reruns only the queries whose inputs it changed.

mod common;

use self::common::execution_log::LoggingDb;
use tribute_front::query::func_names;

const PATH: &str = "reuse.trb";

const ORIGINAL: &str = "\
fn f(x: Int) -> Int { x }
fn g(y: Int) -> Int { y }
";

#[test]
fn moving_code_without_changing_it_reuses_the_parsed_module() {
    let mut db = LoggingDb::default();
    let source = db.source(PATH, ORIGINAL);
    func_names(&db, source);
    db.take_executed();

    let moved = "\
// A comment and blank lines move every declaration.

fn f(x: Int) -> Int {
    // and one inside a body
    x
}
fn g(y: Int) -> Int { y }
";
    db.edit(source, moved);
    func_names(&db, source);
    db.assert_not_executed(&["parsed_ast", "func_names"]);
}
