//! A user edit reuses the frontend results that depend only on package std.

#[path = "../../crates/tribute-front/tests/common/execution_log.rs"]
#[allow(dead_code, reason = "the frontend crate's tests share this log")]
mod execution_log;

use execution_log::LoggingDb;
use tribute::pipeline::{parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_front::SourceCst;

const BEFORE: &str = r#"
struct Point { x: Int, y: Int }

fn shift(point: Point) -> Option(Point) {
    Option::map(.Some(point), fn(p) { Point { x: p.x, y: +1 } })
}

fn main() -> Nil {
    let shifted = shift(Point { x: +1, y: +2 })
    Nil
}
"#;

const AFTER: &str = r#"
struct Point { x: Int, y: Int }

fn shift(point: Point) -> Option(Point) {
    Option::map(.Some(point), fn(p) { Point { x: p.y, y: +2 } })
}

fn main() -> Nil {
    let shifted = shift(Point { x: +1, y: +2 })
    Nil
}
"#;

fn specialize(db: &LoggingDb, source: SourceCst) {
    let typed = parse_and_lower_ast(db, source).expect("the program type checks");
    assert!(
        prepare_frontend_for_lowering(db, typed, source).is_some(),
        "the program specializes"
    );
}

#[test]
fn a_body_edit_does_not_specialize_std_again() {
    let mut db = LoggingDb::default();
    let source = db.source("edit.trb", BEFORE);
    specialize(&db, source);
    db.assert_executed(&["specialized_prelude", "prepare_frontend_details"]);

    db.edit(source, AFTER);
    specialize(&db, source);
    let executed = db.take_executed();
    assert!(
        executed
            .iter()
            .any(|name| name == "prepare_frontend_details"),
        "the user module specializes again; executed: {executed:?}"
    );
    assert!(
        !executed.iter().any(|name| name == "specialized_prelude"),
        "std's specialization is reused; executed: {executed:?}"
    );
}
