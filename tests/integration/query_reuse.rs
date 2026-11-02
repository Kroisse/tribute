//! A user edit reuses the frontend results that depend only on package std.

use tribute::pipeline::{parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_front::SourceCst;
use tribute_front::testing::LoggingDb;

const PROGRAM: &str = r#"
struct Point { x: Int, y: Int }

fn shift(point: Point) -> Option(Point) {
    Option::map(.Some(point), fn(p) { Point { x: p.x, y: +1 } })
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
    let source = db.source("edit.trb", PROGRAM);
    let baseline = db.record(|db| specialize(db, source));

    db.replace(source, "x: p.x, y: +1", "x: p.y, y: +2");
    db.record(|db| specialize(db, source)).assert_reuse(
        &baseline,
        &["prepare_frontend_details"],
        &["specialized_package", "prelude_runtime_types"],
    );
}
