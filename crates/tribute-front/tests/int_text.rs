//! Frontend coverage for the canonical public Int text API.

mod common;

use self::common::run_ast_pipeline;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

#[salsa_test]
fn canonical_int_text_api_resolves_through_prelude(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "int_text_api.trb",
        r#"
fn parse(input: String) -> Result(Int, Int::ParseError) {
    Int::parse(input)
}

fn format(value: Int) -> String {
    Int::to_string(value)
}
"#,
    );

    run_ast_pipeline(db, source);
}

#[salsa_test]
fn generic_constructor_boxes_int_for_its_specialized_layout(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_constructor_int.trb",
        r#"
enum Boxed(a) {
    Box(a),
}

fn format_box(value: Int) -> String {
    case Box(value) {
        Box(inner) -> Int::to_string(inner)
    }
}
"#,
    );

    run_ast_pipeline(db, source);
}
