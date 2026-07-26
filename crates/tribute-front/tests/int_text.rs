//! Frontend coverage for the canonical public Int text API.

mod common;

use self::common::{run_ast_pipeline, run_ast_pipeline_with_ir};
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
    Empty,
    Box(a),
}

fn format_box(value: Int) -> String {
    case Box(value) {
        Box(inner) -> Int::to_string(inner)
        Empty -> "empty"
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    assert!(
        ir.contains("core.unrealized_conversion_cast") && ir.contains("adt.variant_new"),
        "generic constructor must cast Int for its erased field:\n{ir}"
    );
}
