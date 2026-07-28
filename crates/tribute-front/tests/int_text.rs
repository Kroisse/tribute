//! Frontend coverage for the canonical public Int text API.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline, run_ast_pipeline_with_ir};
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

    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "canonical Int text API must type-check through the prelude: {errors:?}"
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
    let format_box = ir
        .split("func.func @format_box")
        .nth(1)
        .expect("format_box function must be lowered");
    let cast = format_box
        .find("core.unrealized_conversion_cast")
        .expect("generic constructor must cast its Int payload to anyref");
    let construct = format_box
        .find("adt.variant_new")
        .expect("generic constructor must construct Box");
    assert!(
        cast < construct,
        "generic constructor must cast its Int payload before constructing Box:\n{format_box}"
    );
}
