//! Tests for record field type checking.
//!
//! These tests verify that record construction properly validates
//! field expression types against declared struct field types.

mod common;

use self::common::{run_ast_pipeline, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// Test basic record construction with correct field types.
/// This should compile successfully.
#[salsa_test]
fn test_record_field_type_correct(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with multiple field types.
#[salsa_test]
fn test_record_mixed_field_types(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Person { name: String, age: Int, active: Bool }

fn make_person() -> Person {
    Person { name: "Alice", age: 30, active: True }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with spread operator.
/// The spread expression should be constrained to the struct type.
#[salsa_test]
fn test_record_spread_same_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with only spread (no explicit fields).
#[salsa_test]
fn test_record_spread_only(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Config { debug: Bool, verbose: Bool }

fn copy_config(c: Config) -> Config {
    Config { ..c }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record with generic type parameter.
#[salsa_test]
fn test_record_generic_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record field type inference in let binding.
#[salsa_test]
fn test_record_field_type_inference(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Int {
    let p = Point { x: 1, y: 2 }
    p.x
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// Snapshot Tests
// ========================================================================

/// Snapshot test for basic record construction IR.
#[salsa_test]
fn test_snapshot_record_construction(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with spread operator.
#[salsa_test]
fn test_snapshot_record_spread(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for generic record construction.
#[salsa_test]
fn test_snapshot_record_generic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with spread only (no explicit fields).
/// All fields should be extracted via `adt.struct_get` from the base.
#[salsa_test]
fn test_snapshot_record_spread_only(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Config { debug: Bool, verbose: Bool }

fn copy_config(c: Config) -> Config {
    Config { ..c }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with all fields explicit plus spread.
/// Explicit fields should take priority over spread values.
#[salsa_test]
fn test_snapshot_record_spread_all_fields_explicit(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn replace_all(p: Point) -> Point {
    Point { x: 1, y: 2, ..p }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test record spread with a function call as the spread expression.
#[salsa_test]
fn test_record_spread_complex_expr(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn origin() -> Point {
    Point { x: 0, y: 0 }
}

fn shift_x() -> Point {
    Point { x: 10, ..origin() }
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// Forward Reference Tests
// ========================================================================

/// Test record construction where the function using the record appears
/// before the struct definition (forward reference).
///
/// This tests that prescan_struct_fields correctly registers field orders
/// before lowering, regardless of declaration order in the source.
#[salsa_test]
fn test_record_forward_reference(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn make_point() -> Point {
    Point { x: 1, y: 2 }
}

struct Point { x: Int, y: Int }
"#,
    );

    // Should compile without ICE, emitting adt.struct_new
    run_ast_pipeline(db, source);
}
