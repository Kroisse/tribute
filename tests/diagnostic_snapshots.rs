//! Snapshot tests for compiler diagnostic messages.
//!
//! These tests capture the current state of error messages as YAML snapshots,
//! providing a baseline for diagnostic quality improvements. When an error
//! message is improved, review and update the corresponding snapshot with
//! `cargo insta review`.

mod common;

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::SourceCst;

// =============================================================================
// Name resolution errors
// =============================================================================

#[salsa_test]
fn diag_unresolved_name(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "test.trb", "fn main() -> Int { undefined_var }");
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_unresolved_name_with_suggestion(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn compute(value: Int) -> Int { value + 1 }

fn main() -> Int { compue(42) }
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_unresolved_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "test.trb", "fn main() -> Foo { 42 }");
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

// =============================================================================
// Parse errors
// =============================================================================

#[salsa_test]
fn diag_syntax_error(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "test.trb", "fn main( { }");
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

// =============================================================================
// Type checking errors
// =============================================================================

#[salsa_test]
fn diag_non_exhaustive_case(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test(x: Nat) -> Nat {
    case x {
        0 -> 1
    }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_type_mismatch_in_function(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn add(x: Int, y: Int) -> Int { x + y }

fn test() -> Int {
    add(1, "hello")
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_main_must_return_nil(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "test.trb", "fn main() -> Int { 42 }");
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

// =============================================================================
// Lowering errors
// =============================================================================

/// Panics: accumulate() called outside tracked function during lowering.
#[ignore = "lowering diagnostic accumulate panics outside tracked function"]
#[salsa_test]
fn diag_unknown_struct_field(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: 1, z: 2 }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

/// Panics: accumulate() called outside tracked function during lowering.
#[ignore = "lowering diagnostic accumulate panics outside tracked function"]
#[salsa_test]
fn diag_missing_struct_field(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: 1 }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

// =============================================================================
// Ability / effect errors
// =============================================================================

#[salsa_test]
fn diag_unhandled_effect_in_main(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability MyEffect {
    fn do_something() -> Int
}

fn main() -> Int {
    MyEffect::do_something()
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_misspelled_ability_name(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability MyEffect {
    fn do_something() -> Int
}

fn test() ->{MyEffec} Int {
    MyEffec::do_something()
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

/// Panics in func_context.rs debug_assert instead of producing a diagnostic.
#[ignore = "arity mismatch panics in effect row merging"]
#[salsa_test]
fn diag_effect_arg_arity_mismatch(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn test() ->{State(Int, Bool)} Int {
    State::get()
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}
