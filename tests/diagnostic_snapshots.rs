//! Snapshot tests for compiler diagnostic messages.
//!
//! These tests capture the current state of error messages as YAML snapshots,
//! providing a baseline for diagnostic quality improvements. When an error
//! message is improved, review and update the corresponding snapshot with
//! `cargo insta review`.

mod common;

use salsa::Database;
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
fn compute(value: Int) -> Int { value + +1 }

fn main() -> Int { compue(+42) }
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
fn diag_unresolved_method_after_tdnr(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Thing { value: Nat }

fn test(thing: Thing) -> Nat {
    thing.missing()
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    assert!(result.module.is_none());
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

#[salsa_test]
fn diag_unhandled_effect_multiple(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Foo {
    op foo() -> Nil
}

ability Bar {
    op bar() -> Nil
}

fn main() -> Nil {
    Foo::foo()
    Bar::bar()
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_effect_row_mismatch(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Foo {
    op foo() -> Nat
}

ability Bar {
    op bar() -> Nat
}

fn test() ->{Foo} Nat {
    Bar::bar()
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_duplicate_effect_in_annotation(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn test() ->{State(Int), State(Int)} Nil {
    Nil
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_residual_effect_rejected_at_handler_boundary(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Foo {
    op foo() -> Nil
}

ability Bar {
    op bar() -> Nil
}

fn comp() ->{Foo, Bar} Nil {
    Foo::foo()
    Bar::bar()
}

fn test() ->{Foo} Nil {
    handle comp() {
        do result { result }
        op Foo::foo() { resume Nil }
    }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_row_unification_mismatch_at_lambda(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Foo {
    op foo() -> Nat
}

ability Bar {
    op bar() -> Nat
}

fn accept_foo(comp: fn() ->{Foo} Nat) ->{Foo} Nat {
    comp()
}

fn test() ->{Foo} Nat {
    accept_foo(fn() { Bar::bar() })
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn valid_distinct_parameterized_effect_annotations(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
}

fn valid() ->{State(Int), State(Bool)} Nil {
    Nil
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}

#[salsa_test]
fn valid_residual_effect_propagates_through_handler(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Foo {
    op foo() -> Nil
}

ability Bar {
    op bar() -> Nil
}

fn comp() ->{Foo, Bar} Nil {
    Foo::foo()
    Bar::bar()
}

fn valid() ->{Bar} Nil {
    handle comp() {
        do result { result }
        op Foo::foo() { resume Nil }
    }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}

#[test]
fn effect_diagnostics_are_deterministic_across_fresh_databases() {
    const SOURCE: &str = r#"
ability State(s) {
    fn get() -> s
}

fn test() ->{State(Int), State(Int)} Nil {
    Nil
}
"#;

    let compile = || {
        salsa::DatabaseImpl::default().attach(|db| {
            let source = SourceCst::from_source_str(db, "test.trb", SOURCE);
            compile_with_diagnostics(db, source).diagnostics
        })
    };

    assert_eq!(compile(), compile());
}

#[salsa_test]
fn diag_missing_handler_arm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn comp() ->{State(Nat)} Nat {
    State::set(1)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
    }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    // Baseline: the compiler does not yet detect missing handler arms.
    assert!(result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_handler_arm_wrong_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn comp() ->{State(Nat)} Nat {
    State::set(1)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(a, b) { run_state(fn() { resume Nil }, a) }
    }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

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
