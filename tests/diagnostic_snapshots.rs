//! Snapshot tests for compiler diagnostic messages.
//!
//! These tests capture the current state of error messages as YAML snapshots,
//! providing a baseline for diagnostic quality improvements. When an error
//! message is improved, review and update the corresponding snapshot with
//! `cargo insta review`.

mod common;

use salsa::Database;
use salsa_test_macros::salsa_test;
use tribute::Diagnostic;
use tribute::pipeline::{compile_ast, compile_frontend, compile_with_diagnostics};
use tribute_front::SourceCst;
use tribute_passes::diagnostic::CompilationPhase;

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
fn diag_heterogeneous_list_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn values() -> List(Nat) {
    [1, True]
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_canonical_list_literal_does_not_match_shadowing_source_list(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
enum List(a) {
    UserList(a),
}

fn source_value() -> List(Nat) {
    UserList(1)
}

fn value() -> Nat {
    case [1] {
        [] -> 0
        [head, ..tail] -> head
    }
}

fn bad() -> List(Nat) {
    [1]
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(!result.diagnostics.is_empty());
    insta::assert_yaml_snapshot!(result.diagnostics);
}

#[salsa_test]
fn diag_non_exhaustive_list_patterns(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn first(xs: List(Nat)) -> Nat {
    case xs {
        [head, ..tail] -> head
    }
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
// Record construction diagnostics
// =============================================================================

#[salsa::tracked]
fn collect_lowering_diagnostics(db: &dyn salsa::Database, source: SourceCst) {
    let _ = compile_ast(db, source);
}

fn lowering_diagnostics(db: &dyn salsa::Database, source: SourceCst) -> Vec<Diagnostic> {
    collect_lowering_diagnostics(db, source);
    collect_lowering_diagnostics::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect()
}

#[salsa_test]
fn diag_unknown_struct_field(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: +1, y: +2, z: +3 }
}
"#,
    );
    let diagnostics = lowering_diagnostics(db, source);
    assert!(!diagnostics.is_empty());
    insta::assert_yaml_snapshot!(diagnostics);
}

#[salsa_test]
fn diag_missing_struct_field(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: +1 }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(result.module.is_none());
    assert_eq!(result.diagnostics.len(), 1, "{:#?}", result.diagnostics);
    assert_eq!(result.diagnostics[0].inner.message, "missing field: y");
    assert!(compile_frontend(db, source).is_none());

    let diagnostics = lowering_diagnostics(db, source);
    assert!(!diagnostics.is_empty());
    insta::assert_yaml_snapshot!(diagnostics);
}

#[salsa_test]
fn record_errors_do_not_gain_missing_field_diagnostics(db: &salsa::DatabaseImpl) {
    let cases = [
        (
            "unknown",
            r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: +1, y: +2, z: +3 }
}
"#,
            "unknown field `z` for struct `Point`",
        ),
        (
            "duplicate",
            r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: +1, x: +2, y: +3 }
}
"#,
            "duplicate field `x`",
        ),
        (
            "field_type",
            r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: True, y: +2 }
}
"#,
            "type error in function 'test': expected `Bool`, found `Int`",
        ),
    ];

    for (name, text, message) in cases {
        let file_name = format!("{name}.trb");
        let source = SourceCst::from_source_str(db, &file_name, text);
        let diagnostics = lowering_diagnostics(db, source);
        assert_eq!(diagnostics.len(), 1, "{name}: {:#?}", diagnostics);
        assert_eq!(diagnostics[0].inner.message, message, "{name}");
    }
}

#[salsa_test]
fn nested_missing_struct_field_is_reported_once(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "nested_missing_struct_field.trb",
        r#"
struct Point { x: Int, y: Int }
struct Wrapper { point: Point }

fn test() -> Wrapper {
    Wrapper { point: Point { x: +1 } }
}
"#,
    );

    let diagnostics = lowering_diagnostics(db, source);
    assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
    assert_eq!(diagnostics[0].inner.message, "missing field: y");
    assert_eq!(diagnostics[0].phase, CompilationPhase::TypeChecking);
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
fn test() ->{abilities::Throw(Int), abilities::Throw(Int)} Nil {
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
