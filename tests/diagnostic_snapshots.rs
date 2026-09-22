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
use tribute::pipeline::{compile_ast, compile_with_diagnostics};
use tribute_core::diagnostic::{CompilationPhase, DiagnosticSeverity};
use tribute_front::SourceCst;
use trunk_ir::Span;

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
// Record field errors
// =============================================================================

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
    let result = compile_with_diagnostics(db, source);
    assert!(result.module.is_none());
    let diagnostics = result.diagnostics;
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
    let diagnostics = result.diagnostics;
    insta::assert_yaml_snapshot!(diagnostics);
}

#[salsa_test]
fn diag_duplicate_struct_field(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Point {
    Point { x: +1, x: +2, y: +3 }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert!(result.module.is_none());
    let diagnostics = result.diagnostics;
    insta::assert_yaml_snapshot!(diagnostics);
}

#[salsa_test]
fn invalid_record_shapes_block_public_compilation_apis(db: &salsa::DatabaseImpl) {
    let cases: &[(&str, &[&str])] = &[
        (
            "Point { x: +1, y: +2, z: +3 }",
            &["unknown field `z` for struct `Point`"],
        ),
        ("Point { x: +1, x: +2, y: +3 }", &["duplicate field `x`"]),
        ("Point { x: +1 }", &["missing field: y"]),
        (
            "Point { x: +1, x: +2, z: +3 }",
            &[
                "duplicate field `x`",
                "missing field: y",
                "unknown field `z` for struct `Point`",
            ],
        ),
    ];
    for (record, messages) in cases {
        let text =
            format!("struct Point {{ x: Int, y: Int }}\nfn test() -> Point {{ {record} }}\n");
        let source = SourceCst::from_source_str(db, "test.trb", &text);
        let result = compile_with_diagnostics(db, source);
        let start = text.find(record).expect("record expression");
        let expected: Vec<_> = messages
            .iter()
            .map(|message| {
                Diagnostic::new(
                    *message,
                    Span::new(start, start + record.len()),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
            })
            .collect();
        // Public diagnostics retain phase/span/message sorting, including mixed errors.
        // Exact equality also rejects spurious missing-field diagnostics.
        assert_eq!(result.diagnostics, expected, "{record}");
        assert!(result.module.is_none(), "{record}");
        assert!(tribute::compile_frontend(db, source).is_none(), "{record}");
        assert!(matches!(compile_ast(db, source), Ok(None)), "{record}");
    }
}

#[salsa_test]
fn generic_record_spread_mismatch_blocks_ir(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "generic_spread.trb",
        r#"
struct Pair(a, b) { first: a, second: b }
fn invalid(base: Pair(Int, Bool)) -> Pair(Int, Int) {
    Pair { first: +1, second: +2, ..base }
}
"#,
    );
    let result = compile_with_diagnostics(db, source);
    assert_eq!(result.diagnostics.len(), 1, "{:?}", result.diagnostics);
    let diagnostic = &result.diagnostics[0];
    assert_eq!(diagnostic.phase, CompilationPhase::TypeChecking);
    assert_eq!(diagnostic.inner.severity, DiagnosticSeverity::Error);
    assert!(diagnostic.inner.message.contains("type error"));
    assert!(diagnostic.inner.message.contains("Int"));
    assert!(diagnostic.inner.message.contains("Bool"));
    assert!(result.module.is_none());
    assert!(tribute::compile_frontend(db, source).is_none());
    assert!(matches!(compile_ast(db, source), Ok(None)));
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
