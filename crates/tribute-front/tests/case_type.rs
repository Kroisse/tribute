//! Tests for case expression type checking.
//!
//! These tests verify that case expressions properly unify
//! scrutinee types with pattern types and arm body types.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::{
    SourceCst,
    ast::{Decl, ExprKind},
};

// ========================================================================
// Basic Case Expression Tests
// ========================================================================

/// Test case expression with Nat literals.
/// Pattern type (Nat) should unify with scrutinee type (Nat).
#[salsa_test]
fn test_case_nat_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn classify(x: Nat) -> Nat {
    case x {
        0 -> 100
        1 -> 200
        _ -> 300
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test case expression with Int literals (negative numbers).
/// Pattern type (Int) should unify with scrutinee type (Int).
#[salsa_test]
fn test_case_int_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn sign(x: Int) -> Int {
    case x {
        -1 -> -100
        0 -> 0
        _ -> 100
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test case expression with Bool patterns.
#[salsa_test]
fn test_case_bool_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn invert(x: Bool) -> Bool {
    case x {
        True -> False
        False -> True
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test case expression with enum variant patterns.
#[salsa_test]
fn test_case_enum_variant(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
enum Option(a) {
    Some(a),
    None,
}

fn unwrap_or(opt: Option(Nat), default: Nat) -> Nat {
    case opt {
        Some(x) -> x
        None -> default
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test case expression result type unification.
/// All arm body types should unify with the case expression's result type.
#[salsa_test]
fn test_case_result_type_unification(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn to_nat(b: Bool) -> Nat {
    case b {
        True -> 1
        False -> 0
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// `as`-wrapped list patterns must participate in list exhaustiveness checking.
#[salsa_test]
fn test_list_as_patterns_are_exhaustive(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn classify(values: List(Nat)) -> Nat {
    case values {
        [] as empty -> 0
        [head, ..tail] as whole -> head
    }
}
"#,
    );

    assert!(
        ast_pipeline_error_messages(db, source).is_empty(),
        "as-wrapped exact and rest list patterns should be exhaustive"
    );
}

/// Literal list elements do not cover every list of the same length, with or
/// without an `as` binding.
#[salsa_test]
fn test_literal_list_patterns_are_not_exhaustive(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn bare(values: List(Nat)) -> Nat {
    case values {
        [0] -> 0
    }
}

fn bound(values: List(Nat)) -> Nat {
    case values {
        [0] as singleton -> 0
    }
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert_eq!(
        errors
            .iter()
            .filter(|error| error.contains("list patterns do not cover all lengths"))
            .count(),
        2,
        "literal list patterns should not count as length-exhaustive: {errors:?}"
    );
}

/// Test nested case expressions.
#[salsa_test]
fn test_case_nested(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn nested(x: Nat, y: Bool) -> Nat {
    case x {
        0 -> case y {
            True -> 10
            False -> 20
        }
        _ -> 30
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

fn nested_case_ids(
    module: &tribute_front::ast::Module<tribute_front::ast::TypedRef<'_>>,
) -> (tribute_front::ast::NodeId, tribute_front::ast::NodeId) {
    let function_body = module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::Function(function) if function.name == trunk_ir::Symbol::new("nested") => {
                Some(&function.body)
            }
            _ => None,
        })
        .expect("nested function should be typechecked");
    let ExprKind::Block { value: outer, .. } = &*function_body.kind else {
        panic!("nested function should begin with a block");
    };
    let ExprKind::Case { arms, .. } = &*outer.kind else {
        panic!("nested function block should end with a case");
    };
    let ExprKind::Case { .. } = &*arms[0].body.kind else {
        panic!("first outer arm should contain the nested case");
    };
    (outer.id, arms[0].body.id)
}

/// Pattern constraints resolve the inner scrutinee only after the body has
/// been checked. Both the already-concrete outer case and the solved inner
/// case must be recorded exactly once.
#[salsa_test]
fn nested_bool_case_is_recorded_after_solving(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn nested(flag: Bool) -> Nat {
    let identity = fn(value) value
    case flag {
        True -> case identity(True) {
            True -> 1
            False -> 0
        }
        False -> 0
    }
}
"#,
    );

    let checked = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let (outer, inner) = nested_case_ids(checked.module(db));
    let exhaustive = checked.exhaustive_cases(db);
    assert_eq!(exhaustive.iter().filter(|&&case| case == outer).count(), 1);
    assert_eq!(exhaustive.iter().filter(|&&case| case == inner).count(), 1);

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert!(
        !ir_text.contains("core.nil"),
        "an exhaustive inner Bool case must not lower an empty nil-cast tail:\n{ir_text}"
    );
    assert!(
        ir_text.contains("scf.yield"),
        "valued case branches must retain their structured yields:\n{ir_text}"
    );
}

#[salsa_test]
fn nested_result_case_is_recorded_after_solving(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
enum Result(a, e) {
    Ok(a),
    Err(e),
}

fn nested(flag: Bool) -> Nat {
    let identity = fn(value) value
    case flag {
        True -> case identity(Ok(1)) {
            Ok(value) -> value
            Err(_) -> 0
        }
        False -> 0
    }
}
"#,
    );

    let checked = tribute_front::query::type_check_output(db, source)
        .expect("type checking should produce output");
    let (outer, inner) = nested_case_ids(checked.module(db));
    let exhaustive = checked.exhaustive_cases(db);
    assert_eq!(exhaustive.iter().filter(|&&case| case == outer).count(), 1);
    assert_eq!(exhaustive.iter().filter(|&&case| case == inner).count(), 1);
}

#[salsa_test]
fn solved_partial_bool_case_reports_one_non_exhaustive_error(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn partial() -> Nat {
    let identity = fn(value) value
    case identity(True) {
        True -> 1
    }
}
"#,
    );

    let errors = ast_pipeline_error_messages(db, source);
    assert_eq!(
        errors
            .iter()
            .filter(|error| error.contains("non-exhaustive case expression"))
            .count(),
        1,
        "partial Bool case should report exactly one error: {errors:?}"
    );
}
