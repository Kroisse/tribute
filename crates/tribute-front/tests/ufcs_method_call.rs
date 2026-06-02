//! Tests for UFCS method call lowering.
//!
//! Verifies that method call arguments are correctly preserved through
//! the pipeline: astgen → typeck → TDNR → IR lowering.
//!
//! Regression tests for #582: multi-argument UFCS calls dropped extra arguments.

mod common;

use self::common::{
    TdnrCall, TdnrSummary, run_ast_pipeline, run_ast_pipeline_with_ir, tdnr_function_summary,
};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

// ========================================================================
// Compilation Tests (no snapshot, just verify no errors)
// ========================================================================

/// Single UFCS method call with multiple arguments.
#[salsa_test]
fn test_ufcs_multi_arg(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Vec3 { x: Nat, y: Nat, z: Nat }

pub mod Vec3 {
    pub fn add(a: Vec3, b: Vec3) -> Vec3 {
        Vec3 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
    }
}

fn test(a: Vec3, b: Vec3) -> Vec3 {
    a.add(b)
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Chained UFCS where intermediate calls have multiple arguments.
#[salsa_test]
fn test_ufcs_chained_multi_arg(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair { x: Nat, y: Nat }
struct Triple { x: Nat, y: Nat, z: Nat }

pub mod Pair {
    pub fn extend(p: Pair, z: Nat) -> Triple {
        Triple { x: p.x, y: p.y, z: z }
    }
}

pub mod Triple {
    pub fn sum(t: Triple) -> Nat { t.x + t.y + t.z }
}

fn test() -> Nat {
    let p = Pair { x: 1, y: 2 }
    p.extend(3).sum()
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Multi-arg UFCS at every step in a chain.
#[salsa_test]
fn test_ufcs_chain_all_multi_arg(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct A { value: Nat }
struct B { value: Nat }
struct C { value: Nat }

pub mod A {
    pub fn to_b(a: A, offset: Nat) -> B {
        B { value: a.value + offset }
    }
}

pub mod B {
    pub fn to_c(b: B, scale: Nat) -> C {
        C { value: b.value * scale }
    }
}

pub mod C {
    pub fn get(c: C) -> Nat { c.value }
}

fn test() -> Nat {
    let a = A { value: 10 }
    a.to_b(5).to_c(2).get()
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// TDNR AST Tests (verify MethodCall -> Call before IR lowering)
// ========================================================================

/// Covers TDNR's resolved, ambiguous, unresolved, and nested-module paths.
#[salsa_test]
fn test_tdnr_method_resolution_cases(db: &salsa::DatabaseImpl) {
    let field_accessor = SourceCst::from_source_str(
        db,
        "field_accessor.trb",
        r#"
struct Point { x: Nat, y: Nat }

fn test(p: Point) -> Nat {
    p.x
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, field_accessor, "test"),
        TdnrSummary {
            method_calls: vec![],
            calls: vec![TdnrCall {
                target: "Point::x".to_owned(),
                arg_count: 1,
            }],
        }
    );

    let user_defined = SourceCst::from_source_str(
        db,
        "user_defined.trb",
        r#"
struct Counter { value: Nat }

pub mod Counter {
    pub fn add(c: Counter, n: Nat) -> Nat {
        c.value
    }
}

fn test(c: Counter) -> Nat {
    c.add(2)
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, user_defined, "test"),
        TdnrSummary {
            method_calls: vec![],
            calls: vec![TdnrCall {
                target: "Counter::add".to_owned(),
                arg_count: 2,
            }],
        }
    );

    let ambiguous = SourceCst::from_source_str(
        db,
        "ambiguous.trb",
        r#"
struct Thing { value: Nat }

pub mod A {
    pub fn pick(t: Thing) -> Nat { t.value }
}

pub mod B {
    pub fn pick(t: Thing) -> Nat { t.value }
}

fn test(t: Thing) -> Nat {
    t.pick()
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, ambiguous, "test"),
        TdnrSummary {
            method_calls: vec!["pick".to_owned()],
            calls: vec![],
        }
    );

    let unresolved = SourceCst::from_source_str(
        db,
        "unresolved.trb",
        r#"
struct Thing { value: Nat }

fn test(t: Thing) -> Nat {
    t.missing()
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, unresolved, "test"),
        TdnrSummary {
            method_calls: vec!["missing".to_owned()],
            calls: vec![],
        }
    );

    let nested_module = SourceCst::from_source_str(
        db,
        "nested_module.trb",
        r#"
struct Item { value: Nat }

pub mod Outer {
    pub fn score(i: Item) -> Nat {
        i.value
    }
}

fn test(i: Item) -> Nat {
    i.score()
}
"#,
    );

    assert_eq!(
        tdnr_function_summary(db, nested_module, "test"),
        TdnrSummary {
            method_calls: vec![],
            calls: vec![TdnrCall {
                target: "Outer::score".to_owned(),
                arg_count: 1,
            }],
        }
    );
}

// ========================================================================
// Snapshot Tests (verify IR output)
// ========================================================================

/// Snapshot: single UFCS call with one extra argument.
/// Verifies `p.extend(3)` becomes `func.call @extend(p, 3)`.
#[salsa_test]
fn test_snapshot_ufcs_single_extra_arg(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair { x: Nat, y: Nat }
struct Triple { x: Nat, y: Nat, z: Nat }

pub mod Pair {
    pub fn extend(p: Pair, z: Nat) -> Triple {
        Triple { x: p.x, y: p.y, z: z }
    }
}

fn test(p: Pair) -> Triple {
    p.extend(3)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot: chained UFCS with multi-arg intermediate call.
/// Verifies `p.extend(3).sum()` lowers correctly with all arguments.
#[salsa_test]
fn test_snapshot_ufcs_chained_multi_arg(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair { x: Nat, y: Nat }
struct Triple { x: Nat, y: Nat, z: Nat }

pub mod Pair {
    pub fn extend(p: Pair, z: Nat) -> Triple {
        Triple { x: p.x, y: p.y, z: z }
    }
}

pub mod Triple {
    pub fn sum(t: Triple) -> Nat { t.x + t.y + t.z }
}

fn test(p: Pair) -> Nat {
    p.extend(3).sum()
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
