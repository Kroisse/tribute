//! End-to-end tests for Ability System (Core) milestone.
//!
//! These tests verify the target code from issue #100 passes the frontend pipeline.
//! The backend (code generation) for abilities is tracked in issues #112-#114.
//!
//! ## Milestone Target
//!
//! The ability system should support:
//! - Ability definitions with type parameters (`ability State(s) { ... }`)
//! - Effect annotations in function signatures (`->{State(Int)}`)
//! - Handler expressions (`handle ... { ... }`)
//! - Handler patterns (`{ State::get() -> k }`)
//!
//! ## Test Strategy
//!
//! Since the wasm backend doesn't yet support abilities, these tests verify:
//! 1. Parsing succeeds
//! 2. Name resolution succeeds
//! 3. Type checking succeeds (including effect inference)
//!
//! Full execution tests will be added when #112-#114 are complete.

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::compile_with_diagnostics;
use tribute_front::SourceCst;

/// Helper to parse source code
fn parse_source(db: &dyn salsa::Database, name: &str, code: &str) -> SourceCst {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    SourceCst::from_path(db, name, source_code, tree)
}

// =============================================================================
// Basic Ability Definition Tests
// =============================================================================

/// Test that ability definitions parse and typecheck.
#[test]
fn test_ability_definition() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "ability_def.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test ability operations with effect annotations.
#[test]
#[ignore = "Ability operation calls not fully supported - parsing issue with arithmetic in call args"]
fn test_ability_operation_with_effect() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "ability_effect.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Handler Expression Tests
// =============================================================================

/// Test basic handle expression parsing and typechecking.
#[test]
#[ignore = "Handler patterns not yet implemented in parser"]
fn test_handle_expression() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn run() -> Int {
    case handle get_state() {
        { result } -> result
        { State::get() -> k } -> 42
        { State::set(v) -> k } -> 0
    }
}

fn main() -> Int { run() }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "handle_expr.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Milestone Target Code Test (Issue #100)
// =============================================================================

/// Test the complete milestone target code from issue #100.
///
/// This is the main acceptance test for the Ability System (Core) milestone.
/// The code should:
/// 1. Parse correctly
/// 2. Pass name resolution
/// 3. Pass type checking with effect inference
///
/// Note: Full execution requires backend support (issues #112-#114).
#[test]
#[ignore = "Handler patterns and ability operation calls not fully implemented"]
fn test_milestone_target_code() {
    // This is the target code from issue #100
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    case handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "milestone_100.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Milestone target code should compile without errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Effect Row Tests
// =============================================================================

/// Test that effect row polymorphism works correctly.
/// The function `run_state` should handle `State(s)` and propagate remaining effects `e`.
#[test]
#[ignore = "Ability operation calls not fully supported - parsing issue with arithmetic in call args"]
fn test_effect_row_polymorphism() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

ability Console {
    fn print(msg: Text) -> Nil
}

fn stateful_print() ->{State(Int), Console} Int {
    Console::print("hello")
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "effect_row.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test that multiple abilities can be combined in effect rows.
#[test]
#[ignore = "Ability operation calls not fully supported in parser"]
fn test_multiple_abilities() {
    let code = r#"ability Reader(r) {
    fn ask() -> r
}

ability Writer(w) {
    fn tell(value: w) -> Nil
}

fn copy() ->{Reader(Int), Writer(Int)} Nil {
    let x = Reader::ask()
    Writer::tell(x)
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "multiple_abilities.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Edge Cases and Error Detection
// =============================================================================

/// Test that unhandled effects are properly tracked.
/// A function using State without declaring it in its effect row should error.
#[test]
#[ignore = "Effect checking not yet enforced - requires #112"]
fn test_unhandled_effect_error() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

// Missing ->{State(Int)} annotation
fn bad_counter() -> Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "unhandled_effect.trb", code);
        let result = compile_with_diagnostics(db, source);

        // Should have an error about unhandled effect
        assert!(
            !result.diagnostics.is_empty(),
            "Expected error for unhandled effect"
        );
    });
}
