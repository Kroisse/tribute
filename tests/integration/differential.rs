//! Differential properties over programs the `tribute-prop` generator emits.
//!
//! Every generated program compiles for both targets, and both the native
//! binary and the Wasm module print exactly the output the generator
//! predicts from each check's value. The expected output also tells which
//! target is wrong when they disagree.
//!
//! The run-and-compare property links and runs every case, so it is named
//! `_slow` and runs few cases; `TRIBUTE_DIFF_CASES` raises the count.

use proptest::prelude::*;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute_prop::program::{Program, program};

use crate::common::{compile_for_both_targets, run_on_both_targets};

thread_local! {
    /// One database for every case, so the prelude is checked once.
    static DB: TributeDatabaseImpl = TributeDatabaseImpl::default();
}

const SOURCE_NAME: &str = "generated.trb";

/// The case count `TRIBUTE_DIFF_CASES` sets, or `default`.
fn cases(default: u32) -> u32 {
    std::env::var("TRIBUTE_DIFF_CASES")
        .ok()
        .and_then(|cases| cases.parse().ok())
        .unwrap_or(default)
}

/// Each shrink step compiles the program again, and the run-and-compare
/// property also runs it, so a failure stops shrinking well inside the test
/// timeout.
fn config(cases: u32, max_shrink_iters: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        max_shrink_iters,
        ..ProptestConfig::default()
    }
}

fn check_compiles(program: &Program) -> Result<(), TestCaseError> {
    let source = program.source();
    DB.with(|db| db.attach(|db| compile_for_both_targets(db, SOURCE_NAME, &source)))
        .map_err(|error| TestCaseError::fail(format!("{error}\n{source}")))?;
    Ok(())
}

fn check_outputs(program: &Program) -> Result<(), TestCaseError> {
    let source = program.source();
    let expected = program.expected_stdout();
    let (native, wasm) = DB
        .with(|db| db.attach(|db| run_on_both_targets(db, SOURCE_NAME, &source)))
        .map_err(|error| TestCaseError::fail(format!("{error}\n{source}")))?;
    for (target, output) in [("native", native), ("wasm", wasm)] {
        prop_assert!(
            output.status.success(),
            "{target}: exit={:?}, stderr={:?}\n{source}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        prop_assert!(
            output.stdout == expected.as_bytes(),
            "{target}: stdout {:?}, expected {expected:?}\n{source}",
            String::from_utf8_lossy(&output.stdout),
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(config(64, 64))]

    /// Every generated program compiles for both targets, and its Wasm
    /// module validates.
    #[test]
    fn generated_programs_compile_for_both_targets(program in program()) {
        check_compiles(&program)?;
    }
}

proptest! {
    #![proptest_config(config(cases(8), 16))]

    /// Both targets run every generated program successfully and print
    /// exactly the output the generator predicts.
    #[test]
    fn generated_programs_print_expected_output_on_both_targets_slow(program in program()) {
        check_outputs(&program)?;
    }
}
