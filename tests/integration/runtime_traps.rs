//! The trap intrinsics of `std::runtime`.
//!
//! They are visible only inside package `std`, so these tests compile a
//! prelude extension that calls them and exposes plain functions to the
//! program.

use super::common::{
    assert_output_in_std_on_both_targets_with_native_asan,
    run_in_std_on_both_targets_with_native_asan, specialized_frontend_ir,
};
use salsa::Database;
use tribute::SourceCst;
use tribute::database::TributeDatabaseImpl;
use tribute::pipeline::compile_with_diagnostics;

/// Functions in package `std` that trap in an arm, in an arm whose result
/// later code uses, in a `let`, and in the right operand of `&&`.
const TRAP_TESTS: &str = r#"
pub mod trap_tests {
    use super::{Nat, absurd}
    use super::runtime::{trap_out_of_range, trap_missing_ability}

    pub fn in_arm(ok: Bool) ->{} Nat {
        case ok {
            True -> 1
            False -> absurd(trap_out_of_range())
        }
    }

    pub fn before_use(ok: Bool) ->{} Nat {
        let found = case ok {
            True -> 1
            False -> absurd(trap_missing_ability())
        }
        found + 1
    }

    pub fn in_let() ->{} Nat {
        let never = trap_out_of_range()
        absurd(never) + 1
    }

    pub fn in_operand(ok: Bool) ->{} Nat {
        case ok && absurd(trap_missing_ability()) {
            True -> 1
            False -> 0
        }
    }
}
"#;

fn program(body: &str) -> String {
    format!("\nfn main() ->{{std::io::Io}} Nil {{\n    {body}\n}}\n")
}

#[test]
fn a_path_that_does_not_trap_keeps_its_value() {
    assert_output_in_std_on_both_targets_with_native_asan(
        "runtime_traps.trb",
        TRAP_TESTS,
        &program(
            "std::io::print_line(std::trap_tests::in_arm(True).to_string)\n    \
             std::io::print_line(std::trap_tests::before_use(True).to_string)\n    \
             std::io::print_line(std::trap_tests::in_operand(False).to_string)",
        ),
        "1\n2\n0",
    );
}

#[test]
fn a_trap_ends_the_execution_on_both_targets() {
    for call in [
        "std::trap_tests::in_arm(False).to_string",
        "std::trap_tests::before_use(False).to_string",
        "std::trap_tests::in_let().to_string",
        "std::trap_tests::in_operand(True).to_string",
    ] {
        let program = program(&format!("std::io::print_line({call})"));
        let (native, wasm) =
            run_in_std_on_both_targets_with_native_asan("runtime_traps.trb", TRAP_TESTS, &program);
        for (target, output) in [("native", native), ("wasm", wasm)] {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success(), "{target}: {call}: {stderr}");
            assert!(output.stdout.is_empty(), "{target}: {call}");
            assert!(
                !stderr.contains("AddressSanitizer"),
                "{target}: {call}: {stderr}"
            );
        }
    }
}

/// The printed body of the function `name` in `ir`, one operation per line.
fn body_of<'a>(ir: &'a str, name: &str) -> Vec<&'a str> {
    let header = format!("@\"{name}\"(");
    let mut lines = ir.lines().skip_while(|line| !line.contains(&header));
    let indent = |line: &str| line.len() - line.trim_start().len();
    let first = lines
        .next()
        .unwrap_or_else(|| panic!("no {name} in:\n{ir}"));
    lines
        .take_while(|line| indent(line) > indent(first))
        .map(str::trim)
        .collect()
}

#[test]
fn the_frontend_lowers_nothing_after_a_trap() {
    TributeDatabaseImpl::default().attach(|db| {
        tribute::prelude::PreludeExtension::new(db, TRAP_TESTS.to_owned());
        let source = SourceCst::from_source_str(
            db,
            "runtime_traps.trb",
            &program(
                "std::io::print_line(std::trap_tests::in_let().to_string)\n    \
                 std::io::print_line(std::trap_tests::in_arm(True).to_string)\n    \
                 std::io::print_line(std::trap_tests::in_operand(True).to_string)",
            ),
        );
        let ir = specialized_frontend_ir(db, source);

        // The trap replaces the call of the intrinsic, the call of `absurd`,
        // the addition after the `let`, and the `return`.
        assert_eq!(
            body_of(&ir, "std::trap_tests::in_let"),
            ["tribute_control.trap out_of_range"],
            "{ir}"
        );
        // An arm ends in the trap instead of a yield.
        let in_arm = body_of(&ir, "std::trap_tests::in_arm");
        let trap = in_arm
            .iter()
            .position(|line| *line == "tribute_control.trap out_of_range")
            .unwrap_or_else(|| panic!("{ir}"));
        assert_eq!(in_arm[trap + 1], "}", "{ir}");
        assert!(in_arm[trap - 1].ends_with('{'), "{ir}");
        let in_operand = body_of(&ir, "std::trap_tests::in_operand");
        let trap = in_operand
            .iter()
            .position(|line| *line == "tribute_control.trap missing_ability")
            .unwrap_or_else(|| panic!("{ir}"));
        assert!(in_operand[trap + 1].starts_with('}'), "{ir}");
        assert!(in_operand[trap - 1].ends_with('{'), "{ir}");
        // No call of an intrinsic is left.
        assert!(!ir.contains("call @\"std::runtime::trap_"), "{ir}");
    });
}

/// The error messages of compiling `source` as a user program.
fn errors(source: &str) -> Vec<String> {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, "user.trb", source);
        compile_with_diagnostics(db, source)
            .diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic.severity == tribute_core::diagnostic::DiagnosticSeverity::Error
            })
            .map(|diagnostic| diagnostic.message.to_owned())
            .collect()
    })
}

#[test]
fn user_packages_cannot_call_a_trap_intrinsic() {
    assert_eq!(
        errors("fn main() -> Nil {\n    absurd(std::runtime::trap_out_of_range())\n}\n"),
        ["`std::runtime::trap_out_of_range` is not visible here: \
             `std::runtime::trap_out_of_range` is declared `pub(pkg)`"],
    );
}
