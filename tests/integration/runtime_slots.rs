//! The compiler-owned runtime array `Slots(a)`.
//!
//! `Slots` is visible only inside package `std`, so these tests compile a
//! prelude extension that uses it and exposes plain functions to the
//! program.

use super::common::{
    assert_output_in_std_on_both_targets_with_native_asan,
    run_in_std_on_both_targets_with_native_asan,
};
use salsa::Database;
use tribute::SourceCst;
use tribute::database::TributeDatabaseImpl;
use tribute::pipeline::compile_with_diagnostics;

/// Functions in package `std` that build, read, and change `Slots` of a
/// scalar, of a managed enum (`String`), and of a managed struct.
const SLOTS_TESTS: &str = r#"
pub mod slots_tests {
    use super::{Nat, String}
    use super::runtime::Slots

    struct Pair { left: Nat, right: String }

    fn nats() ->{} Slots(Nat) {
        Slots::empty()
    }

    fn strings() ->{} Slots(String) {
        Slots::empty()
    }

    fn pairs() ->{} Slots(Pair) {
        Slots::empty()
    }

    fn pair_text(pair: Pair) ->{} String {
        pair.right
    }

    /// Lengths and elements of each step, and of the earlier arrays after
    /// the later steps: every step leaves its argument as it was.
    pub fn scalars() ->{} Nat {
        let one = Slots::insert(nats(), 0, 10)
        let two = Slots::insert(one, 1, 20)
        let three = Slots::insert(two, 0, 5)
        let replaced = Slots::replace(three, 1, 30)
        let removed = Slots::remove(replaced, 0)
        Slots::len(removed) * 100000
            + Slots::get(removed, 0) * 1000
            + Slots::get(removed, 1) * 10
            + Slots::len(one) * 100000000
            + Slots::get(three, 1) * 1000000
            + Slots::len(nats())
    }

    pub fn strings_text() ->{} String {
        let one = Slots::insert(strings(), 0, "b")
        let two = Slots::insert(one, 0, "a")
        let replaced = Slots::replace(two, 1, "c")
        let removed = Slots::remove(replaced, 0)
        Slots::get(two, 0) <> Slots::get(two, 1) <> "|" <> Slots::get(replaced, 1)
            <> "|" <> Slots::get(removed, 0) <> "|" <> Slots::get(one, 0)
    }

    pub fn pairs_text() ->{} String {
        let one = Slots::insert(pairs(), 0, Pair { left: 1, right: "x" })
        let two = Slots::insert(one, 1, Pair { left: 2, right: "y" })
        let replaced = Slots::replace(two, 0, Pair { left: 3, right: "z" })
        let removed = Slots::remove(replaced, 1)
        pair_text(Slots::get(removed, 0)) <> pair_text(Slots::get(two, 0))
            <> pair_text(Slots::get(two, 1))
    }

    pub fn pairs_left() ->{} Nat {
        let one = Slots::insert(pairs(), 0, Pair { left: 7, right: "x" })
        let two = Slots::insert(one, 0, Pair { left: 8, right: "y" })
        Slots::get(two, 0).left * 10 + Slots::get(two, 1).left + Slots::len(two) * 100
    }
}
"#;

#[test]
fn slots_build_read_and_copy_on_both_targets() {
    let program = "
fn main() ->{std::io::Io} Nil {
    std::io::print_line(std::slots_tests::scalars().to_string)
    std::io::print_line(std::slots_tests::strings_text())
    std::io::print_line(std::slots_tests::pairs_text())
    std::io::print_line(std::slots_tests::pairs_left().to_string)
}
";
    assert_output_in_std_on_both_targets_with_native_asan(
        "runtime_slots.trb",
        SLOTS_TESTS,
        program,
        "110230200\nab|c|c|b\nzxy\n287",
    );
}

/// Positions past the end, which the target traps before any access.
const OUT_OF_RANGE: &str = r#"
pub mod slots_bounds {
    use super::{Nat, String}
    use super::runtime::Slots

    fn strings() ->{} Slots(String) {
        Slots::empty()
    }

    pub fn get_at_len() ->{} String {
        Slots::get(Slots::insert(strings(), 0, "a"), 1)
    }

    pub fn insert_past_len() ->{} Nat {
        Slots::len(Slots::insert(strings(), 1, "a"))
    }

    pub fn remove_at_len() ->{} Nat {
        Slots::len(Slots::remove(Slots::insert(strings(), 0, "a"), 1))
    }
}
"#;

#[test]
fn positions_past_the_end_trap_on_both_targets() {
    for call in [
        "std::io::print_line(std::slots_bounds::get_at_len())",
        "std::io::print_line(std::slots_bounds::insert_past_len().to_string)",
        "std::io::print_line(std::slots_bounds::remove_at_len().to_string)",
    ] {
        let program = format!("\nfn main() ->{{std::io::Io}} Nil {{\n    {call}\n}}\n");
        let (native, wasm) =
            run_in_std_on_both_targets_with_native_asan("slots_bounds.trb", OUT_OF_RANGE, &program);
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

/// The error messages of compiling `source` as a user program.
fn errors(source: &str) -> Vec<String> {
    TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, "user.trb", source);
        compile_with_diagnostics(db, source)
            .diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic.inner.severity == tribute_core::diagnostic::DiagnosticSeverity::Error
            })
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect()
    })
}

#[test]
fn user_packages_cannot_name_slots_or_its_intrinsics() {
    let hidden =
        |name: &str| format!("`{name}` is not visible here: `{name}` is declared `pub(pkg)`");
    for (source, name) in [
        (
            "use std::runtime::Slots\nfn main() -> Nil { }\n",
            "std::runtime::Slots",
        ),
        (
            "fn keep(slots: std::runtime::Slots(Nat)) -> Nil { }\nfn main() -> Nil { }\n",
            "std::runtime::Slots",
        ),
        (
            "fn main() -> Nil {\n    let _ = std::runtime::Slots::empty()\n}\n",
            "std::runtime::Slots::empty",
        ),
    ] {
        assert_eq!(errors(source), [hidden(name)], "{source}");
    }
}
