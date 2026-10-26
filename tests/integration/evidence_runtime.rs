//! The evidence runtime written in Tribute source, `std::runtime::evidence`.
//!
//! A prelude extension calls the runtime directly, so these tests check its
//! marker stacks apart from the effect lowering that calls it, on both
//! targets with native under AddressSanitizer.

use super::common::{
    assert_output_in_std_on_both_targets_with_native_asan,
    run_in_std_on_both_targets_with_native_asan,
};

const EVIDENCE_TESTS: &str = r#"
pub mod evidence_tests {
    use super::{Int, Nat}
    use super::runtime::Slots
    use super::runtime::evidence::{Marker, extend, lookup, mask, dup, outer, tail, with_tail, push}

    fn empty() ->{} Slots(Marker) {
        Slots::empty()
    }

    // A prompt below ten as an `Int`.
    fn small(prompt: Nat) ->{} Int {
        case prompt == 0 {
            True -> +0
            False -> small(prompt - 1) + +1
        }
    }

    fn digit(rest: Int, next: Nat) ->{} Int {
        rest * +10 + small(next)
    }

    // Ability 10 at prompt 7, shadowed by prompt 8; ability 5 at prompt 3
    // sorts before it.
    fn one() ->{} Slots(Marker) {
        empty().extend(+10, 7, .None, empty())
    }

    fn two() ->{} Slots(Marker) {
        one().extend(+10, 8, .None, one())
    }

    fn three() ->{} Slots(Marker) {
        two().extend(+5, 3, .None, two())
    }

    /// The prompts each operation leaves on top, one digit each.
    pub fn prompts() ->{} Int {
        let ev = three()
        let tailed = ev.with_tail(-1, one())
        let a = digit(small(ev.lookup(+10)), ev.lookup(+5))
        let b = digit(a, ev.mask(+10).lookup(+10))
        let c = digit(b, two().outer(+10).lookup(+10))
        let d = digit(c, one().dup(+10).mask(+10).lookup(+10))
        let e = digit(d, tailed.tail(-1).lookup(+10))
        let f = digit(e, tailed.lookup(+10))
        let g = digit(f, ev.tail(-1).lookup(+10))
        digit(g, one().push(ev, +5).lookup(+5))
    }

    pub fn missing() ->{} Int {
        small(one().lookup(+4))
    }

    pub fn masked_away() ->{} Int {
        small(one().mask(+10).lookup(+10))
    }
}
"#;

fn program(call: &str) -> String {
    format!(
        "fn main() ->{{std::io::Io}} Nil {{\n    std::io::print_line(Int::to_string({call}))\n}}\n"
    )
}

#[test]
fn markers_stack_per_ability() {
    // lookup 8 and 3; mask uncovers 7; outer is the evidence before; dup then
    // mask is the original; a tail slot holds its evidence, leaves ability 10
    // alone, and is absent from evidence without it; push copies the top.
    assert_output_in_std_on_both_targets_with_native_asan(
        "evidence_runtime.trb",
        EVIDENCE_TESTS,
        &program("std::evidence_tests::prompts()"),
        "837777883",
    );
}

#[test]
fn a_missing_ability_traps() {
    // Never installed, and masked off its last marker.
    for call in [
        "std::evidence_tests::missing()",
        "std::evidence_tests::masked_away()",
    ] {
        let (native, wasm) = run_in_std_on_both_targets_with_native_asan(
            "evidence_runtime.trb",
            EVIDENCE_TESTS,
            &program(call),
        );
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
