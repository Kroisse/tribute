//! The evidence runtime written in Tribute source, `std::runtime::evidence`.
//!
//! A prelude extension calls the runtime directly, so these tests check its
//! marker stacks apart from the effect lowering that calls it. They run on
//! Wasm only: native evidence lowering still maps evidence to the C runtime's
//! pointer, so native cannot run the source runtime yet.

use super::common::run_wasm_in_std;

const EVIDENCE_TESTS: &str = r#"
pub mod evidence_tests {
    use super::{Int, None}
    use super::runtime::Slots
    use super::runtime::evidence::{Marker, extend, lookup, mask, dup, outer, tail, with_tail, push}

    fn empty() ->{} Slots(Marker) {
        Slots::empty()
    }

    fn digit(rest: Int, next: Int) ->{} Int {
        rest * +10 + next
    }

    // Ability 10 at prompt 7, shadowed by prompt 8; ability 5 at prompt 3
    // sorts before it.
    fn one() ->{} Slots(Marker) {
        extend(empty(), +10, +7, None, empty())
    }

    fn two() ->{} Slots(Marker) {
        extend(one(), +10, +8, None, one())
    }

    fn three() ->{} Slots(Marker) {
        extend(two(), +5, +3, None, two())
    }

    /// The prompts each operation leaves on top, one digit each.
    pub fn prompts() ->{} Int {
        let ev = three()
        let tailed = with_tail(ev, -1, one())
        let a = digit(lookup(ev, +10), lookup(ev, +5))
        let b = digit(a, lookup(mask(ev, +10), +10))
        let c = digit(b, lookup(outer(two(), +10), +10))
        let d = digit(c, lookup(mask(dup(one(), +10), +10), +10))
        let e = digit(d, lookup(tail(tailed, -1), +10))
        let f = digit(e, lookup(tailed, +10))
        let g = digit(f, lookup(tail(ev, -1), +10))
        digit(g, lookup(push(one(), ev, +5), +5))
    }

    pub fn missing() ->{} Int {
        lookup(one(), +4)
    }

    pub fn masked_away() ->{} Int {
        lookup(mask(one(), +10), +10)
    }
}
"#;

fn run(call: &str) -> std::process::Output {
    let program = format!(
        "fn main() ->{{std::io::Io}} Nil {{\n    std::io::print_line(Int::to_string({call}))\n}}\n"
    );
    run_wasm_in_std("evidence_runtime.trb", EVIDENCE_TESTS, &program)
}

fn stdout(output: &std::process::Output) -> String {
    assert!(
        output.status.success(),
        "wasmtime failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[test]
fn markers_stack_per_ability() {
    // lookup 8 and 3; mask uncovers 7; outer is the evidence before; dup then
    // mask is the original; a tail slot holds its evidence, leaves ability 10
    // alone, and is absent from evidence without it; push copies the top.
    assert_eq!(stdout(&run("std::evidence_tests::prompts()")), "837777883");
}

#[test]
fn a_missing_ability_traps() {
    // Never installed, and masked off its last marker.
    for call in [
        "std::evidence_tests::missing()",
        "std::evidence_tests::masked_away()",
    ] {
        let output = run(call);
        assert!(!output.status.success(), "{call}: {output:?}");
        assert!(output.stdout.is_empty(), "{call}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("wasm trap"),
            "{call}: {output:?}"
        );
    }
}
