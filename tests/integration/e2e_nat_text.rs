//! End-to-end coverage for public Nat formatting on both targets.

use crate::common;

use common::assert_output_on_both_targets_with_native_asan;

#[test]
fn public_nat_to_string_formats_decimal_text() {
    assert_output_on_both_targets_with_native_asan(
        "nat_text_public_api.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    print_line(Nat::to_string(0))
    print_line(Nat::to_string(7))
    print_line(Nat::to_string(10))
    print_line(Nat::to_string(1200))
    print_line(Nat::to_string(987654321))
    print_line(Nat::to_string(4294967295))
}
"#,
        "0\n7\n10\n1200\n987654321\n4294967295",
    );
}
