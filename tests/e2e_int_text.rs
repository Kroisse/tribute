//! Native end-to-end coverage for public Int parsing and formatting.

mod common;

use common::{compile_and_run_native_asan, compile_and_run_native_with_paired_rc_elimination};
use tribute::pipeline::PairedRcEliminationPolicy;

#[test]
fn public_int_to_string_formats_zero() {
    let output = compile_and_run_native_asan(
        "int_text_format_zero.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    print_line(Int::to_string(+0))
}
"#,
    );

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "0\n");
}

#[test]
fn public_int_parse_handles_zero() {
    let output = compile_and_run_native_asan(
        "int_text_parse_zero.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    case Int::parse("0") {
        Ok(value) -> print_line(Int::to_string(value))
        Error(_) -> print_line("error")
    }
}
"#,
    );

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "0\n");
}

#[test]
fn paired_rc_elimination_preserves_nested_parsed_integers() {
    let output = compile_and_run_native_with_paired_rc_elimination(
        "nested_parsed_integers.trb",
        r##"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    case Int::parse("2") {
        Ok(left) -> case Int::parse("3") {
            Ok(right) -> print_line(Int::to_string(left + right))
            Error(_) -> print_line("error")
        }
        Error(_) -> print_line("error")
    }
}
"##,
        PairedRcEliminationPolicy::Enabled,
    );

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "5\n");
}

#[test]
fn result_int_value_can_flow_to_public_int_to_string() {
    let output = compile_and_run_native_asan(
        "result_int_to_string.trb",
        r#"
use std::io::{Io, print_line}

enum LocalError { Failed }

fn parsed() -> Result(Int, LocalError) { Ok(+0) }

fn main() ->{Io} Nil {
    case parsed() {
        Ok(value) -> print_line(Int::to_string(value))
        Error(_) -> print_line("error")
    }
}
"#,
    );

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "0\n");
}

#[test]
fn generic_constructor_int_payload_uses_specialized_layout() {
    let output = compile_and_run_native_asan(
        "generic_constructor_int_payload.trb",
        r#"
use std::io::{Io, print_line}

enum Boxed(a) { Box(a), Empty }

fn main() ->{Io} Nil {
    case Box(+42) {
        Box(value) -> print_line(Int::to_string(value))
        Empty -> print_line("empty")
    }
}
"#,
    );

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "42\n");
}

#[test]
fn public_int_text_api_covers_decimal_contract_and_boundaries() {
    let output = compile_and_run_native_asan(
        "int_text_public_api.trb",
        r#"
use std::io::{Io, print_line}

fn show_parse(input: String) -> String {
    case Int::parse(input) {
        Ok(value) -> "ok:" <> Int::to_string(value)
        Error(error) -> case error {
            Int::ParseError::InvalidSyntax -> "error:syntax"
            Int::ParseError::OutOfRange -> "error:range"
        }
    }
}

fn show_round_trip(input: String) -> String {
    case Int::parse(input) {
        Ok(value) -> show_parse(Int::to_string(value))
        Error(_) -> "unexpected"
    }
}

fn main() ->{Io} Nil {
    print_line(show_parse("0"))
    print_line(show_parse("+0"))
    print_line(show_parse("-0"))
    print_line(show_parse("42"))
    print_line(show_parse("+42"))
    print_line(show_parse("-42"))
    print_line(show_parse("00042"))
    print_line(show_parse("-00042"))
    print_line(show_parse("1902837465"))
    print_line(show_parse("-1902837465"))

    print_line(Int::to_string(+0))
    print_line(Int::to_string(+7))
    print_line(Int::to_string(-7))

    print_line(show_parse(""))
    print_line(show_parse("+"))
    print_line(show_parse("-"))
    print_line(show_parse(" 1"))
    print_line(show_parse("1 "))
    print_line(show_parse("1 2"))
    print_line(show_parse("12x"))
    print_line(show_parse("1\n"))
    print_line(show_parse("0x10"))
    print_line(show_parse("١"))

    print_line(show_parse("2147483647"))
    print_line(show_parse("-2147483648"))
    print_line(show_parse("2147483648"))
    print_line(show_parse("-2147483649"))
    print_line(show_parse("999999999999999999999999"))
    print_line(show_parse("-999999999999999999999999"))
    print_line(show_parse("2147483648x"))
    print_line(show_parse("-2147483649x"))
    print_line(show_parse("21474836480x"))
    print_line(show_parse("-21474836490x"))

    print_line(show_round_trip("0"))
    print_line(show_round_trip("1"))
    print_line(show_round_trip("-1"))
    print_line(show_round_trip("2147483647"))
    print_line(show_round_trip("-2147483648"))
}
"#,
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit={:?}, stdout='{}', stderr='{}'",
        output.status,
        stdout,
        stderr,
    );
    assert_eq!(
        stdout,
        concat!(
            "ok:0\n",
            "ok:0\n",
            "ok:0\n",
            "ok:42\n",
            "ok:42\n",
            "ok:-42\n",
            "ok:42\n",
            "ok:-42\n",
            "ok:1902837465\n",
            "ok:-1902837465\n",
            "0\n",
            "7\n",
            "-7\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "ok:2147483647\n",
            "ok:-2147483648\n",
            "error:range\n",
            "error:range\n",
            "error:range\n",
            "error:range\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "error:syntax\n",
            "ok:0\n",
            "ok:1\n",
            "ok:-1\n",
            "ok:2147483647\n",
            "ok:-2147483648\n",
        )
    );
}
