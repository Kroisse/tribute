//! Diagnostics printed by the `tribute compile` command.

use std::io::Write as _;
use std::process::Command;

/// A program that compiles with one `unreachable pattern` warning.
const WARNING_PROGRAM: &str = r#"
fn pick(flag: Bool) -> Nat {
    case flag {
        True -> 1
        False -> 2
        _ -> 3
    }
}

fn main() { }
"#;

/// Remove ANSI escape sequences from terminal output.
fn strip_ansi(text: &str) -> String {
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Skip `ESC [ ... <final byte>`.
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}

/// A successful build prints its warnings on every target.
#[test]
fn successful_builds_print_warnings() {
    let mut source = tempfile::Builder::new()
        .suffix(".trb")
        .tempfile()
        .expect("temporary source file");
    source
        .write_all(WARNING_PROGRAM.as_bytes())
        .expect("write source file");
    let output_dir = tempfile::tempdir().expect("temporary output directory");

    for target in ["native", "wasm", "none"] {
        let output = Command::new(env!("CARGO_BIN_EXE_tribute"))
            .arg("compile")
            .arg("--target")
            .arg(target)
            .arg(source.path())
            .arg("-o")
            .arg(output_dir.path().join(format!("out_{target}")))
            .output()
            .expect("invoke tribute compile");
        let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
        let stdout = strip_ansi(&String::from_utf8_lossy(&output.stdout));
        assert!(
            output.status.success(),
            "{target}: exit={:?}\nstdout: {stdout}\nstderr: {stderr}",
            output.status
        );
        let printed = format!("{stdout}{stderr}");
        assert_eq!(
            printed.matches("Warning: unreachable pattern").count(),
            1,
            "{target}:\n{printed}"
        );
    }
}
