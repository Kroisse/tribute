#[path = "../build/string_literal.rs"]
mod string_literal;

#[test]
fn escaped_literals_decode_to_their_rust_values() {
    for (source, expected) in [
        (r#""quoted\"name"), trailing"#, "quoted\"name"),
        (r#""back\\slash""#, r"back\slash"),
        (r#""line\n\t\r\0end""#, "line\n\t\r\0end"),
        (r#""hex\x41_unicode\u{1f980}""#, "hexA_unicode🦀"),
        ("\"continued\\\n    name\"", "continuedname"),
        ("\"한글\"", "한글"),
    ] {
        assert_eq!(string_literal::value(source).as_deref(), Some(expected));
    }
}

#[test]
fn invalid_or_unterminated_literals_are_skipped() {
    for source in [
        "runtime_name",
        "42",
        "\"unterminated",
        r#""invalid\q""#,
        r#""invalid\xFF""#,
    ] {
        assert_eq!(string_literal::value(source), None);
    }
}
