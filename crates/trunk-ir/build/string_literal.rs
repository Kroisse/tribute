//! Decode a quoted Rust string literal at the start of source text.

pub(crate) fn value(text: &str) -> Option<String> {
    let rest = text.strip_prefix('"')?;
    let mut escaped = false;
    for (index, ch) in rest.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => {
                let literal = &text[..index + 2];
                return litrs::StringLit::parse(literal)
                    .ok()
                    .map(|literal| literal.into_value().into_owned());
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
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
            assert_eq!(super::value(source).as_deref(), Some(expected));
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
            assert_eq!(super::value(source), None);
        }
    }
}
