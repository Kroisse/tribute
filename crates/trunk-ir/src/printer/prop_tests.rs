//! Round-trip property tests for escaped literals and symbol references.
//!
//! Each test prints an arbitrary value with the printer's escaping, parses
//! the text back with the raw parser, and checks that the parser consumes the
//! whole literal and recovers exactly the printed value.

use proptest::prelude::*;
use winnow::Parser as _;

use super::*;
use crate::parser::raw::{self, RawAttribute};
use crate::symbol::{Symbol, SymbolPath};

/// Text biased toward characters that need escaping or quoting, mixed with
/// arbitrary Unicode.
fn text() -> impl Strategy<Value = String> {
    let special = prop::sample::select(vec![
        '"', '\\', '\n', '\t', '\r', '\0', '\x01', '\x1f', '\x7f', '\u{80}', '\u{9f}', 'x', '0',
        'a', 'Z', '_', '9', ':', '@', '.', ' ', '-', '{', '}', 'é', '漢', '😀',
    ]);
    prop_oneof![
        prop::collection::vec(prop_oneof![3 => special, 1 => any::<char>()], 0..12)
            .prop_map(|chars| chars.into_iter().collect()),
        any::<String>(),
    ]
}

/// Bytes biased toward the escaped ones and the ASCII boundaries.
fn bytes() -> impl Strategy<Value = Vec<u8>> {
    let special = prop::sample::select(vec![
        b'"', b'\\', b'\n', b'\t', b'\r', 0x00, 0x1f, 0x20, b'x', b'0', 0x7e, 0x7f, 0x80, 0xff,
    ]);
    prop::collection::vec(prop_oneof![special, any::<u8>()], 0..24)
}

/// Whether the printer may write `name` as a bare `@name`.
fn is_bare_symbol(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parse `printed` as one attribute value that must span the whole text.
fn parse_attribute(printed: &str) -> Result<RawAttribute<'_>, TestCaseError> {
    let mut input = printed;
    let attr = raw::raw_attr_value
        .parse_next(&mut input)
        .map_err(|e| TestCaseError::fail(format!("`{printed}` did not parse: {e:?}")))?;
    prop_assert!(input.is_empty(), "`{}` left `{}` unparsed", printed, input);
    Ok(attr)
}

proptest! {
    #[test]
    fn bytes_literals_round_trip(bytes in bytes()) {
        let ctx = IrContext::new();
        let mut printed = String::new();
        write_attribute(&ctx, &mut printed, &Attribute::Bytes(bytes.as_slice().into())).unwrap();

        // The literal is printable ASCII, so it survives any text channel.
        prop_assert!(printed.starts_with("b\"") && printed.ends_with('"'));
        prop_assert!(printed.bytes().all(|b| (0x20..=0x7e).contains(&b)));
        match parse_attribute(&printed)? {
            RawAttribute::Bytes(parsed) => prop_assert_eq!(parsed, bytes),
            other => prop_assert!(false, "`{}` parsed as {:?}", printed, other),
        }
    }

    #[test]
    fn string_literals_round_trip(text in text()) {
        let mut ctx = IrContext::new();
        let attr = ctx.string_attr(&text);
        let mut printed = String::new();
        write_attribute(&ctx, &mut printed, &attr).unwrap();

        prop_assert!(!printed.chars().any(char::is_control));
        match parse_attribute(&printed)? {
            RawAttribute::String(parsed) => prop_assert_eq!(parsed, text),
            other => prop_assert!(false, "`{}` parsed as {:?}", printed, other),
        }
    }

    #[test]
    fn symbol_refs_round_trip_and_quote_only_non_identifiers(name in text()) {
        let mut printed = String::new();
        write_symbol(&mut printed, &Symbol::new(&name)).unwrap();

        let bare = is_bare_symbol(&name);
        prop_assert_eq!(!printed.starts_with("@\""), bare, "printed `{}`", printed);
        if bare {
            prop_assert_eq!(&printed[1..], name.as_str());
        }
        let mut input = printed.as_str();
        let parsed = raw::symbol_ref
            .parse_next(&mut input)
            .map_err(|e| TestCaseError::fail(format!("`{printed}` did not parse: {e:?}")))?;
        prop_assert!(input.is_empty(), "`{}` left `{}` unparsed", printed, input);
        prop_assert_eq!(parsed, name);
    }

    #[test]
    fn symbol_paths_round_trip_component_by_component(
        components in prop::collection::vec(text(), 1..4),
    ) {
        let ctx = IrContext::new();
        let path = SymbolPath::new(components.iter().map(String::as_str));
        let mut printed = String::new();
        write_attribute(&ctx, &mut printed, &Attribute::SymbolRef(path)).unwrap();

        match parse_attribute(&printed)? {
            RawAttribute::SymbolRef(parsed) => prop_assert_eq!(parsed, components),
            other => prop_assert!(false, "`{}` parsed as {:?}", printed, other),
        }
    }
}
