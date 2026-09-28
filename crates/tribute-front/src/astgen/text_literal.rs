//! Decoding of string, bytes, and rune literals.
//!
//! A literal is decoded from its source text following `new-plans/syntax.md`:
//! delimiters are removed, a block literal's indentation is stripped, line
//! breaks are normalized to LF, and escapes are processed. Indentation is
//! decided on the source text, before escapes and interpolation.

use std::fmt;
use std::ops::Range;

use derive_more::{Display, Error};
use itertools::Itertools;
use tree_sitter::Node;

use super::context::AstLoweringCtx;
use super::helpers::report_in_node;

/// An invalid string, bytes, or rune literal.
#[derive(Clone, Debug, PartialEq, Eq, Display, Error)]
#[display("{kind}")]
pub(crate) struct LiteralError {
    /// Byte range of the offending part, relative to the literal text.
    pub range: Range<usize>,
    pub kind: LiteralErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Display)]
pub(crate) enum LiteralErrorKind {
    /// A `\u{…}` escape naming a surrogate code point (`D800`–`DFFF`).
    #[display(
        "invalid Unicode escape: U+{_0:04X} is a surrogate code point, \
         not a Unicode scalar value"
    )]
    Surrogate(u32),
    /// A `\u{…}` escape naming a value above `10FFFF`.
    #[display("invalid Unicode escape: U+{_0:04X} exceeds the maximum U+10FFFF")]
    OutOfRange(u32),
    /// A backslash followed by a character that starts no escape.
    #[display("unknown escape sequence `{_0}`")]
    UnknownEscape(String),
    /// A backslash directly before a line break, reserved for line
    /// continuation.
    #[display("`\\` before a line break is not an escape sequence")]
    LineContinuation,
    /// An escape without the digits or braces its form requires.
    #[display("malformed escape sequence `{_0}`")]
    MalformedEscape(String),
    /// A `\xHH` escape above `7F` in a string.
    #[display("`\\x{_0:02X}` is outside ASCII; write a non-ASCII character as `\\u{{…}}`")]
    NonAsciiByteEscape(u8),
    /// A `\u{…}` escape in a bytes literal.
    #[display("bytes literals cannot contain `\\u{{…}}`; write each byte as `\\xHH`")]
    UnicodeEscapeInBytes,
    /// A block literal whose closing delimiter follows other content on its
    /// line.
    #[display("the closing delimiter of a block literal must be on its own line")]
    ClosingDelimiterNotOnOwnLine,
    /// A block literal line that does not start with the indentation prefix.
    #[display(
        "line is not indented with the block literal's indentation ({}), \
         set by the whitespace before the closing delimiter",
        Indentation(_0)
    )]
    InsufficientIndentation(String),
}

/// Describes an indentation prefix, e.g. `4 spaces` or `1 tab, 2 spaces`.
struct Indentation<'a>(&'a str);

impl fmt::Display for Indentation<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("none");
        }
        let runs = self.0.chars().chunk_by(|&c| c);
        let runs = runs.into_iter().format_with(", ", |(c, run), f| {
            let count = run.count();
            let name = if c == '\t' { "tab" } else { "space" };
            let plural = if count == 1 { "" } else { "s" };
            f(&format_args!("{count} {name}{plural}"))
        });
        write!(f, "{runs}")
    }
}

/// A decoded piece of a literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LiteralPart {
    Text(Vec<u8>),
    /// An interpolation, by its byte range in the literal text.
    Interpolation(Range<usize>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextKind {
    String,
    Bytes,
}

/// Decode a string or bytes literal node, reporting any error.
///
/// Returns `None` when the literal is invalid or interpolates, since
/// interpolation is not lowered yet.
pub(crate) fn lower_text_literal(ctx: &mut AstLoweringCtx<'_>, node: &Node) -> Option<Vec<u8>> {
    let text = ctx.node_text_owned(node);
    let start = node.start_byte();
    let mut cursor = node.walk();
    let interpolations = node
        .named_children(&mut cursor)
        .filter(|child| child.kind().ends_with("interpolation"))
        .map(|child| child.start_byte() - start..child.end_byte() - start)
        .collect_vec();
    let parts = match decode_text_literal(&text, &interpolations) {
        Ok(parts) => parts,
        Err(error) => {
            report_in_node(ctx, node, error.range.clone(), error.to_string());
            return None;
        }
    };
    let mut value = Vec::new();
    let mut valid = true;
    for part in parts {
        match part {
            LiteralPart::Text(bytes) => value.extend(bytes),
            LiteralPart::Interpolation(range) => {
                report_in_node(ctx, node, range, "interpolation is not implemented yet");
                valid = false;
            }
        }
    }
    valid.then_some(value)
}

/// Decode a string or bytes literal's `text`, given the byte ranges of its
/// interpolations within it.
///
/// The grammar guarantees the delimiters; text without them comes from error
/// recovery, which the parser already reported, and decodes as empty.
pub(crate) fn decode_text_literal(
    text: &str,
    interpolations: &[Range<usize>],
) -> Result<Vec<LiteralPart>, LiteralError> {
    let Some(literal) = Delimited::parse(text) else {
        return Ok(Vec::new());
    };
    let segments = segments(literal.content.clone(), interpolations);
    let stripped = if literal.hashes > 0 {
        block_stripped_ranges(text, literal.content.clone(), &segments)?
    } else {
        Vec::new()
    };

    let mut parts = Vec::new();
    let mut buffer = Vec::new();
    for segment in segments {
        match segment {
            Segment::Text(range) => {
                for span in subtract(range, &stripped) {
                    if literal.raw {
                        push_raw(&text[span], &mut buffer);
                    } else {
                        decode_escapes(text, span, literal.kind, &mut buffer)?;
                    }
                }
            }
            Segment::Interpolation(range) => {
                if !buffer.is_empty() {
                    parts.push(LiteralPart::Text(std::mem::take(&mut buffer)));
                }
                parts.push(LiteralPart::Interpolation(range));
            }
        }
    }
    if !buffer.is_empty() || parts.is_empty() {
        parts.push(LiteralPart::Text(buffer));
    }
    Ok(parts)
}

/// The delimiters of a string or bytes literal.
struct Delimited {
    kind: TextKind,
    raw: bool,
    hashes: usize,
    /// Byte range between the quotes.
    content: Range<usize>,
}

impl Delimited {
    fn parse(text: &str) -> Option<Self> {
        let prefix_len = text
            .bytes()
            .take_while(|b| matches!(b, b'r' | b's' | b'b'))
            .count();
        let prefix = &text[..prefix_len];
        let hashes = text[prefix_len..]
            .bytes()
            .take_while(|&b| b == b'#')
            .count();
        let open = prefix_len + hashes;
        let close = text.len().checked_sub(1 + hashes)?;
        let closes = text[close..].strip_prefix('"')?.bytes().all(|b| b == b'#');
        (text[open..].starts_with('"') && open < close && closes).then(|| Self {
            kind: if prefix.contains('b') {
                TextKind::Bytes
            } else {
                TextKind::String
            },
            raw: prefix.contains('r'),
            hashes,
            content: open + 1..close,
        })
    }
}

#[derive(Clone, Debug)]
enum Segment {
    Text(Range<usize>),
    Interpolation(Range<usize>),
}

/// Split `content` into literal text and the interpolations inside it.
fn segments(content: Range<usize>, interpolations: &[Range<usize>]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut start = content.start;
    for interpolation in interpolations {
        segments.push(Segment::Text(start..interpolation.start));
        segments.push(Segment::Interpolation(interpolation.clone()));
        start = interpolation.end;
    }
    segments.push(Segment::Text(start..content.end));
    segments
}

/// The source ranges a `#`-delimited literal drops as a block literal: the
/// opening line break, each line's indentation prefix, and the closing line.
/// A literal that is not a block literal drops nothing.
fn block_stripped_ranges(
    text: &str,
    content: Range<usize>,
    segments: &[Segment],
) -> Result<Vec<Range<usize>>, LiteralError> {
    let opening_ws = text[content.clone()]
        .bytes()
        .take_while(|&b| matches!(b, b' ' | b'\t'))
        .count();
    let Some(opening_break) = line_break_at(text, content.start + opening_ws) else {
        return Ok(Vec::new());
    };

    // Line breaks in literal text, not inside interpolations.
    let breaks = segments
        .iter()
        .filter_map(|segment| match segment {
            Segment::Text(range) => Some(range.clone()),
            Segment::Interpolation(_) => None,
        })
        .flat_map(|range| {
            let mut position = range.start;
            std::iter::from_fn(move || {
                let offset = text[position..range.end].find(['\n', '\r'])?;
                let line_break = line_break_at(text, position + offset)?;
                position = line_break.end;
                Some(line_break)
            })
        })
        .collect_vec();

    let closing_break = breaks.last().expect("the opening line break").clone();
    let closing_line = closing_break.end..content.end;
    let closing_in_text = matches!(
        segments.last(),
        Some(Segment::Text(range)) if range.start <= closing_break.start
    );
    if !closing_in_text || !is_blank(&text[closing_line.clone()]) {
        return Err(LiteralError {
            range: content.end..text.len(),
            kind: LiteralErrorKind::ClosingDelimiterNotOnOwnLine,
        });
    }
    let prefix = &text[closing_line];

    let mut stripped = Vec::new();
    stripped.push(content.start..opening_break.end);
    let body_breaks = &breaks[..breaks.len() - 1];
    let line_starts = std::iter::once(opening_break.end)
        .chain(body_breaks.iter().skip(1).map(|line_break| line_break.end));
    let line_ends = body_breaks
        .iter()
        .skip(1)
        .map(|line_break| line_break.start);
    let line_ends = line_ends.chain(std::iter::once(closing_break.start));
    for (start, end) in line_starts.zip(line_ends) {
        if start >= end {
            continue;
        }
        let line = &text[start..end];
        let interpolated = segments.iter().any(|segment| {
            matches!(segment, Segment::Interpolation(range) if (start..end).contains(&range.start))
        });
        if line.starts_with(prefix) {
            stripped.push(start..start + prefix.len());
        } else if !interpolated && is_blank(line) {
            stripped.push(start..end);
        } else {
            let indent = line.len() - line.trim_start_matches([' ', '\t']).len();
            let width = match indent {
                0 => line.chars().next().map_or(0, char::len_utf8),
                indent => indent,
            };
            return Err(LiteralError {
                range: start..start + width,
                kind: LiteralErrorKind::InsufficientIndentation(prefix.to_string()),
            });
        }
    }
    stripped.push(closing_break.start..content.end);
    Ok(stripped)
}

/// The line break (`\n`, `\r\n`, or `\r`) starting at `position`, if any.
fn line_break_at(text: &str, position: usize) -> Option<Range<usize>> {
    let rest = &text[position..];
    let len = if rest.starts_with("\r\n") {
        2
    } else if rest.starts_with(['\n', '\r']) {
        1
    } else {
        return None;
    };
    Some(position..position + len)
}

fn is_blank(text: &str) -> bool {
    text.bytes().all(|b| matches!(b, b' ' | b'\t'))
}

/// The parts of `range` outside the sorted, disjoint `removed` ranges.
fn subtract(range: Range<usize>, removed: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut start = range.start;
    for removed in removed {
        let (lo, hi) = (removed.start.max(range.start), removed.end.min(range.end));
        if lo >= hi {
            continue;
        }
        if start < lo {
            spans.push(start..lo);
        }
        start = start.max(hi);
    }
    if start < range.end {
        spans.push(start..range.end);
    }
    spans
}

/// Append raw literal text, normalizing line breaks to LF.
fn push_raw(text: &str, out: &mut Vec<u8>) {
    let bytes = text.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        match byte {
            b'\r' if bytes.get(i + 1) == Some(&b'\n') => {}
            b'\r' => out.push(b'\n'),
            _ => out.push(byte),
        }
    }
}

/// Decode `span` of `text`, processing escapes and normalizing line breaks to
/// LF. Errors are reported relative to `text`.
fn decode_escapes(
    text: &str,
    span: Range<usize>,
    kind: TextKind,
    out: &mut Vec<u8>,
) -> Result<(), LiteralError> {
    let bytes = text.as_bytes();
    let mut i = span.start;
    while i < span.end {
        match bytes[i] {
            b'\\' => i = decode_escape(text, i, span.end, kind, out)?,
            b'\r' => {
                if bytes.get(i + 1) != Some(&b'\n') {
                    out.push(b'\n');
                }
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    Ok(())
}

/// Decode the escape starting at the backslash at `start`, returning the
/// position after it.
fn decode_escape(
    text: &str,
    start: usize,
    end: usize,
    kind: TextKind,
    out: &mut Vec<u8>,
) -> Result<usize, LiteralError> {
    let rest = &text[start + 1..end];
    let error = |len: usize, kind| LiteralError {
        range: start..start + 1 + len,
        kind,
    };
    let Some(escape) = rest.chars().next() else {
        return Err(error(
            0,
            LiteralErrorKind::MalformedEscape("\\".to_string()),
        ));
    };
    let simple = match escape {
        'n' => Some(b'\n'),
        'r' => Some(b'\r'),
        't' => Some(b'\t'),
        '0' => Some(b'\0'),
        '\\' => Some(b'\\'),
        '"' => Some(b'"'),
        _ => None,
    };
    if let Some(byte) = simple {
        out.push(byte);
        return Ok(start + 2);
    }
    match escape {
        'x' => {
            let hex = rest
                .get(1..3)
                .filter(|hex| hex.bytes().all(|b| b.is_ascii_hexdigit()));
            let Some(hex) = hex else {
                let len = rest
                    .char_indices()
                    .take_while(|&(i, c)| i < 3 && (i == 0 || c.is_ascii_hexdigit()))
                    .count();
                let malformed = format!("\\{}", &rest[..len]);
                return Err(error(len, LiteralErrorKind::MalformedEscape(malformed)));
            };
            let byte = u8::from_str_radix(hex, 16).expect("two hex digits");
            if kind == TextKind::String && byte > 0x7F {
                return Err(error(3, LiteralErrorKind::NonAsciiByteEscape(byte)));
            }
            out.push(byte);
            Ok(start + 4)
        }
        'u' => {
            let digits = rest
                .strip_prefix("u{")
                .map(|body| body.bytes().take_while(u8::is_ascii_hexdigit).count());
            let len = match digits {
                Some(digits @ 1..=6) if rest[2 + digits..].starts_with('}') => 3 + digits,
                _ => {
                    let len = rest
                        .find('}')
                        .filter(|&len| len < 10)
                        .map_or(1, |len| len + 1);
                    let malformed = format!("\\{}", &rest[..len]);
                    return Err(error(len, LiteralErrorKind::MalformedEscape(malformed)));
                }
            };
            if kind == TextKind::Bytes {
                return Err(error(len, LiteralErrorKind::UnicodeEscapeInBytes));
            }
            let ch = decode_unicode_escape(&rest[2..len - 1])
                .map_err(|kind| error(len, kind))?
                .expect("validated hex digits");
            out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes());
            Ok(start + 1 + len)
        }
        '\n' | '\r' => {
            let len = line_break_at(text, start + 1).map_or(1, |line_break| line_break.len());
            Err(error(len, LiteralErrorKind::LineContinuation))
        }
        other => Err(error(
            other.len_utf8(),
            LiteralErrorKind::UnknownEscape(format!("\\{other}")),
        )),
    }
}

/// Decode the hex digits of a `\u{…}` escape into a Unicode scalar value.
///
/// Returns `Ok(None)` when `hex` is not a hex number; the grammar rejects that
/// shape, so it only occurs for malformed input.
pub(crate) fn decode_unicode_escape(hex: &str) -> Result<Option<char>, LiteralErrorKind> {
    let Ok(value) = u32::from_str_radix(hex, 16) else {
        return Ok(None);
    };
    match char::from_u32(value) {
        Some(ch) => Ok(Some(ch)),
        None if (0xD800..=0xDFFF).contains(&value) => Err(LiteralErrorKind::Surrogate(value)),
        None => Err(LiteralErrorKind::OutOfRange(value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a literal without interpolation into its bytes.
    fn value(text: &str) -> Vec<u8> {
        match decode_text_literal(text, &[]) {
            Ok(parts) => match parts.as_slice() {
                [LiteralPart::Text(bytes)] => bytes.clone(),
                parts => panic!("{text:?}: unexpected parts {parts:?}"),
            },
            Err(error) => panic!("{text:?}: {error}"),
        }
    }

    fn string(text: &str) -> String {
        String::from_utf8(value(text)).unwrap()
    }

    fn error(text: &str) -> LiteralError {
        decode_text_literal(text, &[]).expect_err(text)
    }

    /// Assert that `text` is rejected with `kind` over `part`, the first
    /// occurrence of it at or after `after`.
    fn assert_error(text: &str, after: &str, part: &str, kind: LiteralErrorKind) {
        let from = text.find(after).expect("anchor must occur");
        let start = from + text[from..].find(part).expect("part must occur");
        let error = error(text);
        assert_eq!(
            (error.range, error.kind),
            (start..start + part.len(), kind),
            "{text:?}"
        );
    }

    #[test]
    fn delimiters_are_removed() {
        for (text, expected) in [
            (r#""hello""#, "hello"),
            (r#""""#, ""),
            (r#"s"hi""#, "hi"),
            (r#"r"\d+""#, r"\d+"),
            (r#"rs"\n""#, r"\n"),
            (r##"#"hello"#"##, "hello"),
            (r##"s#"say "hi""#"##, r#"say "hi""#),
            (r###"r##"a "# b"##"###, r##"a "# b"##),
            (r##"r#""#"##, ""),
        ] {
            assert_eq!(string(text), expected, "{text}");
        }
        assert_eq!(value(r#"b"\xFF\x00""#), [0xFF, 0x00]);
        assert_eq!(value(r##"b#"\x41"#"##), b"A");
        assert_eq!(value(r#"rb"\x41""#), br"\x41");
    }

    #[test]
    fn block_literal_strips_closing_indentation() {
        let text = "s#\"\n        SELECT *\n          FROM t\n        \"#";
        assert_eq!(string(text), "SELECT *\n  FROM t");
    }

    #[test]
    fn closing_delimiter_position_keeps_indentation() {
        let text = "s#\"\n    SELECT *\n  \"#";
        assert_eq!(string(text), "  SELECT *");
    }

    #[test]
    fn blank_line_before_closing_delimiter_keeps_final_newline() {
        let text = "#\"\n    a\n\n    \"#";
        assert_eq!(string(text), "a\n");
    }

    #[test]
    fn whitespace_after_opening_delimiter_is_ignored() {
        let text = "#\" \t\n    a\n    \"#";
        assert_eq!(string(text), "a");
    }

    #[test]
    fn blank_lines_may_be_short() {
        let text = "#\"\n    a\n  \n\n\t\n      b  \n    \"#";
        assert_eq!(string(text), "a\n\n\n\n  b  ");
    }

    #[test]
    fn empty_block_literals() {
        assert_eq!(string("#\"\n\"#"), "");
        assert_eq!(string("#\"\n    \"#"), "");
        assert_eq!(string("#\"\n\n    \"#"), "");
    }

    #[test]
    fn escapes_are_content_not_indentation() {
        let text = "s#\"\n    \\tx\n    \\u{20}y\n    \"#";
        assert_eq!(string(text), "\tx\n y");
    }

    #[test]
    fn raw_and_bytes_block_literals_strip_too() {
        assert_eq!(string("r#\"\n    \\d+\n    \"#"), r"\d+");
        assert_eq!(string("rs##\"\n  a\"#b\n  \"##"), "a\"#b");
        assert_eq!(value("b#\"\n    \\x00a\n    \"#"), b"\0a");
        assert_eq!(value("rb#\"\n    \\x00\n    \"#"), br"\x00");
    }

    #[test]
    fn single_line_hash_literals_are_verbatim() {
        assert_eq!(string("#\"  a\n  \"#"), "  a\n  ");
        assert_eq!(string("r#\" x\"#"), " x");
    }

    #[test]
    fn line_breaks_are_normalized() {
        assert_eq!(string("\"a\r\nb\rc\""), "a\nb\nc");
        assert_eq!(string("r\"a\r\nb\rc\""), "a\nb\nc");
        assert_eq!(string("#\"\r\n    a\r\n    b\r\n    \"#"), "a\nb");
        assert_eq!(string(r#""\r\n""#), "\r\n");
    }

    #[test]
    fn indentation_must_match_the_closing_line() {
        let text = "#\"\n    a\n  b\n    \"#";
        assert_error(
            text,
            "a\n",
            "  ",
            LiteralErrorKind::InsufficientIndentation("    ".into()),
        );
        let text = "#\"\n    a\nb\n    \"#";
        assert_error(
            text,
            "a\n",
            "b",
            LiteralErrorKind::InsufficientIndentation("    ".into()),
        );
        let text = "#\"\n\ta\n    \"#";
        assert_error(
            text,
            "",
            "\t",
            LiteralErrorKind::InsufficientIndentation("    ".into()),
        );
    }

    #[test]
    fn closing_delimiter_must_be_on_its_own_line() {
        assert_error(
            "#\"\n    a\n    b\"#",
            "b",
            "\"#",
            LiteralErrorKind::ClosingDelimiterNotOnOwnLine,
        );
        assert_error(
            "s##\"\n  a\"##",
            "a",
            "\"##",
            LiteralErrorKind::ClosingDelimiterNotOnOwnLine,
        );
    }

    #[test]
    fn indentation_message_describes_the_prefix() {
        assert_eq!(
            error("#\"\n a\n\t  \"#").to_string(),
            "line is not indented with the block literal's indentation (1 tab, 2 spaces), \
             set by the whitespace before the closing delimiter"
        );
        assert_eq!(
            LiteralErrorKind::InsufficientIndentation("    ".into()).to_string(),
            "line is not indented with the block literal's indentation (4 spaces), \
             set by the whitespace before the closing delimiter"
        );
    }

    #[test]
    fn interpolations_do_not_take_part_in_indentation() {
        // The interpolation spans lines with less indentation than the prefix.
        let text = "s#\"\n    a \\{f(\nx)} b\n    \\{y}\n    \"#";
        let first = text.find("\\{f").unwrap()..text.find(")}").unwrap() + 2;
        let second = text.find("\\{y").unwrap()..text.find("y}").unwrap() + 2;
        assert_eq!(
            decode_text_literal(text, &[first.clone(), second.clone()]).unwrap(),
            [
                LiteralPart::Text(b"a ".to_vec()),
                LiteralPart::Interpolation(first),
                LiteralPart::Text(b" b\n".to_vec()),
                LiteralPart::Interpolation(second),
            ]
        );
    }

    #[test]
    fn interpolation_must_follow_the_prefix() {
        let text = "s#\"\n    a\n\\{y}\n    \"#";
        let interpolation = text.find("\\{").unwrap()..text.find("}").unwrap() + 1;
        let error = decode_text_literal(text, std::slice::from_ref(&interpolation)).unwrap_err();
        assert_eq!(error.range, interpolation.start..interpolation.start + 1);
    }

    #[test]
    fn interpolation_on_the_closing_line_is_rejected() {
        let text = "s#\"\n    a\n    \\{y}\"#";
        let interpolation = text.find("\\{").unwrap()..text.find("}").unwrap() + 1;
        let error = decode_text_literal(text, &[interpolation]).unwrap_err();
        assert_eq!(error.kind, LiteralErrorKind::ClosingDelimiterNotOnOwnLine);
    }

    #[test]
    fn escapes_decode_to_utf8_bytes() {
        assert_eq!(string(r#""\u{E9}\x41\n\t\r\0\\\"""#), "éA\n\t\r\0\\\"");
    }

    #[test]
    fn invalid_escapes_are_rejected() {
        use LiteralErrorKind::*;
        for (text, part, kind) in [
            (r##"#"a\qb"#"##, r"\q", UnknownEscape(r"\q".into())),
            (r##"#"\é"#"##, r"\é", UnknownEscape(r"\é".into())),
            ("#\"a\\\nb\"#", "\\\n", LineContinuation),
            ("#\"a\\\r\nb\"#", "\\\r\n", LineContinuation),
            (r##"#"\x4"#"##, r"\x4", MalformedEscape(r"\x4".into())),
            (r##"#"\xg0"#"##, r"\x", MalformedEscape(r"\x".into())),
            (r##"#"\u{}"#"##, r"\u{}", MalformedEscape(r"\u{}".into())),
            (r##"#"\u41"#"##, r"\u", MalformedEscape(r"\u".into())),
            (
                r##"#"\u{1234567}"#"##,
                r"\u{1234567}",
                MalformedEscape(r"\u{1234567}".into()),
            ),
            (r##"#"\x80"#"##, r"\x80", NonAsciiByteEscape(0x80)),
            (r##"b#"\u{41}"#"##, r"\u{41}", UnicodeEscapeInBytes),
            (r#""a\u{D800}b""#, r"\u{D800}", Surrogate(0xD800)),
            (r#"s"\u{110000}""#, r"\u{110000}", OutOfRange(0x110000)),
        ] {
            assert_error(text, "", part, kind);
        }
    }

    #[test]
    fn raw_literals_keep_backslashes() {
        assert_eq!(string(r##"r#"\q\u{D800}"#"##), r"\q\u{D800}");
    }

    #[test]
    fn unicode_escape_scalar_value_boundaries() {
        for (hex, expected) in [
            ("0", '\0'),
            ("D7FF", '\u{D7FF}'),
            ("e000", '\u{E000}'),
            ("10FFFF", '\u{10FFFF}'),
            ("000041", 'A'),
        ] {
            assert_eq!(decode_unicode_escape(hex), Ok(Some(expected)), "{hex}");
        }
        assert_eq!(
            decode_unicode_escape("D800"),
            Err(LiteralErrorKind::Surrogate(0xD800))
        );
        assert_eq!(
            decode_unicode_escape("DFFF"),
            Err(LiteralErrorKind::Surrogate(0xDFFF))
        );
        assert_eq!(
            decode_unicode_escape("110000"),
            Err(LiteralErrorKind::OutOfRange(0x110000))
        );
    }
}
