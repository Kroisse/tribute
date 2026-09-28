//! Validation and decoding of numeric literals.
//!
//! The grammar lexes a numeric literal together with every identifier
//! character that follows it, so digit separators, radix digits, exponents,
//! and type suffixes all arrive here as one token. This module checks that
//! token against the literal rules in `new-plans/syntax.md` and decides its
//! final type from its shape and suffix.

use std::ops::Range;

use derive_more::{Display, Error};
use itertools::Itertools;
use winnow::LocatingSlice;
use winnow::combinator::{alt, opt, preceded};
use winnow::prelude::*;
use winnow::token::{one_of, rest, take_while};

/// The decoded value of a numeric literal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NumericValue {
    Nat(u64),
    Int(i64),
    Float(f64),
}

/// An invalid numeric literal.
#[derive(Clone, Debug, PartialEq, Eq, Display, Error)]
#[display("{kind}")]
pub(crate) struct NumericError {
    /// Byte range of the offending part, relative to the literal text.
    pub range: Range<usize>,
    pub kind: NumericErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Display)]
pub(crate) enum NumericErrorKind {
    /// Trailing identifier characters that are not a known suffix.
    #[display("unknown numeric literal suffix `{_0}`; expected `n`, `i`, or `f`")]
    UnknownSuffix(String),
    /// A known suffix on a literal whose shape it cannot apply to.
    #[display("suffix `{suffix}` cannot be used on {conflict}")]
    SuffixNotAllowed {
        suffix: char,
        conflict: SuffixConflict,
    },
    /// Trailing characters after a binary, octal, or hexadecimal literal,
    /// which takes no suffix.
    #[display(
        "{} literals take no suffix, found `{suffix}`{}",
        radix_name(*radix),
        int_form
            .iter()
            .format_with("", |form, f| f(&format_args!("; write `{form}` for an Int")))
    )]
    RadixSuffix {
        radix: u32,
        suffix: String,
        /// The signed spelling that makes the literal an Int, offered for `i`.
        int_form: Option<String>,
    },
    /// A digit outside the literal's radix.
    #[display("invalid digit `{digit}` in {} literal", radix_name(*radix))]
    InvalidDigit { digit: char, radix: u32 },
    /// A radix prefix without any digit.
    #[display("{} literal has no digits", radix_name(*radix))]
    MissingRadixDigits { radix: u32 },
    /// An exponent without any digit.
    #[display("exponent has no digits")]
    MissingExponentDigits,
    /// A negative exponent on a literal that decodes to an integer.
    #[display(
        "an integer literal cannot have a negative exponent; \
         write `{with_point}` or `{with_suffix}` for a Float"
    )]
    NegativeIntegerExponent {
        with_point: String,
        with_suffix: String,
    },
    /// An integer that does not fit the current representation of its type.
    #[display("integer literal exceeds the current implementation limit for `{_0}`")]
    Overflow(IntegerType),
    /// A float literal whose value is not finite.
    #[display("float literal is too large to be represented as `Float`")]
    FloatNotFinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Display)]
pub(crate) enum SuffixConflict {
    #[display("a signed literal")]
    Signed,
    #[display("a literal with a decimal point")]
    Fractional,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Display)]
pub(crate) enum IntegerType {
    Nat,
    Int,
}

fn radix_name(radix: u32) -> &'static str {
    match radix {
        2 => "binary",
        8 => "octal",
        16 => "hexadecimal",
        _ => "decimal",
    }
}

/// Decode a numeric literal token.
///
/// The literal's shape decides its default type: a decimal point makes a
/// Float, a sign makes an Int, and anything else is a Nat. A `n`, `i`, or
/// `f` suffix overrides that default where the rules allow it.
pub(crate) fn parse_numeric_literal(text: &str) -> Result<NumericValue, NumericError> {
    let error = |range: Range<usize>, kind| Err(NumericError { range, kind });
    let Parts {
        sign,
        radix,
        prefix,
        digits,
        fraction,
        exponent,
        suffix,
    } = parts
        .parse(LocatingSlice::new(text))
        .expect("numeric literal lexing accepts any input");
    let (digits, digits_range) = digits;
    let (suffix_text, suffix_range) = suffix;
    let signed = sign.is_some();
    let negative = sign == Some('-');

    if let Some(pos) = digits
        .bytes()
        .position(|b| b != b'_' && !(b as char).is_digit(radix))
    {
        let at = digits_range.start + pos;
        return error(
            at..at + 1,
            NumericErrorKind::InvalidDigit {
                digit: digits.as_bytes()[pos] as char,
                radix,
            },
        );
    }
    if !digits.bytes().any(|b| b != b'_') {
        return error(
            prefix.start..digits_range.end,
            NumericErrorKind::MissingRadixDigits { radix },
        );
    }
    if radix != 10 && !suffix_text.is_empty() {
        return error(
            suffix_range,
            NumericErrorKind::RadixSuffix {
                radix,
                suffix: suffix_text.to_string(),
                int_form: (suffix_text == "i").then(|| {
                    let sign = sign.unwrap_or('+');
                    format!("{sign}{}", &text[prefix.start..digits_range.end])
                }),
            },
        );
    }
    if let Some(exponent) = &exponent
        && !exponent.digits.bytes().any(|b| b != b'_')
    {
        return error(
            exponent.range.clone(),
            NumericErrorKind::MissingExponentDigits,
        );
    }

    let suffix = match suffix_text {
        "" => None,
        "n" => Some('n'),
        "i" => Some('i'),
        "f" => Some('f'),
        other => {
            return error(
                suffix_range,
                NumericErrorKind::UnknownSuffix(other.to_string()),
            );
        }
    };
    let conflict = match suffix {
        Some('n') if signed => Some(SuffixConflict::Signed),
        Some('n' | 'i') if fraction.is_some() => Some(SuffixConflict::Fractional),
        _ => None,
    };
    if let (Some(suffix), Some(conflict)) = (suffix, conflict) {
        return error(
            suffix_range,
            NumericErrorKind::SuffixNotAllowed { suffix, conflict },
        );
    }

    let is_float = match suffix {
        Some('f') => true,
        Some(_) => false,
        None => fraction.is_some(),
    };
    if is_float {
        let mut normalized = String::with_capacity(text.len());
        if negative {
            normalized.push('-');
        }
        normalized.extend(digits.chars().filter(|&c| c != '_'));
        if let Some(fraction) = fraction {
            normalized.push('.');
            normalized.extend(fraction.chars().filter(|&c| c != '_'));
        }
        if let Some(exponent) = &exponent {
            normalized.push('e');
            if exponent.negative {
                normalized.push('-');
            }
            normalized.extend(exponent.digits.chars().filter(|&c| c != '_'));
        }
        let value: f64 = normalized
            .parse()
            .expect("normalized float literal must parse");
        return if value.is_finite() {
            Ok(NumericValue::Float(value))
        } else {
            error(0..text.len(), NumericErrorKind::FloatNotFinite)
        };
    }

    let ty = match suffix {
        Some('i') => IntegerType::Int,
        Some(_) => IntegerType::Nat,
        None if signed => IntegerType::Int,
        None => IntegerType::Nat,
    };
    if let Some(exponent) = &exponent
        && exponent.negative
    {
        let sign = &text[..prefix.start];
        let mantissa = &text[prefix.start..exponent.range.start];
        let exp_text = &text[exponent.range.clone()];
        return error(
            exponent.range.clone(),
            NumericErrorKind::NegativeIntegerExponent {
                with_point: format!("{sign}{mantissa}.0{exp_text}"),
                with_suffix: format!("{sign}{mantissa}{exp_text}f"),
            },
        );
    }

    let overflow = || error(0..text.len(), NumericErrorKind::Overflow(ty));
    let Some(mut magnitude) = digits
        .bytes()
        .filter(|&b| b != b'_')
        .try_fold(0u64, |acc, b| {
            let digit = (b as char).to_digit(radix)?;
            acc.checked_mul(radix as u64)?.checked_add(digit as u64)
        })
    else {
        return overflow();
    };
    if let Some(exponent) = &exponent
        && magnitude != 0
    {
        let scale = exponent
            .digits
            .bytes()
            .filter(|&b| b != b'_')
            .try_fold(0u32, |acc, b| {
                acc.checked_mul(10)?.checked_add((b - b'0') as u32)
            })
            .and_then(|exp| 10u64.checked_pow(exp));
        match scale.and_then(|scale| magnitude.checked_mul(scale)) {
            Some(scaled) => magnitude = scaled,
            None => return overflow(),
        }
    }

    match ty {
        IntegerType::Nat => Ok(NumericValue::Nat(magnitude)),
        IntegerType::Int if negative => {
            // The magnitude of i64::MIN is one more than i64::MAX.
            match 0i64.checked_sub_unsigned(magnitude) {
                Some(value) => Ok(NumericValue::Int(value)),
                None => overflow(),
            }
        }
        IntegerType::Int => match i64::try_from(magnitude) {
            Ok(value) => Ok(NumericValue::Int(value)),
            Err(_) => overflow(),
        },
    }
}

type Input<'a> = LocatingSlice<&'a str>;

/// The lexical parts of a numeric literal token.
///
/// Every part is optional or may be empty, so lexing accepts any text and
/// all validation happens in [`parse_numeric_literal`].
struct Parts<'a> {
    sign: Option<char>,
    radix: u32,
    /// The radix prefix, or an empty range at the magnitude's start.
    prefix: Range<usize>,
    digits: (&'a str, Range<usize>),
    /// Digits after the decimal point; decimal literals only.
    fraction: Option<&'a str>,
    /// Decimal literals only.
    exponent: Option<Exponent<'a>>,
    /// Everything after the number, possibly empty.
    suffix: (&'a str, Range<usize>),
}

struct Exponent<'a> {
    range: Range<usize>,
    negative: bool,
    digits: &'a str,
}

fn parts<'a>(input: &mut Input<'a>) -> ModalResult<Parts<'a>> {
    let sign = opt(one_of(['+', '-'])).parse_next(input)?;
    let (radix, prefix) = opt(preceded(
        '0',
        alt((
            one_of(['x', 'X']).value(16),
            one_of(['o', 'O']).value(8),
            one_of(['b', 'B']).value(2),
        )),
    ))
    .with_span()
    .parse_next(input)?;
    let radix = radix.unwrap_or(10);
    // Binary and octal take any decimal digit so that an out-of-radix digit
    // is reported rather than read as a suffix.
    let digits = take_while(0.., move |c: char| match radix {
        16 => c.is_ascii_hexdigit() || c == '_',
        _ => c.is_ascii_digit() || c == '_',
    })
    .with_span()
    .parse_next(input)?;
    let (fraction, exponent) = if radix == 10 {
        (
            opt(preceded('.', decimal_digits)).parse_next(input)?,
            opt(exponent).parse_next(input)?,
        )
    } else {
        (None, None)
    };
    let suffix = rest.with_span().parse_next(input)?;
    Ok(Parts {
        sign,
        radix,
        prefix,
        digits,
        fraction,
        exponent,
        suffix,
    })
}

fn decimal_digits<'a>(input: &mut Input<'a>) -> ModalResult<&'a str> {
    take_while(0.., |c: char| c.is_ascii_digit() || c == '_').parse_next(input)
}

fn exponent<'a>(input: &mut Input<'a>) -> ModalResult<Exponent<'a>> {
    (one_of(['e', 'E']), opt(one_of(['+', '-'])), decimal_digits)
        .with_span()
        .map(|((_, sign, digits), range)| Exponent {
            range,
            negative: sign == Some('-'),
            digits,
        })
        .parse_next(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(text: &str) -> NumericValue {
        parse_numeric_literal(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    fn error(text: &str) -> NumericError {
        match parse_numeric_literal(text) {
            Ok(value) => panic!("{text} must be rejected, got {value:?}"),
            Err(error) => error,
        }
    }

    #[test]
    fn unsigned_literals_are_nat() {
        for (text, expected) in [
            ("0", 0),
            ("42", 42),
            ("1_000", 1000),
            ("1__0", 10),
            ("1_", 1),
            ("0b1010", 10),
            ("0B1010_1010", 170),
            ("0o777", 511),
            ("0O10", 8),
            ("0xFF", 255),
            ("0x_FF", 255),
            ("0xDEAD_BEEF", 0xDEAD_BEEF),
            ("0x1f", 31),
            ("0x1e", 30),
            ("1e10", 10_000_000_000),
            ("2E+3", 2000),
            ("1_0e1_0", 100_000_000_000),
            ("1e_1", 10),
            ("0e999", 0),
            ("42n", 42),
            ("1e3n", 1000),
            ("18446744073709551615", u64::MAX),
        ] {
            assert_eq!(value(text), NumericValue::Nat(expected), "{text}");
        }
    }

    #[test]
    fn signed_and_suffixed_literals_are_int() {
        for (text, expected) in [
            ("+42", 42),
            ("-42", -42),
            ("-0xFF", -255),
            ("-1e3", -1000),
            ("42i", 42),
            ("-42i", -42),
            ("1e3i", 1000),
            ("1_000_i", 1000),
            ("+9223372036854775807", i64::MAX),
            ("-9223372036854775808", i64::MIN),
        ] {
            assert_eq!(value(text), NumericValue::Int(expected), "{text}");
        }
    }

    #[test]
    fn decimal_point_or_f_suffix_is_float() {
        for (text, expected) in [
            ("1.0", 1.0),
            ("-2.75", -2.75),
            ("1_000.000_1", 1000.0001),
            ("1.5e-3", 0.0015),
            ("+1.0E10", 1.0e10),
            ("42f", 42.0),
            ("-1f", -1.0),
            ("1e-3f", 0.001),
            ("1e10f", 1.0e10),
            ("1.5f", 1.5),
        ] {
            assert_eq!(value(text), NumericValue::Float(expected), "{text}");
        }
    }

    #[test]
    fn rejects_malformed_literals_at_the_offending_part() {
        use NumericErrorKind::*;
        for (text, range, kind) in [
            ("42u8", 2..4, UnknownSuffix("u8".into())),
            ("1_000_x", 6..7, UnknownSuffix("x".into())),
            (
                "0xFFi",
                4..5,
                RadixSuffix {
                    radix: 16,
                    suffix: "i".into(),
                    int_form: Some("+0xFF".into()),
                },
            ),
            (
                "-0o17i",
                5..6,
                RadixSuffix {
                    radix: 8,
                    suffix: "i".into(),
                    int_form: Some("-0o17".into()),
                },
            ),
            (
                "0xFFn",
                4..5,
                RadixSuffix {
                    radix: 16,
                    suffix: "n".into(),
                    int_form: None,
                },
            ),
            (
                "0xFFg",
                4..5,
                RadixSuffix {
                    radix: 16,
                    suffix: "g".into(),
                    int_form: None,
                },
            ),
            (
                "-1n",
                2..3,
                SuffixNotAllowed {
                    suffix: 'n',
                    conflict: SuffixConflict::Signed,
                },
            ),
            (
                "1.5i",
                3..4,
                SuffixNotAllowed {
                    suffix: 'i',
                    conflict: SuffixConflict::Fractional,
                },
            ),
            (
                "0b1f",
                3..4,
                RadixSuffix {
                    radix: 2,
                    suffix: "f".into(),
                    int_form: None,
                },
            ),
            (
                "0b102",
                4..5,
                InvalidDigit {
                    digit: '2',
                    radix: 2,
                },
            ),
            (
                "-0o8",
                3..4,
                InvalidDigit {
                    digit: '8',
                    radix: 8,
                },
            ),
            ("0x", 0..2, MissingRadixDigits { radix: 16 }),
            ("0x_", 0..3, MissingRadixDigits { radix: 16 }),
            ("1e", 1..2, MissingExponentDigits),
            ("1.0e-_", 3..6, MissingExponentDigits),
            (
                "1e-3",
                1..4,
                NegativeIntegerExponent {
                    with_point: "1.0e-3".into(),
                    with_suffix: "1e-3f".into(),
                },
            ),
            (
                "-2e-1i",
                2..5,
                NegativeIntegerExponent {
                    with_point: "-2.0e-1".into(),
                    with_suffix: "-2e-1f".into(),
                },
            ),
            ("18446744073709551616", 0..20, Overflow(IntegerType::Nat)),
            ("1e20", 0..4, Overflow(IntegerType::Nat)),
            ("9223372036854775808i", 0..20, Overflow(IntegerType::Int)),
            ("-9223372036854775809", 0..20, Overflow(IntegerType::Int)),
            ("1e999f", 0..6, FloatNotFinite),
        ] {
            let error = error(text);
            assert_eq!((error.range.clone(), error.kind), (range, kind), "{text}");
        }
    }

    #[test]
    fn negative_exponent_message_suggests_float_forms() {
        assert_eq!(
            error("1e-3").to_string(),
            "an integer literal cannot have a negative exponent; \
             write `1.0e-3` or `1e-3f` for a Float"
        );
    }

    #[test]
    fn radix_suffix_message_suggests_sign() {
        assert_eq!(
            error("0xFFi").to_string(),
            "hexadecimal literals take no suffix, found `i`; write `+0xFF` for an Int"
        );
        assert_eq!(
            error("0b1f").to_string(),
            "binary literals take no suffix, found `f`"
        );
    }
}
