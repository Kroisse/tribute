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
use winnow::combinator::{alt, dispatch, eof, opt, preceded, repeat, terminated};
use winnow::error::{ErrMode, ParserError};
use winnow::prelude::*;
use winnow::stream::Location;
use winnow::token::{one_of, rest};

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
    match numeric_literal.parse_next(&mut LocatingSlice::new(text)) {
        Ok(value) => Ok(value),
        Err(ErrMode::Cut(LexError(Some(error)))) => Err(error),
        Err(_) => unreachable!("numeric literal rules fail with a cut error"),
    }
}

type Input<'a> = LocatingSlice<&'a str>;
type PResult<O> = ModalResult<O, LexError>;

/// Parser error: a backtrack carries nothing, and a violated literal rule is
/// a cut error carrying its diagnostic.
#[derive(Debug)]
struct LexError(Option<NumericError>);

impl<'a> ParserError<Input<'a>> for LexError {
    type Inner = Self;

    fn from_input(_input: &Input<'a>) -> Self {
        Self(None)
    }

    fn into_inner(self) -> Result<Self::Inner, Self> {
        Ok(self)
    }
}

/// Stop parsing and report `kind` at `range` of the literal.
fn reject<O>(range: Range<usize>, kind: NumericErrorKind) -> PResult<O> {
    Err(ErrMode::Cut(LexError(Some(NumericError { range, kind }))))
}

fn numeric_literal(input: &mut Input<'_>) -> PResult<NumericValue> {
    let sign = opt(one_of(['+', '-'])).parse_next(input)?;
    dispatch! {opt(radix_prefix).with_taken().with_span();
        ((Some(radix), prefix), range) => radix_literal(sign, radix, prefix, range.start),
        ((None, _), _) => decimal_literal(sign),
    }
    .parse_next(input)
}

fn radix_prefix(input: &mut Input<'_>) -> PResult<u32> {
    preceded(
        '0',
        alt((
            one_of(['x', 'X']).value(16),
            one_of(['o', 'O']).value(8),
            one_of(['b', 'B']).value(2),
        )),
    )
    .parse_next(input)
}

/// The rest of a binary, octal, or hexadecimal literal after its prefix.
/// Such a literal takes no suffix; a sign makes it an Int.
fn radix_literal<'a>(
    sign: Option<char>,
    radix: u32,
    prefix: &'a str,
    start: usize,
) -> impl Parser<Input<'a>, NumericValue, ErrMode<LexError>> {
    move |input: &mut Input<'a>| {
        let (digits, digits_text) = digits(radix).with_taken().parse_next(input)?;
        let end = input.current_token_start();
        if digits.count == 0 {
            return reject(start..end, NumericErrorKind::MissingRadixDigits { radix });
        }
        let (suffix, suffix_range) = rest.with_span().parse_next(input)?;
        if !suffix.is_empty() {
            let int_form =
                (suffix == "i").then(|| format!("{}{prefix}{digits_text}", sign.unwrap_or('+')));
            return reject(
                suffix_range,
                NumericErrorKind::RadixSuffix {
                    radix,
                    suffix: suffix.to_string(),
                    int_form,
                },
            );
        }
        let ty = match sign {
            Some(_) => IntegerType::Int,
            None => IntegerType::Nat,
        };
        integer_value(ty, sign == Some('-'), digits.value, 0..end)
    }
}

/// The rest of a decimal literal after its sign: digits, an optional
/// fraction and exponent, and an optional suffix.
fn decimal_literal<'a>(
    sign: Option<char>,
) -> impl Parser<Input<'a>, NumericValue, ErrMode<LexError>> {
    move |input: &mut Input<'a>| {
        let start = input.current_token_start();
        let ((integral, fraction, exponent), number) =
            (digits(10), opt(preceded('.', digits(10))), opt(exponent))
                .with_taken()
                .parse_next(input)?;
        if integral.count == 0 {
            return reject(
                start..input.current_token_start(),
                NumericErrorKind::MissingRadixDigits { radix: 10 },
            );
        }
        let suffix = suffix.parse_next(input)?;
        let end = input.current_token_start();
        if let Some((suffix, range)) = &suffix {
            let conflict = match suffix {
                'n' if sign.is_some() => Some(SuffixConflict::Signed),
                'n' | 'i' if fraction.is_some() => Some(SuffixConflict::Fractional),
                _ => None,
            };
            if let Some(conflict) = conflict {
                return reject(
                    range.clone(),
                    NumericErrorKind::SuffixNotAllowed {
                        suffix: *suffix,
                        conflict,
                    },
                );
            }
        }

        let ty = match suffix.map(|(suffix, _)| suffix) {
            Some('f') => None,
            Some('i') => Some(IntegerType::Int),
            Some(_) => Some(IntegerType::Nat),
            None if fraction.is_some() => None,
            None if sign.is_some() => Some(IntegerType::Int),
            None => Some(IntegerType::Nat),
        };
        let Some(ty) = ty else {
            let normalized: String = sign
                .into_iter()
                .chain(number.chars().filter(|&c| c != '_'))
                .collect();
            let value: f64 = normalized
                .parse()
                .expect("a validated decimal literal parses as f64");
            return if value.is_finite() {
                Ok(NumericValue::Float(value))
            } else {
                reject(0..end, NumericErrorKind::FloatNotFinite)
            };
        };

        let mut magnitude = integral.value;
        if let Some(exponent) = exponent {
            if exponent.negative {
                let sign = sign.map(String::from).unwrap_or_default();
                let (mantissa, exponent_text) = number.split_at(exponent.range.start - start);
                return reject(
                    exponent.range,
                    NumericErrorKind::NegativeIntegerExponent {
                        with_point: format!("{sign}{mantissa}.0{exponent_text}"),
                        with_suffix: format!("{sign}{mantissa}{exponent_text}f"),
                    },
                );
            }
            if magnitude != Some(0) {
                magnitude = exponent
                    .digits
                    .value
                    .and_then(|exp| u32::try_from(exp).ok())
                    .and_then(|exp| 10u64.checked_pow(exp))
                    .zip(magnitude)
                    .and_then(|(scale, magnitude)| magnitude.checked_mul(scale));
            }
        }
        integer_value(ty, sign == Some('-'), magnitude, 0..end)
    }
}
/// Digits of a radix with `_` separators, folded into their value.
#[derive(Clone, Copy, Default)]
struct Digits {
    /// Number of digits, not counting separators.
    count: usize,
    /// The value, or `None` once it exceeds `u64`.
    value: Option<u64>,
}

/// Digits of `radix` with `_` separators. Binary and octal also consume the
/// other decimal digits, reporting them as invalid rather than leaving them
/// to be read as a suffix.
fn digits<'a>(radix: u32) -> impl Parser<Input<'a>, Digits, ErrMode<LexError>> {
    let digit = alt((
        '_'.value(None),
        one_of(move |c: char| c.is_digit(radix)).map(move |c: char| c.to_digit(radix)),
        move |input: &mut Input<'a>| {
            let (digit, range) = one_of(|c: char| c.is_ascii_digit())
                .with_span()
                .parse_next(input)?;
            reject(range, NumericErrorKind::InvalidDigit { digit, radix })
        },
    ));
    repeat(0.., digit).fold(
        || Digits {
            count: 0,
            value: Some(0),
        },
        move |digits, digit: Option<u32>| match digit {
            Some(digit) => Digits {
                count: digits.count + 1,
                value: digits.value.and_then(|value| {
                    value
                        .checked_mul(u64::from(radix))?
                        .checked_add(u64::from(digit))
                }),
            },
            None => digits,
        },
    )
}

struct Exponent {
    range: Range<usize>,
    negative: bool,
    digits: Digits,
}

/// A decimal exponent: `e` or `E`, an optional sign, and at least one digit.
fn exponent(input: &mut Input<'_>) -> PResult<Exponent> {
    let ((negative, digits), range) = preceded(
        one_of(['e', 'E']),
        (
            opt(one_of(['+', '-'])).map(|sign| sign == Some('-')),
            digits(10),
        ),
    )
    .with_span()
    .parse_next(input)?;
    if digits.count == 0 {
        return reject(range, NumericErrorKind::MissingExponentDigits);
    }
    Ok(Exponent {
        range,
        negative,
        digits,
    })
}

/// The suffix of a decimal literal: nothing, or `n`, `i`, or `f`.
fn suffix(input: &mut Input<'_>) -> PResult<Option<(char, Range<usize>)>> {
    alt((
        eof.value(None),
        terminated(one_of(['n', 'i', 'f']), eof)
            .with_span()
            .map(Some),
        |input: &mut Input<'_>| {
            let (suffix, range) = rest.with_span().parse_next(input)?;
            reject(range, NumericErrorKind::UnknownSuffix(suffix.to_string()))
        },
    ))
    .parse_next(input)
}

/// An integer literal's value, or an overflow error spanning `range`.
fn integer_value(
    ty: IntegerType,
    negative: bool,
    magnitude: Option<u64>,
    range: Range<usize>,
) -> PResult<NumericValue> {
    let value = magnitude.and_then(|magnitude| match ty {
        IntegerType::Nat => Some(NumericValue::Nat(magnitude)),
        // The magnitude of i64::MIN is one more than i64::MAX.
        IntegerType::Int if negative => 0i64.checked_sub_unsigned(magnitude).map(NumericValue::Int),
        IntegerType::Int => i64::try_from(magnitude).ok().map(NumericValue::Int),
    });
    match value {
        Some(value) => Ok(value),
        None => reject(range, NumericErrorKind::Overflow(ty)),
    }
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
