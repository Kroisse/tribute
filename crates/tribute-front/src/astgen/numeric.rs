//! Validation and decoding of numeric literals.
//!
//! The grammar lexes a numeric literal together with every identifier
//! character that follows it, so digit separators, radix digits, exponents,
//! and type suffixes all arrive here as one token. This module checks that
//! token against the literal rules in `new-plans/syntax.md` and decides its
//! final type from its shape and suffix.

use std::fmt;
use std::ops::Range;

/// The decoded value of a numeric literal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NumericValue {
    Nat(u64),
    Int(i64),
    Float(f64),
}

/// An invalid numeric literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NumericError {
    /// Byte range of the offending part, relative to the literal text.
    pub range: Range<usize>,
    pub kind: NumericErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NumericErrorKind {
    /// Trailing identifier characters that are not a known suffix.
    UnknownSuffix(String),
    /// A known suffix on a literal whose shape it cannot apply to.
    SuffixNotAllowed {
        suffix: char,
        conflict: SuffixConflict,
    },
    /// A digit outside the literal's radix.
    InvalidDigit { digit: char, radix: u32 },
    /// A radix prefix or exponent without any digit.
    MissingDigits(DigitsOf),
    /// A negative exponent on a literal that decodes to an integer.
    NegativeIntegerExponent {
        with_point: String,
        with_suffix: String,
    },
    /// An integer that does not fit the current representation of its type.
    Overflow(IntegerType),
    /// A float literal whose value is not finite.
    FloatNotFinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SuffixConflict {
    Signed,
    Fractional,
    RadixPrefix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DigitsOf {
    Radix(u32),
    Exponent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntegerType {
    Nat,
    Int,
}

impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            NumericErrorKind::UnknownSuffix(suffix) => write!(
                f,
                "unknown numeric literal suffix `{suffix}`; expected `n`, `i`, or `f`"
            ),
            NumericErrorKind::SuffixNotAllowed { suffix, conflict } => {
                let target = match conflict {
                    SuffixConflict::Signed => "a signed literal",
                    SuffixConflict::Fractional => "a literal with a decimal point",
                    SuffixConflict::RadixPrefix => "a binary, octal, or hexadecimal literal",
                };
                write!(f, "suffix `{suffix}` cannot be used on {target}")
            }
            NumericErrorKind::InvalidDigit { digit, radix } => {
                write!(
                    f,
                    "invalid digit `{digit}` in {} literal",
                    radix_name(*radix)
                )
            }
            NumericErrorKind::MissingDigits(DigitsOf::Radix(radix)) => {
                write!(f, "{} literal has no digits", radix_name(*radix))
            }
            NumericErrorKind::MissingDigits(DigitsOf::Exponent) => {
                f.write_str("exponent has no digits")
            }
            NumericErrorKind::NegativeIntegerExponent {
                with_point,
                with_suffix,
            } => write!(
                f,
                "an integer literal cannot have a negative exponent; \
                 write `{with_point}` or `{with_suffix}` for a Float"
            ),
            NumericErrorKind::Overflow(ty) => {
                let ty = match ty {
                    IntegerType::Nat => "Nat",
                    IntegerType::Int => "Int",
                };
                write!(
                    f,
                    "integer literal exceeds the current implementation limit for `{ty}`"
                )
            }
            NumericErrorKind::FloatNotFinite => {
                f.write_str("float literal is too large to be represented as `Float`")
            }
        }
    }
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
    let bytes = text.as_bytes();
    let error = |range: Range<usize>, kind| Err(NumericError { range, kind });

    let (negative, signed, mut i) = match bytes.first() {
        Some(b'-') => (true, true, 1),
        Some(b'+') => (false, true, 1),
        _ => (false, false, 0),
    };

    // Radix prefix
    let prefix_start = i;
    let radix = match (bytes.get(i), bytes.get(i + 1).map(u8::to_ascii_lowercase)) {
        (Some(b'0'), Some(b'x')) => 16,
        (Some(b'0'), Some(b'o')) => 8,
        (Some(b'0'), Some(b'b')) => 2,
        _ => 10,
    };
    if radix != 10 {
        i += 2;
    }

    // Integer digits. Binary and octal scan all decimal digits so that an
    // out-of-radix digit is reported rather than read as a suffix.
    let is_digit = |b: u8| match radix {
        16 => b.is_ascii_hexdigit(),
        _ => b.is_ascii_digit(),
    };
    let digits_start = i;
    while i < bytes.len() && (is_digit(bytes[i]) || bytes[i] == b'_') {
        i += 1;
    }
    let digits = &text[digits_start..i];
    if let Some(pos) = digits
        .bytes()
        .position(|b| b != b'_' && !(b as char).is_digit(radix))
    {
        let at = digits_start + pos;
        return error(
            at..at + 1,
            NumericErrorKind::InvalidDigit {
                digit: bytes[at] as char,
                radix,
            },
        );
    }
    if !digits.bytes().any(|b| b != b'_') {
        return error(
            prefix_start..i,
            NumericErrorKind::MissingDigits(DigitsOf::Radix(radix)),
        );
    }

    // Fraction and exponent (decimal only)
    let mut fraction = None;
    let mut exponent = None;
    if radix == 10 {
        if bytes.get(i) == Some(&b'.') {
            let start = i + 1;
            i = start;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                i += 1;
            }
            fraction = Some(&text[start..i]);
        }
        if matches!(bytes.get(i), Some(b'e' | b'E')) {
            let start = i;
            i += 1;
            let exp_negative = match bytes.get(i) {
                Some(b'-') => {
                    i += 1;
                    true
                }
                Some(b'+') => {
                    i += 1;
                    false
                }
                _ => false,
            };
            let exp_digits_start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                i += 1;
            }
            let exp_digits = &text[exp_digits_start..i];
            if !exp_digits.bytes().any(|b| b != b'_') {
                return error(
                    start..i,
                    NumericErrorKind::MissingDigits(DigitsOf::Exponent),
                );
            }
            exponent = Some(Exponent {
                range: start..i,
                negative: exp_negative,
                digits: exp_digits,
            });
        }
    }

    // Suffix
    let suffix_range = i..bytes.len();
    let suffix = match &text[i..] {
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
        Some('f') if radix != 10 => Some(SuffixConflict::RadixPrefix),
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
        let sign = &text[..prefix_start];
        let mantissa = &text[prefix_start..exponent.range.start];
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

struct Exponent<'a> {
    range: Range<usize>,
    negative: bool,
    digits: &'a str,
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
            ("0xFFn", 255),
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
            ("0xFFi", 255),
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
            ("0xFFg", 4..5, UnknownSuffix("g".into())),
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
                SuffixNotAllowed {
                    suffix: 'f',
                    conflict: SuffixConflict::RadixPrefix,
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
            ("0x", 0..2, MissingDigits(DigitsOf::Radix(16))),
            ("0x_", 0..3, MissingDigits(DigitsOf::Radix(16))),
            ("1e", 1..2, MissingDigits(DigitsOf::Exponent)),
            ("1.0e-_", 3..6, MissingDigits(DigitsOf::Exponent)),
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
}
