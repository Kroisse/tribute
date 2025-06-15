//! Arithmetic operations for Tribute values
//!
//! This module implements dynamic arithmetic operations that handle
//! type checking and coercion at runtime.
//! Uses handle-based API for GC compatibility.

use crate::value::{TrHandle, TrValue, allocation_table};

/// Add two values - supports number + number and string concatenation
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_add(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::unit());
    }

    let result = left
        .with_value(|left_val| {
            right.with_value(|right_val| {
                match (left_val, right_val) {
                    (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                        let sum = ln + rn;
                        TrValue::number(sum)
                    }
                    (TrValue::String(ls), TrValue::String(rs)) => ls.with_string(|left_str| {
                        rs.with_string(|right_str| {
                            let result = format!("{}{}", left_str, right_str);
                            TrValue::string(result)
                        })
                    }),
                    (TrValue::String(ls), &TrValue::Number(rn)) => ls.with_string(|left_str| {
                        let result = format!("{}{}", left_str, rn);
                        TrValue::string(result)
                    }),
                    (&TrValue::Number(ln), TrValue::String(rs)) => rs.with_string(|right_str| {
                        let result = format!("{}{}", ln, right_str);
                        TrValue::string(result)
                    }),
                    _ => {
                        // Other combinations result in unit
                        TrValue::unit()
                    }
                }
            })
        })
        .flatten()
        .unwrap_or_else(TrValue::unit);

    allocation_table().allocate(result)
}

/// Subtract two values - only supports numbers
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_sub(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    let result = left
        .with_value(|left_val| {
            right.with_value(|right_val| {
                match (left_val, right_val) {
                    (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                        let diff = ln - rn;
                        TrValue::number(diff)
                    }
                    _ => {
                        // Non-numeric subtraction results in 0
                        TrValue::number(0.0)
                    }
                }
            })
        })
        .flatten()
        .unwrap_or_else(|| TrValue::number(0.0));

    allocation_table().allocate(result)
}

/// Multiply two values - only supports numbers
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_mul(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    let result = left
        .with_value(|left_val| {
            right.with_value(|right_val| {
                match (left_val, right_val) {
                    (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                        let product = ln * rn;
                        TrValue::number(product)
                    }
                    _ => {
                        // Non-numeric multiplication results in 0
                        TrValue::number(0.0)
                    }
                }
            })
        })
        .flatten()
        .unwrap_or_else(|| TrValue::number(0.0));

    allocation_table().allocate(result)
}

/// Divide two values - only supports numbers
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_div(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    let result = left
        .with_value(|left_val| {
            right.with_value(|right_val| {
                match (left_val, right_val) {
                    (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                        // IEEE 754 floating point handles division by zero automatically
                        // (returns +/-inf or NaN as appropriate)
                        let quotient = ln / rn;
                        TrValue::number(quotient)
                    }
                    _ => {
                        // Non-numeric division results in 0
                        TrValue::number(0.0)
                    }
                }
            })
        })
        .flatten()
        .unwrap_or_else(|| TrValue::number(0.0));

    allocation_table().allocate(result)
}

/// Modulo operation - only supports numbers
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_mod(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    let result = left
        .with_value(|left_val| {
            right.with_value(|right_val| {
                match (left_val, right_val) {
                    (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                        let remainder = ln % rn;
                        TrValue::number(remainder)
                    }
                    _ => {
                        // Non-numeric modulo results in 0
                        TrValue::number(0.0)
                    }
                }
            })
        })
        .flatten()
        .unwrap_or_else(|| TrValue::number(0.0));

    allocation_table().allocate(result)
}

/// Negate a value - only supports numbers
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_neg(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    let result = handle
        .with_value(|val| {
            match val {
                &TrValue::Number(n) => {
                    let negated = -n;
                    TrValue::number(negated)
                }
                _ => {
                    // Non-numeric negation results in 0
                    TrValue::number(0.0)
                }
            }
        })
        .unwrap_or_else(|| TrValue::number(0.0));

    allocation_table().allocate(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::*;
    use crate::value::allocation_table;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_arithmetic_operations() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        let left = tr_value_from_number(10.0);
        let right = tr_value_from_number(5.0);

        // Test addition
        let sum = tr_value_add(left, right);
        assert_eq!(tr_value_to_number(sum), 15.0);
        tr_value_free(sum);

        // Test subtraction
        let diff = tr_value_sub(left, right);
        assert_eq!(tr_value_to_number(diff), 5.0);
        tr_value_free(diff);

        // Test multiplication
        let product = tr_value_mul(left, right);
        assert_eq!(tr_value_to_number(product), 50.0);
        tr_value_free(product);

        // Test division
        let quotient = tr_value_div(left, right);
        assert_eq!(tr_value_to_number(quotient), 2.0);
        tr_value_free(quotient);

        // Clean up original values
        tr_value_free(left);
        tr_value_free(right);
    }

    #[test]
    #[serial]
    fn test_string_arithmetic() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        let hello = tr_value_from_string("Hello ".as_ptr(), 6);
        let world = tr_value_from_string("World!".as_ptr(), 6);

        // Test string concatenation through addition
        let result = tr_value_add(hello, world);
        assert_eq!(tr_value_get_tag(result), 1); // String tag is 1

        tr_value_free(hello);
        tr_value_free(world);
        tr_value_free(result);
    }
}
