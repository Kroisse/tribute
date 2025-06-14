//! Arithmetic operations for Tribute values
//!
//! This module implements dynamic arithmetic operations that handle
//! type checking and coercion at runtime.
//! Uses handle-based API for GC compatibility.

use crate::value::{allocation_table, TrHandle, TrValue};

/// Add two values - supports number + number and string concatenation
#[no_mangle]
pub extern "C" fn tr_value_add(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::unit());
    }

    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();

        let result = match (left_val, right_val) {
            (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                let sum = ln + rn;
                TrValue::number(sum)
            }
            (TrValue::String(ls), TrValue::String(rs)) => {
                let left_str = ls.as_str();
                let right_str = rs.as_str();
                let result = format!("{}{}", left_str, right_str);
                TrValue::string(result)
            }
            (TrValue::String(ls), &TrValue::Number(rn)) => {
                let left_str = ls.as_str();
                let result = format!("{}{}", left_str, rn);
                TrValue::string(result)
            }
            (&TrValue::Number(ln), TrValue::String(rs)) => {
                let right_str = rs.as_str();
                let result = format!("{}{}", ln, right_str);
                TrValue::string(result)
            }
            _ => {
                // Other combinations result in unit
                TrValue::unit()
            }
        };

        allocation_table().allocate(result)
    }
}

/// Subtract two values - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_sub(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();

        let result = match (left_val, right_val) {
            (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                let diff = ln - rn;
                TrValue::number(diff)
            }
            _ => {
                // Non-numeric subtraction results in 0
                TrValue::number(0.0)
            }
        };

        allocation_table().allocate(result)
    }
}

/// Multiply two values - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_mul(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();

        let result = match (left_val, right_val) {
            (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                let product = ln * rn;
                TrValue::number(product)
            }
            _ => {
                // Non-numeric multiplication results in 0
                TrValue::number(0.0)
            }
        };

        allocation_table().allocate(result)
    }
}

/// Divide two values - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_div(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();

        let result = match (left_val, right_val) {
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
        };

        allocation_table().allocate(result)
    }
}

/// Modulo operation - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_mod(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();

        let result = match (left_val, right_val) {
            (&TrValue::Number(ln), &TrValue::Number(rn)) => {
                let remainder = ln % rn;
                TrValue::number(remainder)
            }
            _ => {
                // Non-numeric modulo results in 0
                TrValue::number(0.0)
            }
        };

        allocation_table().allocate(result)
    }
}

/// Negate a value - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_neg(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    unsafe {
        let val = handle.deref();

        let result = match val {
            &TrValue::Number(n) => {
                let negated = -n;
                TrValue::number(negated)
            }
            _ => {
                // Non-numeric negation results in 0
                TrValue::number(0.0)
            }
        };

        allocation_table().allocate(result)
    }
}
