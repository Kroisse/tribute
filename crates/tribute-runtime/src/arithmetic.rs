//! Arithmetic operations for Tribute values
//!
//! This module implements dynamic arithmetic operations that handle
//! type checking and coercion at runtime.
//! Uses handle-based API for GC compatibility.

use crate::value::{TrValue, TrHandle, ValueTag, allocation_table};

/// Add two values - supports number + number and string concatenation
#[no_mangle]
pub extern "C" fn tr_value_add(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::unit());
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        let result = match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let sum = left_val.data.number + right_val.data.number;
                TrValue::number(sum)
            },
            (ValueTag::String, ValueTag::String) => {
                let left_str = left_val.data.string.as_str();
                let right_str = right_val.data.string.as_str();
                let result = format!("{}{}", left_str, right_str);
                TrValue::string(result)
            },
            (ValueTag::String, ValueTag::Number) => {
                let left_str = left_val.data.string.as_str();
                let right_num = right_val.data.number;
                let result = format!("{}{}", left_str, right_num);
                TrValue::string(result)
            },
            (ValueTag::Number, ValueTag::String) => {
                let left_num = left_val.data.number;
                let right_str = right_val.data.string.as_str();
                let result = format!("{}{}", left_num, right_str);
                TrValue::string(result)
            },
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
        
        let result = match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let diff = left_val.data.number - right_val.data.number;
                TrValue::number(diff)
            },
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
        
        let result = match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let product = left_val.data.number * right_val.data.number;
                TrValue::number(product)
            },
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
        
        let result = match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let divisor = right_val.data.number;
                let quotient = if divisor == 0.0 {
                    // Division by zero - return infinity or NaN
                    left_val.data.number / divisor // This will be inf or NaN
                } else {
                    left_val.data.number / divisor
                };
                TrValue::number(quotient)
            },
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
        
        let result = match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let remainder = left_val.data.number % right_val.data.number;
                TrValue::number(remainder)
            },
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
        
        let result = match val.tag {
            ValueTag::Number => {
                let negated = -val.data.number;
                TrValue::number(negated)
            },
            _ => {
                // Non-numeric negation results in 0
                TrValue::number(0.0)
            }
        };
        
        allocation_table().allocate(result)
    }
}