//! Arithmetic operations for Tribute values
//!
//! This module implements dynamic arithmetic operations that handle
//! type checking and coercion at runtime.
//! Uses handle-based API for GC compatibility.

use std::boxed::Box;
use crate::value::{TrValue, TrHandle, ValueTag};

/// Add two values - supports number + number and string concatenation
#[no_mangle]
pub extern "C" fn tr_value_add(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return TrHandle::from_raw(Box::into_raw(Box::new(TrValue::unit())));
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let result = left_val.data.number + right_val.data.number;
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
            },
            (ValueTag::String, ValueTag::String) => {
                let left_str = left_val.data.string.as_str();
                let right_str = right_val.data.string.as_str();
                let result = format!("{}{}", left_str, right_str);
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::string(result))))
            },
            (ValueTag::String, ValueTag::Number) => {
                let left_str = left_val.data.string.as_str();
                let right_num = right_val.data.number;
                let result = format!("{}{}", left_str, right_num);
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::string(result))))
            },
            (ValueTag::Number, ValueTag::String) => {
                let left_num = left_val.data.number;
                let right_str = right_val.data.string.as_str();
                let result = format!("{}{}", left_num, right_str);
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::string(result))))
            },
            _ => {
                // Other combinations result in unit
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::unit())))
            }
        }
    }
}

/// Subtract two values - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_sub(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))));
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let result = left_val.data.number - right_val.data.number;
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
            },
            _ => {
                // Non-numeric subtraction results in 0
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))))
            }
        }
    }
}

/// Multiply two values - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_mul(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))));
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let result = left_val.data.number * right_val.data.number;
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
            },
            _ => {
                // Non-numeric multiplication results in 0
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))))
            }
        }
    }
}

/// Divide two values - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_div(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))));
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let divisor = right_val.data.number;
                if divisor == 0.0 {
                    // Division by zero - return infinity or NaN
                    let result = left_val.data.number / divisor; // This will be inf or NaN
                    TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
                } else {
                    let result = left_val.data.number / divisor;
                    TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
                }
            },
            _ => {
                // Non-numeric division results in 0
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))))
            }
        }
    }
}

/// Modulo operation - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_mod(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))));
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        match (left_val.tag, right_val.tag) {
            (ValueTag::Number, ValueTag::Number) => {
                let result = left_val.data.number % right_val.data.number;
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
            },
            _ => {
                // Non-numeric modulo results in 0
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))))
            }
        }
    }
}

/// Negate a value - only supports numbers
#[no_mangle]
pub extern "C" fn tr_value_neg(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))));
    }
    
    unsafe {
        let val = handle.deref();
        
        match val.tag {
            ValueTag::Number => {
                let result = -val.data.number;
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(result))))
            },
            _ => {
                // Non-numeric negation results in 0
                TrHandle::from_raw(Box::into_raw(Box::new(TrValue::number(0.0))))
            }
        }
    }
}