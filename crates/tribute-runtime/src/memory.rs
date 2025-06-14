//! Memory management for Tribute values
//!
//! This module provides the C-compatible memory management functions
//! that are called by compiled Tribute code.
//! Uses handle-based API for GC compatibility.

use std::boxed::Box;
use crate::value::{TrValue, TrHandle};

/// Allocate a new unit TrValue on the heap
#[no_mangle]
pub extern "C" fn tr_value_new() -> TrHandle {
    let value = Box::new(TrValue::unit());
    TrHandle::from_raw(Box::into_raw(value))
}

/// Free a TrValue allocated on the heap
#[no_mangle]
pub extern "C" fn tr_value_free(handle: TrHandle) {
    if handle.is_null() {
        return;
    }
    
    unsafe {
        // This will call the Drop implementation which handles string cleanup
        let _ = Box::from_raw(handle.raw);
    }
}

/// Create a number value
#[no_mangle]
pub extern "C" fn tr_value_from_number(num: f64) -> TrHandle {
    let value = Box::new(TrValue::number(num));
    TrHandle::from_raw(Box::into_raw(value))
}

/// Create a string value from raw parts
#[no_mangle]
pub extern "C" fn tr_value_from_string(data: *const u8, len: usize) -> TrHandle {
    if data.is_null() {
        // Create empty string
        let value = Box::new(TrValue::string_static(""));
        return TrHandle::from_raw(Box::into_raw(value));
    }
    
    unsafe {
        let slice = std::slice::from_raw_parts(data, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            let value = Box::new(TrValue::string(s.to_owned()));
            TrHandle::from_raw(Box::into_raw(value))
        } else {
            // Invalid UTF-8, create empty string
            let value = Box::new(TrValue::string_static(""));
            TrHandle::from_raw(Box::into_raw(value))
        }
    }
}

/// Extract a number from a TrValue
#[no_mangle]
pub extern "C" fn tr_value_to_number(handle: TrHandle) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    
    unsafe {
        handle.deref().as_number()
    }
}

/// Clone a TrValue (deep copy)
#[no_mangle]
pub extern "C" fn tr_value_clone(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return tr_value_new();
    }
    
    unsafe {
        let cloned = handle.deref().clone_value();
        TrHandle::from_raw(Box::into_raw(Box::new(cloned)))
    }
}

/// Get the tag of a value for type checking
#[no_mangle]
pub extern "C" fn tr_value_get_tag(handle: TrHandle) -> u8 {
    if handle.is_null() {
        return 2; // Unit tag
    }
    
    unsafe {
        handle.deref().tag as u8
    }
}

/// Check if two values are equal (for debugging/testing)
#[no_mangle]
pub extern "C" fn tr_value_equals(left: TrHandle, right: TrHandle) -> bool {
    if left.is_null() && right.is_null() {
        return true;
    }
    if left.is_null() || right.is_null() {
        return false;
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        if left_val.tag != right_val.tag {
            return false;
        }
        
        match left_val.tag {
            crate::value::ValueTag::Number => {
                left_val.data.number == right_val.data.number
            },
            crate::value::ValueTag::String => {
                let left_str = left_val.data.string.as_str();
                let right_str = right_val.data.string.as_str();
                left_str == right_str
            },
            crate::value::ValueTag::Unit => true,
        }
    }
}