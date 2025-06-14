//! String operations for Tribute values
//!
//! This module implements string manipulation functions including
//! concatenation and interpolation support.
//! Uses handle-based API for GC compatibility.

use std::string::String;
use crate::value::{TrValue, TrHandle, ValueTag, allocation_table};

/// Concatenate two string values
#[no_mangle]
pub extern "C" fn tr_string_concat(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::string_static(""));
    }
    
    unsafe {
        let left_val = left.deref();
        let right_val = right.deref();
        
        // Convert both values to strings
        let left_str = match left_val.tag {
            ValueTag::String => left_val.data.string.as_str(),
            ValueTag::Number => {
                // For concatenation, convert number to string
                let num_str = format!("{}", left_val.data.number);
                return tr_string_concat_with_str(num_str.as_ptr(), num_str.len(), right);
            },
            ValueTag::Unit => "()",
        };
        
        let right_str = match right_val.tag {
            ValueTag::String => right_val.data.string.as_str(),
            ValueTag::Number => {
                let num_str = format!("{}", right_val.data.number);
                let result = format!("{}{}", left_str, num_str);
                return allocation_table().allocate(TrValue::string(result));
            },
            ValueTag::Unit => "()",
        };
        
        let result = format!("{}{}", left_str, right_str);
        allocation_table().allocate(TrValue::string(result))
    }
}

/// Helper function to concatenate a C string with a TrValue
#[no_mangle]
pub extern "C" fn tr_string_concat_with_str(str_data: *const u8, str_len: usize, handle: TrHandle) -> TrHandle {
    if str_data.is_null() || handle.is_null() {
        return allocation_table().allocate(TrValue::string_static(""));
    }
    
    unsafe {
        let str_slice = std::slice::from_raw_parts(str_data, str_len);
        let str_part = match std::str::from_utf8(str_slice) {
            Ok(s) => s,
            Err(_) => return allocation_table().allocate(TrValue::string_static("")),
        };
        
        let val = handle.deref();
        let value_str = match val.tag {
            ValueTag::String => val.data.string.as_str(),
            ValueTag::Number => {
                let num_str = format!("{}", val.data.number);
                let result = format!("{}{}", str_part, num_str);
                return allocation_table().allocate(TrValue::string(result));
            },
            ValueTag::Unit => "()",
        };
        
        let result = format!("{}{}", str_part, value_str);
        allocation_table().allocate(TrValue::string(result))
    }
}

/// String interpolation - takes a format string and array of values
#[no_mangle]
pub extern "C" fn tr_string_interpolate(
    format_handle: TrHandle,
    args: *const TrHandle,
    arg_count: usize
) -> TrHandle {
    if format_handle.is_null() {
        return allocation_table().allocate(TrValue::string_static(""));
    }
    
    unsafe {
        let format_val = format_handle.deref();
        let format_str = match format_val.tag {
            ValueTag::String => format_val.data.string.as_str(),
            _ => return allocation_table().allocate(TrValue::string_static("")),
        };
        
        // Simple interpolation: replace {} with arguments in order
        let mut result = String::from(format_str);
        
        if !args.is_null() && arg_count > 0 {
            let arg_slice = std::slice::from_raw_parts(args, arg_count);
            let mut arg_index = 0;
            
            // Find and replace {} patterns
            while let Some(pos) = result.find("{}") {
                if arg_index < arg_slice.len() && !arg_slice[arg_index].is_null() {
                    let arg_val = arg_slice[arg_index].deref();
                    let replacement = match arg_val.tag {
                        ValueTag::String => arg_val.data.string.as_str().to_owned(),
                        ValueTag::Number => format!("{}", arg_val.data.number),
                        ValueTag::Unit => "()".to_owned(),
                    };
                    
                    result.replace_range(pos..pos+2, &replacement);
                    arg_index += 1;
                } else {
                    // No more arguments, leave {} as is
                    break;
                }
            }
        }
        
        allocation_table().allocate(TrValue::string(result))
    }
}

/// Get the length of a string value
#[no_mangle]
pub extern "C" fn tr_string_length(handle: TrHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    
    unsafe {
        let val = handle.deref();
        match val.tag {
            ValueTag::String => val.data.string.len,
            _ => 0,
        }
    }
}

/// Check if a string contains a substring
#[no_mangle]
pub extern "C" fn tr_string_contains(
    haystack: TrHandle,
    needle: TrHandle
) -> bool {
    if haystack.is_null() || needle.is_null() {
        return false;
    }
    
    unsafe {
        let haystack_val = haystack.deref();
        let needle_val = needle.deref();
        
        if haystack_val.tag != ValueTag::String || needle_val.tag != ValueTag::String {
            return false;
        }
        
        let haystack_str = haystack_val.data.string.as_str();
        let needle_str = needle_val.data.string.as_str();
        
        haystack_str.contains(needle_str)
    }
}

/// Convert any value to a string representation
#[no_mangle]
pub extern "C" fn tr_value_to_string(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::string_static("null"));
    }
    
    unsafe {
        let val = handle.deref();
        let result = match val.tag {
            ValueTag::String => {
                // Already a string, clone it
                let s = val.data.string.as_str();
                s.to_owned()
            },
            ValueTag::Number => {
                format!("{}", val.data.number)
            },
            ValueTag::Unit => "()".to_owned(),
        };
        
        allocation_table().allocate(TrValue::string(result))
    }
}