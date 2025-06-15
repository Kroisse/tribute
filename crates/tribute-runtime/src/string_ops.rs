//! String operations for Tribute values
//!
//! This module implements string manipulation functions including
//! concatenation and interpolation support.
//! Uses handle-based API for GC compatibility.

use crate::value::{TrHandle, TrValue, allocation_table};
use std::string::String;

/// Concatenate two string values
#[unsafe(no_mangle)]
pub extern "C" fn tr_string_concat(left: TrHandle, right: TrHandle) -> TrHandle {
    if left.is_null() || right.is_null() {
        return allocation_table().allocate(TrValue::string_static(""));
    }

    let result = left
        .with_value(|left_val| {
            right.with_value(|right_val| {
                // Convert both values to strings
                let left_str = match left_val {
                    TrValue::String(s) => unsafe { s.as_str() },
                    TrValue::Number(n) => {
                        return format!("{}", n);
                    }
                    TrValue::Unit => "()",
                };

                let right_str = match right_val {
                    TrValue::String(s) => unsafe { s.as_str() },
                    TrValue::Number(n) => {
                        let num_str = format!("{}", n);
                        return format!("{}{}", left_str, num_str);
                    }
                    TrValue::Unit => "()",
                };

                format!("{}{}", left_str, right_str)
            })
        })
        .flatten()
        .unwrap_or_default();

    allocation_table().allocate(TrValue::string(result))
}

/// Helper function to concatenate a C string with a TrValue
#[unsafe(no_mangle)]
pub extern "C" fn tr_string_concat_with_str(
    str_data: *const u8,
    str_len: usize,
    handle: TrHandle,
) -> TrHandle {
    if str_data.is_null() || handle.is_null() {
        return allocation_table().allocate(TrValue::string_static(""));
    }

    let str_slice = unsafe { std::slice::from_raw_parts(str_data, str_len) };
    let str_part = match std::str::from_utf8(str_slice) {
        Ok(s) => s,
        Err(_) => return allocation_table().allocate(TrValue::string_static("")),
    };

    let result = handle
        .with_value(|val| {
            let value_str = match val {
                TrValue::String(s) => unsafe { s.as_str() },
                TrValue::Number(n) => {
                    let num_str = format!("{}", n);
                    return format!("{}{}", str_part, num_str);
                }
                TrValue::Unit => "()",
            };

            format!("{}{}", str_part, value_str)
        })
        .unwrap_or_default();

    allocation_table().allocate(TrValue::string(result))
}

/// String interpolation - takes a format string and array of values
#[unsafe(no_mangle)]
pub extern "C" fn tr_string_interpolate(
    format_handle: TrHandle,
    args: *const TrHandle,
    arg_count: usize,
) -> TrHandle {
    if format_handle.is_null() {
        return allocation_table().allocate(TrValue::string_static(""));
    }

    let result = format_handle
        .with_value(|format_val| {
            let format_str = match format_val {
                TrValue::String(s) => unsafe { s.as_str() },
                _ => return String::new(),
            };

            // Simple interpolation: replace {} with arguments in order
            let mut result = String::from(format_str);

            if !args.is_null() && arg_count > 0 {
                let arg_slice = unsafe { std::slice::from_raw_parts(args, arg_count) };
                let mut arg_index = 0;

                // Find and replace {} patterns
                while let Some(pos) = result.find("{}") {
                    if arg_index < arg_slice.len() && !arg_slice[arg_index].is_null() {
                        let replacement = arg_slice[arg_index]
                            .with_value(|arg_val| match arg_val {
                                TrValue::String(s) => unsafe { s.as_str().to_owned() },
                                TrValue::Number(n) => format!("{}", n),
                                TrValue::Unit => "()".to_owned(),
                            })
                            .unwrap_or_else(|| "(invalid)".to_owned());

                        result.replace_range(pos..pos + 2, &replacement);
                        arg_index += 1;
                    } else {
                        // No more arguments, leave {} as is
                        break;
                    }
                }
            }

            result
        })
        .unwrap_or_default();

    allocation_table().allocate(TrValue::string(result))
}

/// Get the length of a string value
#[unsafe(no_mangle)]
pub extern "C" fn tr_string_length(handle: TrHandle) -> usize {
    if handle.is_null() {
        return 0;
    }

    handle
        .with_value(|val| match val {
            TrValue::String(s) => s.len(),
            _ => 0,
        })
        .unwrap_or(0)
}

/// Check if a string contains a substring
#[unsafe(no_mangle)]
pub extern "C" fn tr_string_contains(haystack: TrHandle, needle: TrHandle) -> bool {
    if haystack.is_null() || needle.is_null() {
        return false;
    }

    haystack
        .with_value(|haystack_val| {
            needle.with_value(|needle_val| {
                let (haystack_str, needle_str) = match (haystack_val, needle_val) {
                    (TrValue::String(hs), TrValue::String(ns)) => {
                        (unsafe { hs.as_str() }, unsafe { ns.as_str() })
                    }
                    _ => return false,
                };

                haystack_str.contains(needle_str)
            })
        })
        .flatten()
        .unwrap_or(false)
}

/// Convert any value to a string representation
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_to_string(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::string_static("null"));
    }

    let result = handle
        .with_value(|val| {
            match val {
                TrValue::String(s) => {
                    // Already a string, clone it
                    let str_val = unsafe { s.as_str() };
                    str_val.to_owned()
                }
                TrValue::Number(n) => {
                    format!("{}", n)
                }
                TrValue::Unit => "()".to_owned(),
            }
        })
        .unwrap_or_else(|| "null".to_owned());

    allocation_table().allocate(TrValue::string(result))
}
