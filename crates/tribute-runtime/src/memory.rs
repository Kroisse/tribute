//! Memory management for Tribute values
//!
//! This module provides the C-compatible memory management functions
//! that are called by compiled Tribute code.
//! Uses index-based handle system with allocation table for GC compatibility.

use crate::value::{TrHandle, TrValue, allocation_table};

/// Allocate a new unit TrValue on the heap
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_new() -> TrHandle {
    let value = TrValue::unit();
    allocation_table().allocate(value)
}

/// Free a TrValue allocated on the heap
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_free(handle: TrHandle) {
    allocation_table().free(handle);
}

/// Create a number value
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_from_number(num: f64) -> TrHandle {
    let value = TrValue::number(num);
    allocation_table().allocate(value)
}

/// Create a string value from raw parts (automatically selects optimal mode)
///
/// The function automatically chooses the most efficient storage mode:
/// - Strings ≤ 7 bytes: inline storage (no heap allocation)
/// - Strings > 7 bytes: heap storage
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_from_string(data: *const u8, len: usize) -> TrHandle {
    if data.is_null() {
        let value = TrValue::string_static("");
        return allocation_table().allocate(value);
    }

    unsafe {
        let slice = std::slice::from_raw_parts(data, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            // TrString::new() automatically chooses inline vs heap based on length
            let value = TrValue::string(s.to_owned());
            allocation_table().allocate(value)
        } else {
            // Invalid UTF-8, create empty string
            let value = TrValue::string_static("");
            allocation_table().allocate(value)
        }
    }
}

/// Create a static string value with offset into .rodata section
/// This is used by the compiler for string literals
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_from_static_string(offset: u32, len: u32) -> TrHandle {
    use crate::value::TrString;

    let tr_string = TrString::new_static(offset, len);
    let value = TrValue::String(tr_string);
    allocation_table().allocate(value)
}

/// Get pointer and length from a string value
#[unsafe(no_mangle)]
pub extern "C" fn tr_string_as_ptr(handle: TrHandle, out_len: *mut usize) -> *const u8 {
    if handle.is_null() || out_len.is_null() {
        return std::ptr::null();
    }

    // Use allocation table's safe access method
    allocation_table()
        .with_value(handle, |val| match val {
            TrValue::String(s) => unsafe {
                let (ptr, len) = s.as_ptr_len();
                *out_len = len;
                ptr
            },
            _ => {
                unsafe {
                    *out_len = 0;
                }
                std::ptr::null()
            }
        })
        .unwrap_or_else(|| {
            unsafe {
                *out_len = 0;
            }
            std::ptr::null()
        })
}

/// Extract a number from a TrValue
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_to_number(handle: TrHandle) -> f64 {
    allocation_table().to_number(handle)
}

/// Clone a TrValue (deep copy)
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_clone(handle: TrHandle) -> TrHandle {
    allocation_table().clone_value(handle)
}

/// Get the tag of a value for type checking
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_get_tag(handle: TrHandle) -> u8 {
    allocation_table().get_tag(handle)
}

/// Check if two values are equal (for pattern matching)
#[unsafe(no_mangle)]
pub extern "C" fn tr_value_equals(left: TrHandle, right: TrHandle) -> u8 {
    if allocation_table().values_equal(left, right) {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::allocation_table;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_memory_management() {
        // Clear allocation table for test isolation
        allocation_table().clear();
        // Test number value
        let num_handle = tr_value_from_number(123.0);
        assert!(!num_handle.is_null());
        assert_eq!(tr_value_to_number(num_handle), 123.0);
        tr_value_free(num_handle);

        // Test string value
        let test_str = "test string";
        let str_handle = tr_value_from_string(test_str.as_ptr(), test_str.len());
        assert!(!str_handle.is_null());
        tr_value_free(str_handle);

        // Test cloning
        let original = tr_value_from_number(99.0);
        let cloned = tr_value_clone(original);
        assert_eq!(tr_value_to_number(original), tr_value_to_number(cloned));
        tr_value_free(original);
        tr_value_free(cloned);
    }
}
