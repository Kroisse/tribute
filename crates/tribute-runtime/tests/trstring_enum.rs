//! Tests for the TrString enum implementation using simplified C API

#[test]
fn test_auto_string_creation() {
    use tribute_runtime::{tr_value_from_string, tr_value_free, tr_string_length};
    
    // Test short string (≤ 15 bytes) - should use inline mode automatically
    let short_str = b"Hello";
    let handle = tr_value_from_string(short_str.as_ptr(), short_str.len());
    
    // Verify handle is valid
    assert!(!handle.is_null());
    
    // Verify length
    let len = tr_string_length(handle);
    assert_eq!(len, 5);
    
    // Clean up
    tr_value_free(handle);
}

#[test]
fn test_auto_long_string_creation() {
    use tribute_runtime::{tr_value_from_string, tr_value_free, tr_string_length};
    
    // Test long string (> 15 bytes) - should use heap mode automatically
    let long_str = b"This is a longer string that exceeds inline limit";
    let handle = tr_value_from_string(long_str.as_ptr(), long_str.len());
    
    // Verify handle is valid
    assert!(!handle.is_null());
    
    // Verify length
    let len = tr_string_length(handle);
    assert_eq!(len, long_str.len());
    
    // Clean up
    tr_value_free(handle);
}

#[test]
fn test_static_string_creation() {
    use tribute_runtime::{tr_value_from_static_string, tr_value_free, tr_string_length};
    
    // Test static string creation (for compiler use)
    let handle = tr_value_from_static_string(100, 20);
    
    // Verify handle is valid
    assert!(!handle.is_null());
    
    // Verify length
    let len = tr_string_length(handle);
    assert_eq!(len, 20);
    
    // Clean up
    tr_value_free(handle);
}

#[test]
fn test_string_size_boundary() {
    use tribute_runtime::{tr_value_from_string, tr_value_free, tr_string_length};
    
    // Test exactly 15 bytes (should use inline)
    let boundary_str = b"123456789012345"; // 15 bytes
    let handle = tr_value_from_string(boundary_str.as_ptr(), 15);
    assert!(!handle.is_null());
    assert_eq!(tr_string_length(handle), 15);
    tr_value_free(handle);
    
    // Test 16 bytes (should use heap)
    let over_boundary = b"1234567890123456"; // 16 bytes
    let handle = tr_value_from_string(over_boundary.as_ptr(), 16);
    assert!(!handle.is_null());
    assert_eq!(tr_string_length(handle), 16);
    tr_value_free(handle);
}

#[test]
fn test_string_operations() {
    use tribute_runtime::{tr_value_from_string, tr_string_concat, tr_value_free, tr_string_length};
    
    // Create two strings using auto mode
    let str1 = b"Hello, ";
    let str2 = b"World!";
    
    let handle1 = tr_value_from_string(str1.as_ptr(), str1.len());
    let handle2 = tr_value_from_string(str2.as_ptr(), str2.len());
    
    // Concatenate them
    let result = tr_string_concat(handle1, handle2);
    
    // Verify result
    assert!(!result.is_null());
    let result_len = tr_string_length(result);
    assert_eq!(result_len, str1.len() + str2.len());
    
    // Clean up
    tr_value_free(handle1);
    tr_value_free(handle2);
    tr_value_free(result);
}

#[test]
fn test_empty_and_null_strings() {
    use tribute_runtime::{tr_value_from_string, tr_value_free, tr_string_length};
    
    // Test empty string
    let empty_str = b"";
    let handle = tr_value_from_string(empty_str.as_ptr(), 0);
    assert!(!handle.is_null());
    assert_eq!(tr_string_length(handle), 0);
    tr_value_free(handle);
    
    // Test null pointer (should create empty string)
    let handle = tr_value_from_string(std::ptr::null(), 10);
    assert!(!handle.is_null());
    assert_eq!(tr_string_length(handle), 0);
    tr_value_free(handle);
}