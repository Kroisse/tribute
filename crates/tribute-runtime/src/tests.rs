//! Tests for the runtime library

use crate::value::{TrValue, ValueTag};
use crate::memory::*;
use crate::arithmetic::*;

#[test]
fn test_value_creation() {
    let num_val = TrValue::number(42.0);
    assert_eq!(num_val.tag, ValueTag::Number);
    assert_eq!(num_val.as_number(), 42.0);
    
    let str_val = TrValue::string("hello".to_string());
    assert_eq!(str_val.tag, ValueTag::String);
    assert_eq!(str_val.as_string(), Some("hello"));
    
    let unit_val = TrValue::unit();
    assert_eq!(unit_val.tag, ValueTag::Unit);
    assert!(unit_val.is_unit());
}

#[test]
fn test_memory_management() {
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

#[test]
fn test_arithmetic_operations() {
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
    
    tr_value_free(left);
    tr_value_free(right);
}

#[test]
fn test_string_arithmetic() {
    let hello = tr_value_from_string("Hello ".as_ptr(), 6);
    let world = tr_value_from_string("World!".as_ptr(), 6);
    
    // Test string concatenation through addition
    let result = tr_value_add(hello, world);
    assert_eq!(tr_value_get_tag(result), ValueTag::String as u8);
    
    tr_value_free(hello);
    tr_value_free(world);
    tr_value_free(result);
}

#[test]
fn test_value_equality() {
    let num1 = tr_value_from_number(42.0);
    let num2 = tr_value_from_number(42.0);
    let num3 = tr_value_from_number(43.0);
    
    assert_eq!(tr_value_equals(num1, num2), 1);
    assert_eq!(tr_value_equals(num1, num3), 0);
    
    tr_value_free(num1);
    tr_value_free(num2);
    tr_value_free(num3);
}