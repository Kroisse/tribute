//! Tribute Runtime Library
//!
//! This library provides the runtime support for Tribute programs,
//! including garbage collection, boxing/unboxing, and builtin functions.

pub mod array;
pub mod string;
pub mod list;

use std::{
    alloc::{Layout, alloc, dealloc},
    ptr::{self, NonNull},
    sync::atomic::{AtomicU32, Ordering},
};

use string::TributeString;
use list::TributeList;

/// Boxed value structure with reference counting
#[repr(C)]
pub struct TributeBoxed {
    /// Reference count (atomic for thread safety)
    pub ref_count: AtomicU32,
    /// Value payload
    pub value: TributeValue,
}

/// Value enum for different types
#[repr(C)]
pub enum TributeValue {
    Number(i64),
    Boolean(bool),
    String(TributeString),
    Function(TributeFunction),
    List(TributeList),
    Nil,
}

// Type code constants for C FFI compatibility
impl TributeValue {
    pub const TYPE_NUMBER: u32 = 0;
    pub const TYPE_BOOLEAN: u32 = 1;
    pub const TYPE_STRING: u32 = 2;
    pub const TYPE_FUNCTION: u32 = 3;
    pub const TYPE_LIST: u32 = 4;
    pub const TYPE_NIL: u32 = 5;
}


#[repr(C)]
#[derive(Copy, Clone)]
pub struct TributeFunction {
    pub code_ptr: *mut u8,
    pub env_ptr: *mut TributeBoxed,
}

impl TributeBoxed {
    /// Create a new boxed value with ref_count = 1
    pub fn new(value: TributeValue) -> NonNull<Self> {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc(layout) as *mut Self };

        if ptr.is_null() {
            panic!("Failed to allocate memory for TributeBoxed");
        }

        unsafe {
            ptr::write(
                ptr,
                Self {
                    ref_count: AtomicU32::new(1),
                    value,
                },
            );
        }

        unsafe { NonNull::new_unchecked(ptr) }
    }

    /// Increment reference count
    pub fn retain(&self) -> u32 {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement reference count and deallocate if it reaches 0
    pub fn release(&self) -> bool {
        let old_count = self.ref_count.fetch_sub(1, Ordering::AcqRel);

        if old_count == 1 {
            // Last reference, need to deallocate
            unsafe { self.deallocate() };
            true
        } else {
            false
        }
    }

    /// Deallocate the boxed value and its contents
    unsafe fn deallocate(&self) {
        unsafe {
            // First, deallocate the contents based on type
            match &self.value {
                TributeValue::String(string_data) => {
                    string_data.deallocate();
                }
                TributeValue::Function(func) => {
                    // Release environment if it exists
                    if !func.env_ptr.is_null() {
                        (*func.env_ptr).release();
                    }
                }
                TributeValue::List(list_data) => {
                    list_data.release_all_elements();
                    list_data.deallocate();
                }
                TributeValue::Number(_) | TributeValue::Boolean(_) | TributeValue::Nil => {
                    // Number, Boolean, Nil don't need special cleanup
                }
            }

            // Finally, deallocate the boxed value itself
            let layout = Layout::new::<Self>();
            dealloc(self as *const Self as *mut u8, layout);
        }
    }
}

/// C-compatible functions for MLIR integration
/// Box a number value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_number(value: i64) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeValue::Number(value));
    boxed.as_ptr()
}


/// Box a boolean value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_boolean(value: bool) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeValue::Boolean(value));
    boxed.as_ptr()
}

/// Box a nil value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_nil() -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeValue::Nil);
    boxed.as_ptr()
}

/// Box a function value
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_box_function(
    code_ptr: *mut u8,
    env_ptr: *mut TributeBoxed,
) -> *mut TributeBoxed {
    let function = TributeFunction { code_ptr, env_ptr };

    // Retain the environment if it exists
    if !env_ptr.is_null() {
        unsafe {
            (*env_ptr).retain();
        }
    }

    let boxed = TributeBoxed::new(TributeValue::Function(function));
    boxed.as_ptr()
}


/// Unbox a number (with type checking)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_number(boxed: *mut TributeBoxed) -> i64 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        match &(*boxed).value {
            TributeValue::Number(value) => *value,
            _ => panic!("Type error: expected Number, got different type"),
        }
    }
}


/// Unbox a boolean
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_boolean(boxed: *mut TributeBoxed) -> bool {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        match &(*boxed).value {
            TributeValue::Boolean(value) => *value,
            _ => panic!("Type error: expected Boolean, got different type"),
        }
    }
}

/// Unbox a function (returns code pointer and environment pointer)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_function(
    boxed: *mut TributeBoxed,
    env_out: *mut *mut TributeBoxed,
) -> *mut u8 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        match &(*boxed).value {
            TributeValue::Function(function) => {
                if !env_out.is_null() {
                    *env_out = function.env_ptr;
                    // Retain the environment for the caller
                    if !function.env_ptr.is_null() {
                        (*function.env_ptr).retain();
                    }
                }
                function.code_ptr
            }
            _ => panic!("Type error: expected Function, got different type"),
        }
    }
}

/// Increment reference count
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_retain(boxed: *mut TributeBoxed) -> *mut TributeBoxed {
    unsafe {
        if !boxed.is_null() {
            (*boxed).retain();
        }
        boxed
    }
}

/// Decrement reference count
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_release(boxed: *mut TributeBoxed) {
    unsafe {
        if !boxed.is_null() {
            (*boxed).release();
        }
    }
}

/// Get current reference count (for debugging)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_get_ref_count(boxed: *mut TributeBoxed) -> u32 {
    unsafe {
        if boxed.is_null() {
            return 0;
        }
        (*boxed).ref_count.load(Ordering::Acquire)
    }
}

/// Check type of boxed value
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_get_type(boxed: *mut TributeBoxed) -> u32 {
    unsafe {
        if boxed.is_null() {
            return TributeValue::TYPE_NIL;
        }
        
        match &(*boxed).value {
            TributeValue::Number(_) => TributeValue::TYPE_NUMBER,
            TributeValue::Boolean(_) => TributeValue::TYPE_BOOLEAN,
            TributeValue::String(_) => TributeValue::TYPE_STRING,
            TributeValue::Function(_) => TributeValue::TYPE_FUNCTION,
            TributeValue::List(_) => TributeValue::TYPE_LIST,
            TributeValue::Nil => TributeValue::TYPE_NIL,
        }
    }
}

/// Arithmetic operations on boxed values
/// Add two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_add_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val + right_val;

        // Release input arguments (consumed by the operation)
        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Subtract two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_sub_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val - right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Multiply two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_mul_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val * right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Divide two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_div_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);

        if right_val == 0 {
            panic!("Division by zero");
        }

        let result = left_val / right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Compare two boxed numbers for equality
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_eq_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val == right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_boolean(result)
    }
}


/// Print a boxed value (for debugging)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_print_boxed(boxed: *mut TributeBoxed) {
    unsafe {
        if boxed.is_null() {
            println!("nil");
            return;
        }

        match &(*boxed).value {
            TributeValue::Number(n) => {
                println!("{}", n);
            }
            TributeValue::Boolean(b) => {
                println!("{}", b);
            }
            TributeValue::String(string) => {
                if !string.data.is_null() {
                    let slice = core::slice::from_raw_parts(string.data, string.length);
                    if let Ok(_s) = core::str::from_utf8(slice) {
                        println!("[string content]");
                    } else {
                        println!("[invalid UTF-8 string]");
                    }
                } else {
                    println!("[null string]");
                }
            }
            TributeValue::List(list_data) => {
                if list_data.data.is_null() {
                    println!("[]");
                } else {
                    print!("[");
                    for i in 0..list_data.len() {
                        if i > 0 {
                            print!(", ");
                        }
                        let element = list_data.get_unchecked(i);
                        if element.is_null() {
                            print!("nil");
                        } else {
                            // Recursively print element (without releasing it)
                            match &(*element).value {
                                TributeValue::Number(n) => print!("{}", n),
                                TributeValue::Boolean(b) => print!("{}", b),
                                TributeValue::String(_) => print!("[string]"),
                                TributeValue::Nil => print!("nil"),
                                _ => print!("[nested]"),
                            }
                        }
                    }
                    println!("]");
                }
            }
            TributeValue::Nil => {
                println!("nil");
            }
            TributeValue::Function(_) => {
                println!("[function]");
            }
        }
    }
}

// Uses standard library println! for output

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_unbox_number() {
        unsafe {
            let value = 42i64;

            // Box the number
            let boxed = tribute_box_number(value);
            assert!(!boxed.is_null());

            // Check type
            assert_eq!(tribute_get_type(boxed), TributeValue::TYPE_NUMBER);

            // Unbox the number
            let unboxed = tribute_unbox_number(boxed);
            assert_eq!(unboxed, value);

            // Clean up
            tribute_release(boxed);
        }
    }

    #[test]
    fn test_reference_counting() {
        unsafe {
            let boxed = tribute_box_number(42);
            
            // Initial ref count should be 1
            assert_eq!(tribute_get_ref_count(boxed), 1);
            
            // Retain should increment
            tribute_retain(boxed);
            assert_eq!(tribute_get_ref_count(boxed), 2);
            
            // Release should decrement
            tribute_release(boxed);
            assert_eq!(tribute_get_ref_count(boxed), 1);
            
            // Final release
            tribute_release(boxed);
        }
    }
}
